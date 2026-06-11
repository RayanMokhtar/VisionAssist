"""
Benchmark : Qwen/Qwen3.5-9B (Natif fp16)
=========================================
Lance 10 requêtes de test en séquentiel avec le moteur complet de l'Agent.
Inclut l'injection des Outils (Tool Calling) et le Prompt de production.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List

import torch

# ─── Chemin projet ────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from langchain_core.messages import HumanMessage, SystemMessage
from langage.services.model import ModelService
from langage.services.agent import QwenAgent, BASE_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
)
logger = logging.getLogger("benchmark_qwen9b")

MODELE_ID = "Qwen/Qwen3.5-9B"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "resultats_qwen9b_natif.json")

REQUETES_TEST = [
    "Qu'est-ce que l'intelligence artificielle ?",
    "Quel temps fait-il à Paris ?",
    "Bonjour, comment puis-je t'aider ?",
    "Quelle heure est-il ?",
    "Explique-moi ce qu'est un réseau de neurones en deux phrases.",
    "Donne-moi la météo à Lyon.",
    "Qu'est-ce que le machine learning ?",
    "Quelle heure est-il exactement ?",
    "Qu'est-ce que le deep learning ?",
    "Donne-moi la météo à Bordeaux.",
]

@dataclass
class MesureRequete:
    texte: str
    temps_inference_s: float
    tokens_entree: int
    reponse: str
    tokens_sortie_approx: int = 0
    tools_appeles: List[str] = field(default_factory=list)
    nb_tours: int = 1

@dataclass
class ResultatBenchmark:
    backend: str = "huggingface"
    modele: str = MODELE_ID
    quantification: str = "aucune (fp16 natif)"
    dtype: str = "bfloat16"
    temps_chargement_s: float = 0.0
    vram_avant_chargement_mb: float = 0.0
    vram_apres_chargement_mb: float = 0.0
    vram_modele_mb: float = 0.0
    requetes: List[dict] = field(default_factory=list)
    temps_inference_moyen_s: float = 0.0
    temps_inference_min_s: float = 0.0
    temps_inference_max_s: float = 0.0
    temps_total_s: float = 0.0


def vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def _nettoyer_reponse(texte: str) -> str:
    import re
    texte = re.sub(r"<think>.*?</think>", "", texte, flags=re.DOTALL)
    return texte.strip()

def main() -> None:
    resultat = ResultatBenchmark()
    
    vram_avant = vram_mb()
    resultat.vram_avant_chargement_mb = round(vram_avant, 1)
    logger.info("VRAM avant chargement : %.0f MB", vram_avant)

    logger.info("Chargement de l'Agent et du modèle %s (fp16)...", MODELE_ID)
    t0 = time.time()
    
    # Instance du ModelService customisé pour fp16 natif
    service = ModelService(model_id=MODELE_ID, load_in_4bit=False)
    # L'Agent wrappe le service et orchestre LangGraph + Tools
    agent = QwenAgent(model_service=service)
    
    temps_chargement = time.time() - t0
    vram_apres = vram_mb()

    resultat.temps_chargement_s = round(temps_chargement, 3)
    resultat.vram_apres_chargement_mb = round(vram_apres, 1)
    resultat.vram_modele_mb = round(vram_apres - vram_avant, 1)

    logger.info("Modèle chargé en %.2f s | VRAM : %.0f MB (delta : %.0f MB)",
                temps_chargement, vram_apres, vram_apres - vram_avant)

    temps_inferences = []
    t_total_debut = time.time()

    for i, texte in enumerate(REQUETES_TEST, 1):
        logger.info("Requête %d/%d : %s", i, len(REQUETES_TEST), texte)
        
        t_inf = time.time()
        
        # Invoque le Graphe LangGraph exactement comme le fait chat_ui.py
        # Cela force le LLM à analyser s'il a besoin d'outils, puis les exécute.
        initial_state = {
            "messages": [SystemMessage(content=BASE_PROMPT), HumanMessage(content=texte)],
            "user_id": "benchmark_user",
            "session_id": "benchmark_session",
            "tool_calls_made": []
        }
        
        try:
            final_state = agent.graph.invoke(initial_state)
            last_message = final_state["messages"][-1]
            reponse_nette = _nettoyer_reponse(last_message.content)
            tools_utilises = final_state.get("tool_calls_made", [])
            # Un tour = LLM + Tool + LLM. On approxime en comptant les messages.
            nb_tours = len(final_state["messages"]) // 2
        except Exception as e:
            logger.error("Erreur agent: %s", e)
            reponse_nette = f"ERREUR: {e}"
            tools_utilises = []
            nb_tours = 1
            
        duree_inf = time.time() - t_inf
        tokens_sortie = len(reponse_nette) // 4
        tokens_entree = len(BASE_PROMPT + texte) // 4
        
        temps_inferences.append(duree_inf)
        
        mesure = MesureRequete(
            texte=texte,
            temps_inference_s=round(duree_inf, 3),
            tokens_entree=tokens_entree,
            reponse=reponse_nette,
            tokens_sortie_approx=tokens_sortie,
            tools_appeles=tools_utilises,
            nb_tours=nb_tours,
        )
        resultat.requetes.append(asdict(mesure))

        logger.info("-> %.2f s | Tools: %s | ~%d tokens → %s", duree_inf, tools_utilises, tokens_sortie, reponse_nette[:80])

    resultat.temps_total_s = round(time.time() - t_total_debut, 3)
    resultat.temps_inference_moyen_s = round(sum(temps_inferences) / len(temps_inferences), 3)
    resultat.temps_inference_min_s = round(min(temps_inferences), 3)
    resultat.temps_inference_max_s = round(max(temps_inferences), 3)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(asdict(resultat), f, ensure_ascii=False, indent=2)
    logger.info("Résultats sauvegardés dans : %s", OUTPUT_FILE)

if __name__ == "__main__":
    main()
