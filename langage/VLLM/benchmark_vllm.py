"""
Benchmark vLLM
==============
Lance ce script SEUL dans un processus dédié :
    python langage/VLLM/benchmark_vllm.py

Résultats sauvegardés dans : langage/VLLM/resultats_vllm.json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
)
logger = logging.getLogger("benchmark_vllm")

# ─── Requêtes de test communes aux deux benchmarks ───────────────────────────
# IMPORTANT : Ces requêtes sont identiques à celles de benchmark_hf.py
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

SYSTEM_PROMPT = (
    "Tu es un assistant vocal en français. Réponds directement et brièvement en 1-2 phrases. "
    "NE génère AUCUN 'Thinking Process' ni aucun raisonnement interne. "
    "Donne UNIQUEMENT ta réponse finale."
)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "resultats_vllm.json")


@dataclass
class MesureRequete:
    texte: str
    temps_inference_s: float   # temps total (inclut exécution tools + 2ème appel)
    tokens_entree: int
    reponse: str
    tokens_sortie_approx: int = 0
    tools_appeles: List[str] = field(default_factory=list)
    nb_tours: int = 1


@dataclass
class ResultatVLLM:
    backend: str = "vllm_bitsandbytes_4bit"
    modele: str = ""
    quantification: str = "vLLM bitsandbytes 4-bit"
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    max_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 0.0
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


def main() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    resultat = ResultatVLLM()

    # ── 1. Chargement de la configuration ─────────────────────────────────────
    from configuration import CONFIGURATION
    resultat.modele = CONFIGURATION.qwen.model_id
    resultat.max_tokens = CONFIGURATION.qwen.max_new_tokens
    resultat.temperature = CONFIGURATION.qwen.temperature
    resultat.top_p = CONFIGURATION.qwen.top_p

    # ── 2. Chargement du modèle vLLM ──────────────────────────────────────────
    vram_avant = vram_mb()
    resultat.vram_avant_chargement_mb = vram_avant
    logger.info("[vLLM] VRAM avant chargement : %.0f MB", vram_avant)
    logger.info("[vLLM] Chargement du modèle via vLLM Python API...")

    t0 = time.time()

    # Import ici pour ne pas déclencher le singleton HF
    from langage.VLLM.model_vllm import ModelServiceVLLM
    from langage.services.tools import get_all_tools
    service = ModelServiceVLLM()
    tools_disponibles = get_all_tools()

    temps_chargement = time.time() - t0
    vram_apres = vram_mb()

    resultat.temps_chargement_s = round(temps_chargement, 3)
    resultat.vram_apres_chargement_mb = round(vram_apres, 1)
    resultat.vram_modele_mb = round(vram_apres - vram_avant, 1)

    logger.info("[vLLM] Modèle chargé en %.2f s | VRAM : %.0f MB (delta : %.0f MB)",
                temps_chargement, vram_apres, vram_apres - vram_avant)

    tools_map = {t.name: t for t in tools_disponibles}

    # ── 3. Inférence sur chaque requête ───────────────────────────────────────
    temps_inferences = []
    t_total_debut = time.time()

    for i, texte in enumerate(REQUETES_TEST, 1):
        logger.info("[vLLM] Requête %d/%d : %s", i, len(REQUETES_TEST), texte)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=texte),
        ]

        tokens_entree_approx = len(SYSTEM_PROMPT + texte) // 4
        tools_appeles: List[str] = []
        nb_tours = 0

        t_inf = time.time()

        # ─ Tour 1 : Premier appel modèle ─
        reponse = service.generer_reponse_pour_savoir_si_tool_necessaire_ou_pas(messages, tools=tools_disponibles)
        nb_tours += 1

        # ─ Tour 2 (optionnel) : Si tool_calls présents → exécuter + rappeler le modèle ─
        if reponse.tool_calls:
            for tc in reponse.tool_calls:
                nom = tc["name"]
                args = tc.get("args", {})
                tool_id = tc.get("id", "tc_unknown")
                tools_appeles.append(nom)
                logger.info("[vLLM] Tool appelé : %s(%s)", nom, args)

                if nom in tools_map:
                    try:
                        resultat_tool = tools_map[nom].invoke(args)
                    except Exception as e:
                        resultat_tool = f"Erreur outil {nom}: {e}"
                        logger.error("[vLLM] Erreur exécution tool %s : %s", nom, e)
                else:
                    resultat_tool = f"Outil '{nom}' inconnu."
                    logger.warning("[vLLM] Outil inconnu : %s", nom)

                logger.info("[vLLM] Résultat tool %s : %s", nom, str(resultat_tool)[:80])
                messages = messages + [reponse, ToolMessage(content=str(resultat_tool), tool_call_id=tool_id)]

            reponse = service.generer_reponse_pour_savoir_si_tool_necessaire_ou_pas(messages, tools=tools_disponibles)
            nb_tours += 1

        duree_inf = time.time() - t_inf
        tokens_sortie_approx = len(reponse.content) // 4
        temps_inferences.append(duree_inf)

        mesure = MesureRequete(
            texte=texte,
            temps_inference_s=round(duree_inf, 3),
            tokens_entree=tokens_entree_approx,
            reponse=reponse.content,
            tokens_sortie_approx=tokens_sortie_approx,
            tools_appeles=tools_appeles,
            nb_tours=nb_tours,
        )
        resultat.requetes.append(asdict(mesure))

        flag_tool = f" [TOOLS: {', '.join(tools_appeles)}]" if tools_appeles else ""
        logger.info("[vLLM] -> %.2f s | %d tour(s)%s | ~%d tokens → %s",
                    duree_inf, nb_tours, flag_tool, tokens_sortie_approx, reponse.content[:80])


    # ── 4. Statistiques ───────────────────────────────────────────────────────
    resultat.temps_total_s = round(time.time() - t_total_debut, 3)
    resultat.temps_inference_moyen_s = round(sum(temps_inferences) / len(temps_inferences), 3)
    resultat.temps_inference_min_s = round(min(temps_inferences), 3)
    resultat.temps_inference_max_s = round(max(temps_inferences), 3)

    # ── 5. Affichage résumé ───────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(" RÉSULTATS — vLLM bitsandbytes 4-bit")
    print(sep)
    print(f"  Modèle              : {resultat.modele}")
    print(f"  Quantification      : {resultat.quantification}")
    print(f"  Temps chargement    : {resultat.temps_chargement_s:.2f} s")
    print(f"  VRAM modèle         : {resultat.vram_modele_mb:.0f} MB")
    print(f"  Temps TOTAL (séqu.) : {resultat.temps_total_s:.2f} s")
    print(f"  Inférence moyenne   : {resultat.temps_inference_moyen_s:.2f} s")
    print(f"  Inférence min/max   : {resultat.temps_inference_min_s:.2f} / {resultat.temps_inference_max_s:.2f} s")
    print()
    for r in resultat.requetes:
        print(f"  [{r['texte'][:50]}]")
        print(f"    Durée : {r['temps_inference_s']:.2f} s  | Réponse : {r['reponse'][:70]}...")
    print(f"{sep}\n")

    # ── 6. Sauvegarde JSON ────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(asdict(resultat), f, ensure_ascii=False, indent=2)
    logger.info("[vLLM] Résultats sauvegardés dans : %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
