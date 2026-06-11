"""
Benchmark vLLM — Mode Parallèle (Batching simultané)
=====================================================
Différence clé vs benchmark_vllm.py (séquentiel) :
  - Toutes les requêtes sont soumises EN UNE SEULE fois via llm.chat(list_of_convs)
  - vLLM les traite toutes en parallèle grâce au PagedAttention + Dynamic Batching
  - Le KV cache est partagé efficacement entre toutes les requêtes
  - Cela correspond à la vraie charge de production (N utilisateurs simultanés)

HuggingFace n'a PAS de script équivalent car :
  - HF Transformers n'a pas de scheduler de batching dynamique
  - Chaque requête monopolise tout le GPU séquentiellement
  - N requêtes en parallèle = N fois la VRAM → OOM garanti sur 24GB avec 27B

Résultats sauvegardés dans : langage/VLLM/resultats_vllm_parallel.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# ─── Chemin projet ────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
)
logger = logging.getLogger("benchmark_vllm_parallel")

# ─── 10 Requêtes de test ──────────────────────────────────────────────────────
# Inclut des requêtes nécessitant des tools (météo, heure) et des requêtes directes
REQUETES_TEST = [
    "Qu'est-ce que l'intelligence artificielle ?",          # directe
    "Quel temps fait-il à Paris ?",                         # tool: get_weather
    "Bonjour, comment puis-je t'aider ?",                   # directe
    "Quelle heure est-il ?",                                # tool: get_current_time
    "Explique-moi ce qu'est un réseau de neurones en deux phrases.",  # directe
    "Donne-moi la météo à Lyon.",                           # tool: get_weather
    "Qu'est-ce que le machine learning ?",                  # directe
    "Quelle heure est-il exactement ?",                     # tool: get_current_time
    "Qu'est-ce que le deep learning ?",                     # directe
    "Donne-moi la météo à Bordeaux.",                       # tool: get_weather
]

SYSTEM_PROMPT = (
    "Tu es un assistant vocal en français. Réponds directement et brièvement en 1-2 phrases. "
    "NE génères AUCUN 'Thinking Process' ni aucun raisonnement interne. "
    "Donne UNIQUEMENT ta réponse finale."
)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "resultats_vllm_parallel.json")


# ─── Structures de données ────────────────────────────────────────────────────

@dataclass
class MesureRequete:
    texte: str
    reponse: str
    tokens_entree: int
    tokens_sortie_approx: int = 0
    tools_appeles: List[str] = field(default_factory=list)
    nb_tours: int = 1         # 1 = direct, 2 = tool call + réponse finale

@dataclass
class ResultatParallel:
    backend: str = "vllm_bitsandbytes_4bit_parallel"
    modele: str = ""
    quantification: str = "vLLM bitsandbytes 4-bit"
    nb_requetes: int = 0
    max_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 0.0
    temps_chargement_s: float = 0.0
    vram_modele_mb: float = 0.0
    # Temps globaux (wall-clock)
    temps_total_batch1_s: float = 0.0   # Premier batch (toutes les requêtes)
    temps_total_batch2_s: float = 0.0   # Deuxième batch (requêtes avec tool results)
    temps_total_s: float = 0.0          # Total wall-clock
    debit_requetes_par_s: float = 0.0   # nb_requetes / temps_total_s
    requetes: List[dict] = field(default_factory=list)


def vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# ─── Cycle Tool Calling parallèle ─────────────────────────────────────────────

def executer_tool(nom: str, args: dict, tools_map: dict) -> str:
    """Exécute un outil et retourne le résultat sous forme de chaîne."""
    if nom in tools_map:
        try:
            return str(tools_map[nom].invoke(args))
        except Exception as e:
            logger.error("[parallel] Erreur exécution tool %s : %s", nom, e)
            return f"Erreur outil {nom}: {e}"
    return f"Outil '{nom}' inconnu."


def executer_tools_en_parallele(
    tool_calls_par_requete: List[List[dict]],
    tools_map: dict,
) -> List[List[Tuple[str, str, str]]]:
    """
    Exécute tous les tool calls de toutes les requêtes en parallèle (threads I/O bound).
    Retourne une liste : pour chaque requête, la liste de (nom, tool_id, résultat).
    """
    # Aplatir tous les tool calls avec leur index de requête
    tasks: List[Tuple[int, str, dict, str]] = []  # (req_idx, nom, args, tool_id)
    for req_idx, tcs in enumerate(tool_calls_par_requete):
        for tc in tcs:
            tasks.append((req_idx, tc["name"], tc.get("args", {}), tc.get("id", "tc_unknown")))

    # Exécuter en parallèle
    resultats_par_requete: List[List[Tuple[str, str, str]]] = [[] for _ in tool_calls_par_requete]

    def _run(task: Tuple[int, str, dict, str]):
        req_idx, nom, args, tool_id = task
        logger.info("[parallel] Tool appelé (req#%d) : %s(%s)", req_idx + 1, nom, args)
        result = executer_tool(nom, args, tools_map)
        logger.info("[parallel] Résultat tool %s (req#%d) : %s", nom, req_idx + 1, str(result)[:80])
        return req_idx, nom, tool_id, result

    with ThreadPoolExecutor(max_workers=len(tasks) or 1) as executor:
        futures = [executor.submit(_run, t) for t in tasks]
        for future in futures:
            req_idx, nom, tool_id, result = future.result()
            resultats_par_requete[req_idx].append((nom, tool_id, result))

    return resultats_par_requete


def main() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from vllm import SamplingParams

    resultat = ResultatParallel(nb_requetes=len(REQUETES_TEST))

    # ── 1. Chargement du modèle ───────────────────────────────────────────────
    from configuration import CONFIGURATION
    resultat.modele = CONFIGURATION.qwen.model_id
    resultat.max_tokens = CONFIGURATION.qwen.max_new_tokens
    resultat.temperature = CONFIGURATION.qwen.temperature
    resultat.top_p = CONFIGURATION.qwen.top_p

    vram_avant = vram_mb()
    logger.info("[parallel] VRAM avant chargement : %.0f MB", vram_avant)
    logger.info("[parallel] Chargement du modèle vLLM...")

    t0 = time.time()
    from langage.VLLM.model_vllm import ModelServiceVLLM
    from langage.services.tools import get_all_tools
    service = ModelServiceVLLM()
    tools_disponibles = get_all_tools()

    temps_chargement = time.time() - t0
    vram_apres = vram_mb()
    resultat.temps_chargement_s = round(temps_chargement, 3)
    resultat.vram_modele_mb = round(vram_apres - vram_avant, 1)

    logger.info("[parallel] Modèle chargé en %.2f s", temps_chargement)

    # Paramètres de sampling
    sampling_params = SamplingParams(
        max_tokens=CONFIGURATION.qwen.max_new_tokens,
        temperature=CONFIGURATION.qwen.temperature,
        top_p=CONFIGURATION.qwen.top_p,
    )

    tools_map = {t.name: t for t in tools_disponibles}
    openai_tools = [convert_to_openai_tool(t) for t in tools_disponibles]

    # ── 2. Préparer toutes les conversations en format OpenAI ────────────────
    def build_conversation(texte: str) -> List[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": texte},
        ]

    conversations_initiales = [build_conversation(t) for t in REQUETES_TEST]

    # ── 3. BATCH 1 : Toutes les requêtes en une seule fois ───────────────────
    logger.info("[parallel] === BATCH 1 : %d requêtes envoyées simultanément ===", len(REQUETES_TEST))
    t_batch1 = time.time()

    outputs_batch1 = service.llm.chat(
        messages=conversations_initiales,
        sampling_params=sampling_params,
        tools=openai_tools,
    )

    duree_batch1 = time.time() - t_batch1
    resultat.temps_total_batch1_s = round(duree_batch1, 3)
    logger.info("[parallel] BATCH 1 terminé en %.2f s", duree_batch1)

    # ── 4. Parser les réponses du batch 1 ────────────────────────────────────
    from langage.VLLM.model_vllm import ModelServiceVLLM as MSVLLM
    reponses_batch1: List[AIMessage] = []
    tool_calls_par_requete: List[List[dict]] = []

    for i, output in enumerate(outputs_batch1):
        raw_text = output.outputs[0].text
        tc_list: List[dict] = []
        content = MSVLLM.nettoyer_reponse_llm_brute(raw_text, tc_list)
        msg = AIMessage(content=content, tool_calls=tc_list)
        reponses_batch1.append(msg)
        tool_calls_par_requete.append(tc_list)
        logger.info("[parallel] req#%d → %d tool_call(s) | contenu: %s",
                    i + 1, len(tc_list), content[:60] if content else "(tool call only)")

    # ── 5. Exécuter les tools EN PARALLÈLE (I/O bound) ───────────────────────
    requetes_avec_tool = [i for i, tcs in enumerate(tool_calls_par_requete) if tcs]
    resultats_tools = executer_tools_en_parallele(tool_calls_par_requete, tools_map)

    # ── 6. BATCH 2 : Requêtes nécessitant une 2ème passe ──────────────────────
    conversations_batch2: List[dict] = []  # (req_idx, conv)
    indices_batch2: List[int] = []

    for req_idx in requetes_avec_tool:
        conv = list(conversations_initiales[req_idx])  # copie
        rep1 = reponses_batch1[req_idx]

        # Ajouter le AIMessage avec tool_calls
        ai_msg_openai: Dict[str, Any] = {
            "role": "assistant",
            "content": rep1.content or "",
            "tool_calls": [
                {
                    "id": tc.get("id", "tc_" + str(uuid.uuid4())[:8]),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"]
                    }
                }
                for tc in rep1.tool_calls
            ]
        }
        conv.append(ai_msg_openai)

        # Ajouter les ToolMessages
        for nom, tool_id, tool_result in resultats_tools[req_idx]:
            conv.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result
            })

        conversations_batch2.append(conv)
        indices_batch2.append(req_idx)

    # Lancer le batch 2 si nécessaire
    duree_batch2 = 0.0
    reponses_batch2: Dict[int, AIMessage] = {}

    if conversations_batch2:
        logger.info("[parallel] === BATCH 2 : %d requêtes (avec résultats tools) ===",
                    len(conversations_batch2))
        t_batch2 = time.time()

        outputs_batch2 = service.llm.chat(
            messages=conversations_batch2,
            sampling_params=sampling_params,
            tools=openai_tools,
        )

        duree_batch2 = time.time() - t_batch2
        resultat.temps_total_batch2_s = round(duree_batch2, 3)
        logger.info("[parallel] BATCH 2 terminé en %.2f s", duree_batch2)

        for j, output in enumerate(outputs_batch2):
            req_idx = indices_batch2[j]
            raw_text = output.outputs[0].text
            tc_list: List[dict] = []
            content = MSVLLM.nettoyer_reponse_llm_brute(raw_text, tc_list)
            reponses_batch2[req_idx] = AIMessage(content=content, tool_calls=tc_list)
            logger.info("[parallel] req#%d (post-tool) → %s", req_idx + 1, content[:80])

    # ── 7. Consolider les résultats ───────────────────────────────────────────
    temps_total = duree_batch1 + duree_batch2
    resultat.temps_total_s = round(temps_total, 3)
    resultat.debit_requetes_par_s = round(len(REQUETES_TEST) / temps_total, 3)

    for i, texte in enumerate(REQUETES_TEST):
        # Réponse finale : batch2 si tool call, sinon batch1
        if i in reponses_batch2:
            rep_finale = reponses_batch2[i]
            nb_tours = 2
            tools_appeles = [tc["name"] for tc in tool_calls_par_requete[i]]
        else:
            rep_finale = reponses_batch1[i]
            nb_tours = 1
            tools_appeles = []

        mesure = MesureRequete(
            texte=texte,
            reponse=rep_finale.content,
            tokens_entree=len(SYSTEM_PROMPT + texte) // 4,
            tokens_sortie_approx=len(rep_finale.content) // 4,
            tools_appeles=tools_appeles,
            nb_tours=nb_tours,
        )
        resultat.requetes.append(asdict(mesure))

    # ── 8. Affichage résumé ───────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print(" RÉSULTATS — vLLM BitsAndBytes 4-bit — MODE PARALLÈLE")
    print(sep)
    print(f"  Modèle              : {resultat.modele}")
    print(f"  Quantification      : {resultat.quantification}")
    print(f"  Nb requêtes         : {resultat.nb_requetes} (simultanées)")
    print(f"  Temps chargement    : {resultat.temps_chargement_s:.2f} s")
    print(f"  BATCH 1 (toutes)    : {resultat.temps_total_batch1_s:.2f} s")
    print(f"  BATCH 2 (tools)     : {resultat.temps_total_batch2_s:.2f} s")
    print(f"  Temps TOTAL         : {resultat.temps_total_s:.2f} s")
    print(f"  Débit               : {resultat.debit_requetes_par_s:.2f} req/s")
    print(f"  Latence moy. / req  : {temps_total / len(REQUETES_TEST):.2f} s (estimée)")
    print()
    for r in resultat.requetes:
        flag = f" [TOOL: {', '.join(r['tools_appeles'])}]" if r["tools_appeles"] else ""
        print(f"  [{r['texte'][:55]}]")
        print(f"    Tours: {r['nb_tours']}{flag}")
        print(f"    Réponse: {r['reponse'][:75]}...")
    print(f"{sep}\n")

    # ── 9. Sauvegarde JSON ────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(asdict(resultat), f, ensure_ascii=False, indent=2)
    logger.info("[parallel] Résultats sauvegardés dans : %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
