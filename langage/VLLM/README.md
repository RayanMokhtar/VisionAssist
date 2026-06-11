# VLLM — Scripts de benchmark

Ce dossier contient deux scripts pour comparer les moteurs d'inférence **HuggingFace Transformers** vs **vLLM** sur le même modèle Qwen.

---

## Fichiers

| Fichier | Rôle |
|---|---|
| `model_vllm.py` | Service vLLM — interface identique à `langage/services/model.py` |
| `benchmark.py` | Script de comparatif automatisé |

---

## Paramètres comparés (identiques dans les deux backends)

| Paramètre | Valeur | Fichier source |
|---|---|---|
| Modèle | `./langage/modeles/Qwen3.5-27B` | `configuration.py` |
| Quantification | **4-bit** | BitsAndBytes (HF) / vLLM bitsandbytes |
| `max_tokens` | **500** | `configuration.py` |
| `temperature` | **0.7** | `configuration.py` |
| `top_p` | **0.8** | `configuration.py` |
| dtype | `bfloat16` | Détecté automatiquement |

---

## Différences architecturales à noter

| Aspect | HuggingFace Transformers | vLLM |
|---|---|---|
| Quantification 4-bit | BitsAndBytes (`bnb_4bit_quant_type="fp4"`) | Moteur interne vLLM |
| Gestion mémoire | Allocation PyTorch standard | **PagedAttention** (KV-cache paginée) |
| Batching | Séquentiel (1 requête à la fois) | Continuous batching |
| Attention | `attn_implementation="sdpa"` | Flash Attention 2 (si dispo) |
| `torch.no_grad()` | Requis manuellement | Géré en interne |

---

## Usage

### Tester uniquement vLLM

```bash
cd /home/etudiant_adm/visionAssist/VisionAssist
source .venv/bin/activate
python langage/VLLM/benchmark.py --backend vllm
```

### Tester uniquement HuggingFace

```bash
python langage/VLLM/benchmark.py --backend hf
```

### Comparatif complet (HF puis vLLM)

> ⚠️ Les deux modèles ne peuvent pas être chargés simultanément (24GB VRAM). Le script vide la VRAM entre les deux tests.

```bash
python langage/VLLM/benchmark.py --backend both --output resultats.json
```

### Sauvegarder les résultats

```bash
python langage/VLLM/benchmark.py --backend both --output langage/VLLM/resultats_benchmark.json
```

---

## Installation de vLLM (si nécessaire)

```bash
pip install vllm
# ou
uv add vllm
```

> vLLM requiert CUDA >= 11.8 et Python >= 3.9.
