# VisionAssist – Jetson IA Locale

Pipeline vocal local sur NVIDIA Jetson : **Speech-to-Text → Broker MQTT → Serveur distant → Text-to-Speech**, avec persistance et sélection d'actions à chaque étape.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       VisionAssist (Jetson)                       │
│                                                                    │
│  Micro  ──►  SpeechToText   ──►  ActionRepository  ──►  Broker   │
│              (faster-whisper)     (choix d'actions)    (MQTT)     │
│                                        │                          │
│                                   send_to_server                  │
│                                        │                          │
│                              Serveur distant (HTTP/MQTT)          │
│                                        │                          │
│  Broker  ◄──  ActionRepository  ◄──  Réponse                     │
│                    │                                              │
│                  pass_to_tts                                      │
│                    │                                              │
│              TextToSpeech  ──►  Haut-parleur                     │
│              (piper-tts)                                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Modèles recommandés

### STT – faster-whisper (CTranslate2)
| Modèle     | Taille  | Recommandé pour                          |
|------------|---------|------------------------------------------|
| `tiny`     | 39 MB   | Tests, Jetson Nano CPU                   |
| `base`     | 74 MB   | Jetson Nano, usage léger                 |
| `small` ★  | 244 MB  | **Meilleur équilibre** – tous Jetson     |
| `medium`   | 769 MB  | Haute précision – Jetson Xavier / Orin   |
| `large-v3` | 1.5 GB  | Précision maximale – Jetson Orin 32 GB   |

Configurer dans `configuration.py` : `STT_MODEL_NAME`, `STT_DEVICE`, `STT_COMPUTE_TYPE`

### TTS – piper-tts (ONNX)
| Voix                    | Description                       |
|-------------------------|-----------------------------------|
| `fr_FR-siwis-medium` ★  | Féminine, naturelle, fluide       |
| `fr_FR-upmc-medium`     | Masculine, neutre, professionnelle|
| `fr_FR-tom-medium`      | Masculine, expressive             |

Téléchargement des voix :
```bash
mkdir -p models/tts
wget -O models/tts/fr_FR-siwis-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx
wget -O models/tts/fr_FR-siwis-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json
```

---

## Structure du projet

```
VisionAssist/
├── configuration.py            ← Config centrale (modèles, broker, topics…)
├── main.py                     ← Orchestrateur + menu interactif
├── requirements.txt
│
├── broker/
│   ├── broker_interface.py     ← Interface abstraite IBroker
│   ├── mqtt_broker.py          ← Implémentation MQTT (paho)
│   └── broker_repository.py   ← Opérations métier haut niveau
│
├── speech/
│   ├── speech_to_text.py       ← STT via faster-whisper
│   └── text_to_speech.py       ← TTS via piper-tts
│
├── repository/
│   ├── base_repository.py      ← Interface IRepository + PostOperationAction
│   └── action_repository.py   ← Sélecteur & exécuteur d'actions post-op
│
├── models/tts/                 ← Voix piper (.onnx + .onnx.json)
├── audio_output/               ← Fichiers WAV générés par TTS
├── data/                       ← Persistance JSONL locale
│   ├── stt_history.jsonl
│   ├── tts_history.jsonl
│   ├── server_exchanges.jsonl
│   └── events.jsonl
└── logs/
    └── visionassist.log
```

---

## Actions post-opération (ActionRepository)

Après chaque opération (STT, TTS, réponse serveur), un menu interactif propose :

| Action            | Comportement                                       |
|-------------------|----------------------------------------------------|
| `persist_broker`  | Publie dans le broker MQTT                         |
| `persist_file`    | Écrit dans un fichier JSONL local (`./data/`)      |
| `send_to_server`  | Route vers le serveur distant via MQTT             |
| `pass_to_tts`     | Envoie le texte au moteur TTS local                |
| `broadcast_event` | Diffuse un événement système                       |
| `log_only`        | Journalise uniquement                              |
| `ignore`          | Ne rien faire                                      |

---

## Topics MQTT

| Topic                           | Description                          |
|---------------------------------|--------------------------------------|
| `visionassist/stt/result`       | Résultats de transcription STT       |
| `visionassist/tts/input`        | Demandes / résultats TTS             |
| `visionassist/server/request`   | Requêtes vers le serveur distant     |
| `visionassist/server/response`  | Réponses du serveur distant          |
| `visionassist/events`           | Événements système (supervision)     |
| `visionassist/persist`          | Persistance générale                 |
| `visionassist/errors`           | Erreurs et alertes                   |

---

## Installation

```bash
# 1. Broker MQTT local (Mosquitto)
sudo apt install mosquitto mosquitto-clients
sudo systemctl enable --now mosquitto

# 2. Dépendances Python
pip install -r requirements.txt

# Sur Jetson avec CUDA :
pip install faster-whisper[cuda]

# 3. Dépendances système (capture audio)
sudo apt install python3-pyaudio portaudio19-dev espeak-ng

# 4. Lancement
python main.py
```

## Configuration rapide

Toutes les variables sont dans `configuration.py` et surchargeable via `.env` :

```env
STT_MODEL_NAME=small
STT_DEVICE=cuda
BROKER_HOST=localhost
BROKER_PORT=1883
REMOTE_SERVER_URL=http://192.168.1.100:8080
TTS_ENGINE=piper
```
Projet visant à améliorer la perception de l'environnement des personnes malvoyantes combinant embarqué et intelligence artificielle

liens utiles : 

https://plmlatex.math.cnrs.fr/6657316991jrgdvcxtmtxy : latex rapport

https://www.canva.com/design/DAG7Y4vhYag/kMWcCppjId9w26r_Cp3d2w/edit : lien canvas architecture globale







documentation : 

https://medium.com/@pinaki.brahma/improve-llm-based-response-through-parent-child-retrieval-strategy-part-2-04ef7982b9a3 
