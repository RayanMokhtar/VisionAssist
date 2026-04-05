from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from typing import List, Optional

import configuration as cfg
from broker.mqtt_broker import MQTTBroker
from broker.broker_repository import BrokerRepository
from repository.action_repository import ActionRepository
from repository.base_repository import PostOperationAction
from speech.speech_to_text import SpeechToText, STTResult
from speech.text_to_speech import TextToSpeech, TTSResult


# ─── Journalisation ───────────────────────────────────────────────────────────

def setup_logging() -> None:
    """Configure la rotation de logs fichier + sortie console."""
    os.makedirs(os.path.dirname(os.path.abspath(cfg.LOG_FILE)), exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        cfg.LOG_FILE,
        maxBytes=cfg.LOG_MAX_BYTES,
        backupCount=cfg.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(name)s : %(message)s",
        handlers=[file_handler, console_handler],
    )


logger = logging.getLogger(__name__)


# ─── Application principale ───────────────────────────────────────────────────

class VisionAssistApp:
    """
    Orchestrateur VisionAssist.

    Responsabilités :
        • Démarrer / arrêter la connexion MQTT
        • Charger les modèles STT et TTS
        • Exécuter le pipeline STT → Actions → Serveur → TTS → Actions
        • Présenter le menu interactif de sélection d'actions
        • Relayer les réponses du serveur vers le TTS

    Utilisation rapide :
        app = VisionAssistApp()
        app.start()
        app.run_stt_from_microphone(duration=5)   # parler, transcrire, choisir actions
        app.stop()
    """

    def __init__(self) -> None:
        # ── Couche broker ──────────────────────────────────────────────────
        self._mqtt        = MQTTBroker()
        self._broker_repo = BrokerRepository(self._mqtt)

        # ── Couche speech ──────────────────────────────────────────────────
        self._stt = SpeechToText()
        self._tts = TextToSpeech()

        # ── Couche actions ─────────────────────────────────────────────────
        self._action_repo = ActionRepository(
            broker_repo=self._broker_repo,
            tts_callback=self._handle_tts,   # callback PASS_TO_TTS
        )

        # Abonnement aux réponses du serveur distant
        self._broker_repo.subscribe_server_responses(self._on_server_response)

    # ─── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Démarre l'application : connexion broker + chargement modèle STT."""
        logger.info("╔══ VisionAssist démarrage ═══════════════════════════╗")
        self._mqtt.connect()
        self._stt.load_model()
        logger.info("╚══ Prêt. ════════════════════════════════════════════╝")
        self._broker_repo.broadcast_event("app_started", {"version": "1.0.0"})

    def stop(self) -> None:
        """Arrêt propre : déconnexion broker."""
        self._broker_repo.broadcast_event("app_stopped")
        self._mqtt.disconnect()
        logger.info("VisionAssist arrêté.")

    # ─── Pipeline STT ─────────────────────────────────────────────────────

    def run_stt_from_file(
        self,
        audio_file: str,
        actions: Optional[List[PostOperationAction]] = None,
    ) -> STTResult:
        """
        Transcrit un fichier audio et propose le menu d'actions post-STT.

        Args:
            audio_file: Chemin vers le fichier audio (WAV, MP3, FLAC…).
            actions:    Actions à exécuter. Si None, affiche le menu interactif.

        Returns:
            STTResult de la transcription.
        """
        logger.info("STT depuis fichier : %s", audio_file)
        result = self._stt.transcribe_file(audio_file)
        self._run_post_stt(result, actions)
        return result

    def run_stt_from_microphone(
        self,
        duration: int = 5,
        actions: Optional[List[PostOperationAction]] = None,
    ) -> STTResult:
        """
        Enregistre depuis le microphone, transcrit, et propose les actions.

        Args:
            duration: Durée d'enregistrement en secondes.
            actions:  Actions à exécuter. Si None, affiche le menu interactif.

        Returns:
            STTResult de la transcription.
        """
        logger.info("STT depuis microphone (%ds)…", duration)
        result = self._stt.transcribe_stream(duration_seconds=duration)
        self._run_post_stt(result, actions)
        return result

    def _run_post_stt(
        self,
        result: STTResult,
        actions: Optional[List[PostOperationAction]],
    ) -> None:
        """Affiche le résultat STT, puis exécute les actions choisies."""
        print(f"\n{'─'*56}")
        print(f"  Transcription : {result.text}")
        print(f"  Langue        : {result.language} ({result.language_probability:.0%})")
        print(f"  Durée traitement : {result.duration_ms} ms")
        print(f"{'─'*56}")

        chosen = actions if actions is not None else self._choose_actions("stt")
        self._action_repo.execute("stt", result, chosen)

    # ─── Pipeline TTS ─────────────────────────────────────────────────────

    def run_tts(
        self,
        text: str,
        play_audio: bool = True,
        actions: Optional[List[PostOperationAction]] = None,
    ) -> TTSResult:
        """
        Synthétise un texte en parole, lit l'audio, et propose les actions.

        Args:
            text:       Texte à synthétiser.
            play_audio: Lit le fichier audio après synthèse.
            actions:    Actions à exécuter. Si None, affiche le menu interactif.

        Returns:
            TTSResult de la synthèse.
        """
        logger.info("TTS : %s…", text[:60])
        result = self._tts.synthesize(text)

        if play_audio:
            try:
                self._tts.play(result.output_file)
            except Exception as exc:
                logger.warning("Lecture audio impossible : %s", exc)

        print(f"\n{'─'*56}")
        print(f"  TTS synthétisé : {result.text[:60]}")
        print(f"  Fichier        : {result.output_file}")
        print(f"  Durée          : {result.duration_ms} ms")
        print(f"{'─'*56}")

        chosen = actions if actions is not None else self._choose_actions("tts")
        self._action_repo.execute("tts", result, chosen)
        return result

    # ─── Réponse du serveur distant ───────────────────────────────────────

    def _on_server_response(self, topic: str, payload: Any) -> None:
        """
        Callback déclenché à la réception d'une réponse MQTT du serveur.

        Extrait le texte de réponse, l'exécute via les actions par défaut
        (persist + TTS), puis propose un menu pour actions complémentaires.
        """
        logger.info("← Réponse serveur reçue sur [%s]", topic)

        try:
            content = payload.get("payload", {}) if isinstance(payload, dict) else {}
            response_text = (
                content.get("content", "")
                if isinstance(content, dict)
                else str(content)
            )
        except Exception:
            response_text = str(payload)

        # Actions automatiques pour une réponse serveur
        auto_actions = [
            PostOperationAction.PERSIST_BROKER,
            PostOperationAction.PERSIST_FILE,
        ]
        self._action_repo.execute(
            "server",
            {"text": response_text, "_raw": str(payload)[:200]},
            auto_actions,
        )

        # Synthèse vocale de la réponse si du texte est disponible
        if response_text:
            self.run_tts(response_text, play_audio=True, actions=[
                PostOperationAction.PERSIST_BROKER,
            ])

    def _handle_tts(self, text: str) -> None:
        """Callback utilisé pour l'action PASS_TO_TTS depuis ActionRepository."""
        self.run_tts(text, play_audio=True, actions=[PostOperationAction.PERSIST_BROKER])

    # ─── Sélecteur interactif d'actions ───────────────────────────────────

    def _choose_actions(self, source: str) -> List[PostOperationAction]:
        """
        Affiche un menu numéroté des actions disponibles et retourne
        la sélection de l'utilisateur.

        Saisie :
            « 1,3 »   → actions 1 et 3
            « 0 »     → IGNORE (ne rien faire)
            Entrée    → LOG_ONLY par défaut
        """
        available = self._action_repo.describe_actions(source)
        items = list(available.items())

        print(f"\n┌── Actions après {source.upper()} {'─'*(38 - len(source))}")
        for i, (key, desc) in enumerate(items, 1):
            print(f"│  [{i}] {key:<22} {desc}")
        print("│  [0] Ignorer (ne rien faire)")
        print(f"└{'─'*56}")

        try:
            raw = input("  Votre choix (ex: 1,2 | 0 pour ignorer) : ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return [PostOperationAction.LOG_ONLY]

        if not raw or raw == "0":
            return [PostOperationAction.IGNORE]

        selected: List[PostOperationAction] = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(items):
                    selected.append(PostOperationAction(items[idx][0]))
                else:
                    print(f"  [!] Index {part} hors limites, ignoré.")
            else:
                try:
                    selected.append(PostOperationAction(part))
                except ValueError:
                    print(f"  [!] Action inconnue '{part}', ignorée.")

        return selected or [PostOperationAction.LOG_ONLY]

    # ─── Menu principal interactif ────────────────────────────────────────

    def run_interactive(self) -> None:
        """
        Boucle interactive principale — menu de sélection du mode.

        Options :
            1. STT depuis microphone
            2. STT depuis fichier
            3. TTS direct
            4. Afficher les topics broker
            5. Afficher l'historique des actions
            0. Quitter
        """
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║           VisionAssist – Menu Principal              ║")
        print("╚══════════════════════════════════════════════════════╝")

        while True:
            print("\n  [1] STT depuis microphone")
            print("  [2] STT depuis fichier audio")
            print("  [3] Synthèse TTS directe")
            print("  [4] Afficher les topics MQTT")
            print("  [5] Historique des actions")
            print("  [0] Quitter")

            try:
                choice = input("\n  Choix : ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == "1":
                try:
                    dur = int(input("  Durée enregistrement (secondes, défaut=5) : ").strip() or "5")
                except ValueError:
                    dur = 5
                self.run_stt_from_microphone(duration=dur)

            elif choice == "2":
                path = input("  Chemin du fichier audio : ").strip()
                if os.path.exists(path):
                    self.run_stt_from_file(path)
                else:
                    print(f"  [!] Fichier introuvable : {path}")

            elif choice == "3":
                text = input("  Texte à synthétiser : ").strip()
                if text:
                    self.run_tts(text)
                else:
                    print("  [!] Texte vide.")

            elif choice == "4":
                print("\n  Topics MQTT configurés :")
                for topic, desc in BrokerRepository.list_topics().items():
                    print(f"    {topic:<42} {desc}")

            elif choice == "5":
                history = self._action_repo.find_all()
                if not history:
                    print("  Aucune action dans l'historique.")
                else:
                    print(f"\n  {len(history)} entrée(s) d'historique :")
                    for entry in history[-10:]:   # affiche les 10 dernières
                        print(
                            f"  [{entry['timestamp']}] {entry['source'].upper()} "
                            f"→ {', '.join(entry['actions'])} "
                            f"| {entry.get('data_preview', '')[:40]}"
                        )

            elif choice == "0":
                break
            else:
                print("  [!] Choix invalide.")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    app = VisionAssistApp()
    try:
        app.start()
        app.run_interactive()
    except KeyboardInterrupt:
        logger.info("Interruption clavier.")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
