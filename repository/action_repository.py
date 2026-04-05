"""
ActionRepository – Sélecteur et exécuteur d'actions post-opération.

Rôle central du pipeline VisionAssist :
    Après chaque opération (STT, TTS, réponse serveur…), ce repository :
        1. Expose les actions disponibles pour cette opération
        2. Décrit chaque action en langage naturel
        3. Exécute la ou les actions sélectionnées
        4. Conserve un historique en mémoire
        5. Journalise chaque résultat

Actions disponibles par type d'opération :
    ┌──────────────────┬───────────────────────────────────────────────────────┐
    │ Source           │ Actions disponibles                                   │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ "stt"            │ persist_broker, persist_file, send_to_server,         │
    │                  │ pass_to_tts, broadcast_event, log_only, ignore        │
    │ "tts"            │ persist_broker, persist_file, broadcast_event,        │
    │                  │ log_only, ignore                                      │
    │ "server"         │ persist_broker, persist_file, pass_to_tts,            │
    │                  │ broadcast_event, log_only, ignore                     │
    └──────────────────┴───────────────────────────────────────────────────────┘

Exemple d'utilisation :
    repo = ActionRepository(broker_repo=broker_repository)

    # Après une transcription STT :
    repo.execute(
        source="stt",
        data=stt_result,
        actions=[
            PostOperationAction.PERSIST_BROKER,
            PostOperationAction.SEND_TO_SERVER,
        ],
    )

    # Consulter les actions disponibles :
    repo.describe_actions("stt")
    # → { "persist_broker": "Écrire dans le broker MQTT", ... }
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

import configuration as cfg
from broker.broker_repository import BrokerRepository
from repository.base_repository import IRepository, PostOperationAction

logger = logging.getLogger(__name__)

# Type de retour d'une action
ActionResult = Dict[str, Any]


class ActionRepository(IRepository):
    """
    Repository d'actions post-opération.

    Constructeur :
        broker_repo   – instance de BrokerRepository (obligatoire)
        tts_callback  – fn(text: str) appelée pour l'action PASS_TO_TTS

    Méthodes principales :
        execute(source, data, actions)     – exécute une liste d'actions
        get_available_actions(source)      – liste des actions disponibles
        describe_actions(source)           – descriptions lisibles (dict)
        find_all()                         – historique des actions
        find_by_id(id)                     – une entrée de l'historique
        save(data)                         – persistance directe JSONL
    """

    # ─── Actions disponibles par source ──────────────────────────────────
    AVAILABLE_ACTIONS: Dict[str, List[PostOperationAction]] = {
        "stt": [
            PostOperationAction.PERSIST_BROKER,
            PostOperationAction.PERSIST_FILE,
            PostOperationAction.SEND_TO_SERVER,
            PostOperationAction.PASS_TO_TTS,
            PostOperationAction.BROADCAST_EVENT,
            PostOperationAction.LOG_ONLY,
            PostOperationAction.IGNORE,
        ],
        "tts": [
            PostOperationAction.PERSIST_BROKER,
            PostOperationAction.PERSIST_FILE,
            PostOperationAction.BROADCAST_EVENT,
            PostOperationAction.LOG_ONLY,
            PostOperationAction.IGNORE,
        ],
        "server": [
            PostOperationAction.PERSIST_BROKER,
            PostOperationAction.PERSIST_FILE,
            PostOperationAction.PASS_TO_TTS,
            PostOperationAction.BROADCAST_EVENT,
            PostOperationAction.LOG_ONLY,
            PostOperationAction.IGNORE,
        ],
    }

    # ─── Descriptions lisibles ────────────────────────────────────────────
    ACTION_DESCRIPTIONS: Dict[PostOperationAction, str] = {
        PostOperationAction.PERSIST_BROKER:  "Écrire dans le broker MQTT",
        PostOperationAction.PERSIST_FILE:    "Écrire dans un fichier JSONL local (./data/)",
        PostOperationAction.SEND_TO_SERVER:  "Envoyer vers le serveur distant via MQTT",
        PostOperationAction.PASS_TO_TTS:     "Passer le texte au moteur TTS local",
        PostOperationAction.BROADCAST_EVENT: "Diffuser un événement système",
        PostOperationAction.LOG_ONLY:        "Journaliser uniquement (aucune persistance)",
        PostOperationAction.IGNORE:          "Ne rien faire",
    }

    def __init__(
        self,
        broker_repo: BrokerRepository,
        tts_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Args:
            broker_repo:  BrokerRepository pour les actions MQTT.
            tts_callback: Fonction fn(text) pour l'action PASS_TO_TTS.
        """
        self._broker_repo = broker_repo
        self._tts_callback = tts_callback
        self._history: List[Dict[str, Any]] = []
        os.makedirs(cfg.PERSIST_DIR, exist_ok=True)

    # ─── API principale ───────────────────────────────────────────────────

    def execute(
        self,
        source: str,
        data: Any,
        actions: List[PostOperationAction],
    ) -> List[ActionResult]:
        """
        Exécute une liste d'actions sur une donnée post-opération.

        Args:
            source:  Origine de l'opération : "stt" | "tts" | "server".
            data:    Résultat de l'opération (STTResult, TTSResult, dict…).
            actions: Actions à exécuter dans l'ordre.

        Returns:
            Liste de résultats : [{"action": str, "status": "ok"|"error", "detail": ...}]
        """
        data_dict = self._to_dict(data)
        results: List[ActionResult] = []

        for action in actions:
            try:
                detail = self._dispatch(source, data_dict, action)
                results.append({"action": action.value, "status": "ok", "detail": detail})
                logger.info("[%s] ✓ %s", source.upper(), action.value)
            except Exception as exc:
                results.append({"action": action.value, "status": "error", "detail": str(exc)})
                logger.error("[%s] ✗ %s — %s", source.upper(), action.value, exc)

        self._record_history(source, data_dict, actions, results)
        return results

    def get_available_actions(self, source: str) -> List[PostOperationAction]:
        """
        Retourne les actions disponibles pour une source donnée.

        Args:
            source: "stt" | "tts" | "server"
        """
        return self.AVAILABLE_ACTIONS.get(source, list(PostOperationAction))

    def describe_actions(self, source: str) -> Dict[str, str]:
        """
        Retourne les actions disponibles avec leurs descriptions lisibles.

        Args:
            source: "stt" | "tts" | "server"

        Returns:
            Dict { valeur_action: description_lisible }

        Exemple :
            {
              "persist_broker":  "Écrire dans le broker MQTT",
              "persist_file":    "Écrire dans un fichier JSONL local",
              ...
            }
        """
        available = self.get_available_actions(source)
        return {
            a.value: self.ACTION_DESCRIPTIONS[a]
            for a in available
        }

    # ─── IRepository ──────────────────────────────────────────────────────

    def save(self, data: Any, **kwargs) -> str:
        """Persiste directement dans le fichier d'événements général."""
        record = data if isinstance(data, dict) else {"data": str(data)}
        self._append_jsonl(cfg.PERSIST_EVENTS_FILE, record)
        return cfg.PERSIST_EVENTS_FILE

    def find_all(self) -> List[Dict[str, Any]]:
        """Retourne l'intégralité de l'historique des actions (en mémoire)."""
        return list(self._history)

    def find_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Recherche un enregistrement d'historique par son ID UUID."""
        return next(
            (h for h in self._history if h.get("id") == record_id),
            None,
        )

    # ─── Dispatch interne ─────────────────────────────────────────────────

    def _dispatch(
        self,
        source: str,
        data: Dict[str, Any],
        action: PostOperationAction,
    ) -> Any:
        """Aiguille vers l'implémentation de l'action."""
        if action == PostOperationAction.PERSIST_BROKER:
            return self._action_persist_broker(source, data)

        if action == PostOperationAction.PERSIST_FILE:
            return self._action_persist_file(source, data)

        if action == PostOperationAction.SEND_TO_SERVER:
            return self._action_send_to_server(source, data)

        if action == PostOperationAction.PASS_TO_TTS:
            return self._action_pass_to_tts(data)

        if action == PostOperationAction.BROADCAST_EVENT:
            return self._action_broadcast_event(source, data)

        if action == PostOperationAction.LOG_ONLY:
            logger.info(
                "[%s] LOG_ONLY : %s",
                source.upper(),
                json.dumps(data, ensure_ascii=False)[:200],
            )
            return "logged"

        if action == PostOperationAction.IGNORE:
            return "ignored"

        raise ValueError(f"Action non reconnue : {action!r}")

    # ─── Actions concrètes ────────────────────────────────────────────────

    def _action_persist_broker(self, source: str, data: Dict) -> str:
        """Publie sur le broker MQTT via BrokerRepository selon la source."""
        if source == "stt":
            msg = self._broker_repo.persist_stt_result(
                text=data.get("text", ""),
                language=data.get("language", cfg.STT_LANGUAGE or "fr"),
                confidence=data.get("language_probability"),
                audio_file=data.get("audio_file"),
            )
        elif source == "tts":
            msg = self._broker_repo.persist_tts_request(
                text=data.get("text", ""),
                output_file=data.get("output_file"),
            )
        else:
            # Source inconnue → publication générique
            msg = self._broker_repo.persist_raw(
                topic=cfg.TOPIC_PERSIST,
                data=data,
                source=source,
            )
        return msg.id

    def _action_persist_file(self, source: str, data: Dict) -> str:
        """Écrit une ligne JSON dans le fichier JSONL correspondant à la source."""
        _FILE_MAP = {
            "stt":    cfg.PERSIST_STT_FILE,
            "tts":    cfg.PERSIST_TTS_FILE,
            "server": cfg.PERSIST_SERVER_FILE,
        }
        target = _FILE_MAP.get(source, cfg.PERSIST_EVENTS_FILE)
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "source":    source,
            **data,
        }
        self._append_jsonl(target, record)
        logger.debug("Persisté dans %s", target)
        return target

    def _action_send_to_server(self, source: str, data: Dict) -> str:
        """Route le contenu vers le serveur distant via le broker."""
        content = data.get("text") or data
        msg = self._broker_repo.send_to_remote_server(
            content=content,
            action="process",
            source=source,
        )
        return msg.id

    def _action_pass_to_tts(self, data: Dict) -> str:
        """Passe le texte extrait vers le moteur TTS local."""
        text = data.get("text", "").strip()
        if not text:
            raise ValueError("Aucun texte à passer au TTS (champ 'text' vide).")
        if self._tts_callback:
            self._tts_callback(text)
            return f"TTS déclenché : {text[:60]}"
        logger.warning("PASS_TO_TTS : aucun callback TTS enregistré.")
        return "callback TTS absent"

    def _action_broadcast_event(self, source: str, data: Dict) -> str:
        """Diffuse un événement système sur TOPIC_EVENTS."""
        msg = self._broker_repo.broadcast_event(
            event_type=f"{source}_completed",
            details={"source": source, "summary": data.get("text", "")[:120]},
        )
        return msg.id

    # ─── Utilitaires ──────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(data: Any) -> Dict[str, Any]:
        """Convertit STTResult, TTSResult ou tout autre objet en dict."""
        if isinstance(data, dict):
            return data
        if hasattr(data, "__dataclass_fields__"):
            return asdict(data)
        return {"value": str(data)}

    def _append_jsonl(self, filepath: str, record: Dict) -> None:
        """Ajoute une ligne JSON à un fichier JSONL (crée le répertoire si besoin)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _record_history(
        self,
        source: str,
        data: Dict[str, Any],
        actions: List[PostOperationAction],
        results: List[ActionResult],
    ) -> None:
        """Enregistre l'exécution dans l'historique en mémoire."""
        self._history.append({
            "id":        str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "source":    source,
            "actions":   [a.value for a in actions],
            "results":   results,
            "data_preview": str(data.get("text", ""))[:120],
        })
