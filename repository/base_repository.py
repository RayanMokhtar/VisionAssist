"""
BaseRepository – Contrats abstraits et énumération des actions post-opération.

Ce module définit :
    PostOperationAction  – toutes les actions disponibles après STT / TTS / Serveur
    IRepository          – interface générique pour tout repository de persistance
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional


class PostOperationAction(str, Enum):
    """
    Actions disponibles à la fin de chaque opération.

    Sélectionnables via ActionRepository.execute() ou via l'interface
    interactive de VisionAssistApp._choose_actions().

    ┌────────────────────┬────────────────────────────────────────────────────┐
    │ Action             │ Comportement                                       │
    ├────────────────────┼────────────────────────────────────────────────────┤
    │ PERSIST_BROKER     │ Publie le résultat dans le broker MQTT             │
    │ PERSIST_FILE       │ Écrit dans un fichier JSONL local (./data/)        │
    │ SEND_TO_SERVER     │ Route vers le serveur distant via MQTT             │
    │ PASS_TO_TTS        │ Envoie le texte au moteur TTS local                │
    │ BROADCAST_EVENT    │ Diffuse un événement système sur TOPIC_EVENTS      │
    │ LOG_ONLY           │ Journalise uniquement (aucune persistance)         │
    │ IGNORE             │ Ne fait rien                                       │
    └────────────────────┴────────────────────────────────────────────────────┘
    """
    PERSIST_BROKER   = "persist_broker"
    PERSIST_FILE     = "persist_file"
    SEND_TO_SERVER   = "send_to_server"
    PASS_TO_TTS      = "pass_to_tts"
    BROADCAST_EVENT  = "broadcast_event"
    LOG_ONLY         = "log_only"
    IGNORE           = "ignore"


class IRepository(ABC):
    """
    Interface générique pour tout repository de persistance.

    Méthodes obligatoires :
        save(data)          – persiste une donnée
        find_all()          – récupère toutes les entrées
        find_by_id(id)      – récupère une entrée par son identifiant
    """

    @abstractmethod
    def save(self, data: Any, **kwargs) -> Any:
        """Persiste une donnée et retourne le résultat (ex: id, chemin…)."""

    @abstractmethod
    def find_all(self) -> List[Any]:
        """Récupère toutes les entrées disponibles."""

    @abstractmethod
    def find_by_id(self, record_id: str) -> Optional[Any]:
        """Récupère une entrée par son identifiant unique."""
