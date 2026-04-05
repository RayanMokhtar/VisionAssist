"""
IBroker – Interface abstraite pour tout broker de messages.

Toute implémentation concrète (MQTT, Redis Pub/Sub, RabbitMQ…)
doit respecter ce contrat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class IBroker(ABC):

    @abstractmethod
    def connect(self) -> None:

    @abstractmethod
    def disconnect(self) -> None:
        """Ferme proprement la connexion."""

    @abstractmethod
    def publish(
        self,
        topic: str,
        payload: Any,
        qos: int = 1,
        retain: bool = False,
    ) -> bool:
        """
        Publie un message sur un topic.

        Args:
            topic:   Chemin du topic (ex: "visionassist/stt/result").
            payload: Données à envoyer. Un dict est sérialisé en JSON.
            qos:     Niveau de qualité de service (0, 1 ou 2).
            retain:  Si True, le broker conserve le dernier message du topic.

        Returns:
            True si la publication a réussi, False sinon.
        """

    @abstractmethod
    def subscribe(
        self,
        topic: str,
        callback: Callable[[str, Any], None],
        qos: int = 1,
    ) -> None:
        """
        S'abonne à un topic et appelle callback à chaque message reçu.

        Args:
            topic:    Topic ou pattern (ex: "visionassist/#").
            callback: Fonction (topic: str, payload: Any) → None.
            qos:      Niveau de qualité de service.
        """

    @abstractmethod
    def unsubscribe(self, topic: str) -> None:
        """Se désabonne d'un topic."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Retourne True si la connexion au broker est active."""
