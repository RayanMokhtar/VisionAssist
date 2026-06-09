"""
Schémas des messages échangés via le broker MQTT.

BrokerRequest  – requête entrante (depuis STT ou vision via MQTT)
AgentResponse  – réponse de l'agent (publiée sur results/tts via MQTT)
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BrokerRequest(BaseModel):
    """Requête entrante reçue par l'agent via MQTT.

    Attributes:
        request_id:  Identifiant unique de la requête.
        session_id:  (Legacy) Identifiant de session explicite. Ignoré si user_id est fourni.
        user_id:     Identifiant de l'utilisateur. Utilisé pour résoudre la session du jour.
        text:        Texte transcrit (STT) ou description (vision).
        image_url:   Chemin ou URL vers une image optionnelle.
        latitude:    Latitude GPS du client (rafraîchie à chaque requête). Défaut: Paris.
        longitude:   Longitude GPS du client (rafraîchie à chaque requête). Défaut: Paris.
    """
    request_id: str
    session_id: str = ""
    user_id: str = "default"
    text: str
    image_url: Optional[str] = None
    latitude: Optional[float] = None   # None = pas de GPS fourni → tool utilisera Paris par défaut
    longitude: Optional[float] = None


class AgentResponse(BaseModel):
    """Réponse produite par l'agent après traitement.

    Attributes:
        request_id:       Identifiant de la requête originale.
        session_id:       Identifiant de la session du jour.
        user_id:          Identifiant de l'utilisateur.
        response:         Texte de réponse final (sera envoyé au TTS).
        tool_calls_made:  Noms des outils appelés durant le traitement.
        error:            Message d'erreur éventuel.
    """
    request_id: str
    session_id: str = ""
    user_id: str = "default"
    response: Optional[str] = None
    tool_calls_made: List[str] = Field(default_factory=list)
    error: Optional[str] = None
