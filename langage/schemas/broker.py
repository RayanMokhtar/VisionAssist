"""
Schémas des messages échangés via le broker MQTT.

BrokerRequest  – requête entrante (depuis STT ou vision via MQTT)
AgentResponse  – réponse de l'agent (publiée sur results/tts via MQTT)
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from security.authentification_client import SessionAuthentifiee

class BrokerRequest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    request_id: str
    session_id: Optional[str] = ""
    user_id: Optional[str] = "default"
    text: str
    image_url: Optional[str] = None
    session_authentifiee : Optional[SessionAuthentifiee] = None


class AgentResponse(BaseModel):
    request_id: str
    session_id: Optional[str] = ""
    user_id: str = "default"
    response: Optional[str] = None
    tool_calls_made: List[dict] = Field(default_factory=list)
    error: Optional[str] = None
