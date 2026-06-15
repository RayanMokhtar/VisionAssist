"""
Mémoire court terme — Buffer de conversation de la session du jour.

Charge les messages de la session courante depuis MySQL et les convertit
en objets LangChain (HumanMessage, AIMessage, SystemMessage, ToolMessage)
pour injection dans le contexte du LLM.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from persistance.models import Message
from persistance.repository import REPOSITORIES
from configuration import CONFIGURATION

logger = logging.getLogger(__name__)


class ConversationBuffer:
    """Buffer de conversation qui charge le contexte d'une session depuis MySQL.

    Attributes:
        session_id:    ID de la session (unique par user + date).
        max_messages:  Nombre max de messages dans le buffer.
    """

    def __init__(
        self,
        session_id: str,
    ):
        self.session_id = session_id
        self.max_messages = CONFIGURATION.qwen.max_messages_en_memoire 

    def get_langchain_messages(self) -> List[BaseMessage]:
        """Charge les messages et les convertit en objets LangChain.

        Returns:
            Liste de BaseMessage prête à être injectée dans le prompt du LLM.
        """
        db_messages = REPOSITORIES.messages.list_for_session(
            self.session_id,
            limit=self.max_messages,
        )
        print("nombre de messages récupérés : ",db_messages)
        return self._convert_to_langchain(db_messages)

    def get_message_count(self) -> int:
        """Retourne le nombre de messages dans la session."""
        return len(REPOSITORIES.messages.list_for_session(self.session_id))

    @staticmethod
    def _convert_to_langchain(messages: List[Message]) -> List[BaseMessage]:
        """Convertit les messages ORM en objets LangChain."""
        import re
        lc_messages: List[BaseMessage] = []

        for msg in messages:
            if msg.requete:
                # On purge les images base64 de l'historique pour libérer la VRAM (OOM)
                # L'agent n'a besoin que du texte des conversations passées
                texte_nettoye = re.sub(r'\[Image attachée:.*?\]', '', msg.requete, flags=re.DOTALL).strip()
                if texte_nettoye:
                    lc_messages.append(HumanMessage(content=texte_nettoye))
            if msg.reponse:
                lc_messages.append(AIMessage(content=msg.reponse))

        return lc_messages
