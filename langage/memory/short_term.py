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
from sqlalchemy.orm import Session

from langage.database.models import Message
from langage.database.repositories import message_repo

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
        max_messages: int | None = None,
    ):
        self.session_id = session_id
        self.max_messages = max_messages or 20

    def get_langchain_messages(self, db: Session) -> List[BaseMessage]:
        """Charge les messages et les convertit en objets LangChain.

        Returns:
            Liste de BaseMessage prête à être injectée dans le prompt du LLM.
        """
        db_messages = message_repo.get_session_messages(
            db,
            self.session_id,
            limit=self.max_messages,
        )
        return self._convert_to_langchain(db_messages)

    def get_message_count(self, db: Session) -> int:
        """Retourne le nombre de messages dans la session."""
        return message_repo.count_messages(db, self.session_id)

    @staticmethod
    def _convert_to_langchain(messages: List[Message]) -> List[BaseMessage]:
        """Convertit les messages ORM en objets LangChain."""
        lc_messages: List[BaseMessage] = []

        for msg in messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "tool":
                lc_messages.append(ToolMessage(
                    content=msg.content,
                    tool_call_id=msg.tool_name or "unknown",
                ))
            else:
                logger.warning("Rôle inconnu ignoré : %s", msg.role)

        return lc_messages
