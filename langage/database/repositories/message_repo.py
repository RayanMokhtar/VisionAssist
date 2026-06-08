"""
Repository des messages — CRUD sur l'historique conversationnel.

Gère les messages de toutes les sessions : user, assistant, system, tool.
Utilisé par la mémoire court terme pour charger le contexte d'une session.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from langage.database.models import Message

logger = logging.getLogger(__name__)


def add_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    source: str = "stt",
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    tool_result: Optional[dict] = None,
    tokens_estimate: int = 0,
) -> Message:
    """Ajoute un message à une session.

    Args:
        role: 'user' | 'assistant' | 'system' | 'tool'
        source: 'stt' | 'vision' | 'api' | 'system'
    """
    msg = Message(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        content=content,
        source=source,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        timestamp=datetime.datetime.utcnow(),
        tokens_estimate=tokens_estimate,
    )
    db.add(msg)
    db.flush()
    return msg


def get_session_messages(
    db: Session,
    session_id: str,
    limit: Optional[int] = None,
    roles: Optional[List[str]] = None,
) -> List[Message]:
    """Récupère les messages d'une session, triés par timestamp.

    Args:
        limit: Nombre max de messages à retourner (les plus récents).
        roles: Filtrer par rôles (ex: ['user', 'assistant']).
    """
    query = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.timestamp.asc())
    )
    if roles:
        query = query.filter(Message.role.in_(roles))
    
    if limit:
        # Récupère les N derniers, triés par timestamp asc
        total = query.count()
        if total > limit:
            query = query.offset(total - limit)

    return query.all()


def get_messages_for_date_range(
    db: Session,
    session_ids: List[str],
) -> List[Message]:
    """Récupère tous les messages pour une liste de sessions."""
    if not session_ids:
        return []
    return (
        db.query(Message)
        .filter(Message.session_id.in_(session_ids))
        .order_by(Message.timestamp.asc())
        .all()
    )


def count_messages(db: Session, session_id: str) -> int:
    """Compte le nombre de messages dans une session."""
    return db.query(Message).filter(Message.session_id == session_id).count()
