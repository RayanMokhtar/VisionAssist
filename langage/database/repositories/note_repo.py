"""
Repository des notes utilisateur — CRUD pour les rappels et mémos.

Utilisé par les tools save_note et query_memory.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from langage.database.models import UserNote, User

logger = logging.getLogger(__name__)


def add_note(
    db: Session,
    user_id: str,
    content: str,
    category: str = "general",
    session_id: Optional[str] = None,
    embedding_id: Optional[str] = None,
    expires_at: Optional[datetime.datetime] = None,
) -> UserNote:
    """Crée une nouvelle note pour l'utilisateur."""
    user = db.query(User).filter(User.username == user_id).first()
    if user is None:
        raise ValueError(f"Utilisateur inconnu : {user_id}")

    note = UserNote(
        id=str(uuid.uuid4()),
        user_id=user.id,
        session_id=session_id,
        content=content,
        category=category,
        is_active=True,
        embedding_id=embedding_id,
        expires_at=expires_at,
    )
    db.add(note)
    db.flush()
    logger.info("Note créée (cat=%s) pour user=%s : %s", category, user_id, content[:50])
    return note


def get_active_notes(
    db: Session,
    user_id: str,
    category: Optional[str] = None,
    limit: int = 20,
) -> List[UserNote]:
    """Récupère les notes actives d'un utilisateur."""
    user = db.query(User).filter(User.username == user_id).first()
    if user is None:
        return []

    query = (
        db.query(UserNote)
        .filter(UserNote.user_id == user.id, UserNote.is_active == True)
    )
    if category:
        query = query.filter(UserNote.category == category)

    # Exclure les notes expirées
    now = datetime.datetime.utcnow()
    query = query.filter(
        (UserNote.expires_at == None) | (UserNote.expires_at > now)
    )

    return query.order_by(UserNote.created_at.desc()).limit(limit).all()


def get_all_notes_for_user(db: Session, user_id: str) -> List[UserNote]:
    """Récupère toutes les notes d'un utilisateur (actives et inactives)."""
    user = db.query(User).filter(User.username == user_id).first()
    if user is None:
        return []
    return (
        db.query(UserNote)
        .filter(UserNote.user_id == user.id)
        .order_by(UserNote.created_at.desc())
        .all()
    )


def deactivate_note(db: Session, note_id: str) -> bool:
    """Désactive une note (sans la supprimer)."""
    note = db.query(UserNote).filter(UserNote.id == note_id).first()
    if note:
        note.is_active = False
        db.flush()
        return True
    return False
