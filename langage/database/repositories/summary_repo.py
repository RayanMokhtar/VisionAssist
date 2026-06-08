"""
Repository des résumés journaliers — CRUD sur les daily_summaries.

Chaque utilisateur a au plus 1 résumé par jour.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from langage.database.models import DailySummary, User

logger = logging.getLogger(__name__)


def get_summary_for_date(
    db: Session, user_id: str, target_date: datetime.date
) -> Optional[DailySummary]:
    """Récupère le résumé d'un jour donné pour un utilisateur."""
    user = db.query(User).filter(User.username == user_id).first()
    if user is None:
        return None
    return (
        db.query(DailySummary)
        .filter(DailySummary.user_id == user.id, DailySummary.date == target_date)
        .first()
    )


def get_yesterday_summary(db: Session, user_id: str) -> Optional[DailySummary]:
    """Raccourci pour récupérer le résumé d'hier."""
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    return get_summary_for_date(db, user_id, yesterday)


def get_recent_summaries(
    db: Session, user_id: str, days: int = 7
) -> list[DailySummary]:
    """Récupère les résumés des N derniers jours."""
    user = db.query(User).filter(User.username == user_id).first()
    if user is None:
        return []
    since = datetime.date.today() - datetime.timedelta(days=days)
    return (
        db.query(DailySummary)
        .filter(DailySummary.user_id == user.id, DailySummary.date >= since)
        .order_by(DailySummary.date.desc())
        .all()
    )


def create_or_update_summary(
    db: Session,
    user_id: str,
    target_date: datetime.date,
    summary: str,
    key_events: dict | None = None,
    lieux_visites: list | None = None,
    dangers_notes: list | None = None,
    user_preferences: dict | None = None,
    nb_sessions: int = 1,
    embedding_id: str | None = None,
) -> DailySummary:
    """Crée ou met à jour le résumé d'une journée."""
    user = db.query(User).filter(User.username == user_id).first()
    if user is None:
        raise ValueError(f"Utilisateur inconnu : {user_id}")

    existing = (
        db.query(DailySummary)
        .filter(DailySummary.user_id == user.id, DailySummary.date == target_date)
        .first()
    )

    if existing:
        existing.summary = summary
        existing.key_events = key_events
        existing.lieux_visites = lieux_visites
        existing.dangers_notes = dangers_notes
        existing.user_preferences = user_preferences
        existing.nb_sessions = nb_sessions
        existing.embedding_id = embedding_id
        db.flush()
        logger.info("Résumé mis à jour pour user=%s date=%s", user_id, target_date)
        return existing

    ds = DailySummary(
        id=str(uuid.uuid4()),
        user_id=user.id,
        date=target_date,
        summary=summary,
        key_events=key_events,
        lieux_visites=lieux_visites,
        dangers_notes=dangers_notes,
        user_preferences=user_preferences,
        nb_sessions=nb_sessions,
        embedding_id=embedding_id,
    )
    db.add(ds)
    db.flush()
    logger.info("Résumé créé pour user=%s date=%s", user_id, target_date)
    return ds
