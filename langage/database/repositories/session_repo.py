"""
Repository des sessions — Gestion des sessions journalières par utilisateur.

Logique clé :
    - 1 session unique par utilisateur par jour (clé : user_id + date)
    - Si la session du jour n'existe pas → la créer
    - L'utilisateur est auto-créé s'il n'existe pas
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from langage.database.models import SessionModel, User

logger = logging.getLogger(__name__)


from sqlalchemy.exc import IntegrityError

def get_or_create_user(db: Session, user_id: str) -> User:
    """Récupère un utilisateur ou le crée automatiquement."""
    user = db.query(User).filter(User.username == user_id).first()
    if user is None:
        try:
            with db.begin_nested():
                user = User(
                    id=str(uuid.uuid4()),
                    username=user_id,
                    display_name=user_id,
                )
                db.add(user)
                db.flush()
                logger.info("Nouvel utilisateur créé : %s", user_id)
        except IntegrityError:
            # En cas de condition de course (créé par une autre requête concurrente)
            # Sous MySQL, le REPEATABLE READ peut empêcher de voir la ligne avec db.query().
            # Mais on sait qu'il existe. On peut soit relancer la requête, soit tricher et 
            # forcer une lecture hors de la transaction actuelle si nécessaire, mais dans 
            # 99% des cas on peut juste le relire si la transaction a été commitée ou 
            # lever une erreur contrôlée.
            # Pour éviter que la session soit invalide, begin_nested() annule juste le savepoint.
            user = db.query(User).filter(User.username == user_id).first()
            if user is None:
                # Si toujours None à cause de l'isolation de transaction, on retourne un objet factice 
                # (ou on force la fin de la transaction) - mais normalement la requête suivante passera.
                logger.warning("Conflit de création pour %s, veuillez réessayer la requête.", user_id)
                raise Exception(f"Conflit de création pour l'utilisateur {user_id}. Veuillez réessayer.")
    return user


def get_today_session(db: Session, user_id: str) -> Optional[SessionModel]:
    """Récupère la session du jour pour un utilisateur (si elle existe)."""
    user = get_or_create_user(db, user_id)
    today = datetime.date.today()
    return (
        db.query(SessionModel)
        .filter(SessionModel.user_id == user.id, SessionModel.date == today)
        .first()
    )


def create_today_session(db: Session, user_id: str) -> SessionModel:
    """Crée une nouvelle session pour aujourd'hui."""
    user = get_or_create_user(db, user_id)
    today = datetime.date.today()
    session = SessionModel(
        id=str(uuid.uuid4()),
        user_id=user.id,
        date=today,
        started_at=datetime.datetime.utcnow(),
        total_messages=0,
        yesterday_summary_injected=False,
    )
    db.add(session)
    db.flush()
    logger.info("Nouvelle session créée pour user=%s date=%s", user_id, today)
    return session


def get_or_create_today_session(db: Session, user_id: str) -> tuple[SessionModel, bool]:
    """Récupère ou crée la session du jour.

    Returns:
        (session, is_new) – is_new=True si la session vient d'être créée
    """
    existing = get_today_session(db, user_id)
    if existing is not None:
        return existing, False
    return create_today_session(db, user_id), True


def get_session_by_user_and_date(
    db: Session, user_id: str, target_date: datetime.date
) -> Optional[SessionModel]:
    """Récupère une session pour un utilisateur à une date donnée."""
    user = db.query(User).filter(User.username == user_id).first()
    if user is None:
        return None
    return (
        db.query(SessionModel)
        .filter(SessionModel.user_id == user.id, SessionModel.date == target_date)
        .first()
    )


def update_session_summary(db: Session, session_id: str, summary: str) -> None:
    """Met à jour le résumé d'une session."""
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if session:
        session.summary = summary
        session.ended_at = datetime.datetime.utcnow()
        db.flush()


def mark_yesterday_injected(db: Session, session_id: str) -> None:
    """Marque que le résumé d'hier a été injecté dans cette session."""
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if session:
        session.yesterday_summary_injected = True
        db.flush()


def increment_message_count(db: Session, session_id: str, count: int = 1) -> None:
    """Incrémente le compteur de messages d'une session."""
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if session:
        session.total_messages = (session.total_messages or 0) + count
        db.flush()
