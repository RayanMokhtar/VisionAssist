"""
Moteur SQLAlchemy pour MySQL — Connexion, sessions DB et initialisation.

Utilisation :
    from langage.database.engine import get_db, init_db, setup_database

    # Au démarrage de l'application :
    setup_database()        # crée la base + tables si inexistantes

    # Pour chaque opération :
    with get_db() as db:
        user = session_repo.get_or_create(db, "user_123")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

import os

logger = logging.getLogger(__name__)

DATABASE_URL = "sqlite:///visionassist.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Crée toutes les tables si elles n'existent pas encore."""
    from langage.database.models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Tables SQLite initialisées.")


def setup_database() -> None:
    """Crée la base de données ET les tables.

    Appelé une seule fois au premier démarrage du service LLM.
    Se connecte d'abord sans base pour la créer, puis initialise les tables.
    """
    logger.info("Base de données SQLite vérifiée.")
    init_db()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager pour obtenir une session DB avec commit/rollback automatique.

    Exemple :
        with get_db() as db:
            user = db.query(User).first()
    """
    db = SessionFactory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
