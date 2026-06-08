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

from configuration import CONFIGURATION

logger = logging.getLogger(__name__)
_conf = CONFIGURATION.mysql

DATABASE_URL = (
    f"mysql+pymysql://{_conf.user}:{_conf.password}"
    f"@{_conf.host}:{_conf.port}/{_conf.database}"
    f"?charset=utf8mb4"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=_conf.pool_size,
    pool_recycle=_conf.pool_recycle,
    echo=False,
)

SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Crée toutes les tables si elles n'existent pas encore."""
    from langage.database.models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Tables MySQL initialisées dans '%s'.", _conf.database)


def setup_database() -> None:
    """Crée la base de données ET les tables.

    Appelé une seule fois au premier démarrage du service LLM.
    Se connecte d'abord sans base pour la créer, puis initialise les tables.
    """
    temp_url = (
        f"mysql+pymysql://{_conf.user}:{_conf.password}"
        f"@{_conf.host}:{_conf.port}/"
    )
    temp_engine = create_engine(temp_url)
    with temp_engine.connect() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{_conf.database}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        ))
        conn.commit()
    temp_engine.dispose()
    logger.info("Base de données '%s' créée/vérifiée.", _conf.database)
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
