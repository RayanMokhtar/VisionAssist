"""
DailySummaryJob — Cron job de résumé de fin de journée.

Planifie un job APScheduler qui, chaque soir à l'heure configurée :
    1. Récupère toutes les sessions du jour pour chaque utilisateur actif
    2. Génère un résumé via le LLM
    3. Persiste en MySQL + indexe dans ChromaDB

Utilisation :
    from langage.memory.daily_summary import start_daily_summary_scheduler
    start_daily_summary_scheduler(llm_callable)
"""

from __future__ import annotations

import datetime
import logging
from typing import Callable

from langage.database.engine import get_db
from langage.database.models import User
from langage.memory.summarizer import generate_daily_summary

logger = logging.getLogger(__name__)


def run_daily_summaries(llm_callable: Callable[[str], str]) -> None:
    """Exécute le résumé quotidien pour tous les utilisateurs ayant eu une session aujourd'hui.

    Args:
        llm_callable: Fonction fn(prompt: str) -> str pour appeler le LLM.
    """
    today = datetime.date.today()
    logger.info("=== Démarrage résumé quotidien pour le %s ===", today)

    with get_db() as db:
        # Récupérer tous les utilisateurs
        users = db.query(User).all()

        for user in users:
            try:
                summary = generate_daily_summary(
                    db=db,
                    user_id=user.username,
                    target_date=today,
                    llm_callable=llm_callable,
                )
                if summary:
                    logger.info(
                        "Résumé généré pour user=%s : %s",
                        user.username, summary[:80]
                    )
            except Exception as e:
                logger.error(
                    "Erreur résumé pour user=%s : %s",
                    user.username, e,
                )

    logger.info("=== Résumés quotidiens terminés ===")


def start_daily_summary_scheduler(llm_callable: Callable[[str], str]) -> None:
    """Démarre le planificateur APScheduler pour les résumés de fin de journée.

    Le job s'exécute chaque jour à l'heure configurée (par défaut 23h59).
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning(
            "APScheduler non installé — résumés quotidiens désactivés. "
            "Installez avec : pip install apscheduler>=3.10.0"
        )
        return

    scheduler = BackgroundScheduler()
    trigger = CronTrigger(
        hour=23,
        minute=59,
    )

    scheduler.add_job(
        run_daily_summaries,
        trigger=trigger,
        args=[llm_callable],
        id="daily_summary_job",
        name="Résumé quotidien VisionAssist",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "Planificateur résumé quotidien démarré → %02d:%02d chaque jour",
        23,
        59,
    )
