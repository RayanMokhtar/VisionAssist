"""
Summarizer — Génère des résumés de session et de journée via le LLM.

Utilisé pour :
    1. Résumer une session quand elle se termine
    2. Résumer la journée entière (cron de fin de journée)
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from langage.database.models import Message
from langage.database.repositories import message_repo, summary_repo
from langage.memory.long_term import long_term_memory

logger = logging.getLogger(__name__)


SESSION_SUMMARY_PROMPT = """Tu es VisionAssist, assistant intelligent pour personne malvoyante.
Résume cette conversation de manière concise (3-5 phrases). Mentionne les points importants :
- Questions posées par l'utilisateur
- Informations fournies
- Outils utilisés
- Préférences détectées

Conversation :
{conversation}

Résumé concis :"""


DAILY_SUMMARY_PROMPT = """Tu es VisionAssist, assistant intelligent pour personne malvoyante.
Analyse les conversations de la journée et génère un résumé structuré en JSON.

Format attendu (JSON strict, sans markdown) :
{{
  "resume_general": "2-3 phrases décrivant la journée",
  "lieux_mentionnes": ["lieu1", "lieu2"],
  "questions_frequentes": ["question1"],
  "preferences_utilisateur": {{"cle": "valeur"}},
  "rappels_actifs": ["rappel en attente"],
  "points_attention": ["chose importante à retenir"]
}}

Conversations du {date} :
{conversations}

JSON :"""


def format_messages_for_summary(messages: List[Message]) -> str:
    """Formate les messages en texte lisible pour le prompt de résumé."""
    lines = []
    for msg in messages:
        role = msg.role.upper()
        content = msg.content[:200]
        if msg.tool_name:
            lines.append(f"[{role}] (outil: {msg.tool_name}) {content}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def generate_session_summary(
    db: Session,
    session_id: str,
    llm_callable,
) -> Optional[str]:
    """Génère un résumé textuel d'une session via le LLM.

    Args:
        db: Session SQLAlchemy.
        session_id: ID de la session à résumer.
        llm_callable: Fonction fn(prompt: str) -> str pour appeler le LLM.
    """
    messages = message_repo.get_session_messages(db, session_id)
    if not messages:
        return None

    conversation_text = format_messages_for_summary(messages)
    prompt = SESSION_SUMMARY_PROMPT.format(conversation=conversation_text)

    try:
        summary = llm_callable(prompt)
        logger.info("Résumé de session généré (%d messages)", len(messages))
        return summary.strip()
    except Exception as e:
        logger.error("Erreur génération résumé session : %s", e)
        return None


def generate_daily_summary(
    db: Session,
    user_id: str,
    target_date,
    llm_callable,
) -> Optional[str]:
    """Génère le résumé d'une journée complète et le persiste en BDD + ChromaDB.

    Args:
        db: Session SQLAlchemy.
        user_id: Identifiant de l'utilisateur.
        target_date: Date à résumer (datetime.date).
        llm_callable: Fonction fn(prompt: str) -> str pour appeler le LLM.

    Returns:
        Le résumé textuel ou None en cas d'erreur.
    """
    from langage.database.repositories import session_repo

    session = session_repo.get_session_by_user_and_date(db, user_id, target_date)
    if session is None:
        logger.info("Pas de session pour user=%s date=%s — rien à résumer.", user_id, target_date)
        return None

    messages = message_repo.get_session_messages(db, session.id)
    if not messages:
        logger.info("Session vide pour user=%s date=%s", user_id, target_date)
        return None

    conversation_text = format_messages_for_summary(messages)
    prompt = DAILY_SUMMARY_PROMPT.format(
        date=target_date.isoformat(),
        conversations=conversation_text,
    )

    try:
        raw_response = llm_callable(prompt)
        # Extraire le JSON du résultat
        summary_text = raw_response.strip()
        
        # Tenter de parser le JSON
        parsed = None
        try:
            parsed = json.loads(summary_text)
        except json.JSONDecodeError:
            # Si le LLM n'a pas retourné du JSON pur, garder comme texte
            parsed = {"resume_general": summary_text}

        resume_general = parsed.get("resume_general", summary_text)

        # Persister dans MySQL
        ds = summary_repo.create_or_update_summary(
            db=db,
            user_id=user_id,
            target_date=target_date,
            summary=resume_general,
            key_events=parsed.get("points_attention"),
            lieux_visites=parsed.get("lieux_mentionnes"),
            user_preferences=parsed.get("preferences_utilisateur"),
            nb_sessions=1,
            embedding_id=f"daily_{user_id}_{target_date.isoformat()}",
        )

        # Indexer dans ChromaDB pour le RAG
        long_term_memory.index_daily_summary(
            doc_id=ds.id,
            content=resume_general,
            user_id=user_id,
            date_str=target_date.isoformat(),
        )

        logger.info("Résumé journalier créé : user=%s date=%s", user_id, target_date)
        return resume_general

    except Exception as e:
        logger.error("Erreur génération résumé journalier : %s", e)
        return None
