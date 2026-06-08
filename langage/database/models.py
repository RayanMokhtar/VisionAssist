"""
Modèles ORM SQLAlchemy pour VisionAssist.

Tables :
    users            – Utilisateurs du système
    sessions         – 1 session par utilisateur par jour
    messages         – Historique conversationnel (court terme)
    daily_summaries  – Résumés journaliers générés par le LLM
    user_notes       – Notes et rappels sauvegardés par l'utilisateur
    user_profile     – Profil clé/valeur auto-appris
    tool_calls       – Journal des appels d'outils (audit)
"""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Classe de base pour tous les modèles ORM."""
    pass


# ─── Utilisateurs ─────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    username = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    sessions = relationship("SessionModel", back_populates="user", cascade="all, delete-orphan")
    daily_summaries = relationship("DailySummary", back_populates="user", cascade="all, delete-orphan")
    notes = relationship("UserNote", back_populates="user", cascade="all, delete-orphan")


# ─── Sessions (1 par utilisateur par jour) ────────────────────────────────────

class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    ended_at = Column(DateTime)
    summary = Column(Text)
    total_messages = Column(Integer, default=0)
    yesterday_summary_injected = Column(Boolean, default=False)
    session_metadata = Column("metadata", JSON)

    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_user_date"),
    )

    user = relationship("User", back_populates="sessions")
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.timestamp",
    )


# ─── Messages (historique conversationnel) ────────────────────────────────────

class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)          # user | assistant | system | tool
    content = Column(Text, nullable=False)
    tool_name = Column(String(100))
    tool_args = Column(JSON)
    tool_result = Column(JSON)
    source = Column(String(20), default="stt")          # stt | vision | api | system
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    tokens_estimate = Column(Integer, default=0)

    session = relationship("SessionModel", back_populates="messages")


# ─── Résumés journaliers ──────────────────────────────────────────────────────

class DailySummary(Base):
    __tablename__ = "daily_summaries"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    summary = Column(Text, nullable=False)
    key_events = Column(JSON)
    lieux_visites = Column(JSON)
    dangers_notes = Column(JSON)
    user_preferences = Column(JSON)
    nb_sessions = Column(Integer, default=0)
    embedding_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_summary_user_date"),
    )

    user = relationship("User", back_populates="daily_summaries")


# ─── Notes utilisateur ────────────────────────────────────────────────────────

class UserNote(Base):
    __tablename__ = "user_notes"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True)
    content = Column(Text, nullable=False)
    category = Column(String(50), default="general", index=True)  # rappel | lieu | contact | general
    is_active = Column(Boolean, default=True, index=True)
    embedding_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime)

    user = relationship("User", back_populates="notes")


# ─── Profil utilisateur (clé/valeur) ──────────────────────────────────────────

class UserProfile(Base):
    __tablename__ = "user_profile"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    cle = Column(String(100), nullable=False)
    valeur = Column(JSON, nullable=False)
    source = Column(String(100))
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "cle", name="uq_profile_user_key"),
    )


# ─── Journal des appels d'outils ──────────────────────────────────────────────

class ToolCallLog(Base):
    __tablename__ = "tool_calls"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    tool_name = Column(String(100), nullable=False, index=True)
    args = Column(JSON)
    result = Column(JSON)
    duration_ms = Column(Integer)
    success = Column(Boolean, default=True)
    error_msg = Column(Text)
    called_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
