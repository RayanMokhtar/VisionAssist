# models.py
from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    Text,
    Enum as SAEnum,
    func,
)
from sqlalchemy.types import JSON as SAJSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.mutable import MutableDict
import enum

import uuid
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

Base = declarative_base()


class GUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        else:
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)
    


class CardStatus(enum.Enum):
    active = "active"
    pending = "pending"
    bloquee = "bloquee"
    expiree = "expiree"

JSON_TYPE = SAJSON

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    nom = Column(String(100), nullable=False)
    prenom = Column(String(100), nullable=False)
    adresse = Column(String(255), nullable=True)
    preferences = Column(Text, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)
    derniere_connexion = Column(DateTime(timezone=True), nullable=True)

    cartes = relationship("Carte", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"""<User id={self.id} nom={self.nom} prenom={self.prenom}> preferences={self.preferences} , adresse={self.adresse}>,
        created_at={self.created_at} derniere_connexion={self.derniere_connexion}> , cartes={self.cartes} sessions={self.sessions}>"""

class Carte(Base):
    __tablename__ = "cartes"

    card_id = Column(String(100), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    statut = Column(SAEnum(CardStatus, name="card_status_enum"), nullable=False, default=CardStatus.active)

    secret_chiffre = Column(Text, nullable=False)
    user = relationship("User", back_populates="cartes")
    sessions = relationship("Session", back_populates="carte", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Carte card_id={self.card_id} user_id={self.user_id} statut={self.statut}>"
    
    def to_dict(self):
        return {
            "card_id": self.card_id,
            "user_id": self.user_id,
            "statut": self.statut.value if self.statut else None,
            "secret_configure": self.secret_chiffre is not None

        }

#session_id en uuid
class Session(Base):

    __tablename__ = "sessions"

    session_id = Column(GUID, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    card_id = Column(String(100), ForeignKey("cartes.card_id", ondelete="SET NULL"), nullable=True)

    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    resume = Column(Text, nullable=True)

    user = relationship("User", back_populates="sessions")
    carte = relationship("Carte", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session session_id={self.session_id} user_id={self.user_id} card_id={self.card_id}>"

class Message(Base):
    __tablename__ = "messages"
    message_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(GUID, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    requete = Column(Text, nullable=False)  
    reponse = Column(Text, nullable=True)    
    contexte = Column(MutableDict.as_mutable(JSON_TYPE), nullable=True, default=dict)  
    session = relationship("Session", back_populates="messages")
    def __repr__(self):
        return f"<Message message_id={self.message_id} session_id={self.session_id} ts={self.timestamp}>"