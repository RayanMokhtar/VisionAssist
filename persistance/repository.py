# repositories.py
from typing import List, Optional, Union
from uuid import UUID, uuid4

from sqlalchemy.orm import Session as DBSession
from sqlalchemy.exc import NoResultFound

from persistance.models import User, Carte, Session as SessionModel, Message, CardStatus
from persistance.database import SESSION_ID

class BaseRepository:
    def __init__(self, db: DBSession):
        self.db = db

class UserRepository(BaseRepository):
    def get(self, user_id: int) -> Optional[User]:
        return self.db.query(User).filter_by(id=user_id).first()

    def list(self, limit: int = 100) -> List[User]:
        return self.db.query(User).limit(limit).all()

    def create(self, *, nom: str, prenom: str, adresse: Optional[str] = None, preferences: Optional[dict] = None) -> User:
        preferences = preferences or {}
        user = User(nom=nom, prenom=prenom, adresse=adresse, preferences=preferences)
        self.db.add(user)
        self.db.commit()
        print(f"Created user: {user.nom} {user.prenom} with ID {user.id}")
        return user

    def update(self, user: User, **fields) -> User:
        for k, v in fields.items():
            if hasattr(user, k): # SI attribut existe on fait la mise à jour du repository
                setattr(user, k, v)
        self.db.add(user)
        self.db.commit()
        return user

    def delete(self, user: User) -> None:
        self.db.delete(user)
        self.db.commit()


class CarteRepository(BaseRepository):
    def get(self, card_id: str) -> Optional[Carte]:
        return self.db.query(Carte).filter_by(card_id=card_id).first()

    def list_for_user(self, user_id: int) -> List[Carte]:
        return self.db.query(Carte).filter_by(user_id=user_id).all()

    def create(self, *, card_id: str, user_id: int, statut: CardStatus = CardStatus.active, secret_chiffre: Optional[str] = None,) -> Carte:
        carte = Carte(card_id=card_id, user_id=user_id, statut=statut, secret_chiffre=secret_chiffre) #création du secret au début de l'enrollement ... 
        self.db.add(carte)
        self.db.commit()
        return carte

    def update_status(self, carte: Carte, statut: CardStatus) -> Carte:
        carte.statut = statut
        self.db.add(carte)
        self.db.commit()
        return carte

    def delete(self, carte: Carte) -> None:
        self.db.delete(carte)
        self.db.commit()


class SessionRepository(BaseRepository):
    def get(self, session_id: Union[UUID, str]) -> Optional[SessionModel]:
        if isinstance(session_id, str):
            session_id = UUID(session_id)
        return self.db.query(SessionModel).filter_by(session_id=session_id).first()

    def create(self, *, user_id: Optional[int] = None, card_id: Optional[str] = None, resume: Optional[str] = None, session_id: Optional[Union[UUID, str]] = None) -> SessionModel:
        if session_id is None:
            sid = uuid4()
        else:
            sid = session_id if isinstance(session_id, UUID) else UUID(str(session_id))
        sess = SessionModel(session_id=sid, user_id=user_id, card_id=card_id, resume=resume)
        self.db.add(sess)
        self.db.commit()
        return sess

    def update_resume(self, sess_model: SessionModel, resume: str) -> SessionModel:
        sess_model.resume = resume
        self.db.add(sess_model)
        self.db.commit()
        return sess_model

    def list_for_user(self, user_id: int, limit: int = 100) -> List[SessionModel]:
        return self.db.query(SessionModel).filter_by(user_id=user_id).limit(limit).all()

    def delete(self, sess_model: SessionModel) -> None:
        self.db.delete(sess_model)
        self.db.commit()


class MessageRepository(BaseRepository):
    def get(self, message_id: int) -> Optional[Message]:
        return self.db.query(Message).filter_by(message_id=message_id).first()

    def list_for_session(self, session_id: Union[UUID, str], limit: int = 1000) -> List[Message]:
        if isinstance(session_id, str):
            session_id = UUID(session_id)
        return self.db.query(Message).filter_by(session_id=session_id).limit(limit).all()

    def create(self, *, session_id: Union[UUID, str], requete: str, reponse: Optional[str] = None, contexte: Optional[dict] = None) -> Message:
        if isinstance(session_id, str):
            session_id = UUID(session_id)
        contexte = contexte or {}
        msg = Message(session_id=session_id, requete=requete, reponse=reponse, contexte=contexte)
        self.db.add(msg)
        self.db.commit()
        return msg

    def update_response(self, msg: Message, reponse: str) -> Message:
        msg.reponse = reponse
        self.db.add(msg)
        self.db.commit()
        return msg

    def delete(self, msg: Message) -> None:
        self.db.delete(msg)
        self.db.commit()



class Repositories:
    def __init__(self, db_session: DBSession):
        self.users = UserRepository(db_session)
        self.cartes = CarteRepository(db_session)
        self.sessions = SessionRepository(db_session)
        self.messages = MessageRepository(db_session)


def get_repositories(db_session: DBSession) -> Repositories:
    return Repositories(db_session)


REPOSITORIES = get_repositories(db_session=SESSION_ID)


# test repositorioes factice ... 

# REPOSITORIES.users.create(nom="Image", prenom="test", adresse="123 Main St", preferences={"theme": "dark"})
# REPOSITORIES.sessions.create(user_id=1, card_id=1, resume="Session de test")

# liste = REPOSITORIES.users.list()
# print(liste)