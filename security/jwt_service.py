import jwt
import hashlib 
import secrets
from datetime import datetime, timedelta

from configuration import CONFIGURATION

def create_access_token(user_id : int , card_id : str ) -> str : 
    payload = {
        "user_id": user_id,
        "card_id": card_id,
        "iat":datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=CONFIGURATION.security.jwt.access_token_expire_minutes)
    }
    token = jwt.encode(payload, CONFIGURATION.security.jwt.secret_key, algorithm=CONFIGURATION.security.jwt.algorithm)
    return token

def verify_access_token(token : str) -> dict :
    try : 
        payload = jwt.decode(token, CONFIGURATION.security.jwt.secret_key, algorithms=[CONFIGURATION.security.jwt.algorithm])
        return {"valid": True, "payload": payload}
    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "Token expiré"}
    except jwt.InvalidTokenError:
        return {"valid": False, "error": "Token invalide"}
    
def create_refresh_token(user_id : int , card_id : str ) -> str : 
    #TODO à modifier pour stocker le token de rafraîchissement en base de données et pouvoir le révoquer si besoin
    refresh_token = secrets.token_urlsafe(32) # cette ligne génère un token de rafraîchissement aléatoire sécurisé
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest() # fait un hash du token de rafraîchissement pour le stocker en base de données de manière sécurisée
    expires_at = datetime.utcnow() + timedelta(minutes=CONFIGURATION.security.jwt.refresh_token_expire_minutes)
    return refresh_token
