import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from datetime import datetime, timedelta

from cryptography.fernet import Fernet

# Imports relatifs à votre architecture
from broker.service import get_broker_client
from configuration import CONFIGURATION
from persistance.models import User, Carte, Session as SessionModel, CardStatus
from persistance.repository import REPOSITORIES
from security.jwt_service import create_access_token, create_refresh_token

TOPICS = CONFIGURATION.broker.topics
AUTH_CLIENT_REQUEST_SECURITY = TOPICS.security_client_request_topic
AUTH_ADMIN_REQUEST_SECURITY = TOPICS.security_admin_request_topic
AUTH_RESPONSE_SECURITY = TOPICS.security_response_topic

########################
# Challenges temp: challenge_id -> infos du challenge.
CHALLENGES = {}

# ==========================================
# UTILITAIRES DE SÉCURITÉ ET CRYPTOGRAPHIE
# ==========================================

def get_fernet():
    cle = CONFIGURATION.security.cle_chiffrement_cartes
    if not cle:
        raise RuntimeError("CLE_CHIFFREMENT_CARTES manquante dans la configuration")
    return Fernet(cle.encode("ascii"))

def generer_secret_carte():
    # 32 octets = 256 bits, adapte pour HMAC-SHA256.
    return secrets.token_bytes(32)

def encoder_secret(secret):
    return base64.b64encode(secret).decode("ascii")

def decoder_secret(secret_texte):
    return base64.b64decode(secret_texte.encode("ascii"))

def chiffrer_secret(secret):
    return get_fernet().encrypt(secret).decode("ascii")

def dechiffrer_secret(secret_chiffre):
    return get_fernet().decrypt(secret_chiffre.encode("ascii"))

def generer_challenge():
    return base64.b64encode(secrets.token_bytes(32)).decode("ascii")

def calculer_signature(secret, card_id, challenge_id, challenge):
    message = f"{card_id}:{challenge_id}:{challenge}".encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()

def generer_card_id():
    count_users = REPOSITORIES.users.db.query(User).count()
    print("Nombre d'utilisateurs dans la base de données :", count_users)
    return f"VA-{count_users + 1}"

def publish_response_securite(broker, request, payload):
    response = dict(payload)
    response["request_id"] = request.get("request_id")    
    response["client_id"] = request.get("client_id")
    response["action"] = request.get("action")
    broker.publier(AUTH_RESPONSE_SECURITY, response)


# ==========================================
# CLASSES DE GESTION (LOGIQUE MÉTIER)
# ==========================================

class Authentification:

    @staticmethod
    def handle_request_challenge(broker, _topic, request):
        card_id = str(request.get("card_id", "")).strip().upper()
        if not card_id:
            publish_response_securite(broker, request, {"success": False, "error": "card_id manquant"})
            return

        carte = REPOSITORIES.cartes.get(card_id=card_id)
        if not carte:
            publish_response_securite(broker, request, {"success": False, "error": "card_id inconnu"})
            return

        if carte.statut.value != "active":
            publish_response_securite(broker, request, {"success": False, "error": carte.statut.value})
            return

        if not carte.secret_chiffre:
            publish_response_securite(broker, request, {"success": False, "error": "secret carte non configure"})
            return

        challenge_id = uuid.uuid4().hex
        challenge = generer_challenge()
        expiration = datetime.now() + timedelta(seconds=CONFIGURATION.security.duree_challenge_secondes)

        CHALLENGES[challenge_id] = {
            "card_id": card_id,
            "challenge": challenge,
            "expiration": expiration,
            "utilise": False,
        }

        publish_response_securite(broker, request, {
            "success": True, 
            "card_id": card_id, 
            "challenge_id": challenge_id, 
            "challenge": challenge
        })

    @staticmethod
    def handle_verify_card_hmac(broker, _topic, request):
        try : 
            print("handle verify card hmac", request)
            card_id = str(request.get("card_id", "")).strip().upper()
            challenge_id = str(request.get("challenge_id", "")).strip()
            signature = str(request.get("signature", "")).strip()

            if not card_id or not challenge_id or not signature:
                publish_response_securite(broker, request, {"success": False, "error": "donnees hmac manquantes"})
                return

            challenge_info = CHALLENGES.get(challenge_id)
            if not challenge_info:
                publish_response_securite(broker, request, {"success": False, "error": "challenge inconnu"})
                return

            if challenge_info["utilise"]:
                publish_response_securite(broker, request, {"success": False, "error": "challenge deja utilise"})
                return

            if datetime.now() > challenge_info["expiration"]:
                CHALLENGES.pop(challenge_id, None)
                publish_response_securite(broker, request, {"success": False, "error": "challenge expire"})
                return

            if challenge_info["card_id"] != card_id:
                publish_response_securite(broker, request, {"success": False, "error": "challenge non lie a cette carte"})
                return

            carte = REPOSITORIES.cartes.get(card_id=card_id)
            if not carte:
                publish_response_securite(broker, request, {"success": False, "error": "card_id inconnu"})
                return

            if carte.statut.value != "active":
                publish_response_securite(broker, request, {"success": False, "error": carte.statut.value})
                return

            if not carte.secret_chiffre:
                publish_response_securite(broker, request, {"success": False, "error": "secret carte non configure"})
                return

            try:
                secret = dechiffrer_secret(carte.secret_chiffre)
            except Exception:
                publish_response_securite(broker, request, {"success": False, "error": "secret serveur illisible"})
                return

            signature_attendue = calculer_signature(
                secret,
                card_id,
                challenge_id,
                challenge_info["challenge"],
            )

            if not hmac.compare_digest(signature_attendue, signature):
                publish_response_securite(broker, request, {"success": False, "error": "signature invalide"})
                return

            challenge_info["utilise"] = True

            user = REPOSITORIES.users.get(user_id=carte.user_id)
            if not user:
                publish_response_securite(broker, request, {"success": False, "error": "utilisateur inconnu"})
                return

            user = REPOSITORIES.users.update(user, derniere_connexion=datetime.now())
            if user : 
                verification_de_la_carte = {"success": True,
                    "card_id": card_id,
                    "user_id": user.id,
                    "prenom":user.prenom,
                    "status": carte.statut.value,
                    "derniere_connexion": str(user.derniere_connexion)
                }

                user_id = verification_de_la_carte.get("user_id")
                access_token = create_access_token(user_id=user_id, card_id=card_id)
                refresh_token = create_refresh_token(user_id=user_id, card_id=card_id)
                verification_de_la_carte["access_token"] = access_token
                verification_de_la_carte["refresh_token"] = refresh_token
                
                publish_response_securite(broker, request, verification_de_la_carte)
            else : 
                print("Erreur mise à jour dernière connexion utilisateur")
                publish_response_securite(broker, request, {"success": False, "error": "erreur à la fin du pipeline lors de la mise à jour de l'utilisateur"})
        except Exception as e:
            print("Erreur inattendue dans handle_verify_card_hmac:", str(e))
            publish_response_securite(broker, request, {"success": False, "error": "erreur serveur inattendue"})

class CreationEnrollementCarte:

    @staticmethod
    def preparer_logique_enrollement(message):
        user = REPOSITORIES.users.create(
            nom=message.get("nom"), 
            prenom=message.get("prenom"),
            preferences=message.get("preferences"), 
            adresse=message.get("adresse", "")
        )
        if user is not None: 
            print("Utilisateur créé en base de données : ", user)
        else:
            print("Erreur création utilisateur en base de données")

        carte_id = generer_card_id()
        secret = generer_secret_carte()
        secret_chiffre = chiffrer_secret(secret)

        carte = REPOSITORIES.cartes.create(
            user_id=user.id, 
            statut="pending", 
            card_id=carte_id, 
            secret_chiffre=secret_chiffre
        )
        if carte is not None: 
            print("Carte créée en base de données : ", carte)
        else:
            print("Erreur création carte en base de données")
        
        return carte_id, user.id, encoder_secret(secret)

    @staticmethod
    def handle_logique_enrollement(broker, topic, request):
        card_id, user_id, secret = CreationEnrollementCarte.preparer_logique_enrollement(request)
        publish_response_securite(broker, request, {
            "success": True, 
            "card_id": card_id, 
            "user_id": user_id, 
            "secret": secret
        })

    @staticmethod
    def handle_activate_card(broker, _topic, request):
        print("handle activate card", request)
        card_id = str(request.get("card_id", "")).strip().upper()
        if not card_id:
            publish_response_securite(broker, request, {"success": False, "error": "card_id requis"})
            return
            
        carte_existante = REPOSITORIES.cartes.get(card_id=card_id)
        if not carte_existante:
            publish_response_securite(broker, request, {"success": False, "error": "card_id inconnu"})
            return

        carte = REPOSITORIES.cartes.update_status(carte_existante, CardStatus.active)
            
        # secret = CreationEnrollementCarte.generer_secret_from_card_id(card_id)
        # Vous pouvez éventuellement renvoyer le secret généré si besoin côté client
        publish_response_securite(broker, request, {"success": True, "carte": carte.to_dict()})

    # @staticmethod
    # def generer_secret_from_card_id(card_id):
    #     """Fonction qui va générer un secret"""
    #     secret = hmac.new(CONFIGURATION.jwt.secret_key.encode(), card_id.encode(), hashlib.sha256).hexdigest()
    #     return secret


class AdministrationCartes:

    @staticmethod
    def normaliser_statut(statut):
        statut = str(statut or "").strip().lower()
        if statut not in CardStatus.__members__:
            return None
        return CardStatus[statut]

    @staticmethod
    def carte_complete_to_dict(carte):
        user = carte.user
        derniere_session = (
            REPOSITORIES.sessions.db.query(SessionModel)
            .filter_by(card_id=carte.card_id)
            .order_by(SessionModel.timestamp.desc())
            .first()
        )
        nombre_sessions = (
            REPOSITORIES.sessions.db.query(SessionModel)
            .filter_by(card_id=carte.card_id)
            .count()
        )

        return {
            "card_id": carte.card_id,
            "statut": carte.statut.value if carte.statut else None,
            "secret_configure": carte.secret_chiffre is not None,
            "user": {
                "id": user.id if user else carte.user_id,
                "nom": user.nom if user else None,
                "prenom": user.prenom if user else None,
                "adresse": user.adresse if user else None,
                "preferences": user.preferences if user else None,
                "created_at": str(user.created_at) if user and user.created_at else None,
                "derniere_connexion": str(user.derniere_connexion) if user and user.derniere_connexion else None,
            },
            "authentification": {
                "nombre_sessions": nombre_sessions,
                "derniere_session_id": str(derniere_session.session_id) if derniere_session else None,
                "derniere_session": str(derniere_session.timestamp) if derniere_session else None,
            },
        }

    @staticmethod
    def handle_set_card_status(broker, _topic, request):
        card_id = str(request.get("card_id", "")).strip().upper()
        statut_demande = AdministrationCartes.normaliser_statut(request.get("statut"))

        if not card_id:
            publish_response_securite(broker, request, {"success": False, "error": "card_id requis"})
            return
        if statut_demande is None:
            publish_response_securite(
                broker,
                request,
                {"success": False, "error": "statut invalide. Valeurs: active, pending, bloquee, expiree"},
            )
            return

        carte = REPOSITORIES.cartes.get(card_id=card_id)
        if not carte:
            publish_response_securite(broker, request, {"success": False, "error": "card_id inconnu"})
            return

        ancien_statut = carte.statut
        if ancien_statut == statut_demande:
            publish_response_securite(
                broker,
                request,
                {
                    "success": True,
                    "changed": False,
                    "message": f"Carte deja {statut_demande.value}",
                    "carte": AdministrationCartes.carte_complete_to_dict(carte),
                },
            )
            return

        carte = REPOSITORIES.cartes.update_status(carte, statut_demande)
        publish_response_securite(
            broker,
            request,
            {
                "success": True,
                "changed": True,
                "ancien_statut": ancien_statut.value if ancien_statut else None,
                "nouveau_statut": carte.statut.value,
                "carte": AdministrationCartes.carte_complete_to_dict(carte),
            },
        )

    @staticmethod
    def handle_list_cards(broker, _topic, request):
        cartes = (
            REPOSITORIES.cartes.db.query(Carte)
            .order_by(Carte.user_id.asc(), Carte.card_id.asc())
            .all()
        )
        publish_response_securite(
            broker,
            request,
            {
                "success": True,
                "count": len(cartes),
                "cartes": [AdministrationCartes.carte_complete_to_dict(carte) for carte in cartes],
            },
        )




def traitement_securite_requete(broker, topic, request):
    action = str(request.get("action", "")).strip().lower()
    handlers = {}
    
    # Authentification client => vérifier si la carte est valide et active
    if topic == AUTH_CLIENT_REQUEST_SECURITY:
        handlers = {
            "request-challenge": Authentification.handle_request_challenge,
            "verify-card-hmac": Authentification.handle_verify_card_hmac,
        }

    # Actions admin
    elif topic == AUTH_ADMIN_REQUEST_SECURITY:
        handlers = {
            "prepare-card": CreationEnrollementCarte.handle_logique_enrollement,
            "activate-card": CreationEnrollementCarte.handle_activate_card,
            "set-card-status": AdministrationCartes.handle_set_card_status,
            "list-cards": AdministrationCartes.handle_list_cards,
        }
        
    handler = handlers.get(action)
    if not handler:
        publish_response_securite(broker, request, {"success": False, "error": f"action inconnue: {action}"})
        return
        
    handler(broker, topic, request)


def main():
    broker = get_broker_client("auth-server")
    broker.connexion()
    time.sleep(0.5)
    broker.sabonner(AUTH_CLIENT_REQUEST_SECURITY, lambda topic, payload: traitement_securite_requete(broker, topic, payload))
    broker.sabonner(AUTH_ADMIN_REQUEST_SECURITY, lambda topic, payload: traitement_securite_requete(broker, topic, payload))
    
    print("Auth server MQTT lancé")
    print(f"Topics écoutés : {AUTH_CLIENT_REQUEST_SECURITY}, {AUTH_ADMIN_REQUEST_SECURITY}")
    print(f"Topic réponse : {AUTH_RESPONSE_SECURITY}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nArrêt auth server MQTT")
    finally:
        broker.deconnexion()

if __name__ == "__main__":
    main()
