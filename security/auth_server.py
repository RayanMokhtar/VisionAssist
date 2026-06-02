import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from httpx import request

from broker.service import get_broker_client
from configuration import CONFIGURATION

from persistance.models import User, Carte
from persistance.repository import UserRepository, CarteRepository , REPOSITORIES

# os.environ.setdefault("BROKER_CREDENTIAL_PROFILE", "AUTH_SERVER")


# DB_FILE = Path(__file__).with_name("users.txt")
TOPICS = CONFIGURATION.broker.topics
AUTH_CLIENT_REQUEST_SECURITY = TOPICS.security_client_request_topic
AUTH_ADMIN_REQUEST_SECURITY = TOPICS.security_admin_request_topic
AUTH_RESPONSE_SECURITY = TOPICS.security_response_topic



def generer_card_id():
    count_users = REPOSITORIES.users.db.query(User).count()
    print("Nombre d'utilisateurs dans la base de données :", count_users)
    return f"VA-{count_users + 1}"






def verify_card(card_id):
    carte = REPOSITORIES.cartes.get(card_id=card_id)
    user_id = carte.get("user_id") if carte else None
    user = REPOSITORIES.users.get(user_id=user_id) if user_id else None
    print("user_id",user_id,"user",user)
    if not user:
        return {"success": False,"error": "card_id inconnu"}

    if carte.get("statut") != "active":
        return {
            "success": False,
            "error": f"carte {carte['statut']}",
            "card_id": card_id,
            "user_id": user["user_id"],
            "role": user["role"],
            "status": user["status"],
        }

    user = REPOSITORIES.users.update(user, derniere_connexion=datetime.now())

    dictionnaire_sortie = {
        "success": True,
        "card_id": card_id,
        "user_id": user["user_id"],
        "status": carte["statut"],
        "derniere_connexion": user["derniere_connexion"],
    }
    return dictionnaire_sortie





def handle_verify_card(broker, _topic, request):
    card_id = str(request.get("card_id", "")).strip().upper()
    if not card_id:
        publish_response_securite(broker, request, {"success": False, "error": "card_id manquant"})
        return
    verification_de_la_carte : dict = verify_card(card_id)
    publish_response_securite(broker, request, verification_de_la_carte)



def preparer_logique_enrollement(message):
    user = REPOSITORIES.users.create(nom=message.get("nom"), prenom=message.get("prenom"),preferences=message.get("preferences"), adresse=message.get("adresse", ""))
    if user is not None : 
        print("Utilisateur créé en base de données : ", user)
    else :
        print("Erreur création utilisateur en base de données")

    carte_id = generer_card_id()
    carte = REPOSITORIES.cartes.create(user_id=user.id, statut="pending", card_id=carte_id) #auto increment apres create du user en haut ... 
    if carte is not None : 
        print("Carte créée en base de données : ", carte)
    else :
        print("Erreur création carte en base de données")
    
    return carte_id, user.id


def publish_response_securite(broker, request, payload):
    response = dict(payload)
    response["request_id"] = request.get("request_id")    
    response["client_id"] = request.get("client_id")
    response["action"] = request.get("action")
    broker.publier(AUTH_RESPONSE_SECURITY, response)

def handle_logique_enrollement(broker, topic, request):
    card_id, user_id = preparer_logique_enrollement(request)
    publish_response_securite(broker, request, {"success": True, "card_id": card_id, "user_id": user_id})


def handle_activate_card(broker, _topic, request):
    print("handle activate card",request)
    card_id = str(request.get("card_id", "")).strip().upper()
    if not card_id:
        publish_response_securite(broker, request, {"success": False, "error": "card_id requis"})
        return
    carte = REPOSITORIES.cartes.update_status(REPOSITORIES.cartes.db.query(Carte).filter_by(card_id=card_id).first(), "active")
    if not carte:
        publish_response_securite(broker, request, {"success": False, "error": "card_id inconnu"})
        return
    publish_response_securite(broker, request, {"success": True, "carte": carte.to_dict()})



def traitement_securite_requete(broker, topic, request):
    action = str(request.get("action", "")).strip().lower()
    #authentification client
    if topic == AUTH_CLIENT_REQUEST_SECURITY:
        handlers = {
            "verify-card": handle_verify_card,
        }

    elif topic == AUTH_ADMIN_REQUEST_SECURITY:
        handlers = {
            "prepare-card": handle_logique_enrollement,
            "activate-card": handle_activate_card #activer carte qui était en pending
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
    print("Auth server MQTT lance")
    print(f"Topics ecoutess: {AUTH_CLIENT_REQUEST_SECURITY}, {AUTH_ADMIN_REQUEST_SECURITY}")
    print(f"Topic reponse: {AUTH_RESPONSE_SECURITY}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nArret auth server MQTT")
    finally:
        broker.deconnexion()


if __name__ == "__main__":
    main()
