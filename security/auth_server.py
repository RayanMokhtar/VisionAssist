import hmac
import json
import os
import sys
import time
import hashlib
import secrets
import jwt
import ed25519
from datetime import datetime
from pathlib import Path

from httpx import request

from broker.service import get_broker_client
from configuration import CONFIGURATION

from persistance.models import User, Carte
from persistance.repository import UserRepository, CarteRepository , REPOSITORIES
from security.jwt_service import create_access_token, verify_access_token, create_refresh_token


TOPICS = CONFIGURATION.broker.topics
AUTH_CLIENT_REQUEST_SECURITY = TOPICS.security_client_request_topic
AUTH_ADMIN_REQUEST_SECURITY = TOPICS.security_admin_request_topic
AUTH_RESPONSE_SECURITY = TOPICS.security_response_topic




## Utils region
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



### end utils region

class Authentification : 

    @staticmethod
    def verify_card(card_id):
        carte = REPOSITORIES.cartes.get(card_id=card_id)
        print("carte :",carte)
        user_id = carte.user_id if carte else None
        user = REPOSITORIES.users.get(user_id=user_id) if user_id else None
        print("user_id",user_id,"user",user)
        if not user:
            return {"success": False,"error": "card_id inconnu"}

        if carte.statut.value != "active":
            return {
                "success": False,
                "error": carte.statut.value,
                "card_id": card_id,
                "user_id": user.id
            }

        user = REPOSITORIES.users.update(user, derniere_connexion=datetime.now())
        dictionnaire_sortie = {
            "success": True,
            "card_id": card_id,
            "user_id": user.id,
            "status": carte.statut.value,
            "derniere_connexion": str(user.derniere_connexion),
        }
        return dictionnaire_sortie


    @staticmethod
    def handle_verify_card(broker, _topic, request):
        card_id = str(request.get("card_id", "")).strip().upper()
        if not card_id:
            publish_response_securite(broker, request, {"success": False, "error": "card_id manquant"})
            return
        verification_de_la_carte : dict = Authentification.verify_card(card_id)
        print("verification_de_la_carte",verification_de_la_carte)
        if verification_de_la_carte.get("success") :
            user_id = verification_de_la_carte.get("user_id")
            access_token = create_access_token(user_id=user_id, card_id=card_id)
            refresh_token = create_refresh_token(user_id=user_id, card_id=card_id) # pas forcement utile pour l'instnat le user id et card_id on l'utilisera après
            verification_de_la_carte["access_token"] = access_token
            verification_de_la_carte["refresh_token"] = refresh_token
        publish_response_securite(broker, request, verification_de_la_carte)



## authentification ...


class CreationEnrollementCarte : 

    @staticmethod
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


    @staticmethod
    def handle_logique_enrollement(broker, topic, request):
        card_id, user_id = CreationEnrollementCarte.preparer_logique_enrollement(request)
        publish_response_securite(broker, request, {"success": True, "card_id": card_id, "user_id": user_id})


    @staticmethod
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
        secret = CreationEnrollementCarte.generer_secret_from_card_id(card_id)
        publish_response_securite(broker, request, {"success": True, "carte": carte.to_dict()})

    @staticmethod
    def generer_secret_from_card_id(card_id):
        """fonction qui va générer un secret """
        secret = hmac.new(CONFIGURATION.jwt.secret_key.encode(), card_id.encode(), hashlib.sha256).hexdigest()
        


    
def traitement_securite_requete(broker, topic, request):
    action = str(request.get("action", "")).strip().lower()
    #authentification client => vérifier si la carte est valide et active
    if topic == AUTH_CLIENT_REQUEST_SECURITY:
        handlers = {
            "verify-card": Authentification.handle_verify_card,
        }

    elif topic == AUTH_ADMIN_REQUEST_SECURITY:
        handlers = {
            "prepare-card": CreationEnrollementCarte.handle_logique_enrollement,
            "activate-card": CreationEnrollementCarte.handle_activate_card #activer carte qui était en pending
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
