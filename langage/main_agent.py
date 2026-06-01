import time
import wave
from typing import Literal, Optional
import logging
import uuid

from pydantic import BaseModel, Field

import json

from configuration import CONFIGURATION
from broker.service import get_broker_client 
from broker.broker_interface import IBroker 

from langage.api.schemas.broker import BrokerRequest
from langage.api.services.model import model_service
from langage.api.services.agent import QwenAgent

LOGGER = logging.getLogger(__name__)


INSTANCE_AGENT = QwenAgent(model_service)
CLIENT_BROKER_LLM = get_broker_client(client_id="llm-serveur_client") # à vori si singelton ou pas 


def fonction_trigger_declenchement_llm(topic , message_recu : str | dict ):
    MESSAGE_AVERTISSEMENT = "erreur potentielle dans la récupération du message faites attention"
    try : 
        print("message reçu dans le trigger du llm",message_recu)
        resultat_llm = ""
        print("type message recu",type(message_recu))
        if isinstance(message_recu, dict):
            print("message déjà parsé en dict")
        elif isinstance(message_recu, str):
            print("casting json ....")
            message_recu = json.loads(message_recu)
        else :
            raise ValueError("le message reçu n'est ni une string ni un dict",type(message_recu))
        if topic == CONFIGURATION.broker.topics.llm_topic_ecoute_stt :
            texte = message_recu.get("resultat_stt","").get("texte","")
            entree_agent : BrokerRequest = BrokerRequest(
                request_id=str(uuid.uuid4()),
                session_id=message_recu.get("session_id",""),
                text=texte,
                image_url=None
            )
            resultat_llm = INSTANCE_AGENT.handle(entree_agent)
        elif topic == CONFIGURATION.broker.topics.llm_topic_ecoute_vision :
            texte = message_recu.get("resultat_vision","").get("texte","")
            entree_agent : BrokerRequest = BrokerRequest(
                request_id=str(uuid.uuid4()),
                session_id=message_recu.get("session_id",""),
                text=texte,
                image_url=None # pour l'instant à NOne , on va la récupérer après soit en base 64 , soit via un envoi directement
            )
            resultat_llm = INSTANCE_AGENT.handle(entree_agent)
        else : 
            raise ValueError("la valeur de ce topic est pas attendue",topic)

        if resultat_llm.response != None : 
            resultat_json = resultat_llm.model_dump_json()
            print("résultat json à publier",resultat_json)
        else : 
            resultat_json = {"erreur":MESSAGE_AVERTISSEMENT}
        print("résultat json à publier",resultat_json)
        CLIENT_BROKER_LLM.publier(CONFIGURATION.broker.topics.llm_topic_publication,resultat_json)
    except Exception as e :
        LOGGER.error("erreur dans le déclenchement du llm : %s",str(e))
        resultat_json = {"erreur":MESSAGE_AVERTISSEMENT}
        # CLIENT_BROKER_LLM.publier(CONFIGURATION.broker.topics.llm_topic_publication,resultat_json)


#à faire basculer dans le init , et par ailleurs le topic sur écoute on pourrait ajouter le yolo si on veut une réponse rapide sans passer par le llm ? à voir ou juste un buzzer ? 
def lancement_service_llm(topic_sur_ecoute : str = CONFIGURATION.broker.topics.llm_topic_ecoute_stt):
    CLIENT_BROKER_LLM.connexion()
    CLIENT_BROKER_LLM.sabonner(topic_sur_ecoute,fonction_trigger_declenchement_llm)

    LOGGER.info("llm en écoute sur topic %s...",topic_sur_ecoute)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        CLIENT_BROKER_LLM.deconnexion()
        LOGGER.info("llm cli arrêté.")


#lancement_llm seuleemnt sur stt

lancement_service_llm()


