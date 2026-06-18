import threading
import time
import wave
from typing import Literal, Optional
import logging
import uuid


import json

from configuration import CONFIGURATION
from broker.service import get_broker_client 

from langage.schemas.broker import BrokerRequest
from langage.services.model import MODEL_SERVICE
from langage.services.agent import QwenAgent
from security.schema import SessionAuthentifiee

LOGGER = logging.getLogger(__name__)


INSTANCE_AGENT = QwenAgent(MODEL_SERVICE)
CLIENT_BROKER_LLM = get_broker_client(client_id="llm-serveur_client") # à vori si singelton ou pas 


def transformer_message_broker_en_entree_agent(message_recu : str | dict , texte : str  , image_b64: Optional[str]) -> BrokerRequest :
    try: 
        session_authentifiee = None
        if isinstance(message_recu, dict):
            print("message déjà parsé en dict")
        elif isinstance(message_recu, str):
            print("casting json ....")
            message_recu = json.loads(message_recu)
        else :
            raise ValueError("le message reçu n'est ni une string ni un dict",type(message_recu))
        message_recu.get("session_authentifiee",None)
        if message_recu.get("session_authentifiee",None) != None :
            session_authentifiee_dict = message_recu.get("session_authentifiee",None)
            session_authentifiee = SessionAuthentifiee(
                user_id=session_authentifiee_dict.get("user_id",None),
                card_id=session_authentifiee_dict.get("card_id",None),
                access_token=session_authentifiee_dict.get("access_token",None),
                refresh_token=session_authentifiee_dict.get("refresh_token",None),
                prenom=session_authentifiee_dict.get("prenom",None),
                session_id=session_authentifiee_dict.get("session_id",None),
                expire_at=session_authentifiee_dict.get("expire_at",None)
            )

        entree_agent : BrokerRequest = BrokerRequest(
            request_id=str(uuid.uuid4()),
            session_id=message_recu.get("session_id",""),
            text=texte,
            image_url=image_b64,
            session_authentifiee=session_authentifiee
        )
        return entree_agent
    except Exception as e :
        LOGGER.error("erreur dans la transformation du message broker en entrée agent : %s",str(e))
        raise e

def fonction_trigger_declenchement_llm(topic , message_recu : str | dict ):
    MESSAGE_AVERTISSEMENT = "erreur potentielle dans la récupération du message faites attention"
    print("dans le trigger de l'agent")
    try : 
        resultat_llm = ""
        print("type message recu",message_recu,"\n")
        if isinstance(message_recu, dict):
            print("message déjà parsé en dict")
        elif isinstance(message_recu, str):
            print("casting json ....")
            message_recu = json.loads(message_recu)
        else :
            raise ValueError("le message reçu n'est ni une string ni un dict",type(message_recu))
        if topic == CONFIGURATION.broker.topics.llm_topic_ecoute_stt :
            #print("agent a recu quelque chose : ",message_recu)
            texte = message_recu.get("resultat_stt","").get("texte","")
            image = message_recu.get("resultat_stt","").get("image","")
            entree_agent : BrokerRequest = transformer_message_broker_en_entree_agent(message_recu,texte,image)
        elif topic == CONFIGURATION.broker.topics.llm_topic_ecoute_vision :
            texte = message_recu.get("resultat_vision","").get("texte","")
            image = message_recu.get("resultat_vision","").get("image","")
            entree_agent : BrokerRequest = transformer_message_broker_en_entree_agent(message_recu,texte,image)
        else : 
            raise ValueError("la valeur de ce topic est pas attendue",topic)

        resultat_llm = INSTANCE_AGENT.handle(entree_agent,False)
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
    print("Agent en cours de lancement ...")
    
    CLIENT_BROKER_LLM.connexion()
    
    print("apres connexion")
    CLIENT_BROKER_LLM.sabonner(topic_sur_ecoute, fonction_trigger_declenchement_llm)

    LOGGER.info("LLM en écoute sur le topic %s...", topic_sur_ecoute)
    
    stop_event = threading.Event()
    
    try:
        stop_event.wait() 
        
    except KeyboardInterrupt:
        LOGGER.info("Interruption clavier détectée. Arrêt du script...")
        
    finally:
        CLIENT_BROKER_LLM.deconnexion()
        LOGGER.info("Client LLM arrêté proprement.")


#lancement_llm seuleemnt sur stt
if __name__ == "__main__":
    lancement_service_llm()


