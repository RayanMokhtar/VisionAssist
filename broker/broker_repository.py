import json
import logging
from typing import Any, Callable

import paho.mqtt.client as mqtt

from broker.broker_interface import IBroker
from configuration import CONFIGURATION

logger = logging.getLogger(__name__)

conf_broker = CONFIGURATION.broker


#lien serveur : https://mosquitto.org/download/

class MqttClientBroker(IBroker):
    def __init__(self, client_id: str):
        self.conf = conf_broker
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self.client.enable_logger(logger)
        
        if conf_broker.username:
            self.client.username_pw_set(conf_broker.username, conf_broker.password)
        
        if conf_broker.use_tls:
            self.client.tls_set()
        
        self.est_connecte = False
        self.abonnement_topic_callback : dict[str, Callable] = {}
    
    def connexion(self):
        print("configuration broker : ", self.conf)
        self.client.on_connect = self._wrapper_connexion_personnalisee
        self.client.on_message = self._wrapper_lors_envoi_message
        
        self.client.connect(self.conf.host, self.conf.port, self.conf.keepalive)
        self.client.loop_start()
        logger.info(f"Connecté à {self.conf.host}:{self.conf.port}")
    
    def deconnexion(self):
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Déconnecté du broker")
    
    def publier(self, topic: str, payload: Any) -> bool:
        print("publier dans broker : ", topic, payload , "avec configuration : ", self.conf)
        try:
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload, ensure_ascii=False)
            print("publication topic : ", topic, "payload : ", payload)
            result = self.client.publish(topic, str(payload), qos=self.conf.qos, retain=self.conf.retain)
            result.wait_for_publish(timeout=5.0)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"Erreur publication : {e}")
            return False
    
    def sabonner(self, topic: str, callback: Callable ) -> None:
        self.abonnement_topic_callback[topic] = callback
        self.client.subscribe(topic, qos=self.conf.qos)
        logger.info(f"Abonné à {topic}")
    
    def desabonner(self, topic: str) -> None:
        self.abonnement_topic_callback.pop(topic, None)
        self.client.unsubscribe(topic)
        logger.info(f"Désabonné de {topic}")
    
    def is_connected_to_broker(self) -> bool:
        return self.est_connecte
    
    def _wrapper_connexion_personnalisee(self, client, userdata, flags, rc):
        if rc == 0:
            self.est_connecte = True
            logger.info("Connecté au broker")
        else:
            logger.error(f"Erreur connexion : {rc}")
    
    def _wrapper_lors_envoi_message(self, client, userdata, msg):
        #doti avoir topic et payload ce message
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except:
            payload = msg.payload.decode("utf-8")
        
        if topic in self.abonnement_topic_callback:
            self.abonnement_topic_callback[topic](topic, payload)

    def clean_up_ressources(self,topic):
        try : 
            self.desabonner(topic=topic)
        except Exception as e: 
            logger.error(f"cleanup fail {e}")



# broker = get_broker_factory()
# broker.connexion()

# broker.publier("test/normal", {"solide": "solide"})

# def callback(topic, payload):
#     print(f"Reçu : {topic} - {payload}")

# broker.sabonner("test/topic", callback)