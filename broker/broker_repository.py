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
        
        self._connected = False
        self.abonnement_topic_callback : dict[str, Callable] = {}
    
    def connexion(self):
        self.client.on_connect = self._wrapper_connexion_personnalisee
        self.client.on_message = self._wrapper_hook_lors_envoi_message
        
        self.client.connect(self.conf.host, self.conf.port, self.conf.keepalive)
        self.client.loop_start()
        logger.info(f"Connecté à {self.conf.host}:{self.conf.port}")
    
    def deconnexion(self):
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Déconnecté du broker")
    
    def publier(self, topic: str, payload: Any) -> bool:
        try:
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload, ensure_ascii=False)
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
        return self._connected
    
    def _wrapper_connexion_personnalisee(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("Connecté au broker")
        else:
            logger.error(f"Erreur connexion : {rc}")
    
    def _wrapper_hook_lors_envoi_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except:
            payload = msg.payload.decode("utf-8")
        
        if topic in self.abonnement_topic_callback:
            self.abonnement_topic_callback[topic](topic, payload)

def get_broker_factory() -> IBroker:
    if conf_broker.type_broker == "mosquitto":
        return MqttClientBroker(client_id=conf_broker.client_id)
    
    elif conf_broker.type_broker == "RabbitMQ":
        raise NotImplementedError("RabbitMQ pas implémenté")
    
    else:
        raise ValueError(f"Type broker inconnu : {conf_broker.type_broker}")
    



# broker = get_broker_factory()
# broker.connexion()

# broker.publier("test/normal", {"solide": "solide"})

# def callback(topic, payload):
#     print(f"Reçu : {topic} - {payload}")

# broker.sabonner("test/topic", callback)