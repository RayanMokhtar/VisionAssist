import paho.mqtt.client as mqtt

from broker.broker_repository import MqttClientBroker , conf_broker
from broker.broker_interface import IBroker
from configuration import CONFIGURATION 



def get_broker_client(client_id : str | None ) -> IBroker:
    if not client_id : 
        client_id = conf_broker.client_id
    if conf_broker.type_broker == "mosquitto":
        return MqttClientBroker(client_id=client_id)
    
    elif conf_broker.type_broker == "RabbitMQ":
        raise NotImplementedError("RabbitMQ pas implémenté")
    
    else:
        raise ValueError(f"Type broker inconnu : {conf_broker.type_broker}")
    

# changer client_id côté serveur dans le .env 
BROKER_CLIENT = get_broker_client()