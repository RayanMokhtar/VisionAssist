import paho.mqtt.client as mqtt
import json, ssl, time

client = mqtt.Client()
# Vérification mutuelle via la CA
client.tls_set(ca_certs="certs/ca.crt", tls_version=ssl.PROTOCOL_TLSv1_2)

def on_message(client, userdata, msg):
    status = msg.payload.decode()
    if status == "ACCESS_GRANTED":
        print("ACCÈS AUTORISÉ : Bienvenue Sabine.")
    else:
        print("ACCÈS REFUSÉ : Identifiants incorrects.")

client.on_message = on_message
client.connect("localhost", 8883)
client.subscribe("auth/response")
client.loop_start()

# Simulation d'envoi
payload = {"user": "sabine", "password": "mon_pass_123", "otp": "123456"}
print("Envoi de la requête sécurisée via tunnel MQTTS...")
client.publish("auth/request", json.dumps(payload))

time.sleep(2)
client.loop_stop()