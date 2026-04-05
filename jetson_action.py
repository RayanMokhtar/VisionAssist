import paho.mqtt.client as mqtt
import json, ssl, time, os

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        if data.get("status") == "RE_AUTH_REQUIRED":
            print("\n ALERTE : Votre badge a expiré ou est invalide !")
            if os.path.exists("session_token.jwt"):
                os.remove("session_token.jwt") # On nettoie le fichier inutile
            print(" Action : Relancez 'connexion.py' pour obtenir un nouveau badge.")
        else:
            print(f"\n Réponse Serveur : {data}")
    except Exception as e:
        print(f" Erreur : {e}")

# 1. Lecture du badge
if not os.path.exists("session_token.jwt"):
    print(" Erreur : Aucun badge trouvé. Connectez-vous d'abord.")
    exit()

with open("session_token.jwt", "r") as f:
    token_enregistre = f.read()

# 2. Configuration MQTT
client = mqtt.Client()
client.tls_set(ca_certs="certs/ca.crt", tls_version=ssl.PROTOCOL_TLSv1_2)
client.on_message = on_message
client.connect("localhost", 8883)
client.subscribe("jetson/response")
client.subscribe("auth/response") # Pour écouter les ordres de re-connexion
client.loop_start()

# 3. Tentative d'action avec le token uniquement
print(f" Envoi d'une requête sécurisée avec le JWT...")
payload = {
    "token": token_enregistre,
    "action": "LIRE_DONNEES_SENSIBLES"
}
client.publish("jetson/action", json.dumps(payload))

# Attente pour voir si le serveur accepte ou rejette
time.sleep(3)
client.disconnect()