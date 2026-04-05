import paho.mqtt.client as mqtt
import json, ssl
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

ph = PasswordHasher()

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload)
        user = payload.get("user")
        pwd_attempt = payload.get("password")
        otp_attempt = payload.get("otp")

        with open('users_db.json', 'r') as f:
            db = json.load(f)

        if user in db:
            try:
                # Vérification de l'identité via Argon2id
                ph.verify(db[user]["password_hash"], pwd_attempt)
                # Vérification du second facteur (OTP)
                if otp_attempt == db[user]["otp_secret"]:
                    print(f"AUTHENTIFICATION RÉUSSIE : {user}")
                    client.publish("auth/response", "ACCESS_GRANTED")
                    return
            except VerifyMismatchError:
                pass
        
        print(f"ÉCHEC AUTHENTIFICATION : {user}")
        client.publish("auth/response", "ACCESS_DENIED")

    except Exception as e:
        print(f"Erreur système : {e}")

# Configuration MQTT avec TLS
client = mqtt.Client()
client.tls_set(ca_certs="certs/ca.crt", tls_version=ssl.PROTOCOL_TLSv1_2)
client.on_message = on_message

client.connect("localhost", 8883)
client.subscribe("auth/request")
print("Serveur d'Auth en attente sur MQTTS (Port 8883)...")
client.loop_forever()