import paho.mqtt.client as mqtt
import json
import ssl
import hashlib
import time

def simulate_fingerprint_scan():
    raw_data = "empreinte_digitale_sabine_2024"
    return hashlib.sha256(raw_data.encode()).hexdigest()
def on_message(client, userdata, msg):
    try:
        # On décode la réponse JSON du serveur
        response = json.loads(msg.payload.decode())
        
        if response.get("status") == "OK_ACCES":
            token = response.get("token")
            print(f"\n [AUTH RÉUSSIE] Bienvenue Sabine.")
            print(f" JWT reçu : {token[:30]}...") # On affiche le début du badge
            
            # SAUVEGARDE DU TOKEN (Pour les futures requêtes)
            with open("session_token.jwt", "w") as f:
                f.write(token)
            print(" Token de session enregistré localement.")
            
        else:
            print("\n [ÉCHEC] Identifiants ou empreinte incorrects.")
            
    except Exception as e:
        print(f" Erreur lors de la lecture de la réponse : {e}")

client = mqtt.Client()

client.tls_set(ca_certs="certs/ca.crt", tls_version=ssl.PROTOCOL_TLSv1_2)
client.on_message = on_message
client.connect("localhost", 8883)

# S'abonner AVANT d'envoyer la requête pour recevoir la réponse
client.subscribe("auth/response")
client.loop_start()

print("--- Terminal VisionAssist (Jetson Orin Nano) ---")
pwd = input("Entrez votre mot de passe : ")
print("Posez votre doigt sur le capteur...")
time.sleep(1)

# Option : récupérer l'empreinte stockée dans la BDD locale
try:
    with open("users_db.json", "r") as f:
        db = json.load(f)
    bio_data = db.get("sabine", {}).get("biometric_signature", simulate_fingerprint_scan())
except Exception:
    bio_data = simulate_fingerprint_scan()

payload = {
    "user": "sabine",
    "password": pwd,
    "biometric_data": bio_data
}

print("Envoi de la requête d'authentification sécurisée...")
client.publish("auth/request", json.dumps(payload))

# attendre la réponse (10s)
timeout = 10
start_time = time.time()
while time.time() - start_time < timeout:
    time.sleep(0.1)

client.disconnect()
print("Fermeture du tunnel.")