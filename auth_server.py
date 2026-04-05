import paho.mqtt.client as mqtt
import json, ssl, jwt, datetime
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import os
from dotenv import load_dotenv


load_dotenv()  

JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET non défini dans les variables d'environnement")


ph = PasswordHasher()

def verify_jwt(token):
    """Vérifie le badge envoyé par la Jetson."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return True, payload["sub"]  
    except jwt.ExpiredSignatureError:
        print(" Token expiré présenté.")
        return False, "TOKEN_EXPIRED"
    except jwt.InvalidTokenError:
        print(" Token invalide ou modifié détecté !")
        return False, "INVALID_TOKEN"

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        
        # --- CAS 1 : DEMANDE DE CONNEXION (MFA) ---
        if msg.topic == "auth/request":
            user = data.get("user")
            pwd_attempt = data.get("password")
            bio_attempt = data.get("biometric_data")

            with open('users_db.json', 'r') as f:
                db = json.load(f)

            if user in db:
                # Vérification Argon2id
                try:
                    ph.verify(db[user]["password_hash"], pwd_attempt)
                    pwd_ok = True
                except VerifyMismatchError:
                    pwd_ok = False

                # Vérification Empreinte
                bio_ok = (bio_attempt == db[user]["biometric_signature"])

                if pwd_ok and bio_ok:
                    # Génération du Token (valide 1 minute pour ton test)
                    payload = {
                        "sub": user,
                        "iat": datetime.datetime.utcnow(),
                        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=1),
                        "role": "admin"
                    }
                    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
                    
                    print(f" MFA Validé pour {user} -> Token envoyé.")
                    response = {"status": "OK_ACCES", "token": token}
                    client.publish("auth/response", json.dumps(response))
                else:
                    print(f" Échec MFA pour {user}")
                    client.publish("auth/response", json.dumps({"status": "ERREUR_ACCES"}))

        # --- CAS 2 : DEMANDE D'ACTION (Vérification JWT) ---
        elif msg.topic == "jetson/action":
            token_recu = data.get("token")
            action = data.get("action")
            
            # On appelle notre fonction de vérification
            is_valid, result = verify_jwt(token_recu)

            if is_valid:
                print(f"[JWT VALIDE] Action '{action}' autorisée pour {result}")
                # Réponse de succès à la Jetson
                res = {"status": "SUCCESS", "message": f"Action {action} exécutée."}
                client.publish("jetson/response", json.dumps(res))
            else:
                print(f" [JWT REFUSÉ] Raison : {result}")
                # On informe le client qu'il doit se reconnecter
                res = {"status": "RE_AUTH_REQUIRED", "reason": result}
                client.publish("auth/response", json.dumps(res))

    except Exception as e:
        print(f" Erreur : {e}")

# Configuration MQTT
client = mqtt.Client()
client.tls_set(ca_certs="certs/ca.crt", tls_version=ssl.PROTOCOL_TLSv1_2)
client.on_message = on_message

client.connect("localhost", 8883)

# IMPORTANT : S'abonner aux deux topics
client.subscribe("auth/request")
client.subscribe("jetson/action")

print(" Serveur sécurisé en attente (MFA & JWT)...")

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nArrêt demandé (Ctrl-C). Fermeture propre...")
finally:
    try:
        client.unsubscribe("auth/request")
        client.unsubscribe("jetson/action")
    except Exception:
        pass
    client.disconnect()
    print("Déconnecté du broker. Fin.")