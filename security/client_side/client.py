import json
import os
import sys
import time
import uuid
from getpass import getpass


CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CLIENT_DIR)
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CLIENT_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("ENV_FILE", os.path.join(PROJECT_DIR, ".env"))
os.environ.setdefault("BROKER_CREDENTIAL_PROFILE", "CLIENT")
os.chdir(CLIENT_DIR)

from broker.service import get_broker_client  
from configuration import CONFIGURATION  
from common.utils_client import (calculer_signature, connect_card, is_sw_ok, lire_secret_carte, read_word_bytes, verify_csc1, emuler_mode_utilisateur,
verify_csc0,verifier_mode_emule_par_lecture_secret,)  


TOPICS = CONFIGURATION.broker.topics
AUTH_CLIENT_REQUEST_SECURITY = TOPICS.security_client_request_topic
AUTH_RESPONSE_SECURITY = TOPICS.security_response_topic
PUBLIC_CARD_ID_START = 0x01
PUBLIC_CARD_ID_WORDS = 2
FORBIDDEN_CSC_VALUES = {
    bytes.fromhex("00 00 00 00"),
    bytes.fromhex("80 00 00 00"),
    bytes.fromhex("7F FF FF FF"),
    bytes.fromhex("FF FF FF FF"),
}

# CSC0 hardcode uniquement pour simulation du user mode.
CSC0_SIMULATION = bytes.fromhex("AAAAAAAA")

def pin_to_csc1_bytes(pin):
    return int(pin).to_bytes(4, byteorder="little")


def validate_csc1_pin(pin):
    if not (pin.isdigit() and len(pin) == 4):
        return False, "PIN invalide"
    if pin_to_csc1_bytes(pin) in FORBIDDEN_CSC_VALUES:
        return False, "PIN interdit par la carte"
    return True, "ok"


def read_public_card_id(conn):
    raw = bytearray()
    for offset in range(PUBLIC_CARD_ID_WORDS):
        addr = PUBLIC_CARD_ID_START + offset
        data, sw1, sw2 = read_word_bytes(conn, addr)
        if not is_sw_ok(sw1, sw2) or data is None:
            return None, f"lecture card_id addr=0x{addr:02X} SW={sw1:02X}{sw2:02X}"
        raw.extend(data)

    card_id = bytes(raw).rstrip(b"\x00").decode("ascii", errors="ignore").strip()
    if not card_id:
        return None, "card_id public vide"
    return card_id, "ok"


def mqtt_request(action, payload, timeout=10):
    client_id = f"auth-client-{uuid.uuid4().hex[:8]}"
    request_id = uuid.uuid4().hex
    response_holder = {}

    broker = get_broker_client(client_id)
    broker.connexion()
    try:
        time.sleep(0.5)

        def on_response(_topic, message):
            if isinstance(message, str):
                try:
                    message = json.loads(message)
                except Exception:
                    return
            if message.get("request_id") == request_id and message.get("client_id") == client_id:
                response_holder["response"] = message

        broker.sabonner(AUTH_RESPONSE_SECURITY, on_response)
        request = dict(payload)
        request["action"] = action
        request["request_id"] = request_id
        request["client_id"] = client_id

        if not broker.publier(AUTH_CLIENT_REQUEST_SECURITY, request):
            raise RuntimeError("publication MQTT impossible")

        deadline = time.time() + timeout
        while time.time() < deadline:
            if "response" in response_holder:
                return response_holder["response"]
            time.sleep(0.1)

        raise TimeoutError("timeout reponse MQTT auth")
    finally:
        broker.deconnexion()


def demander_challenge(card_id):
    response = mqtt_request("request-challenge", {"card_id": card_id})
    if not response.get("success"):
        raise RuntimeError(response.get("error", "challenge refuse"))

    challenge_id = response.get("challenge_id")
    challenge = response.get("challenge")

    if not challenge_id or not challenge:
        raise RuntimeError("challenge invalide")

    return challenge_id, challenge

def verifier_carte_hmac(card_id, challenge_id, challenge, signature):
    response = mqtt_request(
        "verify-card-hmac",
        {
            "card_id": card_id,
            "challenge_id": challenge_id,
            "signature": signature,
        },
    )
    return response


def authenticate():
    print("Carte inseree")
    try:
        conn = connect_card()
    except Exception as exc:
        print(f"Erreur connexion carte: {exc}")
        return False
    
    sw1, sw2 = verify_csc0(conn, CSC0_SIMULATION)
    if not is_sw_ok(sw1, sw2):
        print(f"Erreur simulation: CSC0 refuse SW={sw1:02X}{sw2:02X}")
        return False

    sw1, sw2 = emuler_mode_utilisateur(conn)
    if not is_sw_ok(sw1, sw2):
        print(f"Erreur simulation: user mode emule refuse SW={sw1:02X}{sw2:02X}")
        return False

    print("Mode utilisateur emule active")

    if not verifier_mode_emule_par_lecture_secret(conn):
        return False

    card_id, message = read_public_card_id(conn)
    if card_id is None:
        print(f"Acces refuse: {message}")
        return False
    print(f"Card ID lu: {card_id}")

    print("Dites votre code PIN")
    pin = getpass("PIN: ").strip()
    valid, message = validate_csc1_pin(pin)
    if not valid:
        print(f"Acces refuse: {message}")
        return False

    sw1, sw2 = verify_csc1(conn, pin_to_csc1_bytes(pin))
    if not is_sw_ok(sw1, sw2):
        print(f"Acces refuse: VERIFY CSC1 SW={sw1:02X}{sw2:02X}")
        return False
    print("VERIFY CSC1: 90 00")

    try:
        challenge_id, challenge = demander_challenge(card_id)
    except Exception as exc:
        print(f"Erreur challenge serveur: {exc}")
        return False

    secret, message = lire_secret_carte(conn)
    if secret is None:
        print(f"Acces refuse: {message}")
        return False

    signature = calculer_signature(secret, card_id, challenge_id, challenge)

    try:
        response = verifier_carte_hmac(card_id, challenge_id, challenge, signature)
    except Exception as exc:
        print(f"Erreur serveur auth via MQTT: {exc}")
        return False

    if not response.get("success"):
        print(f"Acces refuse serveur: {response.get('error', 'erreur inconnue')}")
        return False
    
    print("Acces autorise")
    print(f"user_id: {response.get('user_id')}")
    print(f"status : {response.get('status')}")
    
    
    return True

if __name__ == "__main__":
    try:
        authenticate()
    except KeyboardInterrupt:
        print("\n[*] Interruption (Ctrl+C) - Fermeture...")
