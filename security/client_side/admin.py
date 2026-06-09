"""
Diagnostic simplifie GemClub-Memo.
Le script admin garde le menu, le broker MQTT et l'orchestration d'enrollement.
Les fonctions bas niveau carte admin sont dans common/utils_admin.py.
"""

import json
import os
import sys
import time
import uuid


CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CLIENT_DIR)
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CLIENT_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PROJECT_DIR)
<<<<<<< HEAD
# os.environ.setdefault("ENV_FILE", os.path.join(PROJECT_DIR, ".env"))
=======
os.environ.setdefault("ENV_FILE", os.path.join(PROJECT_DIR, ".env"))
>>>>>>> feature/Agent-LLM-complet
os.environ.setdefault("BROKER_CREDENTIAL_PROFILE", "ADMIN")
os.chdir(CLIENT_DIR)

from broker.service import get_broker_client
from configuration import CONFIGURATION
from common.utils_admin import (
    afficher_diagnostic,
    ask_user_pin,
    connect_card,
    diagnostic_public,
    is_sw_ok,
    read_public_card_id,
    reset_complet_resilient,
    verifier_detection_carte,
    verify_csc0_resilient,
    write_csc1_pin,
    write_public_card_id,
    ecrire_secret_carte,
    configurer_acces_secret_carte,
)


TOPICS = CONFIGURATION.broker.topics
AUTH_ADMIN_REQUEST_SECURITY = TOPICS.security_admin_request_topic
AUTH_RESPONSE_SECURITY = TOPICS.security_response_topic


def mqtt_request(action: str, payload: dict, timeout=10):
    client_id = f"auth-admin-{uuid.uuid4().hex[:8]}"
    request_id = uuid.uuid4().hex
    response_holder = {}

    broker = get_broker_client(client_id)
    broker.connexion()
    time.sleep(0.5)

    def on_response(_topic, message):
        if isinstance(message, str):
            try:
                print("message", message)
                message = json.loads(message)
            except Exception:
                return
        if message.get("request_id") == request_id and message.get("client_id") == client_id:
            response_holder["response"] = message

    broker.sabonner(AUTH_RESPONSE_SECURITY, on_response)

    if not isinstance(payload, dict):
        print("type avant", type(payload))
        payload = dict(payload)
        print("type apres", type(payload))

    request = payload.copy()
    request["action"] = action
    request["request_id"] = request_id
    request["client_id"] = client_id

    print("request", request)

    if not broker.publier(AUTH_ADMIN_REQUEST_SECURITY, request):
        broker.deconnexion()
        raise RuntimeError("publication MQTT impossible")

    deadline = time.time() + timeout
    while time.time() < deadline:
        if "response" in response_holder:
            broker.deconnexion()
            return response_holder["response"]
        time.sleep(0.1)

    broker.deconnexion()
    raise TimeoutError("timeout reponse MQTT auth")


def enrollement_card(conn, csc0, message_broker):
    print("\n" + "=" * 70)
    print("ENROLLEMENT CARTE")
    print("=" * 70)

    existing_card_id, message = read_public_card_id(conn)
    if existing_card_id is None:
        print(message)
        return
    if existing_card_id:
        print(f"Carte deja initialisee avec card_id={existing_card_id}. Operation arretee.")
        return

    print("\n[*] Verification CSC0...")
    conn, sw1, sw2 = verify_csc0_resilient(conn, csc0)
    if not is_sw_ok(sw1, sw2):
        print(f"CSC0 refuse SW={sw1:02X}{sw2:02X}")
        return

    try:
        response = mqtt_request("prepare-card", message_broker)
    except Exception as exc:
        print(f"Erreur serveur auth via MQTT pendant preparation: {exc}")
        return

    if not response.get("success"):
        print(f"Preparation serveur refusee: {response.get('error')}")
        return

    card_id = response["card_id"]
    user_id = response["user_id"]
    secret = response.get("secret")

    if not secret:
        print("Preparation serveur invalide: secret manquant")
        return

    pin, message = ask_user_pin()
    if pin is None:
        print(message)
        return

    print(f"Serveur: card_id={card_id}, user_id={user_id}")

    conn, ok, message = write_csc1_pin(conn, csc0, pin)
    if not ok:
        print(message)
        return
    
    conn, ok, message = ecrire_secret_carte(conn, csc0, secret)
    if not ok:
        print(message)
        return

    conn, ok, message = write_public_card_id(conn, csc0, card_id)
    if not ok:
        print(message)
        return
    
    conn, ok, message = configurer_acces_secret_carte(conn, csc0)
    if not ok:
        print(message)
        return

    try:
        response = mqtt_request("activate-card", {"card_id": card_id})
    except Exception as exc:
        print(f"Carte ecrite, mais activation serveur MQTT impossible: {exc}")
        return

    if not response.get("success"):
        print(f"Carte ecrite, mais activation serveur refusee: {response.get('error')}")
        return

    print("\nCarte initialisee et activee cote serveur")
    print(f"card_id: {card_id}")
    print(f"user_id: {user_id}")
    print("Retirez et reinserez la carte pour appliquer la protection du secret.")


def demander_csc0():
    csc0_input = input("\nCSC0 (8 hex) : ").strip().upper()

    if len(csc0_input) != 8:
        print("[-] CSC0 invalide (doit faire 8 caracteres hexa)")
        return None

    try:
        return bytes.fromhex(csc0_input)
    except ValueError:
        print("[-] Format hexa incorrect")
        return None


def demander_infos_utilisateur():
    nom = input("entrez nom: ").strip()
    prenom = input("entrez prenom: ").strip()
    adresse = input("entrez adresse: ").strip()
    preferences = input("preferences peut etre vide: ").strip()

    message = {
        "nom": nom,
        "prenom": prenom,
        "adresse": adresse,
        "preferences": preferences,
    }
    print("message", message, type(message))
    return message


def afficher_menu():
    print("\n" + "=" * 70)
    print("MENU PRINCIPAL")
    print("=" * 70)
    print("\n1) Afficher toutes les zones de la carte (dump complet)")
    print("2) Diagnostic public")
    print("3) Reset complet (User Areas + Access Conditions + Protected)")
    print("4) Enrollement carte")
    print("5) Quitter")
    return input("\nChoix : ").strip()


def main():
    """Menu principal."""
    try:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC GEMCLUB-MEMO")
        print("=" * 70)

        if not verifier_detection_carte():
            return

        print("\nAuthentification administrateur requise (CSC0)")
        print("    Necessaire pour acceder a toutes les zones")

        csc0 = demander_csc0()
        if csc0 is None:
            return

        while True:
            choice = afficher_menu()

            if choice == "1":
                try:
                    conn = connect_card()
                    afficher_diagnostic(conn, csc0)
                except Exception as e:
                    print(f"[-] Erreur : {e}")
            elif choice == "2":
                try:
                    conn = connect_card()
                    diagnostic_public(conn)
                except Exception as e:
                    print(f"[-] Erreur : {e}")
            elif choice == "3":
                try:
                    conn = connect_card()
                    reset_complet_resilient(conn, csc0)
                except Exception as e:
                    print(f"[-] Erreur : {e}")
            elif choice == "4":
                try:
                    conn = connect_card()
                    message = demander_infos_utilisateur()
                    enrollement_card(conn, csc0, message)
                except Exception as e:
                    print(f"[-] Erreur : {e}")
            elif choice == "5":
                print("\n[*] Au revoir")
                break
            else:
                print("[-] Choix invalide")
    except KeyboardInterrupt:
        print("\n\n[*] Interruption (Ctrl+C) - Fermeture...")


if __name__ == "__main__":
    main()
