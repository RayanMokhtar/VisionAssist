import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("ENV_FILE", str(PROJECT_DIR / ".env.auth_server"))

from broker.service import get_broker_client
from configuration import CONFIGURATION


DB_FILE = Path(__file__).with_name("users.txt")
TOPICS = CONFIGURATION.broker.topics
AUTH_CLIENT_REQUEST_SECURITY = TOPICS.security_client_request_topic
AUTH_ADMIN_REQUEST_SECURITY = TOPICS.security_admin_request_topic
AUTH_RESPONSE_SECURITY = TOPICS.security_response_topic


def ensure_db():
    if not DB_FILE.exists():
        DB_FILE.write_text("card_id | user_id | role | status | last_auth\n", encoding="utf-8")


def parse_db_line(line):
    parts = [part.strip() for part in line.split("|")]
    if len(parts) != 5:
        return None
    return {
        "card_id": parts[0],
        "user_id": parts[1],
        "role": parts[2],
        "status": parts[3],
        "last_auth": parts[4],
    }


def load_users():
    ensure_db()
    users = {}
    with DB_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("card_id"):
                continue
            row = parse_db_line(line)
            if row:
                users[row["card_id"]] = row
    return users


def save_users(users):
    with DB_FILE.open("w", encoding="utf-8") as f:
        for card_id in sorted(users):
            row = users[card_id]
            f.write(
                f"{row['card_id']} | {row['user_id']} | {row['role']} | "
                f"{row['status']} | {row['last_auth']}\n"
            )


def next_card_id(users):
    max_index = 0
    for card_id in users:
        if not card_id.startswith("VA-"):
            continue
        suffix = card_id[3:]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return f"VA-{max_index + 1:03d}"


def next_user_id(users):
    max_user_id = 0
    for user in users.values():
        user_id = str(user.get("user_id", "")).strip()
        if user_id.isdigit():
            max_user_id = max(max_user_id, int(user_id))
    return str(max_user_id + 1)


def prepare_card():
    users = load_users()
    card_id = next_card_id(users)
    user_id = next_user_id(users)
    users[card_id] = {
        "card_id": card_id,
        "user_id": user_id,
        "role": "user",
        "status": "pending",
        "last_auth": "-",
    }
    save_users(users)
    return users[card_id]


def update_status(card_id, status):
    users = load_users()
    user = users.get(card_id)
    if not user:
        return None
    user["status"] = status
    save_users(users)
    return user


def verify_card(card_id):
    users = load_users()
    user = users.get(card_id)
    if not user:
        return {"success": False, "error": "card_id inconnu"}

    if user["status"] != "active":
        return {
            "success": False,
            "error": f"carte {user['status']}",
            "card_id": card_id,
            "user_id": user["user_id"],
            "role": user["role"],
            "status": user["status"],
        }

    user["last_auth"] = datetime.now().isoformat(timespec="seconds")
    save_users(users)
    return {
        "success": True,
        "card_id": card_id,
        "user_id": user["user_id"],
        "role": user["role"],
        "status": user["status"],
        "last_auth": user["last_auth"],
    }


def build_response(request, payload):
    response = dict(payload)
    response["request_id"] = request.get("request_id")
    return response


def publish_response(broker, request, payload):
    response = build_response(request, payload)
    response["client_id"] = request.get("client_id")
    response["action"] = request.get("action")
    broker.publier(AUTH_RESPONSE_SECURITY, response)


def handle_verify_card(broker, _topic, request):
    card_id = str(request.get("card_id", "")).strip().upper()
    if not card_id:
        publish_response(broker, request, {"success": False, "error": "card_id manquant"})
        return
    publish_response(broker, request, verify_card(card_id))


def handle_prepare_card(broker, _topic, request):
    user = prepare_card()
    publish_response(broker, request, {"success": True, "user": user, "card_id": user["card_id"]})


def handle_activate_card(broker, _topic, request):
    card_id = str(request.get("card_id", "")).strip().upper()
    if not card_id:
        publish_response(broker, request, {"success": False, "error": "card_id requis"})
        return
    user = update_status(card_id, "active")
    if not user:
        publish_response(broker, request, {"success": False, "error": "card_id inconnu"})
        return
    publish_response(broker, request, {"success": True, "user": user})


def handle_status(broker, _topic, request):
    card_id = str(request.get("card_id", "")).strip().upper()
    status = str(request.get("status", "")).strip().lower()
    if not card_id or not status:
        publish_response(broker, request, {"success": False, "error": "card_id et status requis"})
        return
    user = update_status(card_id, status)
    if not user:
        publish_response(broker, request, {"success": False, "error": "card_id inconnu"})
        return
    publish_response(broker, request, {"success": True, "user": user})


def handle_security_request(broker, topic, request):
    action = str(request.get("action", "")).strip().lower()
    if topic == AUTH_CLIENT_REQUEST_SECURITY:
        handlers = {
            "verify-card": handle_verify_card,
        }
    else:
        handlers = {
            "prepare-card": handle_prepare_card,
            "activate-card": handle_activate_card,
            "status": handle_status,
        }
    handler = handlers.get(action)
    if not handler:
        publish_response(broker, request, {"success": False, "error": f"action inconnue: {action}"})
        return
    handler(broker, topic, request)


def main():
    ensure_db()
    broker = get_broker_client("auth-server")
    broker.connexion()
    time.sleep(0.5)
    broker.sabonner(AUTH_CLIENT_REQUEST_SECURITY, lambda topic, payload: handle_security_request(broker, topic, payload))
    broker.sabonner(AUTH_ADMIN_REQUEST_SECURITY, lambda topic, payload: handle_security_request(broker, topic, payload))
    print("Auth server MQTT lance")
    print(f"Topics ecoutes: {AUTH_CLIENT_REQUEST_SECURITY}, {AUTH_ADMIN_REQUEST_SECURITY}")
    print(f"Topic reponse: {AUTH_RESPONSE_SECURITY}")
    print(f"BDD texte: {DB_FILE}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nArret auth server MQTT")
    finally:
        broker.deconnexion()


if __name__ == "__main__":
    main()
