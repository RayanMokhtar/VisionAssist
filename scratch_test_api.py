import requests
import uuid

url = "http://127.0.0.1:8000/chat"
payload = {
    "request_id": str(uuid.uuid4()),
    "text": "quel temps fait til a cergy ?",
    "session_id": "user_default",
    "user_id": "user_default"
}

print("Envoi de la requete...")
try:
    resp = requests.post(url, json=payload, timeout=60)
    print("Status:", resp.status_code)
    print("Response JSON:", resp.json())
except Exception as e:
    print("Error:", e)
