from gradio_client import Client
import time

try:
    client = Client("http://127.0.0.1:7862/")
    result = client.predict(
        message="Bonjour !",
        api_name="/chat"
    )
    print("Result:", result)
except Exception as e:
    print("Error:", e)
