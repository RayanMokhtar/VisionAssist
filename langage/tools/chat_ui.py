"""Temporary web UI to test the Qwen agent without the broker."""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
from requests.exceptions import RequestException


def _save_image(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)
    return tmp.name





def _format_history(
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    return history if history is not None else []


def _chat(
    message: str,
    history: list[dict[str, str]] | None,
    image: Image.Image | None,
    session_id: str,
    latitude: float | None,
    longitude: float | None,
) -> tuple[list[dict[str, str]], str]:
    url = "http://127.0.0.1:8000/chat"
    image_path = _save_image(image)
    payload = {
        "request_id": str(uuid.uuid4()),
        "text": message,
        "image_url": image_path,
        "session_id": session_id,
        "latitude": float(latitude) if latitude is not None else None,
        "longitude": float(longitude) if longitude is not None else None,
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        # Si 'response' est null, on affiche l'erreur
        response_text = data.get("response") or f"Erreur API: {data.get('error', 'Inconnue')}"
    except Exception as e:
        response_text = f"Erreur de communication avec l'API FastAPI : {e}"

    history = _format_history(history)
    user_text = message if image is None else f"{message}\n[image attached]"
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response_text})
    return history, ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen Agent Test") as demo:
        gr.Markdown("# Qwen Agent Test UI")
        session_state = gr.State(str(uuid.uuid4()))
        chatbot = gr.Chatbot(label="Conversation")
        with gr.Row():
            text_input = gr.Textbox(label="Message", placeholder="Ask something…")
            image_input = gr.Image(type="pil", label="Image (optional)")
        with gr.Row():
            lat_input = gr.Number(label="Latitude (Ex: 49.0333 pour Cergy)", value=None)
            lon_input = gr.Number(label="Longitude (Ex: 2.0667 pour Cergy)", value=None)
        send_btn = gr.Button("Send")

        send_btn.click(
            _chat,
            inputs=[text_input, chatbot, image_input, session_state, lat_input, lon_input],
            outputs=[chatbot, text_input],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(share=True)
