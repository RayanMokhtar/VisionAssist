"""Web UI to test the Qwen agent directly using Python handlers."""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

# Add the project root to sys.path to allow absolute imports
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langage.services.agent import QwenAgent
from langage.services.model import MODEL_SERVICE
from langage.schemas.broker import BrokerRequest
from security.authentification_client import SessionAuthentifiee

# Initialize the agent
qwen_agent = QwenAgent(model_service=MODEL_SERVICE)


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
    
    image_path = _save_image(image)
    
    # Mocking authenticated session for the UI testing
    session_auth = SessionAuthentifiee(
        user_id="test_user",
        session_id=session_id,
        card_id="test_card_id",
        access_token="mock_token"
    )
    
    # Creating the broker request simulating an incoming request
    request = BrokerRequest(
        request_id=str(uuid.uuid4()),
        session_id=session_id,
        user_id="test_user",
        text=message,
        image_url=image_path,
        session_authentifiee=session_auth
    )
    
    try:
        # Directly invoking the agent handler
        response = qwen_agent.handle(request)
        
        if response.error:
            response_text = f"Erreur de l'agent: {response.error}"
        else:
            response_text = response.response
            
    except Exception as e:
        response_text = f"Erreur lors de l'exécution de l'agent : {e}"

    history = _format_history(history)
    user_text = message if image is None else f"{message}\n[image attached]"
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response_text})
    return history, ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen Agent Test") as demo:
        gr.Markdown("# Qwen Agent Test UI (Direct Handler)")
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
