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

from langage.api.schemas.broker import BrokerRequest
from langage.api.services.agent import QwenAgent
from langage.api.services.model import model_service


def _ensure_model_loaded() -> None:
    if model_service.model is None or model_service.processor is None:
        model_service.load()


def _save_image(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)
    return tmp.name


_AGENT = QwenAgent(model_service)


def _format_history(
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    return history if history is not None else []


def _chat(
    message: str,
    history: list[dict[str, str]] | None,
    image: Image.Image | None,
    session_id: str,
) -> tuple[list[dict[str, str]], str]:
    _ensure_model_loaded()

    image_path = _save_image(image)
    request = BrokerRequest(
        request_id=str(uuid.uuid4()),
        session_id=session_id,
        text=message,
        image_url=image_path,
    )
    response = _AGENT.handle(request)
    history = _format_history(history)

    user_text = message if image is None else f"{message}\n[image attached]"
    history.append({"role": "user", "content": user_text})
    print("contneu de la réponse", response.response)
    history.append({"role": "assistant", "content": response.response or ""})
    return history, ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen Agent Test") as demo:
        gr.Markdown("# Qwen Agent Test UI")
        session_state = gr.State(str(uuid.uuid4()))
        chatbot = gr.Chatbot(label="Conversation")
        with gr.Row():
            text_input = gr.Textbox(label="Message", placeholder="Ask something…")
            image_input = gr.Image(type="pil", label="Image (optional)")
        send_btn = gr.Button("Send")

        send_btn.click(
            _chat,
            inputs=[text_input, chatbot, image_input, session_state],
            outputs=[chatbot, text_input],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(share=True)
