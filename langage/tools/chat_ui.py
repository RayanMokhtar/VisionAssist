"""Web UI to test the Qwen agent directly using Python handlers."""

from __future__ import annotations

import sys
import json
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


def _chat_stream(
    message: str,
    history: list[dict[str, str]] | None,
    image: Image.Image | None,
    session_id: str,
    latitude: float | None,
    longitude: float | None,
):
    image_path = _save_image(image)
    
    # Mocking authenticated session for the UI testing
    session_auth = SessionAuthentifiee(
        user_id="test_user",
        session_id=session_id,
        card_id="test_card_id",
        access_token="mock_token"
    )
    
    # Creating the broker request
    request = BrokerRequest(
        request_id=str(uuid.uuid4()),
        session_id=session_id,
        user_id="test_user",
        text=message,
        image_url=image_path,
        session_authentifiee=session_auth
    )
    
    history = _format_history(history)
    user_text = message if image is None else f"{message}\n[image attached]"
    history.append({"role": "user", "content": user_text})
    
    # Initialize the outputs for streaming
    history.append({"role": "assistant", "content": "*(L'agent réfléchit...)*"})
    state_display = "Démarrage de l'agent..."
    tools_display = "Aucun outil appelé pour l'instant."
    
    yield history, "", state_display, tools_display
    
    try:
        # We iterate over the streaming events from LangGraph
        for item in qwen_agent.stream_handle(request):
            if item["type"] == "stream_event":
                event = item["event"]
                # Display the internal state dynamically
                state_str = json.dumps(event, default=lambda x: str(x), ensure_ascii=False, indent=2)
                # Show the last 2000 chars to avoid overflowing UI if too large
                if len(state_str) > 2000:
                    state_str = state_str[:2000] + "\n... [tronqué]"
                
                # Determine current node
                current_node = list(event.keys())[0] if event else "unknown"
                state_display = f"**Node actuel:** `{current_node}`\n\n```json\n{state_str}\n```"
                
                yield history, "", state_display, tools_display
                
            elif item["type"] == "final_response":
                response = item["response"]
                if response.error:
                    response_text = f"Erreur de l'agent: {response.error}"
                else:
                    response_text = response.response
                
                # Format the tools display
                if response.tool_calls_made:
                    tools_json = json.dumps(response.tool_calls_made, indent=2, ensure_ascii=False)
                    tools_display = f"```json\n{tools_json}\n```"
                else:
                    tools_display = "Aucun outil n'a été appelé."
                    
                history[-1]["content"] = response_text
                state_display += "\n\n**Agent terminé.**"
                
                yield history, "", state_display, tools_display

    except Exception as e:
        history[-1]["content"] = f"Erreur lors de l'exécution de l'agent : {e}"
        yield history, "", f"Erreur critique: {e}", tools_display


def get_graph_image():
    """Génère l'image PNG du graphe LangGraph."""
    try:
        return qwen_agent.graph.get_graph().draw_mermaid_png()
    except Exception as e:
        print("Could not generate graph image:", e)
        return None


def build_ui() -> gr.Blocks:
    # Use a wide layout to see the chat and the right panel side by side
    with gr.Blocks(title="Qwen Agent Test", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Interface de Test Dynamique (Qwen Agent)")
        session_state = gr.State(str(uuid.uuid4()))
        
        with gr.Row():
            # Left panel: Chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=600)
                with gr.Row():
                    text_input = gr.Textbox(label="Message", placeholder="Posez une question...", scale=4)
                    send_btn = gr.Button("Envoyer", variant="primary", scale=1)
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Image (optionnelle)")
                with gr.Row():
                    lat_input = gr.Number(label="Latitude (Ex: 49.033)", value=None)
                    lon_input = gr.Number(label="Longitude (Ex: 2.066)", value=None)
                    
            # Right panel: Agent internal state and Graph
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("État Dynamique"):
                        state_output = gr.Markdown(value="_En attente du lancement..._")
                    with gr.Tab("Tools Appelés"):
                        tools_output = gr.Markdown(value="_Aucun outil_")
                    with gr.Tab("Graphe de l'Agent"):
                        graph_image = get_graph_image()
                        if graph_image:
                            # Save the bytes to a temp file so Gradio can load it
                            tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                            tmp_img.write(graph_image)
                            tmp_img.flush()
                            gr.Image(value=tmp_img.name, label="LangGraph Flow", interactive=False)
                        else:
                            gr.Markdown("Impossible de générer le graphe (pygraphviz/mermaid-cli requis).")

        send_btn.click(
            _chat_stream,
            inputs=[text_input, chatbot, image_input, session_state, lat_input, lon_input],
            outputs=[chatbot, text_input, state_output, tools_output],
        )
        
        text_input.submit(
            _chat_stream,
            inputs=[text_input, chatbot, image_input, session_state, lat_input, lon_input],
            outputs=[chatbot, text_input, state_output, tools_output],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(share=True)
