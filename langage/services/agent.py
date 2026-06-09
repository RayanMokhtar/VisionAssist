from __future__ import annotations
import os 

import json
import logging
from typing import TypedDict, Annotated, List, Optional

import uuid

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langage.schemas.broker import BrokerRequest, AgentResponse
from langage.services.model import MODEL_SERVICE , ModelService
from langage.services.tools import get_all_tools
from persistance.repository import REPOSITORIES
from langage.memory.short_term import ConversationBuffer
from langage.memory.long_term import long_term_memory
from security.authentification_client import SessionAuthentifiee , session_authentifiee

from persistance.models import Session as SessionModel

logger = logging.getLogger(__name__)


from langgraph.graph.message import add_messages



BASE_PROMPT = """Tu es VisionAssist, un assistant intelligent, bienveillant et proactif conçu pour aider une personne malvoyante dans son quotidien.

Tu peux utiliser tes outils pour obtenir la météo, chercher des lieux, connaître les prochains trains/RER/métros, sauvegarder des notes, ou consulter la mémoire passée.

Sois concis et naturel dans tes réponses vocales. Parle à la 2ème personne du vouvoiement sauf si l'utilisateur préfère le tutoiement.

RÈGLES STRICTES DE RÉPONSE :
1. NE GÉNÈRE AUCUN MONOLOGUE INTERNE. Tu dois donner UNIQUEMENT la réponse finale attendue par l'utilisateur.
2. Il est formellement INTERDIT d'écrire des phrases telles que 'Je dois répondre...', 'L'utilisateur demande...', 'L'outil indique...', ou d'expliquer ce que tu vas faire.
3. Contente-toi de fournir l'information ou la réponse de manière directe, naturelle et fluide.

Pour l'heure et la date, base-toi TOUJOURS sur le résultat le plus récent de l'outil get_current_time présent dans la conversation — jamais sur tes connaissances internes.
Pour les horaires de transport (train, RER, métro, bus), utilise TOUJOURS l'outil get_transit_info avec la destination demandée par l'utilisateur.
Pour sauvegarder un rappel, un mémo ou une note, tu DOIS TOUJOURS utiliser l'outil save_note. Ne dis JAMAIS que tu as noté quelque chose sans avoir appelé l'outil save_note au format XML."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    tool_calls_made: list[str]


class QwenAgent:
    def __init__(self, model_service : ModelService):
        self.model_service = model_service
        self.tools = get_all_tools()
        self.graph = self._build_graph()

    def _build_graph(self):

        tool_node = ToolNode(self.tools)

        def call_model(state: AgentState):

            response = self.model_service.generate_with_tools(state["messages"], self.tools)
            print("la réponse call model ", response)
            if response.tool_calls:
                for tc in response.tool_calls:
                    state["tool_calls_made"].append(tc["name"])
            
            print("state => ",state)

            return {"messages": [response]}

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _build_system_prompt(
        self,
        session_model: SessionModel
    ) -> SystemMessage:
        
        yesterday_context = ""
        if session_model.resume:
            yesterday_context = f"\n\nCONTEXTE PRÉCÉDENT :\n{session_model.resume}"

        profile_context = ""
        try:
            user = REPOSITORIES.users.get(session_model.user_id)
            if user and user.preferences:
                profile_context = f"\n\nPRÉFÉRENCES UTILISATEUR :\n{json.dumps(user.preferences, ensure_ascii=False)}"
        except Exception as e:
            logger.warning(f"Exception lors de la récupération des préférences utilisateur pour user_id={session_model.user_id} : {e}")

        notes_context = ""

        base_prompt = BASE_PROMPT

        full_prompt = base_prompt + yesterday_context + profile_context + notes_context
        return SystemMessage(content=full_prompt)

    def handle(self, request: BrokerRequest) -> AgentResponse:
        """Point d'entrée principal appelé par le main_agent.py."""
        
        logger.info("Réception requête Broker: %s", request.request_id)
        
        # 1. Gestion de la session journalière
        try:
            session = REPOSITORIES.sessions.get(session_id=request.session_authentifiee.session_id) if request.session_authentifiee else None
            user_id = session.user_id if session else None
        except ValueError:
            logger.warning(f"user_id invalide {request.user_id}, session mockée")


        user_content = request.text
        if request.image_url:
            user_content += f"\n[Image attachée: {request.image_url}]"

            
        current_db_msg = REPOSITORIES.messages.create(
            session_id=request.session_authentifiee.session_id,
            requete=user_content
        )

        buffer = ConversationBuffer(str(request.session_authentifiee.session_id))
        lc_messages = buffer.get_langchain_messages()
        
        system_message = self._build_system_prompt(session_model=session)

        if lc_messages:
            last_user_msg = lc_messages[-1]

            # Si une image est fournie, remplacer le dernier message par un HumanMessage multimodal
            # Qwen-VL attend un contenu sous forme de liste [{type: image_url/text, ...}]
            if request.image_url:
                img_path = request.image_url
                # Construire l'URI fichier ou URL selon le contexte
                if os.path.exists(img_path):
                    img_uri = f"file:///{img_path}"
                else:
                    img_uri = img_path  # URL externe

                # Texte du message (sans la mention [Image attachée: ...])
                base_text = last_user_msg.content.replace(f"\n[Image attachée: {img_path}]", "").strip()
                last_user_msg = HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": img_uri}},
                    {"type": "text", "text": base_text},
                ])

            # [system] + [historique sauf dernier msg] + [msg actuel] + [tool_call + tool_result]
            messages_for_llm = [system_message] + lc_messages[:-1] + [last_user_msg]
        else:
            mock_user = HumanMessage(content=request.text)
            print("mock_user => ", mock_user)
            messages_for_llm = [system_message, mock_user]

        initial_state = {
            "messages": messages_for_llm,
            "user_id": request.user_id,
            "session_id": request.session_authentifiee.session_id,
            "tool_calls_made": []
        }

        try:
            final_state = self.graph.invoke(initial_state)
            
            last_message = final_state["messages"][-1]
            response_text = last_message.content

            REPOSITORIES.messages.update_response(current_db_msg, reponse=response_text)
            
            for tc in final_state["tool_calls_made"]:
                logger.info("Tool utilisé: %s", tc)

            return AgentResponse(
                request_id=request.request_id,
                session_id=request.session_authentifiee.session_id,
                user_id=request.user_id,
                response=response_text,
                tool_calls_made=final_state["tool_calls_made"]
            )

        except Exception as e:
            logger.exception("Erreur lors de l'exécution de l'agent")
            return AgentResponse(
                request_id=request.request_id,
                session_id=request.session_authentifiee.session_id,
                user_id=request.user_id,
                error=str(e)
            )

# qwen_agent = QwenAgent(model_service=MODEL_SERVICE)

# broker_request = BrokerRequest(
#     request_id=str(uuid.uuid4()),
#     session_id=str(uuid.uuid4()),
#     user_id="test_user",
#     text="Quel temps fait-il aujourd'hui à Paris ?",
#     session_authentifiee=SessionAuthentifiee(session_id=str(uuid.uuid4()), user_id=1, card_id="test_card_id", expiration="2024-12-31T23:59:59Z") 
# )
# qwen_agent.handle(broker_request)