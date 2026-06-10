from __future__ import annotations

import os 
import json
import logging
import uuid
from typing import TypedDict, Annotated, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from langage.schemas.broker import BrokerRequest, AgentResponse
from langage.services.model import MODEL_SERVICE, ModelService
from langage.services.tools import get_all_tools
from langage.memory.short_term import ConversationBuffer
from langage.memory.long_term import long_term_memory
from persistance.repository import REPOSITORIES
from persistance.models import Session as SessionModel
from security.authentification_client import SessionAuthentifiee
from configuration import CONFIGURATION
logger = logging.getLogger(__name__)

BASE_PROMPT = f"""Tu es {CONFIGURATION.nom_assistant}, un assistant intelligent, bienveillant et proactif conçu pour aider une personne malvoyante dans son quotidien.

Tu peux utiliser tes outils pour obtenir la météo, chercher des lieux, connaître les prochains trains/RER/métros, sauvegarder des notes, ou consulter la mémoire passée.

Sois concis et naturel dans tes réponses vocales. Parle à la 2ème personne du vouvoiement sauf si l'utilisateur préfère le tutoiement.

RÈGLES STRICTES DE RÉPONSE :
1. NE GÉNÈRE AUCUN MONOLOGUE INTERNE. Tu dois donner UNIQUEMENT la réponse finale attendue par l'utilisateur.
2. Il est formellement INTERDIT d'écrire des phrases telles que 'Je dois répondre...', 'L'utilisateur demande...', 'L'outil indique...', ou d'expliquer ce que tu vas faire.
3. Contente-toi de fournir l'information ou la réponse de manière directe, naturelle et fluide.

Pour l'heure et la date, base-toi TOUJOURS sur le résultat le plus récent de l'outil get_current_time présent dans la conversation — jamais sur tes connaissances internes.
Pour les horaires de transport (train, RER, métro, bus), utilise TOUJOURS l'outil get_transit_info avec la destination demandée par l'utilisateur.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    tool_calls_made: list[str]



_MAX_CONTEXT_MESSAGES = 1




class QwenAgent:
    def __init__(self, model_service : ModelService):
        self.model_service = model_service
        self.tools = get_all_tools()
        self.graph = self._build_graph()

    def _build_graph(self):

        tool_node = ToolNode(self.tools)

        def call_model(state: AgentState):

            response = self.model_service.generate_with_tools(state["messages"], self.tools)
            
            if response.tool_calls:
                logger.info(f"[Agent] Le modèle a décidé d'utiliser des outils : {[tc['name'] for tc in response.tool_calls]}")
                for tc in response.tool_calls:
                    state["tool_calls_made"].append(tc["name"])
            else:
                logger.info("[Agent] Le modèle a généré une réponse finale directe.")

            return {"messages": [response]}

        def doit_poursuivre_si_tool_present(state: AgentState):
            print("vérification si on poursuit")
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print("on continue car tool_calls présents dans le message du modèle")
                return "tools"
            else : 
                print("pas de tool_calls présents dans le message du modèle, on termine")
                return END

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", doit_poursuivre_si_tool_present, ["tools", END]) # soit tool soit end
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _build_system_prompt(self, session_model: Optional[SessionModel]) -> SystemMessage:
        prompt = BASE_PROMPT
        if session_model:
            if session_model.resume:
                prompt += f"\n\nCONTEXTE PRÉCÉDENT :\n{session_model.resume}"
            try:
                if (user := REPOSITORIES.users.get(session_model.user_id)) and user.preferences:
                    prompt += f"\n\nPRÉFÉRENCES UTILISATEUR :\n{json.dumps(user.preferences, ensure_ascii=False)}"
            except Exception as e:
                logger.warning(f"Erreur préférences (user={session_model.user_id}): {e}")
        return SystemMessage(content=prompt)

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

            if request.image_url:
                img_b64 = request.image_url

                # Texte du message (sans la mention [Image attachée: ...])
                base_text = last_user_msg.content.replace(f"\n[Image attachée: {img_b64}]", "").strip()

                last_user_msg = HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": img_b64}},
                    {"type": "text", "text": base_text},
                ])

            # [system] + [historique sauf dernier msg] + [msg actuel] + [tool_call + tool_result]
            messages_for_llm = [system_message] + lc_messages[:-1] + [last_user_msg]
        else:
            mock_user = HumanMessage(content=request.text)
            logger.info("[Agent] Nouveau message de l'utilisateur reçu.")
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
