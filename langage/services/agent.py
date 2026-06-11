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

BASE_PROMPT = f"""Tu es {CONFIGURATION.nom_assistant}, un assistant vocal intelligent conçu pour aider une personne malvoyante.

LANGUE : Tu DOIS répondre UNIQUEMENT en français. Ne réponds jamais en anglais, ni dans une autre langue.

Tu peux utiliser tes outils : météo, lieux proches, prochains transports, notes, mémoire.

RÈGLES STRICTES :
1. RÉPONSE DIRECTE UNIQUEMENT. Donne immédiatement la réponse, sans jamais expliquer ta démarche.
2. INTERDIT : 'Je dois...', 'L'utilisateur demande...', 'Wait,', 'Let me...', 'However,', ou tout raisonnement interne.
3. Phrases courtes, naturelles, adaptées à la voix.
4. Vouvoiement par défaut.

Pour l'heure/date : utilise toujours l'outil get_current_time.
Pour les transports : utilise toujours l'outil get_transit_info.
"""


BASE_PROMPT_SIMPLE = f"""Tu es un assistant vocal intelligent conçu pour aider une personne malvoyante. Tu DOIS répondre UNIQUEMENT en français de manière brève."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    tool_calls_made: list[str]



class QwenAgent:

    def __init__(self, model_service : ModelService):
        self.model_service = model_service
        self.tools = get_all_tools()
        self.graph = self.initialisation_graph()    

    def _appel_noeud(self, state: AgentState):

        response = self.model_service.generer_reponse_pour_savoir_si_tool_necessaire_ou_pas(state["messages"], self.tools)

        if response.tool_calls:
            logger.info(f"[Agent] Le modèle a décidé d'utiliser des outils : {[tc['name'] for tc in response.tool_calls]}")
            for tc in response.tool_calls:
                state["tool_calls_made"].append(tc["name"])
        else:
            logger.info("[Agent] Le modèle a généré une réponse finale directe.")

        return {"messages": [response]}

    def _doit_poursuivre_si_tool_present(self,state: AgentState):
        logger.debug("Vérification si on poursuit")
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info("On continue car tool_calls présents dans le message du modèle")
            return "tools"
        else :
            logger.info("Pas de tool_calls présents dans le message du modèle, on termine")
            return END


    def initialisation_graph(self):

        tool_node = ToolNode(self.tools)

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._appel_noeud)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._doit_poursuivre_si_tool_present, ["tools", END])
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _build_system_prompt(self, session_model: Optional[SessionModel]) -> SystemMessage:
        prompt = BASE_PROMPT_SIMPLE
        if session_model:
            if session_model.resume:
                prompt += f"\n\nCONTEXTE PRÉCÉDENT :\n{session_model.resume}"
            try:
                if (user := REPOSITORIES.users.get(session_model.user_id)) and user.preferences:
                    prompt += f"\n\nPRÉFÉRENCES UTILISATEUR :\n{json.dumps(user.preferences, ensure_ascii=False)}"
            except Exception as e:
                logger.warning(f"Erreur préférences (user={session_model.user_id}): {e}")
        return SystemMessage(content=prompt)

    def handle(self, request: BrokerRequest , mode_simple_sans_memoire : bool = False) -> AgentResponse:
        
        if mode_simple_sans_memoire :
            logger.info("Mode SIMPLE SANS MÉMOIRE activé pour la requête %s", request.request_id)
            reponse = self.model_service.poser_question_sur_image(request.text, request.image_url)
            return AgentResponse(
                request_id=request.request_id,
                session_id=request.session_authentifiee.session_id if request.session_authentifiee else "",
                user_id=request.user_id,
                response=reponse
            )
        else : 
            logger.info("Réception requête Broker: %s", request.request_id)
            
            try:
                session = REPOSITORIES.sessions.get(session_id=request.session_authentifiee.session_id) if request.session_authentifiee else None
                user_id = session.user_id if session else request.user_id
            except ValueError:
                logger.warning(f"user_id invalide {request.user_id}, session mockée")


            user_content = request.text
            if request.image_url:
                user_content += f"\n[Image attachée: {request.image_url}]"

            # Charger l'historique AVANT de sauvegarder la requête actuelle
            # => le buffer contient uniquement les échanges précédents, pas le message actuel
            buffer = ConversationBuffer(str(request.session_authentifiee.session_id))
            historique = buffer.get_langchain_messages()

            # Sauvegarder la requête actuelle en base pour la persistance
            current_db_msg = REPOSITORIES.messages.create(
                session_id=request.session_authentifiee.session_id,
                requete=user_content
            )

            system_message = self._build_system_prompt(session_model=session)

            # Construire le message utilisateur courant (toujours depuis request, jamais depuis le buffer)
            if request.image_url:
                img_b64 = request.image_url
                base_text = request.text
                current_user_msg = HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": img_b64}},
                    {"type": "text", "text": base_text},
                ])
            else:
                current_user_msg = HumanMessage(content=request.text)

            # [System] + [historique complet] + [message actuel]
            messages_for_llm = [system_message] + historique + [current_user_msg]

            initial_state = {
                "messages": messages_for_llm,
                "user_id": request.user_id,
                "session_id": request.session_authentifiee.session_id,
                "tool_calls_made": []
            }

            try:
                print("début state messages ", initial_state["messages"])
                final_state = self.graph.invoke(initial_state)
                print("final state messages ", final_state["messages"])
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
