from __future__ import annotations

import time
import json
import logging
from typing import TypedDict, Annotated, Optional
import re as regex

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from langage.schemas.broker import BrokerRequest, AgentResponse
from langage.services.model import MODEL_SERVICE, ModelService
from langage.services.tools import get_all_tools
from langage.memory.short_term import ConversationBuffer
from langage.database.vector_store import vector_store
from persistance.repository import REPOSITORIES
from persistance.models import Session as SessionModel
from configuration import CONFIGURATION

logger = logging.getLogger(__name__)

BASE_PROMPT = f"""Tu es {CONFIGURATION.nom_assistant}, un assistant vocal intelligent conçu pour aider une personne malvoyante.

LANGUE : Tu DOIS répondre UNIQUEMENT en français. Ne réponds jamais en anglais, ni dans une autre langue.

Tu peux utiliser tes outils : météo, lieux proches, prochains transports, mémoire conversationnelle, besoin_image.

RÈGLES STRICTES :
1. RÉPONSE DIRECTE UNIQUEMENT. Donne immédiatement la réponse, sans jamais expliquer ta démarche.
2. INTERDIT : 'Je dois...', 'L'utilisateur demande...', 'Wait,', 'Let me...', 'However,', ou tout raisonnement interne.
3. Phrases courtes, naturelles, adaptées à la voix.
4. Vouvoiement par défaut.
5. VISION : Tu ne vois pas l'image par défaut. Si la question de l'utilisateur requiert de voir ce qu'il a devant lui (ex: "qu'est-ce que c'est ?", "lis ce texte"), .

Pour l'heure/date : utilise toujours l'outil get_current_time.
Pour les transports : utilise toujours l'outil get_transit_info.
Pour se souvenir d'une conversation passée : utilise l'outil query_memory.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    tool_calls_made: list[dict]
    pending_image_url: Optional[str]


class QwenAgent:

    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.tools = get_all_tools()
        self.graph = self._build_graph()

    def _call_model(self, state: AgentState):
        response = self.model_service.generer_reponse_pour_savoir_si_tool_necessaire_ou_pas(
            state["messages"], self.tools
        )
        if response.tool_calls:
            logger.info("[Agent] Outils appelés : %s", [tc["name"] for tc in response.tool_calls])
            for tc in response.tool_calls:
                state["tool_calls_made"].append({"name": tc["name"], "args": tc["args"]})
        else:
            logger.info("[Agent] Réponse directe générée.")
            
        return {"messages": [response]}

    def _should_continue(self, state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    def _route_after_tools(self, state: AgentState):
        messages = state["messages"]
        last_msg = messages[-1] if messages else None
        # Si le dernier message vient de 'besoin_image', l'outil a généré la réponse finale
        if getattr(last_msg, "name", "") == "besoin_image":
            return END
        return "agent"

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._should_continue, ["tools", END])
        workflow.add_conditional_edges("tools", self._route_after_tools, ["agent", END])
        return workflow.compile()

    def _build_system_prompt(self, session_model: Optional[SessionModel]) -> SystemMessage:
        prompt = BASE_PROMPT
        if session_model:
            if session_model.resume:
                prompt += f"\n\nCONTEXTE PRÉCÉDENT :\n{session_model.resume}"
            try:
                user = REPOSITORIES.users.get(session_model.user_id)
                if user and user.preferences:
                    prompt += f"\n\nPRÉFÉRENCES UTILISATEUR :\n{json.dumps(user.preferences, ensure_ascii=False)}"
            except Exception as e:
                logger.warning("Erreur préférences (user=%s): %s", session_model.user_id, e)
        return SystemMessage(content=prompt)


    def nettoyer_texte_reponse(self, response_text: str) -> str:
        """
        Nettoie le texte généré par le LLM pour le rendre lisible par un TTS.
        """
        if not response_text:
            return ""

        # 1. Fallback historique : nettoyage des balises <think> au cas où l'agent principal les génère
        if "</think>" in response_text:
            response_text = response_text.split("</think>")[-1]
        elif "<think>" in response_text:
            response_text = response_text.split("<think>")[0]
            if not response_text.strip():
                response_text = "Je suis désolé, je n'ai pas pu terminer mon analyse."
        
        # Par sécurité, on supprime aussi toute paire <tool_call>...<tool_call> qui pourrait rester
        response_text = regex.sub(r'<tool_call>.*?<tool_call>', '', response_text, flags=regex.DOTALL)

        # 2. Supprimer la syntaxe Markdown courante (gras, italique)
        # Supprime les astérisques (*) et les underscores (_)
        response_text = regex.sub(r'[*_]+', '', response_text)

        # 3. Supprimer d'autres caractères Markdown (titres #, citations >)
        response_text = regex.sub(r'[#>]+', '', response_text)

        # 4. Nettoyer les espaces et les sauts de ligne multiples
        # Remplace tous les blocs d'espaces/sauts de ligne par un espace unique
        response_text = regex.sub(r'\s+', ' ', response_text).strip()

        return response_text

    def handle(self, request: BrokerRequest, mode_simple_sans_memoire: bool = False) -> AgentResponse:
        if mode_simple_sans_memoire:
            logger.info("Mode SIMPLE SANS MÉMOIRE : %s", request.request_id)
            reponse = self.model_service.poser_question_sur_image(request.text, request.image_url)
            return AgentResponse(
                request_id=request.request_id,
                session_id=request.session_authentifiee.session_id if request.session_authentifiee else "",
                user_id=request.user_id,
                response=reponse,
            )

        logger.info("Réception requête Broker: %s", request.request_id)
        logger.info("Mode agent AVEC MÉMOIRE : %s", request.request_id)

        try:
            session = REPOSITORIES.sessions.get(
                session_id=request.session_authentifiee.session_id
            ) if request.session_authentifiee else None
        except ValueError:
            logger.warning("user_id invalide %s, session mockée", request.user_id)
            session = None

        user_content = request.text
        if request.image_url:
            user_content += "\n[Image attachée: "+request.image_url+"]"

        buffer = ConversationBuffer(str(request.session_authentifiee.session_id))
        historique = buffer.get_langchain_messages()

        current_db_msg = REPOSITORIES.messages.create(
            session_id=request.session_authentifiee.session_id,
            requete=user_content,
        )

        system_message = self._build_system_prompt(session_model=session)

        # On n'injecte PAS l'image ici pour économiser la VRAM sur les requêtes simples
        current_user_msg = HumanMessage(content=request.text)

        initial_state = {
            "messages": [system_message] + historique + [current_user_msg],
            "user_id": request.user_id,
            "session_id": request.session_authentifiee.session_id,
            "tool_calls_made": [],
            "pending_image_url": request.image_url,
        }

        try:
            final_state = self.graph.invoke(initial_state)
            response_text = final_state["messages"][-1].content

            reponse_nettoyee = self.nettoyer_texte_reponse(response_text)

            REPOSITORIES.messages.update_response(current_db_msg, reponse=response_text)

            try:
                t0 = time.time()
                vector_store.index_document(
                    collection_name="conversation_messages",
                    doc_id=str(current_db_msg.message_id),
                    content=f"Utilisateur: {user_content}\nAssistant: {response_text}",
                    metadata={
                        "user_id": str(request.user_id),
                        "session_id": str(request.session_authentifiee.session_id),
                        "timestamp": current_db_msg.timestamp.isoformat() if current_db_msg.timestamp else "",
                    },
                )
                logger.info("Indexation FAISS : %.2f s", time.time() - t0)
            except Exception as e:
                logger.warning("Erreur indexation FAISS : %s", e)

            for tc in final_state["tool_calls_made"]:
                logger.info("Tool utilisé: %s", tc)

            return AgentResponse(
                request_id=request.request_id,
                session_id=request.session_authentifiee.session_id,
                user_id=request.user_id,
                response=reponse_nettoyee,
                tool_calls_made=final_state["tool_calls_made"],
            )

        except Exception as e:
            logger.exception("Erreur lors de l'exécution de l'agent")
            return AgentResponse(
                request_id=request.request_id,
                session_id=request.session_authentifiee.session_id,
                user_id=request.user_id,
                error=str(e),
            )

    def stream_handle(self, request: BrokerRequest):
        """Yields intermediate state events from LangGraph, then final AgentResponse."""
        logger.info("Début stream_handle pour requête: %s", request.request_id)

        try:
            session = REPOSITORIES.sessions.get(
                session_id=request.session_authentifiee.session_id
            ) if request.session_authentifiee else None
        except ValueError:
            session = None

        user_content = request.text
        if request.image_url:
            user_content += f"\n[Image attachée: {request.image_url}]"

        buffer = ConversationBuffer(str(request.session_authentifiee.session_id))
        historique = buffer.get_langchain_messages()

        current_db_msg = REPOSITORIES.messages.create(
            session_id=request.session_authentifiee.session_id,
            requete=user_content,
        )

        system_message = self._build_system_prompt(session_model=session)

        # On n'injecte PAS l'image ici pour économiser la VRAM sur les requêtes simples
        current_user_msg = HumanMessage(content=request.text)

        initial_state = {
            "messages": [system_message] + historique + [current_user_msg],
            "user_id": request.user_id,
            "session_id": request.session_authentifiee.session_id,
            "tool_calls_made": [],
            "pending_image_url": request.image_url,
        }

        try:
            # Yield events from the graph
            for event in self.graph.stream(initial_state):
                yield {"type": "stream_event", "event": event}

            # Retrieve final state by invoking again or using the last event
            final_state = self.graph.invoke(initial_state)
            response_text = final_state["messages"][-1].content

            REPOSITORIES.messages.update_response(current_db_msg, reponse=response_text)

            try:
                t0 = time.time()
                vector_store.index_document(
                    collection_name="conversation_messages",
                    doc_id=str(current_db_msg.message_id),
                    content=f"Utilisateur: {user_content}\nAssistant: {response_text}",
                    metadata={
                        "user_id": str(request.user_id),
                        "session_id": str(request.session_authentifiee.session_id),
                        "timestamp": current_db_msg.timestamp.isoformat() if current_db_msg.timestamp else "",
                    },
                )
            except Exception as e:
                logger.warning("Erreur indexation FAISS : %s", e)

            response_obj = AgentResponse(
                request_id=request.request_id,
                session_id=request.session_authentifiee.session_id,
                user_id=request.user_id,
                response=response_text,
                tool_calls_made=final_state["tool_calls_made"],
            )
            yield {"type": "final_response", "response": response_obj}

        except Exception as e:
            logger.exception("Erreur lors de l'exécution en stream de l'agent")
            yield {"type": "final_response", "response": AgentResponse(
                request_id=request.request_id,
                session_id=request.session_authentifiee.session_id,
                user_id=request.user_id,
                error=str(e),
            )}
