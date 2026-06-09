from __future__ import annotations

import json
import logging
from typing import TypedDict, Annotated, List, Optional
import datetime

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from sqlalchemy.orm import Session as DbSession
from langage.schemas.broker import BrokerRequest, AgentResponse
from langage.services.model import model_service , ModelService
from langage.services.tools import get_all_tools,current_user_id,current_session_id
from langage.database.engine import get_db
from langage.database.repositories import session_repo, message_repo, summary_repo, note_repo
from langage.database.models import UserProfile
from langage.memory.short_term import ConversationBuffer
from langage.memory.long_term import long_term_memory

logger = logging.getLogger(__name__)


from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    tool_calls_made: list[str]


class QwenAgent:
    def __init__(self, model_service : ModelService = model_service):
        self.model_service = model_service
        self.tools = get_all_tools()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construit le graphe d'exécution LangGraph."""
        # 1. Le ToolNode qui exécute les outils
        tool_node = ToolNode(self.tools)

        # 2. Le noeud du LLM
        def call_model(state: AgentState):
            # Utilise la méthode personnalisée pour HF (Singleton)
            response = self.model_service.generate_with_tools(state["messages"], self.tools)
            
            # Enregistrer les tools appelés pour les stats/réponse
            if response.tool_calls:
                for tc in response.tool_calls:
                    state["tool_calls_made"].append(tc["name"])
                    
            return {"messages": [response]}

        # 3. Routage conditionnel
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
        db: DbSession,
        user_id: str,
        session_model,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> SystemMessage:
        """Construit le prompt système initial avec l'historique et la mémoire."""
        
        # 1. Résumé d'hier (si c'est une nouvelle session et non injecté)
        yesterday_context = ""
        if not session_model.yesterday_summary_injected:
            yesterday_summary = summary_repo.get_yesterday_summary(db, user_id)
            if yesterday_summary:
                yesterday_context = f"\n\nCONTEXTE D'HIER :\n{yesterday_summary.summary}"
            else:
                yesterday_context = "\n\nCONTEXTE : C'est votre premier échange (ou aucune donnée hier)."
            
            # Marquer comme injecté pour ne pas le répéter en base
            session_repo.mark_yesterday_injected(db, session_model.id)

        # 2. Profil utilisateur (préférences)
        user_profile_records = db.query(UserProfile).filter(UserProfile.user_id == session_model.user_id).all()
        profile_context = ""
        if user_profile_records:
            prefs = {p.cle: p.valeur for p in user_profile_records}
            profile_context = f"\n\nPRÉFÉRENCES UTILISATEUR :\n{json.dumps(prefs, ensure_ascii=False)}"

        # 3. Rappels actifs
        active_notes = note_repo.get_active_notes(db, user_id)
        notes_context = ""
        if active_notes:
            notes_lines = [f"- {n.content}" for n in active_notes]
            notes_context = "\n\nRAPPELS ACTIFS :\n" + "\n".join(notes_lines)

        base_prompt = (
            "Tu es VisionAssist, un assistant intelligent, bienveillant et proactif conçu pour "
            "aider une personne malvoyante dans son quotidien.\n"
            "Tu peux utiliser tes outils pour obtenir la météo, chercher des lieux, "
            "connaître les prochains trains/RER/métros, "
            "sauvegarder des notes, ou consulter la mémoire passée.\n"
            "Sois concis et naturel dans tes réponses vocales. Parle à la 2ème personne du vouvoiement "
            "sauf si l'utilisateur préfère le tutoiement.\n"
            "RÈGLES STRICTES DE RÉPONSE :\n"
            "1. NE GÉNÈRE AUCUN MONOLOGUE INTERNE. Tu dois donner UNIQUEMENT la réponse finale attendue par l'utilisateur.\n"
            "2. Il est formellement INTERDIT d'écrire des phrases telles que 'Je dois répondre...', 'L'utilisateur demande...', 'L'outil indique...', ou d'expliquer ce que tu vas faire.\n"
            "3. Contente-toi de fournir l'information ou la réponse de manière directe, naturelle et fluide.\n"
            "Pour l'heure et la date, base-toi TOUJOURS sur le résultat le plus récent de l'outil "
            "get_current_time présent dans la conversation — jamais sur tes connaissances internes.\n"
            "Pour les horaires de transport (train, RER, métro, bus), utilise TOUJOURS l'outil "
            "get_transit_info avec la destination demandée par l'utilisateur.\n"
            "Pour sauvegarder un rappel, un mémo ou une note, tu DOIS TOUJOURS utiliser l'outil save_note. "
            "Ne dis JAMAIS que tu as noté quelque chose sans avoir appelé l'outil save_note au format XML."
        )

        # Injection de la position GPS si disponible
        if latitude is not None and longitude is not None:
            gps_context = (
                f"\n\nPOSITION GPS ACTUELLE DE L'UTILISATEUR : latitude={latitude:.6f}, longitude={longitude:.6f}. "
                "Les outils de recherche de lieux et de transport utilisent automatiquement cette position — "
                "tu n'as pas besoin de la passer en argument."
            )
        else:
            gps_context = (
                "\n\nPOSITION GPS : non disponible (géolocalisation non activée). "
                "Si l'utilisateur cherche un lieu proche ou des horaires de transport, "
                "invite-le à activer la géolocalisation dans l'interface."
            )

        full_prompt = base_prompt + gps_context + yesterday_context + profile_context + notes_context
        return SystemMessage(content=full_prompt)

    def handle(self, request: BrokerRequest) -> AgentResponse:
        """Point d'entrée principal appelé par le main_agent.py."""
        
        logger.info("Réception requête Broker: %s", request.request_id)
        
        # Setup contextvars pour les outils
        current_user_id.set(request.user_id)

        # GPS injecté dans le texte du message (voir étape 2 ci-dessous)
        if request.latitude is not None and request.longitude is not None:
            logger.info("📍 [GPS] Position reçue : lat=%.4f lon=%.4f", request.latitude, request.longitude)
        else:
            logger.info("📍 [GPS] Pas de coordonnées fournies dans cette requête")
        
        with get_db() as db:
            # 1. Gestion de la session journalière
            session_model, is_new_session = session_repo.get_or_create_today_session(db, request.user_id)
            session_id = session_model.id
            current_session_id.set(session_id)
            
            # 2. Enregistrer le message de l'utilisateur
            user_content = request.text
            if request.image_url:
                user_content += f"\n[Image attachée: {request.image_url}]"

            # Injecter les coordonnées GPS dans le texte → le LLM sait TOUJOURS où l'utilisateur est
            if request.latitude is not None and request.longitude is not None:
                user_content = (
                    f"[Ma position GPS actuelle : latitude={request.latitude:.6f}, longitude={request.longitude:.6f}]\n"
                    + user_content
                )
                
            message_repo.add_message(
                db=db,
                session_id=session_id,
                role="user",
                content=user_content,
                source="api" if request.image_url else "stt"
            )
            session_repo.increment_message_count(db, session_id)

            # 3. Construire le contexte LangChain
            buffer = ConversationBuffer(session_id)
            lc_messages = buffer.get_langchain_messages(db)
            
            # Récupérer les informations nécessaires pour le prompt système
            system_message = self._build_system_prompt(
                db, request.user_id, session_model,
                latitude=request.latitude,
                longitude=request.longitude,
            )

        # =====================================================================
        # APPEL AU LLM (HORS TRANSACTION BDD POUR NE PAS BLOQUER)
        # =====================================================================
        # 4. Récupérer l'heure exacte via la commande système WSL
            import subprocess as _sp
            import uuid as _uuid
            try:
                _time_result = _sp.check_output(
                    ["date", "+%H:%M|%u|%d|%m|%Y"], text=True
                ).strip()
                _tp = _time_result.split("|")
                _hm = _tp[0]          # ex: "10:20"
                _h, _m = _hm.split(":")
                _jours_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
                _mois_fr  = ["janvier", "février", "mars", "avril", "mai", "juin",
                             "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
                _jour_nom = _jours_fr[int(_tp[1]) - 1]
                _mois_nom = _mois_fr[int(_tp[3]) - 1]
                _time_str = (
                    f"Il est {int(_h)}h{_m}. "
                    f"Nous sommes {_jour_nom} {int(_tp[2])} {_mois_nom} {_tp[4]}."
                )
            except Exception as _e:
                logger.warning("Fallback datetime pour l'heure : %s", _e)
                _now = datetime.datetime.now()
                _jours_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
                _mois_fr  = ["janvier", "février", "mars", "avril", "mai", "juin",
                             "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
                _time_str = (
                    f"Il est {_now.hour}h{_now.minute:02d}. "
                    f"Nous sommes {_jours_fr[_now.weekday()]} {_now.day} "
                    f"{_mois_fr[_now.month - 1]} {_now.year}."
                )

            logger.info("🕐 [TOOL INJECTÉ] get_current_time → %s", _time_str)

        # 5. Ajouter le SystemMessage en tête
        _tool_call_id = f"injected_time_{_uuid.uuid4().hex[:8]}"
        injected_tool_call = AIMessage(
            content="",
            tool_calls=[{
                "name": "get_current_time",
                "args": {},
                "id": _tool_call_id,
                "type": "tool_call"
            }]
        )
        injected_tool_result = ToolMessage(
            content=_time_str,
            tool_call_id=_tool_call_id,
            name="get_current_time"
        )

        if lc_messages:
            # Récupérer le dernier message (message de l'utilisateur courant)
            last_user_msg = lc_messages[-1]

            # Si une image est fournie, remplacer le dernier message par un HumanMessage multimodal
            # Qwen-VL attend un contenu sous forme de liste [{type: image_url/text, ...}]
            if request.image_url:
                import os as _os
                img_path = request.image_url
                # Construire l'URI fichier ou URL selon le contexte
                if _os.path.exists(img_path):
                    img_uri = f"file://{img_path}"
                else:
                    img_uri = img_path  # URL externe

                # Texte du message (sans la mention [Image attachée: ...])
                base_text = last_user_msg.content.replace(f"\n[Image attachée: {img_path}]", "").strip()
                last_user_msg = HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": img_uri}},
                    {"type": "text", "text": base_text},
                ])

            # [system] + [historique sauf dernier msg] + [msg actuel] + [tool_call + tool_result]
            messages_for_llm = (
                [system_message]
                + lc_messages[:-1]
                + [last_user_msg]
                + [injected_tool_call, injected_tool_result]
            )
        else:
            mock_user = HumanMessage(content=request.text or "Bonjour")
            messages_for_llm = [system_message, mock_user, injected_tool_call, injected_tool_result]

        # 6. Invoquer l'agent (LangGraph)
        initial_state = {
            "messages": messages_for_llm,
            "user_id": request.user_id,
            "session_id": session_id,
            "tool_calls_made": []
        }

        try:
            # Exécution du graphe (LLM + Tools)
            final_state = self.graph.invoke(initial_state)
            
            # Le dernier message est la réponse finale de l'assistant
            last_message = final_state["messages"][-1]
            response_text = last_message.content

            # =====================================================================
            # SAUVEGARDE EN BDD DE LA RÉPONSE DE L'ASSISTANT
            # =====================================================================
            with get_db() as db:
                # 7. Sauvegarder la réponse en BDD
                message_repo.add_message(
                    db=db,
                    session_id=session_id,
                    role="assistant",
                    content=response_text,
                    source="api"
                )
                session_repo.increment_message_count(db, session_id)
                
                # Si des tools ont été utilisés, on les sauvegarde aussi dans l'historique
                # (Dans une v2, on pourrait itérer sur final_state["messages"] pour sauvegarder 
                # spécifiquement les ToolMessages et AIMessages intermédiaires)
                for tc in final_state["tool_calls_made"]:
                    # Juste pour log basique
                    logger.info("Tool utilisé: %s", tc)

            return AgentResponse(
                request_id=request.request_id,
                session_id=session_id,
                user_id=request.user_id,
                response=response_text,
                tool_calls_made=final_state["tool_calls_made"]
            )

        except Exception as e:
            logger.exception("Erreur lors de l'exécution de l'agent")
            return AgentResponse(
                request_id=request.request_id,
                session_id=session_id,
                user_id=request.user_id,
                error=str(e)
            )
