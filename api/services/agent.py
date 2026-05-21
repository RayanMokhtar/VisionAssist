"""LangChain-based multimodal agent for Qwen3.5."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from api.config import settings
from api.schemas.broker import BrokerRequest, BrokerResponse
from api.services.model import ModelService, model_service

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the agent runtime."""

    system_prompt: str = "You are a helpful multimodal assistant."
    request_topic: str = "qwen.requests"
    response_topic: str = "qwen.responses"


class BrokerClient(Protocol):
    """Contract compatible with the VisionAssist broker."""

    def connexion(self) -> None:
        ...

    def deconnexion(self) -> None:
        ...

    def publier(self, topic: str, payload: Any) -> bool:
        ...

    def sabonner(self, topic: str, fonction_apres_trigger: Any) -> None:
        ...

    def desabonner(self, topic: str) -> None:
        ...


class SessionMemory:
    """Minimal in-memory conversation buffer."""

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []

    def load_history(self) -> list[BaseMessage]:
        return list(self._messages)

    def save_turn(self, user_text: str, assistant_text: str) -> None:
        self._messages.append(HumanMessage(content=user_text))
        self._messages.append(AIMessage(content=assistant_text))


class SessionMemoryStore:
    """Keep a memory buffer per session id."""

    def __init__(self) -> None:
        self._memories: dict[str, SessionMemory] = {}

    def get(self, session_id: str) -> SessionMemory:
        if session_id not in self._memories:
            self._memories[session_id] = SessionMemory()
        return self._memories[session_id]


class QwenMultimodalChatModel(BaseChatModel):
    """LangChain chat model wrapper around Qwen3.5."""

    def __init__(self, model: ModelService) -> None:
        super().__init__()
        self._model = model

    @property
    def _llm_type(self) -> str:
        return "qwen3.5-multimodal"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        response_text = self._invoke_model(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])

    def _invoke_model(self, messages: list[BaseMessage]) -> str:
        qwen_messages = _messages_to_qwen(messages)
        inputs = self._prepare_inputs(qwen_messages)
        return self._generate_text(inputs)

    def _prepare_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self._model.processor is None or self._model.model is None:
            raise RuntimeError("ModelService is not loaded.")
        inputs = self._model.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=settings.enable_thinking,
        )
        return {k: v.to(self._model.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _generate_text(self, inputs: dict[str, Any]) -> str:
        if self._model.processor is None or self._model.model is None:
            raise RuntimeError("ModelService is not loaded.")
        output_ids = self._model.model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.do_sample,
            temperature=settings.temperature,
            top_p=settings.top_p,
        )
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            prompt_length = input_ids.shape[-1]
            output_ids = output_ids[:, prompt_length:]
        decoded = self._model.processor.batch_decode(output_ids, skip_special_tokens=True)
        return decoded[0] if decoded else ""


class QwenAgent:
    """Orchestrates memory and model calls."""

    def __init__(self, model: ModelService, config: AgentConfig | None = None) -> None:
        self._model = model
        self._config = config or AgentConfig()
        self._memory_store = SessionMemoryStore()
        self._chat_model = QwenMultimodalChatModel(model)

    def handle(self, request: BrokerRequest) -> BrokerResponse:
        memory = self._memory_store.get(request.session_id)
        history = memory.load_history()
        messages = _build_messages(
            system_prompt=self._config.system_prompt,
            history=history,
            user_text=request.text,
            image_url=request.image_url,
            context=request.context,
        )

        response_message = self._chat_model.invoke(messages)
        response_text = _coerce_text(getattr(response_message, "content", response_message))
        memory.save_turn(request.text, response_text)
        return BrokerResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            response=response_text,
        )


class AgentWorker:
    """Worker that bridges broker messages to the agent."""

    def __init__(
        self,
        agent: QwenAgent,
        broker: BrokerClient,
        config: AgentConfig | None = None,
    ) -> None:
        self._agent = agent
        self._broker = broker
        self._config = config or AgentConfig()

    def start(self) -> None:
        self._broker.connexion()
        self._broker.sabonner(self._config.request_topic, self._on_message)

    def stop(self) -> None:
        self._broker.desabonner(self._config.request_topic)
        self._broker.deconnexion()

    def _on_message(self, topic: str, payload: Any) -> None:
        response = self._process_payload(payload)
        self._broker.publier(self._config.response_topic, response)

    def _process_payload(self, payload: Any) -> dict[str, Any]:
        try:
            request = _load_request(payload)
            response = self._agent.handle(request)
        except Exception as exc:
            logger.exception("Agent failed to process payload.")
            response = BrokerResponse(
                request_id=_safe_request_id(payload),
                session_id=_safe_session_id(payload),
                error=str(exc),
            )
        return response.model_dump()


def _build_messages(
    system_prompt: str,
    history: list[BaseMessage],
    user_text: str,
    image_url: str | None,
    context: list[str] | None,
) -> list[BaseMessage]:
    system_text = _build_system_prompt(system_prompt, context)
    messages: list[BaseMessage] = [SystemMessage(content=system_text)]
    messages.extend(history)
    messages.append(_build_user_message(user_text, image_url))
    return messages


def _build_user_message(text: str, image_url: str | None) -> HumanMessage:
    if image_url:
        content = [
            {"type": "image", "url": image_url},
            {"type": "text", "text": text},
        ]
        return HumanMessage(content=content)
    return HumanMessage(content=text)


def _build_system_prompt(base_prompt: str, context: list[str] | None) -> str:
    if not context:
        return base_prompt
    joined = "\n".join(f"- {item}" for item in context)
    return f"{base_prompt}\n\nContext:\n{joined}"


def _messages_to_qwen(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    qwen_messages: list[dict[str, Any]] = []
    for message in messages:
        role = _map_role(message)
        if isinstance(message, HumanMessage) and isinstance(message.content, list):
            qwen_messages.append({"role": role, "content": message.content})
        else:
            qwen_messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": str(message.content)}],
                }
            )
    return qwen_messages


def _map_role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, AIMessage):
        return "assistant"
    return "user"


def _load_request(payload: Any) -> BrokerRequest:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        return BrokerRequest.model_validate_json(payload)
    if isinstance(payload, dict):
        return BrokerRequest.model_validate(payload)
    return BrokerRequest.model_validate({})


def _safe_request_id(payload: Any) -> str:
    try:
        if isinstance(payload, dict):
            data = payload
        else:
            data = json.loads(payload)
        return str(data.get("request_id") or "unknown")
    except json.JSONDecodeError:
        return "unknown"


def _safe_session_id(payload: Any) -> str:
    try:
        if isinstance(payload, dict):
            data = payload
        else:
            data = json.loads(payload)
        return str(data.get("session_id") or "unknown")
    except json.JSONDecodeError:
        return "unknown"


def _coerce_text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content)


agent_service = QwenAgent(model_service)
