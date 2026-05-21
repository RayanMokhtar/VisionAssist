"""Routes /chat et /chat/stream — inférence complète et streaming."""

import asyncio
import logging
from threading import Thread

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas.chat import ChatRequest, ChatResponse
from api.services.model import model_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Reçoit un texte (et une image optionnelle) et renvoie la réponse complète."""
    try:
        response_text = model_service.generate(
            text=request.text,
            image_url=request.image_url,
        )
    except Exception as exc:
        logger.exception("Erreur lors de la génération.")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la génération.") from exc

    return ChatResponse(response=response_text)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Reçoit un texte (et une image optionnelle) et renvoie la réponse en streaming token par token."""
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def producer() -> None:
        """Tourne dans un thread : push chaque token dans la queue."""
        try:
            for token in model_service.generate_stream(
                text=request.text,
                image_url=request.image_url,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception:
            logger.exception("Erreur lors du streaming.")
            loop.call_soon_threadsafe(queue.put_nowait, "\n[Erreur interne lors de la génération]")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel de fin

    async def token_generator():
        """Lit la queue de manière asynchrone et yield les tokens."""
        thread = Thread(target=producer)
        thread.start()
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")
