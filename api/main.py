"""Point d'entrée de l'API Qwen3-VL."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import settings
from api.routes.chat import router as chat_router
from api.services.model import model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage et libère les ressources à l'arrêt."""
    logger.info("Démarrage de l'API — chargement du modèle …")
    model_service.load()
    yield
    logger.info("Arrêt de l'API.")


app = FastAPI(
    title="Qwen3-VL API",
    description="API de chat multimodale (texte + image) propulsée par Qwen3-VL.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router)


@app.get("/health")
async def health():
    """Vérifie que le service est opérationnel."""
    return {"status": "ok"}
