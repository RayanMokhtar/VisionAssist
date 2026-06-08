"""
Serveur FastAPI pour tester l'Agent Qwen en mode HTTP.

Permet de dialoguer avec l'agent via le port 8000, indépendamment de MQTT.
Inclut l'initialisation de la base de données et du cron de résumé journalier.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langage.api.schemas.broker import BrokerRequest, AgentResponse
from langage.api.services.agent import QwenAgent
from langage.api.services.model import model_service
from langage.database.engine import setup_database
from langage.memory.daily_summary import start_daily_summary_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instanciation de l'agent
agent = QwenAgent(model_service=model_service)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarrage du service VisionAssist API...")
    # Initialisation DB (crée DB et tables si inexistantes)
    setup_database()
    
    # Démarrage du cron de résumé journalier
    try:
        start_daily_summary_scheduler(llm_callable=model_service.invoke_simple)
    except Exception as e:
        logger.warning(f"Erreur au démarrage du cron de résumé quotidien : {e}")
        
    yield
    logger.info("Arrêt du service.")

app = FastAPI(title="VisionAssist Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=AgentResponse)
def chat_endpoint(request: BrokerRequest):
    """Reçoit une requête (ex: depuis Gradio ou TTS), la passe à l'agent et retourne la réponse."""
    try:
        response = agent.handle(request)
        return response
    except Exception as e:
        logger.exception("Erreur dans le endpoint /chat")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
