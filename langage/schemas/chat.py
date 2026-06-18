"""Schémas Pydantic pour la route /chat."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Corps de la requête envoyée à /chat."""

    text: str = Field(..., min_length=1, description="Texte / prompt envoyé au modèle.")
    image_url: str | None = Field(
        default=None,
        description="URL ou chemin local vers une image (optionnel).",
    )
    session_id: str | None = Field(
        default=None,
        description="Identifiant de session pour maintenir l'historique de conversation.",
    )


class ChatResponse(BaseModel):
    """Réponse renvoyée par /chat."""

    response: str = Field(..., description="Texte généré par le modèle.")
