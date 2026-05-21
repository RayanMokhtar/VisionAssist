"""Schémas Pydantic pour la route /chat."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Corps de la requête envoyée à /chat."""

    text: str = Field(..., min_length=1, description="Texte / prompt envoyé au modèle.")
    image_url: str | None = Field(
        default=None,
        description="URL ou chemin local vers une image (optionnel).",
    )


class ChatResponse(BaseModel):
    """Réponse renvoyée par /chat."""

    response: str = Field(..., description="Texte généré par le modèle.")
