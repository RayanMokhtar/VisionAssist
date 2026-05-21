"""Configuration centralisée de l'API."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Paramètres de l'application, surchargeables via variables d'environnement."""

    # --- Modèle ---
    model_id: str = "./modeles/Qwen3.5-27B"
    load_in_4bit: bool = True

    # --- Génération ---
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8
    do_sample: bool = True
    enable_thinking: bool = False

    # --- Serveur ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Limites ---
    max_image_size_mb: int = 10

    model_config = {"env_prefix": "QWEN_"}


settings = Settings()
