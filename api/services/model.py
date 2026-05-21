"""Service de chargement et d'inférence du modèle Qwen3.5."""

import logging
from collections.abc import Iterator
from threading import Thread

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from api.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """Encapsule le modèle et le processeur pour l'inférence."""

    def __init__(self) -> None:
        self.model: AutoModelForImageTextToText | None = None
        self.processor: AutoProcessor | None = None

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Charge le modèle et le processeur sur GPU avec quantification 4-bit."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA n'est pas disponible — un GPU est requis.")

        logger.info("Chargement du modèle %s (4-bit) …", settings.model_id)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=settings.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.processor = AutoProcessor.from_pretrained(settings.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            settings.model_id,
            quantization_config=quantization_config,
            device_map="cuda",
        )
        logger.info("Modèle chargé sur : %s", self.model.device)

    # ------------------------------------------------------------------
    # Construction des messages
    # ------------------------------------------------------------------
    @staticmethod
    def _build_messages(text: str, image_url: str | None) -> list[dict]:
        """Construit la liste de messages au format attendu par le modèle."""
        content: list[dict] = []
        if image_url:
            content.append({"type": "image", "url": image_url})
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, text: str, image_url: str | None) -> dict:
        messages = self._build_messages(text, image_url)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=settings.enable_thinking,
        )
        return {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Inférence complète
    # ------------------------------------------------------------------
    def generate(self, text: str, image_url: str | None = None) -> str:
        """Génère une réponse complète (bloquant)."""
        inputs = self._prepare_inputs(text, image_url)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.do_sample,
            temperature=settings.temperature,
            top_p=settings.top_p,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        chunks = [token for token in streamer]
        thread.join()
        return "".join(chunks)

    # ------------------------------------------------------------------
    # Inférence en streaming
    # ------------------------------------------------------------------
    def generate_stream(self, text: str, image_url: str | None = None) -> Iterator[str]:
        """Génère une réponse token par token (générateur)."""
        inputs = self._prepare_inputs(text, image_url)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.do_sample,
            temperature=settings.temperature,
            top_p=settings.top_p,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()


# Singleton utilisé par l'ensemble de l'application.
model_service = ModelService()
