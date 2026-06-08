"""
ModelService — Abstraction du modèle de langage (LLM).

Charge Qwen3-VL directement via HuggingFace Transformers en tant que Singleton.
Gère le formatage des outils pour Qwen.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Optional, List, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.utils.function_calling import convert_to_openai_tool

from configuration import CONFIGURATION

logger = logging.getLogger(__name__)
_conf = CONFIGURATION.llm


class ModelService:
    """Service de chargement et d'accès au modèle LLM via Transformers."""

    def __init__(self):
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Charge le modèle directement via Transformers (singleton)."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError:
            raise ImportError(
                "transformers et torch requis. "
                "Installez avec : pip install transformers torch accelerate"
            )

        model_id = _conf.model_name
        logger.info("Chargement du modèle Transformers : %s...", model_id)
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype
            )
            # Utilisation de 'cuda' au lieu de 'auto' pour éviter les erreurs d'offload CPU avec BitsAndBytes
            device_map = "cuda"
            logger.info("Utilisation de BitsAndBytes (4-bit) pour optimiser la VRAM. device_map forcé sur 'cuda'.")
        except ImportError:
            quantization_config = None
            device_map = "auto"
            logger.warning("BitsAndBytes non installé. Chargement en précision standard.")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="sdpa",
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._loaded = True
        logger.info("LLM chargé avec succès.")

    def get_chat_model(self):
        raise NotImplementedError("Utilisez generate_with_tools directement en mode transformers.")

    def invoke_simple(self, prompt: str) -> str:
        """Appel simple au LLM (sans tool calling) — utilisé par le summarizer."""
        if not self._loaded:
            self.load()

        messages = [{"role": "user", "content": prompt}]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(text=[text_prompt], return_tensors="pt").to(self.model.device)

        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=_conf.max_tokens,
                do_sample=True,
                temperature=_conf.temperature,
            )
            
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated_ids, skip_special_tokens=True)

    def generate_with_tools(self, lc_messages: List[BaseMessage], tools: List[Any]) -> AIMessage:
        """
        Génère une réponse en supportant le tool calling manuellement.
        Parse les messages LangChain en format dict HF, injecte les descriptions d'outils,
        et parse la sortie pour identifier si le modèle a décidé d'utiliser un outil.
        """
        if not self._loaded:
            self.load()

        hf_messages = []
        pil_images = []  # images PIL à passer au processeur

        for m in lc_messages:
            if isinstance(m, HumanMessage):
                # Détecter contenu multimodal (liste avec image_url + text)
                if isinstance(m.content, list):
                    hf_content = []
                    for part in m.content:
                        if part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            # Charger l'image PIL depuis un chemin fichier ou URL
                            try:
                                from PIL import Image as _PILImage
                                if url.startswith("file://"):
                                    img_path = url[7:]
                                    pil_img = _PILImage.open(img_path).convert("RGB")
                                else:
                                    import io as _io
                                    import requests as _req
                                    r = _req.get(url, timeout=10)
                                    pil_img = _PILImage.open(_io.BytesIO(r.content)).convert("RGB")
                                pil_images.append(pil_img)
                                hf_content.append({"type": "image"})
                            except Exception as _img_err:
                                logger.warning("Impossible de charger l'image : %s", _img_err)
                        elif part.get("type") == "text":
                            hf_content.append({"type": "text", "text": part["text"]})
                    hf_messages.append({"role": "user", "content": hf_content})
                else:
                    hf_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                if m.tool_calls:
                    # Qwen s'attend à voir l'appel sous forme textuelle si c'est de l'historique
                    calls_str = "\n".join(
                        f'<tool_call>\n{{"name": "{tc["name"]}", "arguments": {json.dumps(tc["args"])}}}\n</tool_call>'
                        for tc in m.tool_calls
                    )
                    hf_messages.append({"role": "assistant", "content": calls_str})
                else:
                    hf_messages.append({"role": "assistant", "content": m.content})
            elif isinstance(m, ToolMessage):
                hf_messages.append({"role": "tool", "name": m.name, "content": m.content})
            elif isinstance(m, SystemMessage):
                hf_messages.append({"role": "system", "content": m.content})

        # Conversion des outils au format JSON Schema standard (OpenAI-like)
        hf_tools = [convert_to_openai_tool(t) for t in tools]
        
        # Pour Qwen3-VL, on injecte les outils dans le system prompt si on gère à la main
        # Mais le processeur Qwen 2.5/3 supporte l'argument tools dans apply_chat_template !
        try:
            text_prompt = self.processor.apply_chat_template(
                hf_messages, 
                tools=hf_tools, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except TypeError:
            # Fallback si le processeur ne supporte pas 'tools' nativement
            logger.warning("Le processor ne supporte pas l'argument tools, injection manuelle.")
            tools_desc = json.dumps(hf_tools, indent=2, ensure_ascii=False)
            sys_msg = (
                "Tu as accès aux outils suivants:\n" + tools_desc + 
                "\nPour utiliser un outil, réponds UNIQUEMENT avec le format XML suivant:\n"
                "<tool_call>\n{\"name\": \"nom_outil\", \"arguments\": {\"arg1\": \"valeur1\"}}\n</tool_call>"
            )
            # Ajout au début
            if hf_messages and hf_messages[0]["role"] == "system":
                hf_messages[0]["content"] += "\n\n" + sys_msg
            else:
                hf_messages.insert(0, {"role": "system", "content": sys_msg})
                
            text_prompt = self.processor.apply_chat_template(
                hf_messages, tokenize=False, add_generation_prompt=True
            )

        # Passer les images PIL au processeur si présentes (Qwen-VL multimodal)
        if pil_images:
            logger.info("🖼️ [MODEL] Traitement multimodal : %d image(s) fournie(s)", len(pil_images))
            inputs = self.processor(
                text=[text_prompt],
                images=pil_images,
                return_tensors="pt"
            ).to(self.model.device)
        else:
            inputs = self.processor(text=[text_prompt], return_tensors="pt").to(self.model.device)

        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=_conf.max_tokens,
                do_sample=True,
                temperature=_conf.temperature,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Parse du format tool call de Qwen : <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
        tool_calls = []
        content = response_text

        # Regex pour matcher le bloc tool_call
        match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", response_text, re.DOTALL)
        if match:
            try:
                tc_data = json.loads(match.group(1))
                tool_calls.append({
                    "name": tc_data["name"],
                    "args": tc_data["arguments"],
                    "id": "tc_" + str(uuid.uuid4())[:8]
                })
                # On retire le bloc de tool call du contenu textuel retourné à l'utilisateur
                content = response_text.replace(match.group(0), "").strip()
            except Exception as e:
                logger.error("Erreur de parsing du tool_call Qwen : %s", e)

        return AIMessage(content=content, tool_calls=tool_calls)


# Singleton global
model_service = ModelService()
