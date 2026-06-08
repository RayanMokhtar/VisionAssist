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

from langage.api.config import settings

logger = logging.getLogger(__name__)


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
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError(
                "transformers et torch requis. "
                "Installez avec : pip install transformers torch accelerate"
            )

        model_id = settings.model_id
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

        self.model = AutoModelForImageTextToText.from_pretrained(
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
                max_new_tokens=settings.max_new_tokens,
                do_sample=settings.do_sample,
                temperature=settings.temperature,
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
        for m in lc_messages:
            if isinstance(m, HumanMessage):
                # Détecter contenu multimodal (liste avec image_url + text)
                if isinstance(m.content, list):
                    content_list = []
                    for part in m.content:
                        if part.get("type") == "text":
                            content_list.append({"type": "text", "text": part["text"]})
                        elif part.get("type") == "image_url":
                            # Qwen-VL attend "image"
                            img_uri = part["image_url"]["url"]
                            content_list.append({"type": "image", "image": img_uri})
                    hf_messages.append({"role": "user", "content": content_list})
                else:
                    hf_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                if m.tool_calls:
                    hf_messages.append({
                        "role": "assistant", 
                        "content": m.content or "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["args"]
                                }
                            } for tc in m.tool_calls
                        ]
                    })
                else:
                    hf_messages.append({"role": "assistant", "content": m.content})
            elif isinstance(m, ToolMessage):
                # Qwen 2.5/3 attend le rôle 'tool' avec le nom et le contenu
                hf_messages.append({
                    "role": "tool", 
                    "name": m.name, 
                    "content": str(m.content)
                })
            elif isinstance(m, SystemMessage):
                hf_messages.append({"role": "system", "content": m.content})

        # Conversion des outils au format JSON Schema standard (OpenAI-like)
        hf_tools = [convert_to_openai_tool(t) for t in tools]
        
        # Pour Qwen3-VL, on injecte les outils dans le system prompt si on gère à la main
        # Mais le processeur Qwen 2.5/3 supporte l'argument tools dans apply_chat_template !
        try:
            try:
                text_prompt = self.processor.apply_chat_template(
                    hf_messages, 
                    tools=hf_tools, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.error("Erreur lors de l'appel à apply_chat_template. Messages: %s, Tools: %s", hf_messages, hf_tools)
                raise e
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

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(hf_messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
        except Exception as e:
            logger.warning("Impossible de traiter l'image avec qwen_vl_utils: %s", e)
            inputs = self.processor(text=[text_prompt], return_tensors="pt").to(self.model.device)

        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=settings.do_sample,
                temperature=settings.temperature,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Parse du format tool call de Qwen : <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
        tool_calls = []
        content = response_text

        # Regex pour matcher le bloc tool_call (format JSON classique ou format XML natif Qwen3)
        match_json = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", response_text, re.DOTALL)
        match_xml = re.search(r"<tool_call>\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*</tool_call>", response_text, re.DOTALL)

        if match_json:
            try:
                tc_data = json.loads(match_json.group(1))
                tool_calls.append({
                    "name": tc_data["name"],
                    "args": tc_data["arguments"],
                    "id": "tc_" + str(uuid.uuid4())[:8]
                })
                content = response_text.replace(match_json.group(0), "").strip()
            except Exception as e:
                logger.error("Erreur de parsing du tool_call Qwen JSON : %s", e)
        elif match_xml:
            try:
                name = match_xml.group(1).strip()
                args_str = match_xml.group(2).strip()
                args = {}
                if args_str:
                    try:
                        args = json.loads(args_str)
                    except Exception:
                        try:
                            import ast
                            args = ast.literal_eval(args_str)
                        except Exception:
                            # Fallback pour format XML attributs (ex: location="Cergy")
                            pairs = re.findall(r'(\w+)=["\']([^"\']+)["\']', args_str)
                            if pairs:
                                args = dict(pairs)
                            else:
                                # Parsing du format <parameter=nom>valeur</parameter>
                                param_matches = re.findall(r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>', args_str, re.DOTALL)
                                if param_matches:
                                    args = {k: v.strip() for k, v in param_matches}
                                else:
                                    logger.error("Impossible de parser les arguments: %r", args_str)
                tool_calls.append({
                    "name": name,
                    "args": args if isinstance(args, dict) else {},
                    "id": "tc_" + str(uuid.uuid4())[:8]
                })
                content = response_text.replace(match_xml.group(0), "").strip()
            except Exception as e:
                logger.error("Erreur de parsing du tool_call Qwen XML : %s | args: %r", e, match_xml.group(2) if match_xml else "")

        # Nettoyage des balises de "pensée" (Chain-of-Thought) du modèle Qwen
        content = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL)
        content = re.sub(r"</?think>", "", content).strip()

        return AIMessage(content=content, tool_calls=tool_calls)


# Singleton global
model_service = ModelService()
