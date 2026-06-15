from __future__ import annotations

import json
import logging
import ast
import re
import uuid
from typing import Optional, List

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from qwen_vl_utils import process_vision_info

from configuration import CONFIGURATION

logger = logging.getLogger(__name__)


class ModelService:
    """Chargement et inférence du modèle Qwen via Transformers."""

    def __init__(self, model_id: str = None, load_in_4bit: bool = True):
        self.model = None
        self.processor = None
        self.est_charge = False
        self.config_agent = CONFIGURATION.qwen
        self.model_id = model_id or self.config_agent.model_id
        self.load_in_4bit = load_in_4bit
        self.load()

    def load(self) -> None:
        if self.est_charge:
            return

        logger.info("Chargement du modèle Transformers : %s...", self.model_id)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("Format de bits du modèle: %s", dtype)

        quantization_config = None
        if self.load_in_4bit:
            try:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
                logger.info("BitsAndBytes 4-bit activé, device_map='cuda'.")
            except ImportError:
                logger.warning("BitsAndBytes non installé, chargement en précision standard.")
        else:
            logger.info("Chargement en précision native (sans BitsAndBytes).")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=self.config_agent.device,
            attn_implementation="sdpa",
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.est_charge = True
        logger.info("LLM chargé avec succès.")

    def generer_reponse_pour_savoir_si_tool_necessaire_ou_pas(
        self, langchain_messages: list, tools: list
    ) -> AIMessage:
        """Prend l'historique LangChain, envoie au modèle, retourne un AIMessage (avec ou sans tool_calls)."""

        qwen_messages = []
        for msg in langchain_messages:
            if isinstance(msg, HumanMessage):
                if isinstance(msg.content, list):
                    parts = []
                    for part in msg.content:
                        if part.get("type") == "image_url":
                            print("part")
                            parts.append({"type": "image", "image": part["image_url"]["url"]})
                        else:
                            parts.append(part)
                    qwen_messages.append({"role": "user", "content": parts})
                else:
                    qwen_messages.append({"role": "user", "content": [{"type": "text", "text": msg.content}]})
            elif isinstance(msg, AIMessage):
                qwen_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                qwen_messages.append({"role": "tool", "name": msg.name, "content": str(msg.content)})
            elif isinstance(msg, SystemMessage):
                qwen_messages.append({"role": "system", "content": msg.content})

        qwen_tools = [convert_to_openai_tool(t) for t in tools]

        text_prompt = self.processor.apply_chat_template(
            qwen_messages, tools=qwen_tools, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7)

        reponse_brute = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        del inputs, output_ids
        torch.cuda.empty_cache()

        tool_calls: list = []
        texte = self.nettoyer_reponse_llm_brute(reponse_brute, tool_calls)
        return AIMessage(content=texte, tool_calls=tool_calls)

    @staticmethod
    def nettoyer_reponse_llm_brute(reponse_brute: str, tool_calls: Optional[List] = None) -> str:
        if tool_calls is None:
            tool_calls = []

        # Parser les tool_calls JSON
        for m in re.finditer(r"<tool_call>\s*({.*?})\s*</tool_call>", reponse_brute, re.DOTALL):
            try:
                tc = json.loads(m.group(1))
                tool_calls.append({"name": tc.get("name", "unknown"), "args": tc.get("arguments", {}), "id": "tc_" + str(uuid.uuid4())[:8]})
            except Exception as e:
                logger.error("Erreur parsing tool_call JSON : %s", e)

        # Parser les tool_calls XML
        for m in re.finditer(r"<tool_call>\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*</tool_call>", reponse_brute, re.DOTALL):
            try:
                name = m.group(1).strip()
                args_str = m.group(2).strip()
                args = {}
                if args_str:
                    try:
                        args = json.loads(args_str)
                    except Exception:
                        if pairs := re.findall(r'(\w+)=["\']([^"\']+)["\']', args_str):
                            args = dict(pairs)
                        elif params := re.findall(r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>', args_str, re.DOTALL):
                            args = {k: v.strip() for k, v in params}
                        else:
                            try:
                                args = ast.literal_eval(args_str)
                            except Exception:
                                logger.error("Impossible de parser les arguments XML: %r", args_str)
                tool_calls.append({"name": name, "args": args if isinstance(args, dict) else {}, "id": "tc_" + str(uuid.uuid4())[:8]})
            except Exception as e:
                logger.error("Erreur parsing tool_call XML : %s", e)

        # Nettoyer le texte (supprimer <think> et <tool_call>)
        content = re.sub(r"<think>.*?</think>", "", reponse_brute, flags=re.DOTALL)
        content = content.split("</think>", 1)[-1] if "</think>" in content else content
        content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
        content = content.split("<tool_call>", 1)[0] if "<tool_call>" in content else content

        # Détecter raisonnement interne non balisé
        MARQUEURS = (
            "Thinking Process:", "The user", "Let me", "Let's", "Wait,", "Wait ", "Hmm",
            "However,", "Actually,", "Looking at", "I need to", "I should", "I must",
            "First,", "So,", "OK,", "Okay,", "Alright,",
        )
        stripped = content.strip()
        if any(stripped.startswith(m) for m in MARQUEURS):
            for i, ligne in enumerate(stripped.splitlines()):
                l = ligne.strip()
                if l and not any(l.startswith(m) for m in MARQUEURS) and not l.startswith("-") and len(l) > 5:
                    content = "\n".join(stripped.splitlines()[i:]).strip()
                    logger.warning("[ModelService] Raisonnement non balisé supprimé (%d lignes).", i)
                    break
            else:
                content = "Désolé, je n'ai pas pu formuler une réponse claire. Pouvez-vous reformuler votre demande ?"
                logger.error("[ModelService] Toute la réponse est du raisonnement interne.")

        return content.strip()

    def poser_question_sur_image(self, prompt: str, chemin_image: str) -> str:
        print( chemin_image,"image du model")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": chemin_image},
            {"type": "text", "text": prompt},
        ]}]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=CONFIGURATION.qwen.max_new_tokens,
                temperature=CONFIGURATION.qwen.temperature,
                do_sample=CONFIGURATION.qwen.do_sample,
            )

        result = self.processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        del inputs, output_ids
        torch.cuda.empty_cache()
        return result





MODEL_SERVICE: ModelService = ModelService()
