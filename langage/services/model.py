from __future__ import annotations

import json
import logging
import ast
import re
import uuid
from typing import Optional, List, Any

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText , BitsAndBytesConfig

from langchain_core.messages import AIMessage,BaseMessage,HumanMessage,SystemMessage,ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from qwen_vl_utils import process_vision_info 


from configuration import CONFIGURATION

logger = logging.getLogger(__name__)



class ModelService:
    """Service de chargement et d'accès au modèle LLM via Transformers."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.est_charge = False
        self.config_agent = CONFIGURATION.qwen
        self.load()

    def load(self) -> bool:
        """Charge le modèle directement via Transformers (singleton)."""
        if self.est_charge:
            return True

        model_id = self.config_agent.model_id
        logger.info("Chargement du modèle Transformers : %s...", model_id)
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(" format bits : ",dtype)

        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype
            )
            logger.info("Utilisation de BitsAndBytes (4-bit) pour optimiser la VRAM. device_map forcé sur 'cuda'.")
        except ImportError:
            quantization_config = None
            logger.warning("BitsAndBytes non installé. Chargement en précision standard.")

        print("quantization config", quantization_config)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.config_agent.device,
            attn_implementation="sdpa", #TODO : scaled dot product attention à revoir ??? 
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.est_charge = True
        logger.info("LLM chargé avec succès.")

    def get_chat_model(self):
        raise NotImplementedError("Utilisez generate_with_tools directement en mode transformers.")

    def invoke_simple(self, prompt: str) -> str:

        messages = [{"role": "user", "content": prompt}]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(text=[text_prompt], return_tensors="pt").to(self.config_agent.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config_agent.max_new_tokens,
                do_sample=self.config_agent.do_sample,
                temperature=self.config_agent.temperature,
            )
            
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated_ids, skip_special_tokens=True)

    def generate_with_tools(self, lc_messages: List[BaseMessage], tools: List[Any]) -> AIMessage:
        print("Messages reçus pour génération avec outils :", lc_messages)
        hf_messages = []
        for m in lc_messages:
            if isinstance(m, HumanMessage):
                if isinstance(m.content, list):
                    content_list = []
                    for part in m.content:
                        if part.get("type") == "text":
                            content_list.append({"type": "text", "text": part["text"]})
                        elif part.get("type") == "image_url":
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

        hf_tools = [convert_to_openai_tool(t) for t in tools]
        
        # try:
        print("tools :",tools)
        print("hf tools :",hf_tools)
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
        try:
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

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config_agent.max_new_tokens,
                do_sample=self.config_agent.do_sample,
                temperature=self.config_agent.temperature,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        tool_calls = []
        content = response_text

        print("Réponse brute du modèle :", response_text)


        reponse_nettoyee = ModelService.nettoyer_reponse_llm_brute(content,tool_calls)

        print("réponse nettoyée : " , reponse_nettoyee)

        return AIMessage(content=reponse_nettoyee, tool_calls=tool_calls)

    @staticmethod
    def nettoyer_reponse_llm_brute(reponse_brute: str, tool_calls: Optional[List] = None) -> str:
        if tool_calls is None:
            tool_calls = []
            
        for match_json in re.finditer(r"<tool_call>\s*({.*?})\s*</tool_call>", reponse_brute, re.DOTALL):
            try:
                tc_data = json.loads(match_json.group(1))
                tool_calls.append({
                    "name": tc_data.get("name", "unknown"),
                    "args": tc_data.get("arguments", {}),
                    "id": "tc_" + str(uuid.uuid4())[:8]
                })
            except Exception as e:
                logger.error("Erreur de parsing du tool_call Qwen JSON : %s", e)

        for match_xml in re.finditer(r"<tool_call>\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*</tool_call>", reponse_brute, re.DOTALL):
            try:
                name = match_xml.group(1).strip()
                args_str = match_xml.group(2).strip()
                args = {}
                if args_str:
                    try:
                        args = json.loads(args_str)
                    except Exception:
                        try:
                            args = ast.literal_eval(args_str)
                        except Exception:
                            pairs = re.findall(r'(\w+)=["\']([^"\']+)["\']', args_str)
                            if pairs:
                                args = dict(pairs)
                            else:
                                param_matches = re.findall(r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>', args_str, re.DOTALL)
                                if param_matches:
                                    args = {k: v.strip() for k, v in param_matches}
                                else:
                                    logger.error("Impossible de parser les arguments XML: %r", args_str)
                
                tool_calls.append({
                    "name": name,
                    "args": args if isinstance(args, dict) else {},
                    "id": "tc_" + str(uuid.uuid4())[:8]
                })
            except Exception as e:
                logger.error("Erreur de parsing du tool_call Qwen XML : %s", e)

        
        # 3. Nettoyage du texte (suppression des balises <think> et <tool_call>)
        content = re.sub(r"<think>.*?</think>", "", reponse_brute, flags=re.DOTALL)
        if "</think>" in content:
            content = content.split("</think>", 1)[-1]
            
        content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
        if "<tool_call>" in content:
            content = content.split("<tool_call>", 1)[0]
            
        return content.strip()


MODEL_SERVICE = ModelService()

if __name__ == "__main__":
    MODEL_SERVICE.generate_with_tools(lc_messages=[
        HumanMessage(content=[
            {"type": "text", "text": "quelle est la météo ?"},
            {"type": "image_url", "image_url": {"url": "./langage/missile.png"}}
        ])
    ], tools=[
        {
            "name": "get_weather",
            "description": "Récupère la météo pour une localisation donnée.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "La ville ou région pour la météo."}
                },
                "required": ["location"]
            }
        }
    ])
