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
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

from qwen_vl_utils import process_vision_info 


from configuration import CONFIGURATION

logger = logging.getLogger(__name__)



class ModelService:
    """Service de chargement et d'accès au modèle LLM via Transformers."""

    def __init__(self, model_id: str = None, load_in_4bit: bool = True):
        self.model = None
        self.processor = None
        self.est_charge = False
        self.config_agent = CONFIGURATION.qwen
        self.model_id = model_id or self.config_agent.model_id
        self.load_in_4bit = load_in_4bit
        self.load()

    def load(self) -> bool:
        """Charge le modèle directement via Transformers (singleton)."""
        if self.est_charge:
            return True

        model_id = self.model_id
        logger.info("Chargement du modèle Transformers : %s...", model_id)
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("Format de bits du modèle: %s", dtype)

        quantization_config = None
        if self.load_in_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype
                )
                logger.info("Utilisation de BitsAndBytes (4-bit) pour optimiser la VRAM. device_map forcé sur 'cuda'.")
            except ImportError:
                logger.warning("BitsAndBytes non installé. Chargement en précision standard.")
        else:
            logger.info("Chargement en précision native (sans BitsAndBytes).")

        logger.info("Quantization config: %s", quantization_config)
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



    def generer_reponse_pour_savoir_si_tool_necessaire_ou_pas(self, langchain_messages_du_graph: list, tools: list) -> AIMessage:
        """
        Prend l'historique LangChain, l'envoie à Qwen, et renvoie un AIMessage
        compris par LangGraph (avec ou sans tool_calls).
        """
        
        qwen_messages_parses = []
        for message in langchain_messages_du_graph:
            if isinstance(message, HumanMessage):
                # C'est ici qu'on gère tes images base64 ou tes textes
                if isinstance(message.content, list):
                    content_parts = []
                    for part in message.content:
                        if part.get("type") == "image_url":
                            # Conversion LangChain (OpenAI) vers format Qwen-VL
                            content_parts.append({
                                "type": "image",
                                "image": part["image_url"]["url"]
                            })
                        elif part.get("type") == "text":
                            content_parts.append(part)
                        else:
                            content_parts.append(part)
                    qwen_messages_parses.append({"role": "user", "content": content_parts})
                else:
                    qwen_messages_parses.append({"role": "user", "content": [{"type": "text", "text": message.content}]})

            elif isinstance(message, AIMessage):
                qwen_messages_parses.append({"role": "assistant", "content": message.content})

            elif isinstance(message, ToolMessage):
                qwen_messages_parses.append({"role": "tool", "name": message.name, "content": str(message.content)})

            elif isinstance(message, SystemMessage):
                qwen_messages_parses.append({"role": "system", "content": message.content})

        qwen_tools = [convert_to_openai_tool(t) for t in tools]

        #qwen + inférence
        text_prompt = self.processor.apply_chat_template(
            qwen_messages_parses, 
            tools=qwen_tools, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(qwen_messages_parses)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
            )
            
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        reponse_brute = self.processor.decode(generated_ids, skip_special_tokens=True)

        #parsing pour langchain
        tool_calls_langchain = []
        texte_nettoye = reponse_brute

        texte_nettoye = self.nettoyer_reponse_llm_brute(reponse_brute, tool_calls_langchain)

        return AIMessage(
            content=texte_nettoye, 
            tool_calls=tool_calls_langchain
            )

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
                        if pairs := re.findall(r'(\w+)=["\']([^"\']+)["\']', args_str):
                            args = dict(pairs)
                        elif param_matches := re.findall(r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>', args_str, re.DOTALL):
                            args = {k: v.strip() for k, v in param_matches}
                        else:
                            try:
                                args = ast.literal_eval(args_str)
                            except Exception:
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
        content = content.split("</think>", 1)[-1] if "</think>" in content else content
        content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
        content = content.split("<tool_call>", 1)[0] if "<tool_call>" in content else content

        # 4. Détecter un raisonnement interne non bailisé (anglais, style chain-of-thought)
        #    Si la réponse commence par des marqueurs de raisonnement, on cherche la vraie réponse.
        MARQUEURS_RAISONNEMENT = (
            "Thinking Process:", "Thinking Process", "The user", "Let me", "Let's", "Wait,", "Wait ", "Hmm", "However,",
            "Actually,", "Looking at", "I need to", "I should", "I must",
            "First,", "So,", "OK,", "Okay,", "Alright,",
        )
        contenu_stripped = content.strip()
        if any(contenu_stripped.startswith(m) for m in MARQUEURS_RAISONNEMENT):
            # Chercher la première ligne qui ressemble à une vraie réponse française
            lignes = contenu_stripped.splitlines()
            for i, ligne in enumerate(lignes):
                ligne_stripped = ligne.strip()
                # Une ligne française valide : non vide, ne commence pas par un marqueur anglais
                if (ligne_stripped
                        and not any(ligne_stripped.startswith(m) for m in MARQUEURS_RAISONNEMENT)
                        and not ligne_stripped.startswith("-")
                        and len(ligne_stripped) > 5):
                    content = "\n".join(lignes[i:]).strip()
                    logger.warning("[ModelService] Raisonnement non bailisé détecté et supprimé (%d lignes).", i)
                    break
            else:
                # Tout le contenu est du raisonnement : réponse générique
                content = "Désolé, je n'ai pas pu formuler une réponse claire. Pouvez-vous reformuler votre demande ?"
                logger.error("[ModelService] Toute la réponse semble être du raisonnement interne. Réponse générique utilisée.")

        return content.strip()


    def poser_question_sur_image(self, prompt: str, chemin_image: str) -> str:

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": chemin_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True 
        )

        image_inputs, video_inputs = process_vision_info(messages)  
        print("images inputs ",image_inputs)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        print("Génération en cours...")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=CONFIGURATION.qwen.max_new_tokens,
                temperature=CONFIGURATION.qwen.temperature,
                do_sample=CONFIGURATION.qwen.do_sample)
            
            
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        reponse_texte = self.processor.decode(generated_ids, skip_special_tokens=True)

        return reponse_texte

# Singleton paresseux : le modèle n'est chargé qu'au premier appel de get_model_service()
# Cela évite de charger 18 Go en VRAM lors d'un simple `import` du module.
_model_service_instance: ModelService | None = None

def get_model_service(model_id: str = None, load_in_4bit: bool = True) -> ModelService:
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService(model_id=model_id, load_in_4bit=load_in_4bit)
    return _model_service_instance

# Alias de compatibilité : les modules existants peuvent continuer à importer MODEL_SERVICE
# sans modification. L'instanciation reste différée au premier accès via __getattr__.
class _LazyProxy:
    """Proxy transparent qui instancie ModelService au premier accès d'attribut."""
    def __getattr__(self, name):
        return getattr(get_model_service(), name)

MODEL_SERVICE: ModelService = _LazyProxy()  # type: ignore
