import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./langage/modeles/Qwen3.5-27B")

messages = [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "Quelle heure est-il ?"},
    {"role": "assistant", "content": "Il est 16h55. Nous sommes lundi 8 juin 2026."},
    {"role": "user", "content": "quelle heure est t'il ?"},
    {"role": "assistant", "content": "L'utilisateur me demande l'heure. L'outil get_current_time m'a retourné l'heure actuelle : 16h59, lundi 8 juin 2026. Je dois répondre de manière concise et naturelle à la 2ème personne du vouvoiement.\n\n\nIl est 16h59. Nous sommes lundi 8 juin 2026."},
    {"role": "user", "content": "quelle heure est t'il ?"},
    {"role": "assistant", "content": "L'outil get_current_time m'a retourné l'heure actuelle : 17h00, lundi 8 juin 2026. Je dois répondre de manière concise et naturelle à la 2ème personne du vouvoiement.\n\n\nIl est 17h00. Nous sommes lundi 8 juin 2026."},
    {"role": "user", "content": "quel temps fait t'il a cergy ?"},
    {"role": "assistant", "content": "<tool_call>\n<function=get_weather>\n<parameter=city>\nCergy\n</parameter>\n</function>\n</tool_call>"},
    {"role": "user", "content": "quelle heure est t'il ?"},
    {"role": "assistant", "content": "L'utilisateur demande l'heure. L'outil get_current_time m'a fourni l'heure actuelle : 17h04, lundi 8 juin 2026. Je dois répondre de manière concise et naturelle en utilisant cette information fraîche.\n\n\nIl est 17h04. Nous sommes lundi 8 juin 2026."},
    {"role": "user", "content": "quel temps fait til a cergy ?"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_current_time\", \"arguments\": {}}\n</tool_call>"},
    {"role": "user", "content": "<tool_response>\n{\"name\": \"get_current_time\", \"content\": \"17:06\"}\n</tool_response>"}
]

try:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("SUCCESS 7:")
except Exception as e:
    print("FAILED 7:", type(e), e)
