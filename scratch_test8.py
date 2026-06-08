import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./langage/modeles/Qwen3.5-27B")

messages = [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "quelle heure est il ?"},
    {"role": "assistant", "content": ""},
    {"role": "user", "content": "quel temps fait til a cergy ?"}
]

try:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("SUCCESS 8:")
except Exception as e:
    print("FAILED 8:", type(e), e)
