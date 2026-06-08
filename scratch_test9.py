import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./langage/modeles/Qwen3.5-27B")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns current time",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

messages = [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "quelle heure est il ?"},
    {"role": "assistant", "content": ""},
    {"role": "user", "content": "quel temps fait til a cergy ?"}
]

try:
    prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    print("SUCCESS 9:")
except Exception as e:
    print("FAILED 9:", type(e), e)
