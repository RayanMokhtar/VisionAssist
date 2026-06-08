import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./langage/modeles/Qwen3.5-27B")

messages = [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "question 1"},
    {"role": "assistant", "content": "<tool_call>\n<function=get_weather>\n<parameter=city>\nCergy\n</parameter>\n</function>\n</tool_call>"},
    {"role": "user", "content": "quelle heure est il ?"}
]

try:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("SUCCESS 6:")
except Exception as e:
    print("FAILED 6:", e)
