import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./langage/modeles/Qwen3.5-27B")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_current_time", "arguments": "{}"}}]},
    {"role": "tool", "name": "get_current_time", "content": "16:45"},
    {"role": "user", "content": "quelle heure est il ?"}
]

try:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("SUCCESS 1")
except Exception as e:
    print("FAILED 1:", e)

messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "quelle heure est il ?"},
    {"role": "assistant", "content": "I need to check.", "tool_calls": [{"type": "function", "function": {"name": "get_current_time", "arguments": "{}"}}]},
    {"role": "tool", "name": "get_current_time", "content": "16:45"}
]

try:
    prompt = tokenizer.apply_chat_template(messages2, add_generation_prompt=True, tokenize=False)
    print("SUCCESS 2")
except Exception as e:
    print("FAILED 2:", e)
