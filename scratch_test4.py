import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./langage/modeles/Qwen3.5-27B")

messages = [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "question 1"},
    {"role": "assistant", "content": "answer 1"},
    {"role": "user", "content": "question 2"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_current_time\", \"arguments\": {}}\n</tool_call>"},
    {"role": "user", "content": "<tool_response>\n{\"name\": \"get_current_time\", \"content\": \"17:00\"}\n</tool_response>"}
]

try:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("SUCCESS 4:")
    print(prompt)
except Exception as e:
    print("FAILED 4:", e)
