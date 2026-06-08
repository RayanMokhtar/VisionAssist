import transformers
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("./langage/modeles/Qwen3.5-27B-4bit-bitsandbytes")
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }
    }
}]
messages = [
    {"role": "user", "content": "Quel temps fait-il à Paris ?"},
    {"role": "assistant", "content": "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>"},
    {"role": "tool", "name": "get_weather", "content": "Il fait 20°C et ensoleillé."}
]
res = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
print("Observation formatting:")
print(res)
