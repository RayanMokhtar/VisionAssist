import transformers
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("./langage/modeles/Qwen3.5-27B-4bit-bitsandbytes")
print("Processor loaded.")
try:
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}}
            }
        }
    }]
    messages = [{"role": "user", "content": "Quel temps fait-il à Paris ?"}]
    res = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    print("Success with tools!")
    print(res)
except Exception as e:
    print("Error:", e)
