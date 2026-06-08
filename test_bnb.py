from transformers import Qwen3VLForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch

model_id = "Qwen/Qwen3-VL-8B-Instruct"

# Try loading in 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

try:
    print(f"Loading {model_id} in 4-bit...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config
    )
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
