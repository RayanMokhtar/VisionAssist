from transformers import Qwen3VLForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch

model_id = "Qwen/Qwen3-VL-8B-Instruct"

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=dtype,
    llm_int8_enable_fp32_cpu_offload=True
)

try:
    print(f"Loading {model_id} in 4-bit with CPU offload...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=dtype
    )
    print("Success with CPU offload!")
except Exception as e:
    print(f"Failed with CPU offload: {e}")

try:
    print(f"Loading {model_id} in 4-bit forcing CUDA...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
        quantization_config=quantization_config,
        torch_dtype=dtype
    )
    print("Success with forced CUDA!")
except Exception as e:
    print(f"Failed with forced CUDA: {e}")

