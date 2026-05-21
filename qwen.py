from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model_id = "Qwen/Qwen3-VL-4B-Instruct"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA n'est pas disponible.")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype=dtype,
    device_map={"": 0},
    attn_implementation="sdpa"
)

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Décris cette image en français."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs.pop("token_type_ids", None)
inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.8
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(output_text[0])