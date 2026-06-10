import time
import sys
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from PIL import Image
import os
from threading import Thread

# 1. Configuration
MODEL_ID = "mistral-community/pixtral-12b"  # Version compatible avec Transformers
IMAGE_PATH = "table_ronde.png"           # Modifiez l'extension si c'est un .jpg

print("=== Initialisation Pixtral 12B sur CPU ===")

# Optimisation CPU : utiliser tous les cœurs physiques
torch.set_num_threads(torch.get_num_threads())

# Utiliser bfloat16 si le CPU le supporte (très rapide), sinon float32
dtype = torch.bfloat16 if torch.backends.cpu.get_cpu_capability() else torch.float32
print(f"[*] Type de tenseurs : {dtype}")

# 2. Chargement du Processeur et du Modèle
print(f"[*] Chargement du processeur depuis {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("[*] Chargement du modèle (cela nécessite ~24 Go de RAM et prend quelques minutes)...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    torch_dtype=dtype,
    low_cpu_mem_usage=True # Indispensable pour ne pas saturer la RAM au chargement
)
model.eval()

# 3. Préparation de l'image et du prompt
if not os.path.exists(IMAGE_PATH):
    # Fallback si l'image png n'existe pas, on tente jpg
    if os.path.exists("table_ronde.jpg"):
        IMAGE_PATH = "table_ronde.jpg"
    else:
        raise FileNotFoundError(f"L'image {IMAGE_PATH} est introuvable dans le dossier.")

print(f"\n[*] Traitement de l'image : {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("RGB")

PROMPT = "Décris en détail ce que tu vois dans cette image."
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    },
]

# Formatage du prompt selon le template de Mistral
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Préparation des tenseurs
inputs = processor(
    text=text_prompt,
    images=[image],
    return_tensors="pt"
)

# Le processeur Pixtral retourne pixel_values comme une liste de listes de tenseurs :
#   pixel_values = [[tensor_img1, tensor_img2, ...]]   (batch x images x C,H,W)
# Le modèle attend une simple liste de tenseurs (une par image).
if "pixel_values" in inputs and isinstance(inputs["pixel_values"], list):
    flat = []
    for batch_item in inputs["pixel_values"]:
        if isinstance(batch_item, list):
            for img_tensor in batch_item:
                flat.append(img_tensor.to(dtype))
        elif isinstance(batch_item, torch.Tensor):
            flat.append(batch_item.to(dtype))
    inputs["pixel_values"] = flat

# IMPORTANT SUR CPU : S'assurer que tous les tenseurs sont dans le même type que le modèle
for key in inputs:
    if key == "pixel_values":
        continue  # Déjà converti ci-dessus
    if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point():
        inputs[key] = inputs[key].to(dtype)

# 4. Génération en streaming
print("\n[*] Génération en cours (sur CPU, patientez quelques minutes)...")
print("=" * 60)
print("📝 Description de l'image :")
print("=" * 60)

start_time = time.time()

# Streamer pour afficher les tokens au fur et à mesure
streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

# Lancer la génération dans un thread séparé pour pouvoir streamer
generation_kwargs = dict(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    use_cache=True,
    streamer=streamer,
)

thread = Thread(target=lambda: model.generate(**generation_kwargs))
with torch.inference_mode():
    thread.start()

# Afficher chaque morceau de texte dès qu'il est généré
for text_chunk in streamer:
    print(text_chunk, end="", flush=True)

thread.join()

elapsed = time.time() - start_time

print(f"\n\n⏱️  Temps de génération : {elapsed:.1f} secondes")
