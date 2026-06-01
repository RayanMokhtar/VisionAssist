import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download


##Script installation 
REPO_ID = "rhasspy/piper-voices"  # Repo Hugging Face officiel pour Piper
MODEL_NAME = "fr_FR-siwis-medium"  # Voix française (medium quality)
LOCAL_DIR = "./data/tts/modeles/tts"  # Dossier de destination (comme dans TTSConfig)

# Chemins des fichiers dans le repo
ONNX_FILENAME = f"fr/fr_FR/siwis/medium/{MODEL_NAME}.onnx"
JSON_FILENAME = f"fr/fr_FR/siwis/medium/{MODEL_NAME}.onnx.json"

def nettoyer_dossier_vide(dossier):
    """Supprime un dossier s'il est vide (récursif)."""
    try:
        os.rmdir(dossier)  # Ne supprime que si vide
    except OSError:
        pass  # Pas vide, on laisse

def installer_modele():
    """
    Télécharge et installe le modèle Piper automatiquement.
    - Vérifie si les fichiers existent déjà (évite re-téléchargement).
    - Crée le dossier si besoin.
    - Télécharge depuis Hugging Face.
    - Déplace les fichiers au bon endroit.
    - Affiche la progression.
    """
    print("Installation du modèle Piper en cours...")

    # Crée le dossier local s'il n'existe pas
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Dossier de destination : {LOCAL_DIR}")

    # Chemin final des fichiers locaux (sans sous-dossiers)
    onnx_path = os.path.join(LOCAL_DIR, f"{MODEL_NAME}.onnx")
    json_path = os.path.join(LOCAL_DIR, f"{MODEL_NAME}.onnx.json")

    # Télécharge le modèle .onnx si pas déjà présent
    if not os.path.exists(onnx_path):
        print(f"Téléchargement du modèle .onnx ({ONNX_FILENAME})...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=ONNX_FILENAME,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False  # Télécharge vraiment le fichier
        )
        # Déplace le fichier du sous-dossier vers la racine
        actual_onnx_path = os.path.join(LOCAL_DIR, ONNX_FILENAME)
        if os.path.exists(actual_onnx_path):
            shutil.move(actual_onnx_path, onnx_path)
            # Nettoie les sous-dossiers vides
            subdirs = os.path.dirname(actual_onnx_path)
            while subdirs != LOCAL_DIR:
                nettoyer_dossier_vide(subdirs)
                subdirs = os.path.dirname(subdirs)
        print(f"Modèle .onnx installé : {onnx_path}")
    else:
        print(f"Modèle .onnx déjà présent : {onnx_path}")

    # Télécharge le fichier .json si pas déjà présent
    if not os.path.exists(json_path):
        print(f"Téléchargement de la config .json ({JSON_FILENAME})...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=JSON_FILENAME,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        # Déplace le fichier
        actual_json_path = os.path.join(LOCAL_DIR, JSON_FILENAME)
        if os.path.exists(actual_json_path):
            shutil.move(actual_json_path, json_path)
            # Nettoie les sous-dossiers vides
            subdirs = os.path.dirname(actual_json_path)
            while subdirs != LOCAL_DIR:
                nettoyer_dossier_vide(subdirs)
                subdirs = os.path.dirname(subdirs)
        print(f"Config .json installée : {json_path}")
    else:
        print(f"Config .json déjà présente : {json_path}")

    # Vérification finale
    if os.path.exists(onnx_path) and os.path.exists(json_path):
        taille_onnx = os.path.getsize(onnx_path) / (1024 * 1024)  # En Mo
        taille_json = os.path.getsize(json_path) / 1024  # En Ko
        print("Installation réussie !")
        print(f"- Modèle : {onnx_path} ({taille_onnx:.1f} Mo)")
        print(f"- Config : {json_path} ({taille_json:.1f} Ko)")
        print("Tu peux maintenant utiliser TextToSpeech avec Piper.")
    else:
        print("Erreur : Un ou plusieurs fichiers n'ont pas été téléchargés.")

if __name__ == "__main__":
    installer_modele()