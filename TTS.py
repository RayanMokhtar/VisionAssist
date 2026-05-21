import asyncio
import re
import subprocess

import requests
import edge_tts

API_URL = "http://localhost:8000/chat"
TTS_VOICE = "fr-FR-DeniseNeural"
TTS_OUTPUT = "/tmp/tts_response.mp3"


def play_audio_windows(file_path: str) -> None:
    """Joue un fichier MP3 via PowerShell Windows (compatible WSL)."""
    win_path = subprocess.check_output(["wslpath", "-w", file_path]).decode().strip()
    ps_script = f"""
Add-Type -AssemblyName presentationCore
$mp = New-Object System.Windows.Media.MediaPlayer
$mp.Open([uri]'{win_path}')
$mp.Play()
Start-Sleep -Milliseconds 300
while(-not $mp.NaturalDuration.HasTimeSpan){{Start-Sleep -Milliseconds 50}}
$duration = [int]$mp.NaturalDuration.TimeSpan.TotalMilliseconds
Start-Sleep -Milliseconds ($duration + 200)
$mp.Stop()
"""
    subprocess.run(["powershell.exe", "-Command", ps_script], check=True)


def synthesize_and_play(text: str) -> None:
    """Synthétise le texte avec edge-tts et le joue."""
    print("[TTS] Génération de l'audio …")
    async def _generate():
        tts = edge_tts.Communicate(text, voice=TTS_VOICE)
        await tts.save(TTS_OUTPUT)

    asyncio.run(_generate())
    print("[TTS] Lecture …")
    play_audio_windows(TTS_OUTPUT)


def get_response_and_speak(prompt: str, image_url: str | None = None) -> None:
    """Récupère la réponse complète de l'API et synthétise le TTS."""
    payload = {"text": prompt, "image_url": image_url}

    print(f"[INFO] Envoi de la requête à {API_URL} …")
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        # Récupérer la réponse JSON
        data = response.json()
        full_response = data.get("response", "")
        
        # Supprimer les astérisques
        clean_response = full_response.replace("*", "")
        
        print(f"\n[RÉPONSE]\n{clean_response}\n")
        
        # Synthétiser et jouer
        synthesize_and_play(clean_response)
        print("[Terminé]")

    except requests.exceptions.ConnectionError:
        print("[ERREUR] Impossible de se connecter à l'API. Le service est-il démarré ?")
    except requests.exceptions.Timeout:
        print("[ERREUR] Timeout (la génération prend trop longtemps).")
    except Exception as e:
        print(f"[ERREUR] {e}")


if __name__ == "__main__":
    get_response_and_speak(
        prompt="Que vois-tu sur l'image en français ?",
        image_url="./table_ronde.jpg",
    )




tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()


def stream_and_speak(text: str, image_url: str | None = None) -> None:
    """Consomme le stream de l'API et envoie les phrases au TTS au fur et à mesure."""
    payload = {"text": text, "image_url": image_url}
    buffer = ""
    sentence_end = re.compile(r"(?<=[.!?:\n])\s")

    print(f"[INFO] Envoi de la requête à {API_URL} …")
    try:
        with requests.post(API_URL, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            print("[INFO] Stream démarré, réception des tokens :\n")
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)
                    buffer += chunk
                    parts = sentence_end.split(buffer)
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if sentence:
                            tts_queue.put(sentence)
                    buffer = parts[-1]

    except requests.exceptions.ConnectionError:
        print("[ERREUR] Impossible de se connecter à l'API. Le service est-il démarré ?")
        return
    except Exception as e:
        print(f"[ERREUR] {e}")
        return

    # Synthétiser le reste du buffer
    if buffer.strip():
        tts_queue.put(buffer.strip())

    # Signal de fin + attente
    tts_queue.put(None)
    tts_thread.join()
    print("\n[Terminé]")


if __name__ == "__main__":
    stream_and_speak(
        text="Que vois-tu sur l'image en français ?",
        image_url="./table_ronde.jpg",
    )
