import pyaudio
from typing import Optional

def trouver_index_audio_par_nom(mot_cle: Optional[str], type_peripherique: str) -> Optional[int]:
    """Cherche dynamiquement l'index d'un périphérique audio par son nom."""
    if not mot_cle:
        return None
        
    p = pyaudio.PyAudio()
    mot_cle = mot_cle.lower()
    index_trouve = None

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        # print(f"device {i} : {info}")
        nom_device = info.get('name', '').lower()
        
        is_input = info.get('maxInputChannels') > 0
        is_output = info.get('maxOutputChannels') > 0
        
        if mot_cle in nom_device:
            if type_peripherique == "input" and is_input:
                index_trouve = i
                print("trouvé input")
                break
            elif type_peripherique == "output" and is_output:
                print("trouvé output")
                index_trouve = i
                break
                
    p.terminate()
    if index_trouve is None:
        print(f"AVERTISSEMENT : Aucun périphérique {type_peripherique} trouvé contenant '{mot_cle}'.")
    return index_trouve


# import pyaudio

# p = pyaudio.PyAudio()
# print("--- TOUS LES PÉRIPHÉRIQUES VUS PAR PYAUDIO ---")
# for i in range(p.get_device_count()):
#     info = p.get_device_info_by_index(i)
#     nom = info.get('name')
#     entrees = info.get('maxInputChannels')
#     sorties = info.get('maxOutputChannels')
#     print(f"Index {i} : {nom} (Micro: {entrees}, Enceinte: {sorties})")
# p.terminate()
