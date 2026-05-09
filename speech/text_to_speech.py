from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import wave
from typing import Literal, Optional
import logging

import pyaudio
from piper import PiperVoice , SynthesisConfig
from pydantic import BaseModel, Field

from configuration import CONFIGURATION
from broker.service import get_broker_client 
from broker.broker_interface import IBroker 


### TTS piper : https://arxiv.org/html/2512.08006v1

#Documentation choix TTS : https://www.datacamp.com/blog/best-open-source-text-to-speech-tts-engines?utm_cid=23552157100&utm_aid=188237542530&utm_campaign=230119_1-ps-other~dsa-tofu~ai_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9056593-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~emea-en~dsa~tofu~blog~artificial-intelligence&gad_source=1&gad_campaignid=23552157100&gbraid=0AAAAADQ9WsFrYlpVH5Kp-ES7XFIPO7iEZ&gclid=CjwKCAjwpcTNBhA5EiwAdO1S9tnpoHkvvrWhSv92EZYExFu1k8D_lvjfX5-Hk7QYHOa8dY13zlenBBoCdd0QAvD_BwE

## fichier entrainement piper : https://raw.githubusercontent.com/rhasspy/piper/master/TRAINING.md

# https://arxiv.org/pdf/2106.06103


#lien vers la démo : https://rhasspy.github.io/piper-samples/demo.html



logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
    handlers=[
        logging.StreamHandler(), #console 
        logging.FileHandler(CONFIGURATION.paths.log_file), #fichier log
    ]
)

LOGGER = logging.getLogger(__name__)

configuration_tts = CONFIGURATION.tts

class TTSResult(BaseModel):
    texte: str
    chemin_fichier_audio: str
    duree_secondes : int = 0
    moteur: str = configuration_tts.moteur_tts
    multiplicateur_lenteur : float = configuration_tts.multiplicateur_lenteur

    def __str__(self) -> str:
        return f"[{self.texte} {self.chemin_fichier_audio} | {self.duree_secondes} s]" 

def charger_modele(configuration_tts=configuration_tts):
    if os.path.exists(configuration_tts.chemin_modele):
        print(f"{configuration_tts.chemin_modele}")
        #doc : https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
        modele  = PiperVoice.load(configuration_tts.chemin_modele, config_path=configuration_tts.configuration_modele)
        LOGGER.info("Modèle chargé" ) 
        return modele
    else : 
        LOGGER.error("modele pas chargé ??")

MODELE_TTS = charger_modele()


#chargement se fait via onnx
class TextToSpeech:
    def __init__(self) -> None:
        self.configuration_tts  = configuration_tts
        self.audio_config = CONFIGURATION.audio
        self.modele = MODELE_TTS
        os.makedirs(self.configuration_tts.dossier_sortie, exist_ok=True)
        self.dossier_sortie = self.configuration_tts.dossier_sortie

    def synthetiser(self, texte: str , nom_fichier_sortie : str) -> TTSResult:
    
        if not texte.strip():
            raise ValueError("Texte vide — rien à synthétiser.")

        t0 = time.perf_counter()

        chemin_sortie = f'{self.dossier_sortie}/{nom_fichier_sortie}'
        print("chemin_sortie : ",chemin_sortie)
        with wave.open(chemin_sortie, "wb") as f:
            self.modele.synthesize_wav(texte,f)

        t1 = time.perf_counter()

        duree = int((t1-t0))

        resultat = TTSResult(texte=texte,chemin_fichier_audio=chemin_sortie,duree_secondes=duree)
        print(f"TTS synthétisé : {resultat}")
        return resultat

    def lire_audio(self, chemin_fichier: str) -> None:
    
        audio = pyaudio.PyAudio()
        try:
           with wave.open(chemin_fichier, "rb") as f:
               stream = audio.open(
                   format=audio.get_format_from_width(f.getsampwidth()),
                   channels=f.getnchannels(),
                   rate=f.getframerate(),
                   output=True,
                   output_device_index=self.audio_config.device_index,
                   frames_per_buffer=self.audio_config.taille_chunk,
               )
               data = f.readframes(self.audio_config.taille_chunk)
               while data:
                   stream.write(data)
                   data = f.readframes(self.audio_config.taille_chunk)
               stream.stop_stream()
               stream.close()
        finally:
           audio.terminate()


    def pipeline(self, texte: str, nom_fichier_sortie : str = "test.wav", supprimer_fichier: bool = False) -> TTSResult:
        resultat = self.synthetiser(texte, nom_fichier_sortie=nom_fichier_sortie)
        try:
            self.lire_audio(resultat.chemin_fichier_audio)
        finally:
            if supprimer_fichier and os.path.exists(resultat.chemin_fichier_audio):
                os.unlink(resultat.chemin_fichier_audio)
        return resultat
    
    def fonction_trigger(self , topic , message_recu : str):
        MESSAGE_AVERTISSEMENT = "erreur potentielle dans la récupération du message faites attention"
        texte = message_recu.get("resultat_llm","")
        if texte : 
            LOGGER.info(f"tts demandé pour synthetiser ce texte : {texte}")
            resultat = self.pipeline(texte)
            #TODO : à voir si on publie dans le broker ou pas ??   
        else : 
            LOGGER.warning("Message TTS sans texte")
            resultat = self.pipeline(MESSAGE_AVERTISSEMENT)


#à faire basculer dans le init , et par ailleurs le topic sur écoute on pourrait ajouter le yolo si on veut une réponse rapide sans passer par le llm ? à voir ou juste un buzzer ? 
def lancement_service_tts(client_id : str = "tts", topic_sur_ecoute : str = CONFIGURATION.broker.topics.tts_topic):
    broker = get_broker_client(client_id) # à vori si singelton ou pas 
    TTS = TextToSpeech()
    broker.connexion()
    broker.sabonner(topic_sur_ecoute,TTS.fonction_trigger)

    LOGGER.info("TTS en écoute sur topic %s...",topic_sur_ecoute)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        broker.deconnexion()
        LOGGER.info("TTS arrêté.")


lancement_service_tts()