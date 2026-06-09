import os
import tempfile
import time
import wave
import pyaudio
import audioop
import collections


from typing import Callable, Optional , Literal 


from pydantic import BaseModel, Field
from faster_whisper import WhisperModel

from configuration import CONFIGURATION
from broker.service import get_broker_client 
from security.utils import extraire_pin_4_chiffres


#Lien de la cdoc de doc de la librairie faster_whisper :  https://github.com/SYSTRAN/faster-whisper


class STTSegment(BaseModel):
    debut: float
    fin: float
    texte: str
    score_confiance_moyen: float #de -infini , à 0 , TODO : ajouter seuil de confiance sinon demander à l'utilisateur de réessayer.
    probabilite_audio_silence: float #p = 0 =>  parole

    def __str__(self) -> str:# pour afficher l'objet
        return f"[{self.debut} {self.fin} | {self.texte} | scoreconfiance moyen : {self.score_confiance_moyen}, probabilite que ce soit un audio : {self.probabilite_audio_silence}] {self.text}"


class STTResult(BaseModel):
    texte: str
    language: str
    language_probability: float = Field(ge=0.0, le=1.0)
    segments: list[STTSegment] = []
    duration_ms: int = 0
    chemin_fichier_audio: Optional[str] = None
    def __str__(self) -> str:# pour afficher l'objet
        return f"[{self.language} {self.language_probability:.0%} | {self.duration_ms}ms] {self.texte} , segments : \n {self.segments}" 

print("configuration  dans stt,",CONFIGURATION.broker)


MODELE_STT = WhisperModel(CONFIGURATION.stt.model_name,device=CONFIGURATION.stt.device,compute_type=CONFIGURATION.stt.quantization_modele)

CLIENT_BROKER_STT = get_broker_client(client_id="stt-jetson")

class SpeechToText:

    def __init__(self) -> None:
        self.stt_configuration = CONFIGURATION.stt
        self.audio_config = CONFIGURATION.audio
        self.modele = MODELE_STT         
        self.charger_modele()      

    def charger_modele(self) -> None:
        if self.modele is None : 
            print("chargement modele stt selon conf ", self.stt_configuration)
            self.modele = WhisperModel(
                self.stt_configuration.model_name,
                device=self.stt_configuration.device,
                quantization_modele=self.stt_configuration.quantization_modele,
            )
        else : 
            print("modele déjà chargé avec la configuration ",self.stt_configuration)

    def transcrire_fichier_audio(self, chemin_fichier_audio: str) -> STTResult:
        segments = []
        if self.modele is None:
            raise RuntimeError("modele pas chargé ...")
        if not os.path.exists(chemin_fichier_audio):
            raise FileNotFoundError(chemin_fichier_audio)

        t0 = time.perf_counter()
        segments_iter, info = self.modele.transcribe(
            chemin_fichier_audio,
            language=self.stt_configuration.langue,
            beam_size=self.stt_configuration.exploration_possibilites_decodage,
            vad_filter=self.stt_configuration.detection_activite_de_la_voix,
            vad_parameters={"min_silence_duration_ms": self.stt_configuration.duree_min_fin_parole},
        )

        for s in segments_iter : 
            segments.append(STTSegment(
                    debut=round(s.start, 3),
                    fin=round(s.end, 3),
                    texte=s.text.strip(),
                    score_confiance_moyen=round(s.avg_logprob, 4),
                    probabilite_audio_silence=round(s.no_speech_prob, 4),
            ))
        
        t1 = time.perf_counter()
        result = STTResult(
            texte=" ".join(s.texte for s in segments).strip(),
            language=info.language,
            language_probability=round(info.language_probability, 3),
            segments=segments, #on laisse pour l'instant pour debug
            duration_ms=int((t1 - t0) * 1000),
            chemin_fichier_audio=chemin_fichier_audio,
        )

        return result


    
    def enregistrer_audio_microphone_apres_activation(self) -> Optional[str]:
        print("micro en écoute ...")
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=self.audio_config.canaux_ecoute,
            rate=self.audio_config.taux_echantillonnage_hz, #humain entre 300 et 3400hz
            input=True,
            input_device_index=self.audio_config.device_index,
            frames_per_buffer=self.audio_config.taille_chunk,
        )

        #duree chunk : 64 ms car c la taille d'un chunk 1024 / taille échantillon valeur par seconde
        chunk_duree_secondes = self.audio_config.taille_chunk / self.audio_config.taux_echantillonnage_hz
        silence_chunks_max = int(self.audio_config.silence_duree_max_secondes / chunk_duree_secondes)
        max_chunks = int(self.audio_config.duree_max_enregistrement_theorique / chunk_duree_secondes)
        pre_roll_chunks = int(self.audio_config.pre_roll_parole_avant_enregistrement_secondes / chunk_duree_secondes)

        pre_buffer = collections.deque(maxlen=pre_roll_chunks)
        frames = []
        
        print(" taille file buffer : ",pre_buffer)
        print("silence chunks ",silence_chunks_max)
        parole_detectee = False
        silence_chunks = 0

        try:
            for _ in range(max_chunks):
                data = stream.read(
                    self.audio_config.taille_chunk,
                    exception_on_overflow=False
                )
                # print("data : ",data)  data en little endian
                energie = audioop.rms(data, 2)  # 2 octets par échantillon
                # print("energie :", energie)  
                if not parole_detectee:
                    pre_buffer.append(data)
                    if energie > self.audio_config.seuil_energie:
                        print("parole détectée")
                        parole_detectee = True
                        frames.extend(pre_buffer) #ajout du début (taille fixe de la file car les nouveaux éléments se poussent)
                        frames.append(data)
                    continue

                frames.append(data)

                if energie < self.audio_config.seuil_energie:
                    silence_chunks += 1
                else:
                    silence_chunks = 0 #reprise de parole

                if silence_chunks >= silence_chunks_max:
                    print("fin de parole détectée")
                    break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        fichier_temporaire_stockage = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        # Vérifier si des frames ont été enregistrées
        if not frames or len(frames) < 10:  # Au moins quelques frames
            print("Aucune parole détectée - fichier vide non retourné")
            os.unlink(fichier_temporaire_stockage.name)
            return None

        with wave.open(fichier_temporaire_stockage.name, "wb") as f:
            f.setnchannels(self.audio_config.canaux_ecoute)
            f.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            f.setframerate(self.audio_config.taux_echantillonnage_hz)
            f.writeframes(b"".join(frames))

        return fichier_temporaire_stockage.name
    
    #après authentification on l'active, quand il a rentré le pin ...sinon enregistrer_audio_microphone_apres_activation
    def ecouter_en_continu_avec_mot_activation(self) -> None : 
        print("mode écoute active en cours ...")
        actif = False
        CLIENT_BROKER_STT.connexion()
        while True : 
            chemin_audio = self.enregistrer_audio_microphone_apres_activation()
            if chemin_audio is None : 
                continue
            try : 
                resultat = self.transcrire_fichier_audio(chemin_fichier_audio=chemin_audio)
                texte = resultat.texte.lower().strip()
                print("texte renvoyé par écoute : ",texte)
                if not texte : 
                    continue
                if not actif :  
                    if CONFIGURATION.nom_assistant.lower() in texte:
                        actif = True
                        print("assistant activé car présent dans texte : ",texte)
                        message_payload = {"resultat_stt":{"texte":texte.replace(CONFIGURATION.nom_assistant.lower(),"")}}
                        pub = CLIENT_BROKER_STT.publier(CONFIGURATION.broker.topics.stt_topic,message_payload)
                        print("message publié sur le broker : ",CLIENT_BROKER_STT)
                        print("état payload publié : ",pub )
                        actif = False
            except Exception as e : 
                print("erreur inattenue dans ecoute continue stt",str(e))
            finally : 
                os.unlink(chemin_audio)

    def pipeline_authentification_stt(self , nombre_tentatives : int = 0): #pour tester on garde à 1 mais sera à injecter
        while nombre_tentatives <= CONFIGURATION.security.max_tentatives_avant_blocage_carte_gemalto : 
            chemin_audio = self.enregistrer_audio_microphone_apres_activation()
            if chemin_audio is None : 
                continue
            try : 
                resultat = self.transcrire_fichier_audio(chemin_fichier_audio=chemin_audio)
                texte = resultat.texte.strip()
                print("résultat transcript ",texte)
                pin_potentiel = extraire_pin_4_chiffres(texte)
                return pin_potentiel
            except Exception as e : 
                print("erreur inattenue dans pipeline authentification stt",str(e))
                return None
            finally : 
                os.unlink(chemin_audio)
            


    def pipeline(self, type: Literal["micro", "fichier"] = "micro", chemin_fichier_audio: Optional[str] = None) -> STTResult:
        if type == "micro":
            chemin_fichier = self.enregistrer_audio_microphone_apres_activation()
        elif type == "fichier":
            chemin_fichier = chemin_fichier_audio
        else:
            raise ValueError(f"type entrée non supporté : {type}")
        try:
            transcription = self.transcrire_fichier_audio(chemin_fichier)
            print("résultat transcription" , transcription)
        finally:
            if type == "micro":
                os.unlink(chemin_fichier)

INSTANCE_STT = SpeechToText()
# resultat = INSTANCE_STT.ecouter_en_continu_avec_mot_activation()

if __name__ == "__main__":
    INSTANCE_STT.ecouter_en_continu_avec_mot_activation()
  