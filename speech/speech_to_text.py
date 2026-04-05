import os
import tempfile
import time
import wave
import pyaudio

from typing import Callable, Optional , Literal 


from pydantic import BaseModel, Field
from faster_whisper import WhisperModel

from configuration import CONFIGURATION


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


MODELE_STT = WhisperModel(CONFIGURATION.stt.model_name,device=CONFIGURATION.stt.device,compute_type=CONFIGURATION.stt.quantization_modele)



class SpeechToText:

    def __init__(self) -> None:
        self.stt_configuration = CONFIGURATION.stt
        self.audio_config = CONFIGURATION.audio
        self.modele = MODELE_STT         
        self.charger_modele()      

    def charger_modele(self) -> None:
        if self.modele is None : 
            print("chargement modele stt")
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

    def enregistrer_audio_microphone(self, duree_record: int = 5) -> str:
        print("enregistrement audio en cours : ")
        #ouvrir streamaudio , enregistrer depuis micrphone ... 
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16, #chaqueéchantillon sur 16 bits 
            channels=self.audio_config.canaux_ecoute,
            rate=self.audio_config.taux_echantillonnage_hz,
            input=True,#micro en lecture
            input_device_index=self.audio_config.device_index,
            frames_per_buffer=self.audio_config.taille_chunk,
        )

        nb_chunks_obtenus = int((self.audio_config.taux_echantillonnage_hz / self.audio_config.taille_chunk) * duree_record) #combien de chunk : sur un seconde * nbr seconde

        #lire flux audio  nb_chunks fois , si buffer plain continue quand meme 
        frames = [stream.read(self.audio_config.taille_chunk, exception_on_overflow=False) for _ in range(nb_chunks_obtenus)]

        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("enregistrement fini ")
        fichier_temporaire_stockage = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)#suppression après via le unlink
        with wave.open(fichier_temporaire_stockage.name, "wb") as f:
            #écriture entête pour fastWhisper
            f.setnchannels(self.audio_config.canaux_ecoute)
            f.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            f.setframerate(self.audio_config.taux_echantillonnage_hz)
            f.writeframes(b"".join(frames))#écritures des frames

        return fichier_temporaire_stockage.name

    def pipeline(self, type: Literal["micro", "fichier"] = "micro", duree_record: Optional[int] = 5, chemin_fichier_audio: Optional[str] = None) -> STTResult:
        if type == "micro":
            chemin_fichier = self.enregistrer_audio_microphone(duree_record)
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

instance_stt = SpeechToText()
resultat = instance_stt.pipeline(type="micro", duree_record=10)

  