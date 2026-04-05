from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import wave
from typing import Literal, Optional

import pyaudio
from piper.voice import PiperVoice
from pydantic import BaseModel, Field

from configuration import CONFIGURATION


### TTS piper : https://arxiv.org/html/2512.08006v1

#Documentation choix TTS : https://www.datacamp.com/blog/best-open-source-text-to-speech-tts-engines?utm_cid=23552157100&utm_aid=188237542530&utm_campaign=230119_1-ps-other~dsa-tofu~ai_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9056593-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~emea-en~dsa~tofu~blog~artificial-intelligence&gad_source=1&gad_campaignid=23552157100&gbraid=0AAAAADQ9WsFrYlpVH5Kp-ES7XFIPO7iEZ&gclid=CjwKCAjwpcTNBhA5EiwAdO1S9tnpoHkvvrWhSv92EZYExFu1k8D_lvjfX5-Hk7QYHOa8dY13zlenBBoCdd0QAvD_BwE


class TTSResult(BaseModel):
    texte: str
    chemin_fichier_audio: str
    duration_ms: int = 0
    moteur: str = "piper"
    vitesse: float = 1.0

    def __str__(self) -> str:
        return f"[{self.moteur} | {self.duration_ms}ms | x{self.vitesse}] {self.texte[:60]}"




def charger_modele():
    configuration = CONFIGURATION.tts
    if not os.path.exists(configuration.model)
    return PiperVoice.load(cfg.model_path, config_path=cfg.config_path)

MODELE_TTS = charger_modele()


class TextToSpeech:
    """
    Moteur TTS basé sur Piper (ONNX, ARM-optimisé, 100% local).

    Méthodes principales :
        synthetiser(texte)              → génère un fichier WAV, retourne TTSResult
        lire_audio(chemin)              → joue un fichier WAV via PyAudio
        pipeline(texte)                 → synthetiser + lire_audio en une seule étape
        pipeline_streaming(texte)       → synthèse + lecture chunk par chunk (latence réduite)

    Exemple :
        tts = TextToSpeech()
        result = tts.pipeline("Bonjour, je suis VisionAssist")
        print(result)
    """

    def __init__(self) -> None:
        self.tts_config  = CONFIGURATION.tts
        self.audio_config = CONFIGURATION.audio
        self.modele = MODELE_TTS
        os.makedirs(self.tts_config.output_dir, exist_ok=True)

    # ─── Synthèse ─────────────────────────────────────────────────────────

    def synthetiser(self, texte: str, chemin_sortie: Optional[str] = None) -> TTSResult:
        """
        Synthétise le texte en fichier WAV.

        Args:
            texte:         Texte à synthétiser.
            chemin_sortie: Chemin du fichier WAV de sortie.
                           Si None, crée un fichier temporaire dans output_dir.

        Returns:
            TTSResult avec le chemin du fichier généré et la durée de traitement.
        """
        if not texte.strip():
            raise ValueError("Texte vide — rien à synthétiser.")

        if chemin_sortie is None:
            os.makedirs(self.tts_config.output_dir, exist_ok=True)
            fd, chemin_sortie = tempfile.mkstemp(
                suffix=".wav",
                dir=self.tts_config.output_dir,
            )
            os.close(fd)

        t0 = time.perf_counter()

        with wave.open(chemin_sortie, "wb") as fichier_wav:
            # Piper écrit directement dans le fichier WAV (entête + PCM 16-bit)
            self.modele.synthesize(
                texte,
                fichier_wav,
                length_scale=1.0 / self.tts_config.speed,  # speed=2.0 → length_scale=0.5
            )

        duration_ms = int((time.perf_counter() - t0) * 1000)

        result = TTSResult(
            texte=texte,
            chemin_fichier_audio=chemin_sortie,
            duration_ms=duration_ms,
            moteur=self.tts_config.engine,
            vitesse=self.tts_config.speed,
        )
        print(f"TTS synthétisé : {result}")
        return result

    # ─── Lecture audio ────────────────────────────────────────────────────

    def lire_audio(self, chemin_fichier: str) -> None:
        """
        Joue un fichier WAV via PyAudio (haut-parleur local).

        Args:
            chemin_fichier: Chemin du fichier WAV à lire.
        """
        if not os.path.exists(chemin_fichier):
            raise FileNotFoundError(f"Fichier audio introuvable : {chemin_fichier}")

        audio = pyaudio.PyAudio()
        try:
            with wave.open(chemin_fichier, "rb") as wf:
                stream = audio.open(
                    format=audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=self.audio_config.device_index,
                    frames_per_buffer=self.audio_config.taille_chunk,
                )
                data = wf.readframes(self.audio_config.taille_chunk)
                while data:
                    stream.write(data)
                    data = wf.readframes(self.audio_config.taille_chunk)

                stream.stop_stream()
                stream.close()
        finally:
            audio.terminate()

    # ─── Pipeline complet ─────────────────────────────────────────────────

    def pipeline(
        self,
        texte: str,
        supprimer_fichier: bool = True,
        chemin_sortie: Optional[str] = None,
    ) -> TTSResult:
        """
        Synthétise et joue le texte en une seule étape.

        Args:
            texte:            Texte à synthétiser et lire.
            supprimer_fichier: Si True, supprime le WAV après lecture.
            chemin_sortie:    Chemin de sortie optionnel (sinon fichier temporaire).

        Returns:
            TTSResult de la synthèse.
        """
        result = self.synthetiser(texte, chemin_sortie=chemin_sortie)
        try:
            self.lire_audio(result.chemin_fichier_audio)
        finally:
            if supprimer_fichier and os.path.exists(result.chemin_fichier_audio):
                os.unlink(result.chemin_fichier_audio)
        return result

    # ─── Pipeline streaming (latence réduite) ─────────────────────────────

    def pipeline_streaming(self, texte: str) -> TTSResult:
        """
        Synthèse et lecture en parallèle via un thread producteur/consommateur.

        Avantage : commence à jouer dès que les premiers chunks audio sont prêts.
        Latence perçue réduite de ~40% vs pipeline() séquentiel.

        Principe :
            Thread 1 (producteur) : Piper génère les chunks PCM → file d'attente
            Thread 2 (consommateur) : PyAudio lit la file et joue en temps réel
        """
        if not texte.strip():
            raise ValueError("Texte vide.")

        t0 = time.perf_counter()
        file_audio: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=20)
        sample_rate = self.tts_config.sample_rate

        def _produire() -> None:
            """Génère les chunks PCM et les pousse dans la file."""
            try:
                for chunk_audio in self.modele.synthesize_stream_raw(
                    texte,
                    length_scale=1.0 / self.tts_config.speed,
                ):
                    file_audio.put(chunk_audio)
            finally:
                file_audio.put(None)  # signal de fin

        def _consommer() -> None:
            """Lit la file et envoie à PyAudio."""
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=self.audio_config.device_index,
                frames_per_buffer=self.audio_config.taille_chunk,
            )
            try:
                while True:
                    chunk = file_audio.get()
                    if chunk is None:
                        break
                    stream.write(chunk)
            finally:
                stream.stop_stream()
                stream.close()
                audio.terminate()

        producteur  = threading.Thread(target=_produire,  daemon=True)
        consommateur = threading.Thread(target=_consommer, daemon=True)

        producteur.start()
        consommateur.start()

        producteur.join()
        consommateur.join()

        duration_ms = int((time.perf_counter() - t0) * 1000)

        return TTSResult(
            texte=texte,
            chemin_fichier_audio="",   # pas de fichier en mode streaming
            duration_ms=duration_ms,
            moteur=self.tts_config.engine,
            vitesse=self.tts_config.speed,
        )