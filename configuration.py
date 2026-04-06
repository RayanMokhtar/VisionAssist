from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class STTConfig(BaseModel):
    model_name: Literal["tiny", "base", "small", "medium", "large-v3"] = "tiny"
    device: Literal["cuda", "cpu"] = "cpu"
    quantization_modele: Literal["int8", "float16", "int8_float16"] = "int8"
    langue: Optional[str] = "fr"
    exploration_possibilites_decodage: int = Field(5, ge=1, le=10)
    detection_activite_de_la_voix: bool = True
    duree_min_fin_parole : int = 500



class TTSConfig(BaseModel):
    moteur_tts: Literal["piper", "espeak","voxtral"] = "piper" #à voir si on migre sur du voxtral ? 
    chemin_modele: str = "./data/tts/modeles/fr_FR-siwis-medium.onnx"
    configuration_modele : Optional[str] = "./data/tts/modeles/fr_FR-siwis-medium.onnx.json"
    taux_echantillonnage_hz: int = 22050
    multiplicateur_lenteur: float = 1.0 # 2 alors 2 fois plus lent
    dossier_sortie: str = "./data/tts/sortie_modeles"


class AudioConfig(BaseModel):
    taux_echantillonnage_hz: int = 16000  #recommandé pour le tts à voir si on unifie pas
    canaux_ecoute : int = 1 # 2 si stéréo 
    taille_chunk : int = 1024
    device_index: Optional[int] = None




class TopicConfig(BaseModel):
    stt_topic  : str = "results/stt"
    tts_topic : str = "results/tts"
    vision_topic : str = "results/vision"
    erreurs_topic : str = "results/erreurs" 



class BrokerConfig(BaseModel):
    type_broker : Literal["RabbitMQ","mosquitto"] = "mosquitto"
    host: str = "localhost"
    port: int = Field(1883, ge=1, le=65535)
    keepalive: int = 60
    client_id: str = "visionassist-jetson" #TODO à modifier dans serveurito
    username: Optional[str] = None
    password: Optional[str] = None
    use_tls: bool = False
    qos: Literal[0, 1, 2] = 1
    retain: bool = True
    topics:TopicConfig=Field(default_factory=TopicConfig,description="configuration topics")


class PathConfig(BaseModel):
    data_dir: str = "./data"
    stt_file: str = "./data/stt_history.jsonl"
    tts_file: str = "./data/tts_history.jsonl"
    events_file: str = "./data/events.jsonl"
    server_file: str = "./data/server_exchanges.jsonl"
    log_file: str = "./logs/visionassist.log"
    log_level: str = "INFO"
    log_max_bytes: int = 5 * 1024 * 1024
    log_backup_count: int = 3


class Configuration(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )
    stt:STTConfig=Field(default_factory=STTConfig,description="configuration modele stt")
    tts:TTSConfig=Field(default_factory=TTSConfig,description="configuration modele tts")
    audio:AudioConfig=Field(default_factory=AudioConfig,description="configuration modele audio")
    broker:BrokerConfig=Field(default_factory=BrokerConfig,description="configuration broker")
    paths: PathConfig=Field(default_factory=PathConfig,description="conf paths chemin fichiers")
    

def get_configuration():
    return Configuration()


CONFIGURATION = get_configuration()