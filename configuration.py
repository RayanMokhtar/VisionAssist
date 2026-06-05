from __future__ import annotations# annotation de type sont converties en string et pas évaluées directement , évite erreur qd import circulaires

from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class STTConfig(BaseModel):
    model_name: Literal["tiny", "base", "small", "medium", "large-v3"] = "small"
    device: Literal["cuda", "cpu"] = "cpu"
    quantization_modele: Literal["int8", "float16", "int8_float16"] = "int8"
    langue: Optional[str] = "fr"
    exploration_possibilites_decodage: int = Field(10, ge=1, le=10)
    detection_activite_de_la_voix: bool = True
    duree_min_fin_parole : int = 1200



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
    silence_duree_max_secondes: float = 1.2
    duree_max_enregistrement_theorique: float = 20
    seuil_energie: int = 300 #pour le RMS - réduit pour meilleure détection de parole
    pre_roll_parole_avant_enregistrement_secondes: float = 0.8





class TopicConfig(BaseModel):
    stt_topic  : str = "results/stt"
    tts_topic : str = "results/tts"
    vision_topic : str = "results/vision"
    erreurs_topic : str = "results/erreurs" 
    llm_topic_ecoute_stt : str = "results/stt"
    llm_topic_ecoute_vision : str = "results/vision"
    llm_topic_publication : str = "results/tts"
    #partie sécurité
    security_client_request_topic : str = "results/security/client/request" # pour authentification
    security_admin_request_topic : str = "results/security/admin/request" # pour l'enrollement de la carte ( génération workflow card_id)
    security_response_topic : str = "results/security/response" # réponses des deux cas ... 


class BrokerConfig(BaseModel):
    type_broker : Literal["RabbitMQ","mosquitto"] = "mosquitto"
    host: str = "172.20.10.4"
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


class DBConfig(BaseModel):
    url  : Optional[str] = Field(None, description="URL de connexion à la base de données")
    host : str = Field("localhost", description="Adresse du serveur de base de données")
    port : int = Field(5432, description="Port de connexion à la base de données")
    username : str = Field("qlq chose", description="Nom d'utilisateur pour la base de données")
    password : str = Field("password", description="Mot de passe pour la base de données")
    echo : bool = Field(False, description="Afficher les requêtes SQL dans les logs (True/False)")
    pool_size : int = Field(5, description="Nombre de connexions dans le pool de la base de données")
    dialecte : str = Field("sqlite", description="Dialecte de la base de données (ex: postgresql, mysql, sqlite)")
    db_file : Optional[str] = Field("db_file.db", description="Chemin du fichier de base de données SQLite (si dialecte sqlite)")
    db_name : str = Field("visionassist_db", description="Nom de la base de données (si dialecte postgresql ou mysql)")

#TODO récupérer ça depuis la machine pour tester 
class JWTConfig(BaseModel):
    pass
 
class SecurityConfig(BaseModel):
    jwtConfig : JWTConfig = Field(default_factory=JWTConfig,description="jwt config ")
    max_tentatives_avant_blocage_carte_gemalto : int = 3


class Configuration(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )
    nom_assistant : str = "LAZIBUS"
    stt:STTConfig=Field(default_factory=STTConfig,description="configuration modele stt")
    tts:TTSConfig=Field(default_factory=TTSConfig,description="configuration modele tts")
    audio:AudioConfig=Field(default_factory=AudioConfig,description="configuration modele audio")
    broker:BrokerConfig=Field(default_factory=BrokerConfig,description="configuration broker")
    paths: PathConfig=Field(default_factory=PathConfig,description="conf paths chemin fichiers")
    db : DBConfig = Field(default_factory=DBConfig,description="configuration base de données")
    security : SecurityConfig = Field(default_factory=SecurityConfig,description="configuration de la sécurité")

def get_configuration():
    return Configuration()


CONFIGURATION = get_configuration()