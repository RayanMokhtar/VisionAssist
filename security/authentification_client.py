import os
from typing import Optional
import uuid
from pathlib import Path
import subprocess

from security.client_side.client import CSC0_SIMULATION, demander_challenge, pin_to_csc1_bytes, read_public_card_id, validate_csc1_pin, verifier_carte_hmac
from security.client_side.client import CSC0_SIMULATION
from speech.speech_to_text import INSTANCE_STT
from speech.text_to_speech import INSTANCE_TTS , lancement_service_tts
from configuration import CONFIGURATION
from persistance.repository import REPOSITORIES

from multiprocessing import Process
from threading import Thread
from pydantic import BaseModel



from broker.service import get_broker_client  
from configuration import CONFIGURATION  
from security.client_side.common.utils_client import (calculer_signature, connect_card, is_sw_ok, lire_secret_carte, read_word_bytes, verify_csc1, emuler_mode_utilisateur,
verify_csc0,verifier_mode_emule_par_lecture_secret,)  


# from security.client_side.client import 

DEMANDE_INSERTION_PIN = "BONJOUR , Veuillez insérer votre pinne de 4 chiffres lentement"
CODE_PIN_ERRONE = "Code pinne erroné"
CARTE_BLOQUEE = "La carte est bloquée après 3 tentatives veuillez vous rapprocher de l'administrateur désormais"
MESSAGE_ATTENTE = "Merci de patienter pendant que je vérifie votre carte et prépare tout pour vous"
MESSAGE_BIENVENUE = "BONJOUR BONJOUR"

class SessionAuthentifiee(BaseModel):
    user_id: Optional[int] = None, 
    card_id: Optional[str] = None, 
    access_token: Optional[str] = None, 
    refresh_token: Optional[str] = None, 
    prenom: Optional[str] = None, 
    session_id: Optional[str] = None,
    expire_at: Optional[float] = None


def mode_emule_pour_user(conn):
    sw1, sw2 = verify_csc0(conn, CSC0_SIMULATION)
    sw1, sw2 = emuler_mode_utilisateur(conn)
    if not verifier_mode_emule_par_lecture_secret(conn):
        return False
    return True


def lecture_card_id(conn):
    card_id, message = read_public_card_id(conn)
    return card_id, message

def verifier_format_pin_dans_csc1(conn , pin):
    sw1, sw2 = verify_csc1(conn, pin_to_csc1_bytes(pin)) # après saisie pour le format de la carte
    if not is_sw_ok(sw1, sw2):
        print(f"Acces refuse: VERIFY CSC1 SW={sw1:02X}{sw2:02X}")
        return False
    else : 
        return True
    


def logique_creation_session_agent_ia(reponse_authentification_apres_hmac : dict) -> SessionAuthentifiee :
    session_id = uuid.uuid4()
    session_repo = REPOSITORIES.sessions.create(session_id=session_id, user_id=reponse_authentification_apres_hmac.get("user_id"), card_id=reponse_authentification_apres_hmac.get("card_id"))
    print("sessions créee en base de données : ", session_repo)

    return session_id

    


def lancement_scripts_terminaux():
    # 1. On trouve automatiquement la racine du projet (le dossier VisionAssist)
    # Assure-toi que ce fichier est bien dans un sous-dossier, sinon ajuste les .parent
    print("lancement scripts")
    dossier_racine = Path(__file__).resolve().parent.parent 
    
    # 2. On construit le chemin vers Python de manière fiable
    if CONFIGURATION.environnement == "linux":
        chemin_python = dossier_racine / ".venv" / "bin" / "python"
    else:
        chemin_python = dossier_racine / ".venv" / "Scripts" / "python.exe"

    chemin_python_absolu = str(chemin_python)
    dossier_racine_absolu = str(dossier_racine)

    if CONFIGURATION.environnement == "linux": 
        # J'ai retiré le 2>/dev/null temporairement pour que tu puisses voir les erreurs 
        # dans les nouvelles fenêtres qui vont s'ouvrir. Tu pourras les remettre plus tard !
        scripts_linux = [
            # f'"{chemin_python_absolu}" -m speech.text_to_speech', 
            # f'"{chemin_python_absolu}" -m speech.speech_to_text',
            # f'"{dossier_racine_absolu}/worker_vision/build/main"' 
        ]
    
        for cmd_str in scripts_linux:
            try:
                commande_complete = ["gnome-terminal", "--", "bash", "-c", f"{cmd_str}; exec bash"]
                subprocess.Popen(commande_complete, preexec_fn=os.setpgrp, cwd=dossier_racine_absolu)
                print(f"Terminal Linux lancé pour : {cmd_str}")
            except Exception as e:
                print(f"Erreur lors du lancement Linux de {cmd_str} : {e}")
                
    else: # Cas Windows
        scripts_windows = [
            f'"{chemin_python_absolu}" -m speech.text_to_speech',
            f'"{chemin_python_absolu}" -m speech.speech_to_text'
        ]
        
        for cmd in scripts_windows:
            try:
                commande_complete = f"start cmd /K {cmd}"
                subprocess.Popen(commande_complete, shell=True, cwd=dossier_racine_absolu)
                print(f"Terminal Windows lancé pour : {cmd}")
            except Exception as e:
                print(f"Erreur lors du lancement Windows de {cmd} : {e}")



def lancement_worker_vision():
    print("Lancement du worker_vision...")
    
    # 1. On trouve les dossiers
    dossier_racine = Path(__file__).resolve().parent.parent 
    dossier_build_absolu = str(dossier_racine / "worker_vision" / "build")
    
    chemin_activate = str(dossier_racine / ".venv" / "bin" / "activate")

    if CONFIGURATION.environnement == "linux": 
        cmd_vision = f"source {chemin_activate} && deactivate && ./main"
        
        try:
            commande_complete = ["gnome-terminal", "--", "bash", "-c", f"{cmd_vision}; exec bash"]
            subprocess.Popen(commande_complete, preexec_fn=os.setpgrp, cwd=dossier_build_absolu)
            print(f"Terminal Linux lancé dans {dossier_build_absolu}")
        except Exception as e:
            print(f"Erreur : {e}")


def pipeline_authentification(ficher_tts_sortie : str = "synthese.wav") -> Optional[SessionAuthentifiee]:
    card_id = None
    conn = None
    carte_lecteur_non_lue = True 
    if not CONFIGURATION.mode_degrade : #pas besoin de la carte si dégradé
        print("conf mode" , CONFIGURATION.mode_degrade)
        while carte_lecteur_non_lue :
            is_ok , conn = connect_card()
            if is_ok : 
                mode_emule_actif = mode_emule_pour_user(conn)
                card_id , message = lecture_card_id(conn)
                if card_id is not None  and mode_emule_actif : 
                    carte_lecteur_non_lue = False
                else :
                    print(f"Carte lecteur non lue ou mode emule non actif: {message}")
                    INSTANCE_TTS.pipeline(texte="Carte lecteur non lue ou mode emule non actif, veuillez réessayer" , nom_fichier_sortie=ficher_tts_sortie)

        nombre_tentatives = 0
        pin_valide = False
        print("ici")
        card_id , message = lecture_card_id(conn) # duplicat à voir si on enlève ou pas ?
        while not pin_valide and  nombre_tentatives <= CONFIGURATION.security.max_tentatives_avant_blocage_carte_gemalto :
            print("nombre tentatives restantes ", nombre_tentatives)
            tts_texte = INSTANCE_TTS.pipeline(texte=DEMANDE_INSERTION_PIN ,nom_fichier_sortie=ficher_tts_sortie)
            code_pin_potentiel = INSTANCE_STT.pipeline_authentification_stt(nombre_tentatives)
            try : 
                valid, message = validate_csc1_pin(code_pin_potentiel)
            except Exception as e:
                print(f"Erreur lors de la validation du pin: {e}")
                nombre_tentatives += 1
                _ = INSTANCE_TTS.pipeline(texte=CODE_PIN_ERRONE , nom_fichier_sortie=ficher_tts_sortie)
                continue

            pin_valide = verifier_format_pin_dans_csc1(conn , code_pin_potentiel)
            if not pin_valide : 
                nombre_tentatives += 1
                _ = INSTANCE_TTS.pipeline(texte=CODE_PIN_ERRONE , nom_fichier_sortie=ficher_tts_sortie)
        
        if nombre_tentatives >= CONFIGURATION.security.max_tentatives_avant_blocage_carte_gemalto : 
            INSTANCE_TTS.pipeline(texte=CODE_PIN_ERRONE , nom_fichier_sortie=ficher_tts_sortie)
    
    INSTANCE_TTS.pipeline(texte=MESSAGE_ATTENTE , nom_fichier_sortie=ficher_tts_sortie)
    #vérificaiton validité de la carte + secret + challenge crytpo côté serveur 
    is_carte_valide = True
    try:
        challenge_id, challenge = demander_challenge(card_id)
    except Exception as exc:
        is_carte_valide = False
        print(f"Erreur challenge serveur: {exc}")
        return False

    secret, message = lire_secret_carte(conn)
    if secret is None:
        is_carte_valide = False
        print(f"Acces refuse: {message}")
        return False

    signature = calculer_signature(secret, card_id, challenge_id, challenge)

    try:
        response_authentification_apres_hmac = verifier_carte_hmac(card_id, challenge_id, challenge, signature)
    except Exception as exc:
        is_carte_valide = False
        print(f"Erreur serveur auth via MQTT: {exc}")
        return False

    if not response_authentification_apres_hmac.get("success"):
        print(f"Acces refuse serveur: {response_authentification_apres_hmac.get('error', 'erreur inconnue')}")
        is_carte_valide = False
        return False
    
    print("Acces autorise")
    print(f"user_id: {response_authentification_apres_hmac.get('user_id')}")
    print(f"status : {response_authentification_apres_hmac.get('status')}")

    prenom_utilisateur = response_authentification_apres_hmac.get("prenom")
    access_token , refresh_token = response_authentification_apres_hmac.get("access_token") , response_authentification_apres_hmac.get("refresh_token")
    session_id = response_authentification_apres_hmac.get("session_id")
    
    print("is carte valide :", is_carte_valide)
    if is_carte_valide : 
        print("session_id = ",session_id)
        session_authentifiee = SessionAuthentifiee(
            user_id=response_authentification_apres_hmac.get("user_id"),
            card_id=response_authentification_apres_hmac.get("card_id"),
            access_token=access_token,
            refresh_token=refresh_token,
            prenom=prenom_utilisateur,
            session_id=str(session_id),
            expire_at=176000000# par défaut
        )
        INSTANCE_TTS.pipeline(f"{MESSAGE_BIENVENUE} {prenom_utilisateur} moi c'est {CONFIGURATION.nom_assistant}, je serai votre assistant, commencez à parler une fois le bip" , ficher_tts_sortie)
        print(f"Session authentifiée créée avec ID: {session_id}")
        #création de la session 
        # session = REPOSITORIES.sessions.create(user_id=response_authentification_apres_hmac.get("user_id"), card_id=response_authentification_apres_hmac.get("card_id"), session_id=session_id)
        # lancement_scripts_terminaux()
        return session_authentifiee # à ajouter l'expiration du token et la logique de refresh token dans le pipeline d'authentification ou dans un décorateur à part pour les méthodes qui nécessitent une authentification
    else : 
        print("carte pas valide vérifier une des étapes du pipeline")
        print(f"Session authentifiée créée avec ID: {session_id}")
        INSTANCE_TTS.pipeline("Accès refusé, carte invalide" , ficher_tts_sortie)
        return None

#ajouter le required_auth comme décorateur dans la méthode du llm où on doit soumettre nos trucs
   
# SESSION_UTILISATEUR = SessionAuthentifiee(
#     user_id="user_123",
#     card_id="card_456",
#     access_token="access_token_mock_abc123",
#     refresh_token="refresh_token_mock_def456",
#     prenom="Jean",
#     session_id="c30c6528-400c-408e-8c2a-dd1e4d701a36",
#     expire_at=1760000000.0  # timestamp futur simulé
# )
    
SESSION_UTILISATEUR = None

if __name__ == "__main__":
    print("session_authentification avant : ", SESSION_UTILISATEUR)
    SESSION_UTILISATEUR = pipeline_authentification()
    if SESSION_UTILISATEUR is not None and SESSION_UTILISATEUR :
        print("session_authentification après : ", SESSION_UTILISATEUR) 
        lancement_worker_vision()
        process_stt = Thread(target=INSTANCE_STT.ecouter_en_continu_avec_mot_activation, args=(SESSION_UTILISATEUR,), name="STT_WORKER")
        process_tts = Thread(target=lancement_service_tts,name="TTS_WORKER")
        process_stt.start() ; process_tts.start()
    



    
    # lancement_scripts_terminaux()