from typing import Optional

from speech.speech_to_text import INSTANCE_STT
from speech.text_to_speech import INSTANCE_TTS
from configuration import CONFIGURATION


# from security.client_side.client import 

DEMANDE_INSERTION_PIN = "veuillez insérer votre pin de 4 chiffres lentement"
CODE_PIN_ERRONE = "Code pin erroné"
CARTE_BLOQUEE = "La carte est bloquée après 3 tentatives veuillez vous rapprocher de l'administrateur désormais"
MESSAGE_BIENVENUE = "YOKOSSOU monsieur"

class SessionAuthentifiee:
    def __init__(self):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.user_id: Optional[str] = None
        self.carte_id: Optional[str] = None
        self.expire_at: Optional[float] = None
        self.session_id : Optional[str] = None
        self.prenom : Optional[str] = None

#modele de données 
def pipeline_authentification(ficher_tts_sortie : str = "synthese.wav"):
    carte_lecteur_non_lue = True 
    while carte_lecteur_non_lue :
        carte_lecteur_non_lue = False #insérer fonction ici

    nombre_tentatives = 0
    pin_valide = False
    print("ici")
    while not pin_valide and  nombre_tentatives <= CONFIGURATION.security.max_tentatives_avant_blocage_carte_gemalto :
        print("nombre tentatives restantes ", nombre_tentatives)
        tts_texte = INSTANCE_TTS.pipeline(texte=DEMANDE_INSERTION_PIN ,nom_fichier_sortie=ficher_tts_sortie)
        code_pin_potentiel = INSTANCE_STT.pipeline_authentification_stt(nombre_tentatives)
        #ajouter méthode vérification code pin (validité et vérification du pin dans la carte)
        pin_valide = True # admettons mais en réalité on enverra probablement le secret dans la fonction également et le card_id ???
        if not pin_valide : 
            nombre_tentatives += 1
            _ = INSTANCE_TTS.pipeline(texte=CODE_PIN_ERRONE , nom_fichier_sortie=ficher_tts_sortie)
    
    if nombre_tentatives >= CONFIGURATION.security.max_tentatives_avant_blocage_carte_gemalto : 
        INSTANCE_TTS.pipeline(texte=CODE_PIN_ERRONE , nom_fichier_sortie=ficher_tts_sortie)
    
    #vérificaiton validité de la carte // envoi dans broker etc ...
    is_carte_valide = True
    prenom_utilisateur = "LAZIB"
    access_token , refresh_token = "a" , "b" # à voir où ils seront stocké côté embarqué
    if is_carte_valide : 
        #initialisation du worker vision avec os.subprocess ... dans un autre terminal , et script python INSTANCE_STT.ecouter_en_continue_avec_mot_cle_activation et le TTS_service en écoute que du broker ...
        #creation objet session rempli
        INSTANCE_TTS.pipeline(f"{MESSAGE_BIENVENUE} {prenom_utilisateur} Comment puis je vous aider aujourd'hui ?" , ficher_tts_sortie)


#ajouter le required_auth comme décorateur dans la méthode du llm où on doit soumettre nos trucs

        
        
pipeline_authentification()