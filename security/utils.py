import re
import unicodedata
from typing import Optional

CHIFFRES_FR = {
    "zero": "0",
    "zéro": "0",
    "un": "1",
    "une": "1",
    "deux": "2",
    "trois": "3",
    "quatre": "4",
    "cinq": "5",
    "six": "6",
    "sept": "7",
    "huit": "8",
    "neuf": "9",
}

# pas obligatoire on peut gérer l'accent dans le dictionnaire
# def normaliser_texte(texte: str) -> str:
#     texte = texte.lower().strip()
#     texte = unicodedata.normalize("NFD", texte)
#     texte = "".join(
#         caractere for caractere in texte
#         if unicodedata.category(caractere) != "Mn"
#     )
#     return texte #suppression des accents bizarres dans unicode

def extraire_pin_4_chiffres(texte: str) -> Optional[str]:

    """ TODO : à voir si le STT retourne des mélanges dans le stt dans ce cas ajouter une méthode plus robuste"""
    # texte = normaliser_texte(texte)

    # Cas 1 : "1234" ou "mon pin est 1234"
    match = re.search(r"\b\d{4}\b", texte)
    if match:
        print(" par matching on a eu : ",match.group())
        return match.group()

    # Cas 2 : "1 2 3 4" ou "mon code est 1 2 3 4"
    chiffres_separes = re.findall(r"\b\d\b", texte)
    if len(chiffres_separes) == 4:
        print(f' les chiffres sont séparés : {"".join(chiffres_separes)}')
        return "".join(chiffres_separes)

    # Cas 3 : "un deux trois quatre" ou "mon code est un deux trois quatre"
    mots = re.findall(r"\b\w+\b", texte)

    chiffres_trouves = []

    for mot in mots:
        if mot in CHIFFRES_FR:
            chiffres_trouves.append(CHIFFRES_FR[mot])

    print("chiffre retrouvés par voix texte => :" , chiffres_trouves)

    if len(chiffres_trouves) == 4:
        return "".join(chiffres_trouves)
    else :
        return None