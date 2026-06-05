import hashlib
import hmac

from smartcard.System import readers
from smartcard.util import toHexString


# =========================
# Constantes carte
# =========================

P2_VERIFY_CSC0 = 0x07
P2_VERIFY_CSC1 = 0x39
P2_EMULATE_USER_MODE = 0x3A

PUBLIC_CARD_ID_START = 0x01
PUBLIC_CARD_ID_WORDS = 2

SECRET_CARTE_START = 0x10
SECRET_CARTE_WORDS = 8
SECRET_CARTE_TAILLE = SECRET_CARTE_WORDS * 4


# =========================
# Connexion et APDU simples
# =========================

def connect_card():
    available_readers = readers()
    if not available_readers:
        raise Exception("Aucun lecteur detecte")
    reader = available_readers[0]
    conn = reader.createConnection()
    conn.connect()
    print(f"Lecteur : {reader}")
    print(f"ATR : {toHexString(conn.getATR())}")
    return conn


def is_sw_ok(sw1, sw2):
    return (sw1, sw2) == (0x90, 0x00)


def read_word_bytes(conn, addr):
    apdu = [0x80, 0xBE, 0x00, addr, 0x04]
    data, sw1, sw2 = conn.transmit(apdu)
    if not is_sw_ok(sw1, sw2):
        return None, sw1, sw2
    return data, sw1, sw2


# =========================
# CSC et mode utilisateur
# =========================

def verify_csc0(conn, csc0_bytes):
    apdu = [0x00, 0x20, 0x00, P2_VERIFY_CSC0, 0x04] + list(csc0_bytes)
    _, sw1, sw2 = conn.transmit(apdu)
    return sw1, sw2


def verify_csc1(conn, csc1_bytes):
    if len(csc1_bytes) != 4:
        raise ValueError("CSC1 doit faire 4 octets")
    apdu = [0x00, 0x20, 0x00, P2_VERIFY_CSC1, 0x04] + list(csc1_bytes)
    _, sw1, sw2 = conn.transmit(apdu)
    return sw1, sw2


def emuler_mode_utilisateur(conn):
    # Le contenu data est obligatoire mais non verifie par la carte.
    apdu = [0x00, 0x20, 0x00, P2_EMULATE_USER_MODE, 0x04, 0x00, 0x00, 0x00, 0x00]
    _, sw1, sw2 = conn.transmit(apdu)
    return sw1, sw2


# =========================
# Secret carte et HMAC
# =========================

def lire_secret_carte(conn):
    raw = bytearray()

    for offset in range(SECRET_CARTE_WORDS):
        addr = SECRET_CARTE_START + offset
        data, sw1, sw2 = read_word_bytes(conn, addr)

        if not is_sw_ok(sw1, sw2) or data is None:
            return None, f"Erreur lecture secret addr=0x{addr:02X} SW={sw1:02X}{sw2:02X}"

        raw.extend(data)

    secret = bytes(raw)
    if len(secret) != SECRET_CARTE_TAILLE:
        return None, "secret carte invalide"

    return secret, "ok"


def calculer_signature(secret, card_id, challenge_id, challenge):
    message = f"{card_id}:{challenge_id}:{challenge}".encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def verifier_mode_emule_par_lecture_secret(conn):
    # En user mode avec 0x05=0x30, le secret ne doit pas etre lisible sans CSC1.
    secret, message = lire_secret_carte(conn)

    if secret is None:
        print(f"[+] Mode emule confirme: secret refuse sans CSC1 ({message})")
        return True

    print("[-] Mode emule non confirme: secret lisible sans CSC1")
    return False
