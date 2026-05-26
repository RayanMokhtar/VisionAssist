"""
Diagnostic simplifié GemClub-Memo
Affiche toutes les zones de la carte en hexa + compteurs CSC
Basé sur la documentation officielle GemClub-Memo (DPD10370A00)
"""

import sys
import os
import json
import time
import uuid
from getpass import getpass

# Ajouter le répertoire racine au path
CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CLIENT_DIR)
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CLIENT_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("ENV_FILE", os.path.join(PROJECT_DIR, ".env.admin"))
os.chdir(CLIENT_DIR)

from smartcard.util import toHexString
from smartcard.Exceptions import CardConnectionException
from broker.service import get_broker_client
from configuration import CONFIGURATION
from common.reader import connect_card, read_word_bytes, update_zone, is_sw_ok, verify_csc0
from common.card_config import WORD_ZERO, ADDR_CSC1_WORD

# Adresses selon documentation GemClub-Memo
ADDR_MANUFACTURER = 0x00            # Manufacturer Area
ADDR_ISSUER_START = 0x01            # Issuer Area (0x01-0x04)
ADDR_ISSUER_END = 0x04
ADDR_ACCESS_CONDITIONS = 0x05       # Access Conditions Area
ADDR_CSC0 = 0x06                    # Card Secret Code 0
ADDR_CSC0_COUNTER = 0x07            # CSC 0 Ratification Counter
ADDR_CTC1_START = 0x08              # CTC 1 (0x08-0x0A)
ADDR_BALANCE1_START = 0x0B          # Balance 1 (0x0B-0x0F)
ADDR_USER_AREA1_START = 0x10        # User Area 1 (0x10-0x1F)
ADDR_USER_AREA1_END = 0x1F
ADDR_CTC2_START = 0x20              # CTC 2 (0x20-0x22)
ADDR_BALANCE2_START = 0x23          # Balance 2 (0x23-0x27)
ADDR_USER_AREA2_START = 0x28        # User Area 2 (0x28-0x37)
ADDR_USER_AREA2_END = 0x37
ADDR_CSC1 = 0x38                    # Card Secret Code 1
ADDR_CSC1_COUNTER = 0x39            # CSC 1 Ratification Counter
ADDR_CSC2 = 0x3A                    # Card Secret Code 2
ADDR_CSC2_COUNTER = 0x3B            # CSC 2 Ratification Counter
ADDR_PROTECTED_START = 0x3C         # Protected Area (0x3C-0x3F)
ADDR_PROTECTED_END = 0x3F
TOPICS = CONFIGURATION.broker.topics
AUTH_ADMIN_REQUEST_SECURITY = TOPICS.security_admin_request_topic
AUTH_RESPONSE_SECURITY = TOPICS.security_response_topic
PUBLIC_CARD_ID_START = 0x01
PUBLIC_CARD_ID_WORDS = 2
FORBIDDEN_CSC_VALUES = {
    bytes.fromhex("00 00 00 00"),
    bytes.fromhex("80 00 00 00"),
    bytes.fromhex("7F FF FF FF"),
    bytes.fromhex("FF FF FF FF"),
}


def is_card_reset_error(exc):
    text = str(exc).lower()
    return (
        isinstance(exc, CardConnectionException)
        or "card was reset" in text
        or "0x80100068" in text
    )


def reconnect_and_verify_csc0(csc0):
    print("[*] Carte reset par le lecteur: reconnexion...")
    conn = connect_card()
    print("[*] Nouvelle verification CSC0...")
    sw1, sw2 = verify_csc0(conn, csc0)
    if not is_sw_ok(sw1, sw2):
        raise RuntimeError(f"VERIFY CSC0 apres reconnexion echoue (SW={sw1:02X}{sw2:02X})")
    print("[+] CSC0 verifie apres reconnexion")
    return conn


def verify_csc0_resilient(conn, csc0):
    try:
        sw1, sw2 = verify_csc0(conn, csc0)
        return conn, sw1, sw2
    except Exception as exc:
        if not is_card_reset_error(exc):
            raise
        conn = reconnect_and_verify_csc0(csc0)
        return conn, 0x90, 0x00


def update_zone_resilient(conn, csc0, addr, data):
    for attempt in range(2):
        try:
            sw1, sw2 = update_zone(conn, addr, data)
            return conn, sw1, sw2
        except Exception as exc:
            if not is_card_reset_error(exc) or attempt == 1:
                raise
            conn = reconnect_and_verify_csc0(csc0)
    return conn, 0x6F, 0x00


def read_word_bytes_resilient(conn, addr):
    for attempt in range(2):
        try:
            data, sw1, sw2 = read_word_bytes(conn, addr)
            return conn, data, sw1, sw2
        except Exception as exc:
            if not is_card_reset_error(exc) or attempt == 1:
                raise
            print("[*] Carte reset par le lecteur pendant une lecture publique: reconnexion...")
            conn = connect_card()
    return conn, None, 0x6F, 0x00


def pin_to_csc1_bytes(pin):
    return int(pin).to_bytes(4, byteorder="little")


def validate_csc1_pin(pin):
    if not (pin.isdigit() and len(pin) == 4):
        return False, "PIN invalide: il faut 4 chiffres"
    if pin_to_csc1_bytes(pin) in FORBIDDEN_CSC_VALUES:
        return False, "PIN interdit par la datasheet"
    return True, "ok"


def ask_user_pin():
    pin = getpass("PIN utilisateur CSC1 (4 chiffres): ").strip()
    confirm = getpass("Confirmer PIN CSC1: ").strip()
    if pin != confirm:
        return None, "Les PIN ne correspondent pas"
    valid, message = validate_csc1_pin(pin)
    if not valid:
        return None, message
    return pin, "ok"


def read_public_card_id(conn):
    raw = bytearray()
    for offset in range(PUBLIC_CARD_ID_WORDS):
        addr = PUBLIC_CARD_ID_START + offset
        conn, data, sw1, sw2 = read_word_bytes_resilient(conn, addr)
        if not is_sw_ok(sw1, sw2) or data is None:
            return None, f"Erreur lecture card_id addr=0x{addr:02X} SW={sw1:02X}{sw2:02X}"
        raw.extend(data)

    raw_bytes = bytes(raw)
    if raw_bytes in (b"\x00" * len(raw_bytes), b"\xFF" * len(raw_bytes)):
        return "", "ok"
    card_id = raw_bytes.rstrip(b"\x00").rstrip(b"\xFF").decode("ascii", errors="ignore").strip()
    return card_id, "ok"


def encode_card_id_words(card_id):
    raw = card_id.encode("ascii")
    max_len = PUBLIC_CARD_ID_WORDS * 4
    if len(raw) > max_len:
        raise ValueError(f"card_id trop long: max {max_len} caracteres ASCII")
    raw = raw.ljust(max_len, b"\x00")
    return [raw[i:i + 4] for i in range(0, max_len, 4)]


def write_public_card_id(conn, csc0, card_id):
    for offset, word in enumerate(encode_card_id_words(card_id)):
        addr = PUBLIC_CARD_ID_START + offset
        conn, sw1, sw2 = update_zone_resilient(conn, csc0, addr, word)
        if not is_sw_ok(sw1, sw2):
            return conn, False, f"Erreur ecriture card_id addr=0x{addr:02X} SW={sw1:02X}{sw2:02X}"
    return conn, True, "ok"


def write_csc1_pin(conn, csc0, pin):
    conn, sw1, sw2 = update_zone_resilient(conn, csc0, ADDR_CSC1_WORD, pin_to_csc1_bytes(pin))
    if not is_sw_ok(sw1, sw2):
        return conn, False, f"Erreur ecriture CSC1 SW={sw1:02X}{sw2:02X}"
    return conn, True, "ok"


def mqtt_request(action, payload, timeout=10):
    client_id = f"auth-admin-{uuid.uuid4().hex[:8]}"
    request_id = uuid.uuid4().hex
    response_holder = {}

    broker = get_broker_client(client_id)
    broker.connexion()
    time.sleep(0.5)

    def on_response(_topic, message):
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except Exception:
                return
        if message.get("request_id") == request_id and message.get("client_id") == client_id:
            response_holder["response"] = message

    broker.sabonner(AUTH_RESPONSE_SECURITY, on_response)
    request = dict(payload)
    request["action"] = action
    request["request_id"] = request_id
    request["client_id"] = client_id
    if not broker.publier(AUTH_ADMIN_REQUEST_SECURITY, request):
        broker.deconnexion()
        raise RuntimeError("publication MQTT impossible")

    deadline = time.time() + timeout
    while time.time() < deadline:
        if "response" in response_holder:
            broker.deconnexion()
            return response_holder["response"]
        time.sleep(0.1)

    broker.deconnexion()
    raise TimeoutError("timeout reponse MQTT auth")


def decode_counter(counter_word):
    """
    Décode un compteur CSC selon la doc GemClub-Memo.
    Utilise uniquement les 4 bits MSB (bits 31-28) du word.
    
    0x0 (0000b) = 3 tentatives restantes
    0x8 (1000b) = 2 tentatives restantes
    0xC (1100b) = 1 tentative restante
    0xE (1110b) = 0 tentative
    0xF (1111b) = BLOQUÉ DÉFINITIVEMENT
    
    Returns: (tentatives_restantes, status_message)
    """
    msb_4bits = (counter_word >> 28) & 0x0F
    
    if msb_4bits == 0x0:  # 0000b
        return 3, "✓ 3 tentatives restantes"
    elif msb_4bits == 0x8:  # 1000b
        return 2, "⚠ 2 tentatives restantes"
    elif msb_4bits == 0xC:  # 1100b
        return 1, "⚠ 1 tentative restante"
    elif msb_4bits == 0xE:  # 1110b
        return 0, "🔴 0 tentative - BLOQUÉ au prochain échec!"
    elif msb_4bits == 0xF:  # 1111b
        return -1, "🔒 BLOQUÉ DÉFINITIVEMENT"
    else:
        return -2, f"? Valeur inattendue (bits 31-28 = 0x{msb_4bits:X})"

def read_and_display_word(conn, addr, label):
    """Lit et affiche un word en hexa"""
    data, sw1, sw2 = read_word_bytes(conn, addr)
    if is_sw_ok(sw1, sw2) and data is not None:
        hex_str = toHexString(data)
        print(f"   0x{addr:02X}: {hex_str}")
        return data
    else:
        print(f"   0x{addr:02X}: ERROR (SW={sw1:02X}{sw2:02X})")
        return None

def read_range(conn, start, end, title):
    """Lit et affiche une plage d'adresses"""
    print(f"\n{title}")
    print("-" * 70)
    for addr in range(start, end + 1):
        read_and_display_word(conn, addr, "")

def afficher_diagnostic(conn, csc0):
    """Affiche le diagnostic complet: toutes les zones en hexa + compteurs CSC"""
    print("\n" + "="*70)
    print("DIAGNOSTIC GEMCLUB-MEMO - DUMP COMPLET")
    print("="*70)
    
    # Vérifier CSC0
    print("\n[*] Vérification CSC0...")
    conn, sw1, sw2 = verify_csc0_resilient(conn, csc0)
    if not is_sw_ok(sw1, sw2):
        print(f"[-] VERIFY CSC0 échoué (SW={sw1:02X}{sw2:02X})")
        if sw1 == 0x63 and sw2 == 0x00:
            print("    ⚠️  Code incorrect")
        elif sw1 == 0x69 and sw2 == 0x82:
            print("    🔒 CSC0 BLOQUÉ")
        return
    
    print("[+] CSC0 vérifié - Accès complet OK\n")
    
    # MANUFACTURER AREA
    print("\n📌 MANUFACTURER AREA (0x00)")
    print("-" * 70)
    read_and_display_word(conn, ADDR_MANUFACTURER, "Manufacturer")
    
    # ISSUER AREA
    print("\n📌 ISSUER AREA (0x01-0x04)")
    print("-" * 70)
    for addr in range(ADDR_ISSUER_START, ADDR_ISSUER_END + 1):
        read_and_display_word(conn, addr, "")
    
    # ACCESS CONDITIONS
    print("\n📌 ACCESS CONDITIONS AREA (0x05)")
    print("-" * 70)
    data = read_and_display_word(conn, ADDR_ACCESS_CONDITIONS, "")
    if data:
        acc_word = int.from_bytes(data, byteorder='little')
        acc_byte = (acc_word >> 24) & 0xFF
        print(f"   → Byte conditions (bits 31-24): 0x{acc_byte:02X} = {acc_byte:08b}b")
    
    # CSC0 & COMPTEUR
    print("\n📌 CSC0 & COMPTEUR (0x06-0x07)")
    print("-" * 70)
    read_and_display_word(conn, ADDR_CSC0, "")
    
    data, sw1, sw2 = read_word_bytes(conn, ADDR_CSC0_COUNTER)
    if is_sw_ok(sw1, sw2) and data is not None:
        word = int.from_bytes(data, byteorder='little')
        tentatives, message = decode_counter(word)
        hex_str = toHexString(data)
        print(f"   0x{ADDR_CSC0_COUNTER:02X}: {hex_str}  → {message}")
    else:
        print(f"   0x{ADDR_CSC0_COUNTER:02X}: ERROR (SW={sw1:02X}{sw2:02X})")
    
    # CTC 1 & BALANCE 1
    print("\n📌 CTC 1 & BALANCE 1 (0x08-0x0F)")
    print("-" * 70)
    for addr in range(0x08, 0x10):
        read_and_display_word(conn, addr, "")
    
    # USER AREA 1
    print("\n📌 USER AREA 1 (0x10-0x1F)")
    print("-" * 70)
    for addr in range(ADDR_USER_AREA1_START, ADDR_USER_AREA1_END + 1):
        read_and_display_word(conn, addr, "")
    
    # CTC 2 & BALANCE 2
    print("\n📌 CTC 2 & BALANCE 2 (0x20-0x27)")
    print("-" * 70)
    for addr in range(0x20, 0x28):
        read_and_display_word(conn, addr, "")
    
    # USER AREA 2
    print("\n📌 USER AREA 2 (0x28-0x37)")
    print("-" * 70)
    for addr in range(ADDR_USER_AREA2_START, ADDR_USER_AREA2_END + 1):
        read_and_display_word(conn, addr, "")
    
    # CSC1 & COMPTEUR
    print("\n📌 CSC1 & COMPTEUR (0x38-0x39)")
    print("-" * 70)
    read_and_display_word(conn, ADDR_CSC1, "")
    
    data, sw1, sw2 = read_word_bytes(conn, ADDR_CSC1_COUNTER)
    if is_sw_ok(sw1, sw2) and data is not None:
        word = int.from_bytes(data, byteorder='little')
        tentatives, message = decode_counter(word)
        hex_str = toHexString(data)
        print(f"   0x{ADDR_CSC1_COUNTER:02X}: {hex_str}  → {message}")
    else:
        print(f"   0x{ADDR_CSC1_COUNTER:02X}: ERROR (SW={sw1:02X}{sw2:02X})")
    
    # CSC2 & COMPTEUR
    print("\n📌 CSC2 & COMPTEUR (0x3A-0x3B)")
    print("-" * 70)
    read_and_display_word(conn, ADDR_CSC2, "")
    
    data, sw1, sw2 = read_word_bytes(conn, ADDR_CSC2_COUNTER)
    if is_sw_ok(sw1, sw2) and data is not None:
        word = int.from_bytes(data, byteorder='little')
        tentatives, message = decode_counter(word)
        hex_str = toHexString(data)
        print(f"   0x{ADDR_CSC2_COUNTER:02X}: {hex_str}  → {message}")
    else:
        print(f"   0x{ADDR_CSC2_COUNTER:02X}: ERROR (SW={sw1:02X}{sw2:02X})")
    
    # PROTECTED AREA
    print("\n📌 PROTECTED AREA (0x3C-0x3F)")
    print("-" * 70)
    for addr in range(ADDR_PROTECTED_START, ADDR_PROTECTED_END + 1):
        read_and_display_word(conn, addr, "")
    
    # LÉGENDE
    print("\n" + "="*70)
    print("📖 LÉGENDE")
    print("="*70)
    print("ACCESS CONDITIONS (0x05 - byte bits 31-24):")
    print("  Rb1 Ub1 Ru1 Uu1 Rb2 Ub2 Ru2 Uu2")
    print("  Rb1: Read CTC/Balance 1 (0=libre, 1=CSC1)")
    print("  Ub1: Update Balance 1 (0=CSC1, 1=interdit)")
    print("  Ru1: Read User Area 1 (0=libre, 1=CSC1)")
    print("  Uu1: Update User Area 1 (0=CSC1, 1=interdit)")
    print("  Rb2, Ub2, Ru2, Uu2: idem pour Area 2 avec CSC2")
    print()
    print("COMPTEURS CSC (bits 31-28 seulement):")
    print("  0x00000000 = 3 tentatives | 0x80000000 = 2 tentatives")
    print("  0xC0000000 = 1 tentative  | 0xE0000000 = 0 tentative")
    print("  0xF0000000 = BLOQUÉ définitivement")
    print("="*70)

def reset_complet(conn, csc0):
    """Reset complet: User Areas 1&2, Access Conditions, Protected Areas"""
    print("\n" + "="*70)
    print("RESET COMPLET DE LA CARTE")
    print("="*70)
    print("\n⚠️  Cette opération va remettre à 0x00000000:")
    print("  • User Area 1 (0x10-0x1F)")
    print("  • User Area 2 (0x28-0x37)")
    print("  • Access Conditions (0x05)")
    print("  • Protected Areas (bits 23-0 de 0x05 + 0x3C-0x3F)")
    
    confirm = input("\nConfirmer le reset complet ? (oui/non) : ").strip().lower()
    if confirm not in ['oui', 'o', 'y', 'yes']:
        print("[*] Annulé")
        return
    
    # Vérifier CSC0
    print("\n[*] Vérification CSC0...")
    sw1, sw2 = verify_csc0(conn, csc0)
    if not is_sw_ok(sw1, sw2):
        print(f"[-] VERIFY CSC0 échoué (SW={sw1:02X}{sw2:02X})")
        return
    
    print("[+] CSC0 vérifié\n")
    
    errors = 0
    
    # Reset User Area 1 (0x10-0x1F)
    print("[*] Reset User Area 1 (0x10-0x1F)...")
    for addr in range(ADDR_USER_AREA1_START, ADDR_USER_AREA1_END + 1):
        conn, sw1, sw2 = update_zone_resilient(conn, csc0, addr, WORD_ZERO)
        if is_sw_ok(sw1, sw2):
            print(f"    [+] 0x{addr:02X} → 00 00 00 00")
        else:
            print(f"    [-] 0x{addr:02X} échoué (SW={sw1:02X}{sw2:02X})")
            errors += 1
    
    # Reset User Area 2 (0x28-0x37)
    print("\n[*] Reset User Area 2 (0x28-0x37)...")
    for addr in range(ADDR_USER_AREA2_START, ADDR_USER_AREA2_END + 1):
        conn, sw1, sw2 = update_zone_resilient(conn, csc0, addr, WORD_ZERO)
        if is_sw_ok(sw1, sw2):
            print(f"    [+] 0x{addr:02X} → 00 00 00 00")
        else:
            print(f"    [-] 0x{addr:02X} échoué (SW={sw1:02X}{sw2:02X})")
            errors += 1
    
    # Reset Access Conditions (0x05)
    print("\n[*] Reset Access Conditions (0x05)...")
    conn, sw1, sw2 = update_zone_resilient(conn, csc0, ADDR_ACCESS_CONDITIONS, WORD_ZERO)
    if is_sw_ok(sw1, sw2):
        print(f"    [+] 0x{ADDR_ACCESS_CONDITIONS:02X} → 00 00 00 00 (accès libre)")
    else:
        print(f"    [-] 0x{ADDR_ACCESS_CONDITIONS:02X} échoué (SW={sw1:02X}{sw2:02X})")
        errors += 1
    
    # Reset Protected Areas (0x3C-0x3F)
    print("\n[*] Reset Protected Areas (0x3C-0x3F)...")
    for addr in range(ADDR_PROTECTED_START, ADDR_PROTECTED_END + 1):
        conn, sw1, sw2 = update_zone_resilient(conn, csc0, addr, WORD_ZERO)
        if is_sw_ok(sw1, sw2):
            print(f"    [+] 0x{addr:02X} → 00 00 00 00")
        else:
            print(f"    [-] 0x{addr:02X} échoué (SW={sw1:02X}{sw2:02X})")
            errors += 1
    
    # Résumé
    print("\n" + "="*70)
    if errors == 0:
        print("✅ RESET COMPLET TERMINÉ AVEC SUCCÈS")
    else:
        print(f"⚠️  RESET TERMINÉ AVEC {errors} ERREUR(S)")
    print("="*70)


def reset_complet_resilient(conn, csc0):
    """Reset complet avec reconnexion automatique si PC/SC reset la carte."""
    print("\n" + "="*70)
    print("RESET COMPLET DE LA CARTE")
    print("="*70)
    print("\nCette operation va remettre a 0x00000000:")
    print("  - Issuer Area utilisee par card_id (0x01-0x02)")
    print("  - User Area 1 (0x10-0x1F)")
    print("  - User Area 2 (0x28-0x37)")
    print("  - Access Conditions (0x05)")
    print("  - Protected Areas (0x3C-0x3F)")
    print("  - ATTENTION: 0x04 n'est pas touche pour preserver les bits de mode")

    confirm = input("\nConfirmer le reset complet ? (oui/non) : ").strip().lower()
    if confirm not in ["oui", "o", "y", "yes"]:
        print("[*] Annule")
        return

    print("\n[*] Verification CSC0...")
    conn, sw1, sw2 = verify_csc0_resilient(conn, csc0)
    if not is_sw_ok(sw1, sw2):
        print(f"[-] VERIFY CSC0 echoue (SW={sw1:02X}{sw2:02X})")
        return

    print("[+] CSC0 verifie\n")
    errors = 0

    print("[*] Reset Issuer Area utilisee par card_id (0x01-0x02)...")
    for addr in range(0x01, 0x03):
        conn, sw1, sw2 = update_zone_resilient(conn, csc0, addr, WORD_ZERO)
        if is_sw_ok(sw1, sw2):
            print(f"    [+] 0x{addr:02X} -> 00 00 00 00")
        else:
            print(f"    [-] 0x{addr:02X} echoue (SW={sw1:02X}{sw2:02X})")
            errors += 1
    print()

    reset_ranges = [
        ("User Area 1", ADDR_USER_AREA1_START, ADDR_USER_AREA1_END),
        ("User Area 2", ADDR_USER_AREA2_START, ADDR_USER_AREA2_END),
        ("Protected Areas", ADDR_PROTECTED_START, ADDR_PROTECTED_END),
    ]

    for title, start, end in reset_ranges:
        print(f"[*] Reset {title} (0x{start:02X}-0x{end:02X})...")
        for addr in range(start, end + 1):
            conn, sw1, sw2 = update_zone_resilient(conn, csc0, addr, WORD_ZERO)
            if is_sw_ok(sw1, sw2):
                print(f"    [+] 0x{addr:02X} -> 00 00 00 00")
            else:
                print(f"    [-] 0x{addr:02X} echoue (SW={sw1:02X}{sw2:02X})")
                errors += 1
        print()

    print("[*] Reset Access Conditions (0x05)...")
    conn, sw1, sw2 = update_zone_resilient(conn, csc0, ADDR_ACCESS_CONDITIONS, WORD_ZERO)
    if is_sw_ok(sw1, sw2):
        print(f"    [+] 0x{ADDR_ACCESS_CONDITIONS:02X} -> 00 00 00 00")
    else:
        print(f"    [-] 0x{ADDR_ACCESS_CONDITIONS:02X} echoue (SW={sw1:02X}{sw2:02X})")
        errors += 1

    print("\n" + "="*70)
    if errors == 0:
        print("RESET COMPLET TERMINE AVEC SUCCES")
    else:
        print(f"RESET TERMINE AVEC {errors} ERREUR(S)")
    print("="*70)


def enrollement_card(conn, csc0):
    print("\n" + "="*70)
    print("ENROLLEMENT CARTE")
    print("="*70)

    existing_card_id, message = read_public_card_id(conn)
    if existing_card_id is None:
        print(message)
        return
    if existing_card_id:
        print(f"Carte deja initialisee avec card_id={existing_card_id}. Operation arretee.")
        return

    print("\n[*] Verification CSC0...")
    conn, sw1, sw2 = verify_csc0_resilient(conn, csc0)
    if not is_sw_ok(sw1, sw2):
        print(f"CSC0 refuse SW={sw1:02X}{sw2:02X}")
        return

    try:
        response = mqtt_request("prepare-card", {})
    except Exception as exc:
        print(f"Erreur serveur auth via MQTT pendant preparation: {exc}")
        return

    if not response.get("success"):
        print(f"Preparation serveur refusee: {response.get('error')}")
        return

    card_id = response["card_id"]
    user_id = response["user"]["user_id"]

    pin, message = ask_user_pin()
    if pin is None:
        print(message)
        return

    print(f"Serveur: card_id={card_id}, user_id={user_id}")

    conn, ok, message = write_csc1_pin(conn, csc0, pin)
    if not ok:
        print(message)
        return

    conn, ok, message = write_public_card_id(conn, csc0, card_id)
    if not ok:
        print(message)
        return

    try:
        response = mqtt_request("activate-card", {"card_id": card_id})
    except Exception as exc:
        print(f"Carte ecrite, mais activation serveur MQTT impossible: {exc}")
        return

    if not response.get("success"):
        print(f"Carte ecrite, mais activation serveur refusee: {response.get('error')}")
        return

    print("\nCarte initialisee et activee cote serveur")
    print(f"card_id: {card_id}")
    print(f"user_id: {user_id}")


def diagnostic_public(conn):
    print("\n" + "="*70)
    print("DIAGNOSTIC PUBLIC")
    print("="*70)

    print("\nZone publique card_id (0x01-0x02)")
    raw = bytearray()
    for offset in range(PUBLIC_CARD_ID_WORDS):
        addr = PUBLIC_CARD_ID_START + offset
        conn, data, sw1, sw2 = read_word_bytes_resilient(conn, addr)
        if not is_sw_ok(sw1, sw2) or data is None:
            print(f"0x{addr:02X}: ERROR SW={sw1:02X}{sw2:02X}")
            return
        print(f"0x{addr:02X}: {' '.join(f'{b:02X}' for b in data)}")
        raw.extend(data)

    card_id = bytes(raw).rstrip(b"\x00").decode("ascii", errors="ignore").strip()
    print(f"card_id: {card_id or '<vide>'}")

    conn, data, sw1, sw2 = read_word_bytes_resilient(conn, ADDR_CSC1_COUNTER)
    if is_sw_ok(sw1, sw2) and data is not None:
        word = int.from_bytes(data, byteorder="little")
        _, message = decode_counter(word)
        print(f"\nCompteur CSC1 0x39: {toHexString(data)} -> {message}")
    else:
        print(f"\nCompteur CSC1 0x39: ERROR SW={sw1:02X}{sw2:02X}")


def main():
    """Menu principal"""
    try:
        # Demander CSC0 au début (nécessaire pour les deux options)
        print("\n" + "="*70)
        print("DIAGNOSTIC GEMCLUB-MEMO")
        print("="*70)
        print("\n🔐 Authentification administrateur requise (CSC0)")
        print("    Nécessaire pour accéder à toutes les zones")
        
        csc0_input = input("\nCSC0 (8 hex, ex: AAAAAAAA) : ").strip().upper()
        
        if len(csc0_input) != 8:
            print("[-] CSC0 invalide (doit faire 8 caractères hexa)")
            return
        
        try:
            csc0 = bytes.fromhex(csc0_input)
        except ValueError:
            print("[-] Format hexa incorrect")
            return
        
        # Menu principal
        while True:
            print("\n" + "="*70)
            print("MENU PRINCIPAL")
            print("="*70)
            print("\n1) Afficher toutes les zones de la carte (dump complet)")
            print("2) Diagnostic public")
            print("3) Reset complet (User Areas + Access Conditions + Protected)")
            print("4) Enrollement carte")
            print("5) Quitter")
            
            choice = input("\nChoix : ").strip()
            
            if choice == "1":
                try:
                    conn = connect_card()
                    afficher_diagnostic(conn, csc0)
                except Exception as e:
                    print(f"[-] Erreur : {e}")
            elif choice == "2":
                try:
                    conn = connect_card()
                    diagnostic_public(conn)
                except Exception as e:
                    print(f"[-] Erreur : {e}")
            elif choice == "3":
                try:
                    conn = connect_card()
                    reset_complet_resilient(conn, csc0)
                except Exception as e:
                    print(f"[-] Erreur : {e}")
            elif choice == "4":
                try:
                    conn = connect_card()
                    enrollement_card(conn, csc0)
                except Exception as e:
                    print(f"[-] Erreur : {e}")
            elif choice == "5":
                print("\n[*] Au revoir")
                break
            else:
                print("[-] Choix invalide")
    except KeyboardInterrupt:
        print("\n\n[*] Interruption (Ctrl+C) - Fermeture...")

if __name__ == "__main__":
    main()
