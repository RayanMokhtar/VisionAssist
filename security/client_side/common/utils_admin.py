import base64

from getpass import getpass

from smartcard.Exceptions import CardConnectionException
from smartcard.System import readers
from smartcard.util import toHexString


# =========================
# Constantes carte
# =========================

P2_VERIFY_CSC0 = 0x07
P2_VERIFY_CSC1 = 0x39
P2_EMULATE_USER_MODE = 0x3A

ADDR_CSC1_WORD = 0x38
WORD_ZERO = bytes.fromhex("00 00 00 00")

# Adresses selon documentation GemClub-Memo
ADDR_MANUFACTURER = 0x00
ADDR_ISSUER_START = 0x01
ADDR_ISSUER_END = 0x04
ADDR_ACCESS_CONDITIONS = 0x05
# User Area 1: lecture avec CSC1, ecriture interdite en mode utilisateur.
ACCESS_SECRET_LECTURE_CSC1 = bytes.fromhex("00 00 00 30")
ADDR_CSC0 = 0x06
ADDR_CSC0_COUNTER = 0x07
ADDR_CTC1_START = 0x08
ADDR_BALANCE1_START = 0x0B
ADDR_USER_AREA1_START = 0x10
ADDR_USER_AREA1_END = 0x1F
ADDR_CTC2_START = 0x20
ADDR_BALANCE2_START = 0x23
ADDR_USER_AREA2_START = 0x28
ADDR_USER_AREA2_END = 0x37
ADDR_CSC1 = 0x38
ADDR_CSC1_COUNTER = 0x39
ADDR_CSC2 = 0x3A
ADDR_CSC2_COUNTER = 0x3B
ADDR_PROTECTED_START = 0x3C
ADDR_PROTECTED_END = 0x3F

PUBLIC_CARD_ID_START = 0x01
PUBLIC_CARD_ID_WORDS = 2

SECRET_CARTE_START = 0x10
SECRET_CARTE_WORDS = 8
SECRET_CARTE_TAILLE = SECRET_CARTE_WORDS * 4

FORBIDDEN_CSC_VALUES = {
    bytes.fromhex("00 00 00 00"),
    bytes.fromhex("80 00 00 00"),
    bytes.fromhex("7F FF FF FF"),
    bytes.fromhex("FF FF FF FF"),
}


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


def verifier_detection_carte():
    try:
        conn = connect_card()
        try:
            conn.disconnect()
        except Exception:
            pass
        print("[+] Carte detectee")
        return True
    except Exception as exc:
        print(f"[-] Carte non detectee: {exc}")
        return False


def is_sw_ok(sw1, sw2):
    return (sw1, sw2) == (0x90, 0x00)


def read_word_bytes(conn, addr):
    apdu = [0x80, 0xBE, 0x00, addr, 0x04]
    data, sw1, sw2 = conn.transmit(apdu)
    if not is_sw_ok(sw1, sw2):
        return None, sw1, sw2
    return data, sw1, sw2


def update_zone(conn, addr, data_bytes):
    if len(data_bytes) != 4:
        raise ValueError("data_bytes doit faire 4 octets")
    apdu = [0x80, 0xDE, 0x00, addr, 0x04] + list(data_bytes)
    _, sw1, sw2 = conn.transmit(apdu)
    return sw1, sw2


# =========================
# CSC0, CSC1 et reconnexion
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


# =========================
# PIN utilisateur CSC1
# =========================

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


def write_csc1_pin(conn, csc0, pin):
    conn, sw1, sw2 = update_zone_resilient(conn, csc0, ADDR_CSC1_WORD, pin_to_csc1_bytes(pin))
    if not is_sw_ok(sw1, sw2):
        return conn, False, f"Erreur ecriture CSC1 SW={sw1:02X}{sw2:02X}"
    return conn, True, "ok"


# =========================
# Card ID public
# =========================

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


# =========================
# Secret carte et access conditions
# =========================

def decoder_secret_texte(secret_texte):
    return base64.b64decode(secret_texte.encode("ascii"))


def ecrire_secret_carte(conn, csc0, secret_texte):
    secret = decoder_secret_texte(secret_texte)

    if len(secret) != SECRET_CARTE_TAILLE:
        return conn, False, f"secret invalide: {len(secret)} octets au lieu de {SECRET_CARTE_TAILLE}"

    for offset in range(SECRET_CARTE_WORDS):
        addr = SECRET_CARTE_START + offset
        debut = offset * 4
        fin = debut + 4
        word = secret[debut:fin]

        conn, sw1, sw2 = update_zone_resilient(conn, csc0, addr, word)
        if not is_sw_ok(sw1, sw2):
            return conn, False, f"Erreur ecriture secret addr=0x{addr:02X} SW={sw1:02X}{sw2:02X}"

    return conn, True, "ok"


def configurer_acces_secret_carte(conn, csc0):
    # Protege le secret stocke dans User Area 1 (0x10-0x1F).
    # Ru1=1: lecture avec CSC1, Uu1=1: ecriture interdite en mode utilisateur.
    conn, sw1, sw2 = update_zone_resilient(conn, csc0, ADDR_ACCESS_CONDITIONS, ACCESS_SECRET_LECTURE_CSC1)
    if not is_sw_ok(sw1, sw2):
        return conn, False, f"Erreur configuration acces secret SW={sw1:02X}{sw2:02X}"
    return conn, True, "ok"


# =========================
# Diagnostic carte
# =========================

def decode_counter(counter_word):
    """
    Decode un compteur CSC selon la doc GemClub-Memo.
    Utilise uniquement les 4 bits MSB (bits 31-28) du word.
    """
    msb_4bits = (counter_word >> 28) & 0x0F

    if msb_4bits == 0x0:
        return 3, "3 tentatives restantes"
    if msb_4bits == 0x8:
        return 2, "2 tentatives restantes"
    if msb_4bits == 0xC:
        return 1, "1 tentative restante"
    if msb_4bits == 0xE:
        return 0, "0 tentative - BLOQUE au prochain echec!"
    if msb_4bits == 0xF:
        return -1, "BLOQUE DEFINITIVEMENT"
    return -2, f"? Valeur inattendue (bits 31-28 = 0x{msb_4bits:X})"


def read_and_display_word(conn, addr, label):
    """Lit et affiche un word en hexa."""
    data, sw1, sw2 = read_word_bytes(conn, addr)
    if is_sw_ok(sw1, sw2) and data is not None:
        hex_str = toHexString(data)
        print(f"   0x{addr:02X}: {hex_str}")
        return data
    print(f"   0x{addr:02X}: ERROR (SW={sw1:02X}{sw2:02X})")
    return None


def afficher_diagnostic(conn, csc0):
    """Affiche le diagnostic complet: toutes les zones en hexa + compteurs CSC."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC GEMCLUB-MEMO - DUMP COMPLET")
    print("=" * 70)

    print("\n[*] Verification CSC0...")
    conn, sw1, sw2 = verify_csc0_resilient(conn, csc0)
    if not is_sw_ok(sw1, sw2):
        print(f"[-] VERIFY CSC0 echoue (SW={sw1:02X}{sw2:02X})")
        if sw1 == 0x63 and sw2 == 0x00:
            print("    Code incorrect")
        elif sw1 == 0x69 and sw2 == 0x82:
            print("    CSC0 BLOQUE")
        return

    print("[+] CSC0 verifie - Acces complet OK\n")

    print("\nMANUFACTURER AREA (0x00)")
    print("-" * 70)
    read_and_display_word(conn, ADDR_MANUFACTURER, "Manufacturer")

    print("\nISSUER AREA (0x01-0x04)")
    print("-" * 70)
    for addr in range(ADDR_ISSUER_START, ADDR_ISSUER_END + 1):
        read_and_display_word(conn, addr, "")

    print("\nACCESS CONDITIONS AREA (0x05)")
    print("-" * 70)
    data = read_and_display_word(conn, ADDR_ACCESS_CONDITIONS, "")
    if data:
        acc_word = int.from_bytes(data, byteorder="little")
        acc_byte = (acc_word >> 24) & 0xFF
        print(f"   -> Byte conditions (bits 31-24): 0x{acc_byte:02X} = {acc_byte:08b}b")

    print("\nCSC0 & COMPTEUR (0x06-0x07)")
    print("-" * 70)
    read_and_display_word(conn, ADDR_CSC0, "")
    data, sw1, sw2 = read_word_bytes(conn, ADDR_CSC0_COUNTER)
    if is_sw_ok(sw1, sw2) and data is not None:
        word = int.from_bytes(data, byteorder="little")
        _, message = decode_counter(word)
        print(f"   0x{ADDR_CSC0_COUNTER:02X}: {toHexString(data)}  -> {message}")
    else:
        print(f"   0x{ADDR_CSC0_COUNTER:02X}: ERROR (SW={sw1:02X}{sw2:02X})")

    print("\nCTC 1 & BALANCE 1 (0x08-0x0F)")
    print("-" * 70)
    for addr in range(0x08, 0x10):
        read_and_display_word(conn, addr, "")

    print("\nUSER AREA 1 (0x10-0x1F)")
    print("-" * 70)
    for addr in range(ADDR_USER_AREA1_START, ADDR_USER_AREA1_END + 1):
        read_and_display_word(conn, addr, "")

    print("\nCTC 2 & BALANCE 2 (0x20-0x27)")
    print("-" * 70)
    for addr in range(0x20, 0x28):
        read_and_display_word(conn, addr, "")

    print("\nUSER AREA 2 (0x28-0x37)")
    print("-" * 70)
    for addr in range(ADDR_USER_AREA2_START, ADDR_USER_AREA2_END + 1):
        read_and_display_word(conn, addr, "")

    print("\nCSC1 & COMPTEUR (0x38-0x39)")
    print("-" * 70)
    read_and_display_word(conn, ADDR_CSC1, "")
    data, sw1, sw2 = read_word_bytes(conn, ADDR_CSC1_COUNTER)
    if is_sw_ok(sw1, sw2) and data is not None:
        word = int.from_bytes(data, byteorder="little")
        _, message = decode_counter(word)
        print(f"   0x{ADDR_CSC1_COUNTER:02X}: {toHexString(data)}  -> {message}")
    else:
        print(f"   0x{ADDR_CSC1_COUNTER:02X}: ERROR (SW={sw1:02X}{sw2:02X})")

    print("\nCSC2 & COMPTEUR (0x3A-0x3B)")
    print("-" * 70)
    read_and_display_word(conn, ADDR_CSC2, "")
    data, sw1, sw2 = read_word_bytes(conn, ADDR_CSC2_COUNTER)
    if is_sw_ok(sw1, sw2) and data is not None:
        word = int.from_bytes(data, byteorder="little")
        _, message = decode_counter(word)
        print(f"   0x{ADDR_CSC2_COUNTER:02X}: {toHexString(data)}  -> {message}")
    else:
        print(f"   0x{ADDR_CSC2_COUNTER:02X}: ERROR (SW={sw1:02X}{sw2:02X})")

    print("\nPROTECTED AREA (0x3C-0x3F)")
    print("-" * 70)
    for addr in range(ADDR_PROTECTED_START, ADDR_PROTECTED_END + 1):
        read_and_display_word(conn, addr, "")

    print("\n" + "=" * 70)
    print("LEGENDE")
    print("=" * 70)
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
    print("  0xF0000000 = BLOQUE definitivement")
    print("=" * 70)


def diagnostic_public(conn):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC PUBLIC")
    print("=" * 70)

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


# =========================
# Reset carte
# =========================

def reset_complet_resilient(conn, csc0):
    """Reset complet avec reconnexion automatique si PC/SC reset la carte."""
    print("\n" + "=" * 70)
    print("RESET COMPLET DE LA CARTE")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    if errors == 0:
        print("RESET COMPLET TERMINE AVEC SUCCES")
    else:
        print(f"RESET TERMINE AVEC {errors} ERREUR(S)")
    print("=" * 70)
