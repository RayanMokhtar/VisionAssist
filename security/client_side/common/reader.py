from smartcard.System import readers
from smartcard.util import toHexString

from common.card_config import P2_VERIFY_CSC0, P2_VERIFY_CSC1


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


def update_zone(conn, addr, data_bytes):
    if len(data_bytes) != 4:
        raise ValueError("data_bytes doit faire 4 octets")
    apdu = [0x80, 0xDE, 0x00, addr, 0x04] + list(data_bytes)
    _, sw1, sw2 = conn.transmit(apdu)
    return sw1, sw2


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
