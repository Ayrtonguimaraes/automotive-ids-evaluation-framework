"""Rebuild the first 58 bytes of each SOME/IP message from the CSV files of Alkhatib et al. (2021).

The CSV files of github.com/Alkhatibnatasha/SOMEIP_IDS hold one row per message with the header fields
(MAC and IP addresses, UDP ports, SOME/IP service, method, length, client, session, versions, message
type and return code), but not the raw bytes. The feature extractor of the paper reads the first 58
bytes of each frame, which for SOME/IP over UDP are exactly the four headers:

    Ethernet  0-13   destination MAC, source MAC, EtherType 0x0800
    IPv4     14-33   version/IHL, DSCP, total length, ID, flags, TTL, protocol, checksum, source, destination
    UDP      34-41   source port, destination port, length, checksum
    SOME/IP  42-57   service ID, method ID, length, client ID, session ID, protocol version,
                     interface version, message type, return code

Fields taken from the CSV: MACs, IPs, protocol, ports, service, method, length, client, session, the
two versions, message type and return code. Fields computed from them: IPv4 total length and UDP
length (from the SOME/IP length, which counts the 8 bytes after it plus the payload), and the IPv4
header checksum. Fields the CSV does not have are fixed: IPv4 ID 0, flags "don't fragment", TTL 64 and
UDP checksum 0 (allowed in IPv4, meaning "no checksum"). The extractor uses the difference between
consecutive messages, so a fixed field becomes 0 for every message and carries no information.
"""
import numpy as np
import pandas as pd

NUMBER_OF_BYTES = 58
TTL = 64
DONT_FRAGMENT = 0x4000


def _mac(text):
    return [int(part, 16) for part in text.split(":")]


def _ip(text):
    return [int(part) for part in text.split(".")]


def _u16(value):
    return [(value >> 8) & 0xFF, value & 0xFF]


def _u32(value):
    return [(value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF]


def ipv4_checksum(header):
    """Internet checksum of a 20-byte IPv4 header whose checksum bytes are zero."""
    total = sum((header[i] << 8) + header[i + 1] for i in range(0, len(header), 2))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def row_to_bytes(row):
    """The first 58 bytes of the frame described by one CSV row (a dict-like with the CSV columns)."""
    someip_length = int(row["length"])
    udp_length = 8 + 8 + someip_length          # UDP header + SOME/IP message ID and length fields + the rest
    ip_length = 20 + udp_length

    ethernet = _mac(row["mac_dst"]) + _mac(row["mac_src"]) + [0x08, 0x00]
    ipv4 = [0x45, 0x00] + _u16(ip_length) + _u16(0) + _u16(DONT_FRAGMENT) + [TTL, int(row["proto"])] + [0, 0] \
        + _ip(row["ip_src"]) + _ip(row["ip_dst"])
    ipv4[10:12] = _u16(ipv4_checksum(ipv4))
    udp = _u16(int(row["sport"])) + _u16(int(row["dport"])) + _u16(udp_length) + _u16(0)
    someip = _u16(int(row["service_id"])) + _u16(int(row["method_id"])) + _u32(someip_length) \
        + _u16(int(row["Client_id"])) + _u16(int(row["session_id"])) \
        + [int(row["proto_ver"]), int(row["iface_ver"]), int(row["msg_type"]), int(row["retcode"])]

    frame = ethernet + ipv4 + udp + someip
    assert len(frame) == NUMBER_OF_BYTES, len(frame)
    return frame


def read_capture(csv_path):
    """Return (packets, labels) for one CSV file: a (N, 58) uint8 array and N labels, 1 = attack."""
    frame = pd.read_csv(csv_path)
    # The first, unnamed column is the row number: the file must already be in capture order
    order = frame.iloc[:, 0].to_numpy()
    if not np.array_equal(order, np.arange(len(frame))):
        raise ValueError(f"{csv_path}: rows are not in capture order")
    packets = np.array([row_to_bytes(row) for row in frame.to_dict("records")], dtype=np.uint8)
    # 0 is normal and 1-4 are the attack classes; the paper's IDS is binary, so any attack is 1
    labels = (frame["label"].to_numpy() != 0).astype(np.uint8)
    return packets, labels
