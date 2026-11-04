"""Safely inspect one capture of the SOME/IP dataset (a *_x.pickle / *_y.pickle pair).

Pickle files can run arbitrary code when they are loaded, so this uses the restricted
unpickler of feature_generator/safe_pickle.py (numpy arrays only; anything else is blocked).

Usage: python tools/inspect_someip_pickle.py error_on_error_1_x.pickle error_on_error_1_y.pickle
"""
import importlib.util
import os
import sys

import numpy as np

# Load feature_generator/safe_pickle.py by path, so the heavy feature generator package is not imported
_spec = importlib.util.spec_from_file_location(
    "safe_pickle", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feature_generator", "safe_pickle.py"))
_safe_pickle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_safe_pickle)
safe_load = _safe_pickle.safe_load


def describe_row(row):
    """Decode a 58-byte row as Ethernet + IPv4 + UDP + SOME/IP header (14 + 20 + 8 + 16 bytes)."""
    r = [int(v) for v in row]
    return {
        "ethertype": hex(r[12] << 8 | r[13]),
        "ip_proto": r[23],
        "src_port": r[34] << 8 | r[35],
        "dst_port": r[36] << 8 | r[37],
        "message_id": bytes(r[42:46]).hex(),
        "length": int.from_bytes(bytes(r[46:50]), "big"),
        "client_id": bytes(r[50:52]).hex(),
        "session_id": bytes(r[52:54]).hex(),
        "protocol_version": r[54],
        "interface_version": r[55],
        "message_type": hex(r[56]),
        "return_code": r[57],
    }


def main(x_path, y_path):
    x = safe_load(x_path)
    y = safe_load(y_path)

    print(f"x: {type(x).__name__}, dtype={getattr(x, 'dtype', None)}, shape={getattr(x, 'shape', None)}")
    print(f"y: {type(y).__name__}, dtype={getattr(y, 'dtype', None)}, shape={getattr(y, 'shape', None)}")

    y = np.asarray(y).ravel()
    values, counts = np.unique(y, return_counts=True)
    print("labels:", dict(zip(values.tolist(), counts.tolist())))

    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] < 58:
        print("x does not look like N x 58 bytes; stopping here")
        return

    print("first 2 rows (hex):")
    for row in x[:2]:
        print(" ", bytes(int(v) & 0xFF for v in row[:58]).hex())
    print("decoded first 3 rows:")
    for row in x[:3]:
        print(" ", describe_row(row))

    # Which SOME/IP message types appear in normal (0) and attack (1) rows
    for label in values.tolist():
        types = np.unique(x[y == label][:, 56], return_counts=True)
        print(f"message types where label={label}:", {hex(int(t)): int(c) for t, c in zip(*types)})


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
