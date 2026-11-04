"""Safely inspect one capture of the SOME/IP dataset (a *_x.pickle / *_y.pickle pair).

Pickle files can run arbitrary code when they are loaded, so this uses an unpickler
that only allows the few numpy classes needed to rebuild arrays; anything else is
blocked and reported.

Usage: python tools/inspect_someip_pickle.py error_on_error_1_x.pickle error_on_error_1_y.pickle
"""
import pickle
import sys

import numpy as np

ALLOWED = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked: {module}.{name}")


def safe_load(path):
    with open(path, "rb") as pickle_file:
        return SafeUnpickler(pickle_file).load()


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
