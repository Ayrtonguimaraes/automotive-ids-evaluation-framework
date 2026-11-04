"""Load pickles from third parties without letting them execute code.

Pickle files can run arbitrary code when they are loaded, so this unpickler only allows
the few numpy classes needed to rebuild arrays; anything else is blocked.
"""
import pickle

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
