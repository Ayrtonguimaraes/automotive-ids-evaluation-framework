"""List the saved out-of-fold prediction files and tell which dataset each run belongs to.

Every training run saves, per fold, the validation predictions (index, y_true, proba). The total number of
windows over the 5 folds identifies the dataset (AEID train 643,264 windows; TOW-IDS train ~1.2 million;
SOME/IP pickles 1,263,490; SOME/IP CSV a few thousand) and the number of outputs tells binary (1 or 2
columns) from the TOW-IDS multiclass CNN (6).

Usage: python tools/list_oof.py
"""
import collections
import glob
import os

import numpy as np

outputs = os.environ.get("IDS_ARTIFACTS_DIR", os.path.expanduser("~/outputs"))
runs = collections.defaultdict(list)
for path in sorted(glob.glob(f"{outputs}/*/metrics/oof_proba_*.npz")):
    runs[os.path.basename(os.path.dirname(os.path.dirname(path)))].append(path)

print(f"{'run':45s} {'file':45s} {'windows':>9} {'outputs':>7} {'attack share':>12}")
for run, paths in runs.items():
    for path in paths:
        data = np.load(path)
        proba, y = data["proba"], np.asarray(data["y_true"])
        n_out = proba.shape[1] if proba.ndim == 2 else 1
        attack = (y.argmax(axis=1) != 4).mean() if y.ndim == 2 and y.shape[1] == 6 else (y.ravel() != 0).mean()
        print(f"{run:45s} {os.path.basename(path):45s} {len(y):9d} {n_out:7d} {attack:12.3f}")
