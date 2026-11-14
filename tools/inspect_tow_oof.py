"""Check the format of the TOW-IDS out-of-fold predictions before the feedback-loop retraining uses them.

Prints, for the RF (binary) and the CNN (6 outputs) training runs: the shapes of index, y_true and proba,
the column sums of y_true, the first rows, and whether the CNN labels agree with the RF binary labels
once both are aligned by window index. Reads only training predictions.

Usage: python tools/inspect_tow_oof.py <RF run folder> <first CNN run folder> ... (5 CNN fold folders)
"""
import glob
import sys

import numpy as np

rf_run, cnn_runs = sys.argv[1], sys.argv[2:]
rf = [np.load(p) for p in sorted(glob.glob(f"{rf_run}/metrics/oof_proba_*_fold_*.npz"))]
cnn = [np.load(glob.glob(f"{r}/metrics/oof_proba_*_fold_*.npz")[0]) for r in cnn_runs]
for name, parts in (("RF", rf), ("CNN", cnn)):
    d = parts[0]
    print(f"{name}: keys {list(d.keys())}; index {d['index'].shape}, y_true {d['y_true'].shape} {d['y_true'].dtype}, "
          f"proba {d['proba'].shape}")
    y = np.concatenate([p["y_true"] for p in parts])
    print(f"  y_true column sums {y.sum(axis=0) if y.ndim == 2 else np.bincount(y.astype(int).ravel())}")
    print(f"  first rows y_true {d['y_true'][:3].tolist()}\n  first rows proba {np.round(d['proba'][:3], 3).tolist()}")

n = sum(len(p["index"]) for p in rf)
rf_y = np.full(n, -1); cnn_y = np.full((n, cnn[0]["y_true"].shape[1]), -1.0)
for p in rf:
    rf_y[p["index"]] = np.asarray(p["y_true"]).ravel()
for p in cnn:
    cnn_y[p["index"]] = p["y_true"]
print(f"every window has one RF and one CNN prediction: {(rf_y >= 0).all() and (cnn_y >= 0).all()}")
for col in range(cnn_y.shape[1]):
    agree = ((cnn_y[:, col] == 1) == (rf_y == 0)).mean()
    print(f"  CNN label column {col} == 1  <=>  RF label normal: agreement {agree:.4f}")
