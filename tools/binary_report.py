"""Test metrics that stay honest when one class dominates, from the saved test probabilities.

With the paper's window label (attack if any message in the window is an attack), most SOME/IP windows
are attacks, so accuracy and F1 reward a model that calls everything an attack. This prints, per fold
and on average: ROC-AUC, balanced accuracy, recall on attacks, specificity (recall on normal windows),
precision, F1, and the same metrics for the trivial "everything is an attack" model.

Usage: python tools/binary_report.py <metrics folder of a test run>
"""
import glob
import os
import re
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

folder = os.path.expanduser(sys.argv[1])
paths = sorted(glob.glob(f"{folder}/*test_proba*.npz"), key=lambda p: int(re.search(r"fold_(\d+)", p).group(1)))
if not paths:
    sys.exit(f"no test_proba files in {folder}")

rows = []
for path in paths:
    data = np.load(path)
    proba, y = data["proba"], np.asarray(data["y_true"]).ravel().astype(int)
    p = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()
    pred = (p >= 0.5).astype(int)
    rows.append([roc_auc_score(y, p), balanced_accuracy_score(y, pred), recall_score(y, pred),
                 recall_score(y, pred, pos_label=0), precision_score(y, pred, zero_division=0), f1_score(y, pred)])
    fold = re.search(r"fold_(\d+)", path).group(1)
    print(f"fold {fold}: AUC {rows[-1][0]:.4f}  balanced acc {rows[-1][1]:.4f}  recall {rows[-1][2]:.4f}  "
          f"specificity {rows[-1][3]:.4f}  precision {rows[-1][4]:.4f}  F1 {rows[-1][5]:.4f}")

m, s = np.mean(rows, axis=0), np.std(rows, axis=0)
names = ["AUC", "balanced acc", "recall", "specificity", "precision", "F1"]
print("mean ± std: " + "  ".join(f"{n} {a:.4f}±{b:.4f}" for n, a, b in zip(names, m, s)))
attack_share = y.mean()
print(f"test windows: {len(y)}, attack share {attack_share:.4f}. 'Everything is an attack' gives AUC 0.5, "
      f"balanced acc 0.5, specificity 0, precision {attack_share:.4f}, F1 {2 * attack_share / (1 + attack_share):.4f}")
