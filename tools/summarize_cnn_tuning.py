"""Rank the CNN tuning candidates by their mean validation ROC-AUC (the selection rule decided beforehand).

Reads the val_metrics_*.csv of the training runs started at or after <not before> (run folders are
named <timestamp>_pytorch_train...), one row per fold, and averages them per candidate (batch size and
learning rate, which are in the file name).

Usage: python tools/summarize_cnn_tuning.py <not before, e.g. 2026_11_12_20_00_00>
"""
import collections
import glob
import os
import re
import sys

import pandas as pd

not_before = sys.argv[1]
outputs = os.environ.get("IDS_ARTIFACTS_DIR", os.path.expanduser("~/outputs"))
rows = collections.defaultdict(list)
for path in sorted(glob.glob(f"{outputs}/*_pytorch_train*/metrics/val_metrics_PrunedCNNIDS_*.csv")):
    run = os.path.basename(os.path.dirname(os.path.dirname(path)))
    if run < not_before:
        continue
    candidate = re.search(r"BS(\d+)_EP\d+_LR([0-9.e-]+)\.csv$", path).groups()
    rows[candidate].append(pd.read_csv(path))

if not rows:
    sys.exit(f"no tuning runs found after {not_before} in {outputs}")
print(f"{'batch':>6} {'lr':>8} {'folds':>6} {'val AUC mean':>13} {'std':>6} {'min':>6} {'F1 mean':>8}")
ranking = []
for (batch, lr), frames in rows.items():
    df = pd.concat(frames)
    ranking.append((df["roc_auc"].mean(), batch, lr, len(df), df["roc_auc"].std(ddof=0), df["roc_auc"].min(), df["f1"].mean()))
for auc, batch, lr, n, std, low, f1 in sorted(ranking, reverse=True):
    print(f"{batch:>6} {lr:>8} {n:>6} {auc:13.4f} {std:6.4f} {low:6.4f} {f1:8.4f}")
best = max(ranking)
print(f"\nselected (highest mean validation AUC): batch_size={best[1]}, learning_rate={best[2]}")
