"""Rank the SOME/IP CSV window search variants by mean validation ROC-AUC, per model and per grouping.

Selection rule, decided before running (same as the hyperparameter search): for each model and each
grouping, the highest mean validation AUC over the 5 folds; variants within TIE of the best count as
tied, and the smallest window among them wins (it decides sooner and leaves fewer messages without a
decision). The test is then run once, only for the selected variants.

Reads ~/outputs/window_search/rf_<variant>.csv (tools/tune_rf.py --only) and the CNN files
val_metrics_PrunedCNNIDS_<variant>_*.csv of the training runs (jobs/window_search.sbatch).
Usage: python tools/summarize_window_search.py
"""
import glob
import os

import pandas as pd

TIE = 0.002
outputs = os.environ.get("IDS_ARTIFACTS_DIR", os.path.expanduser("~/outputs"))
rows = []
for grouping in ("arrival", "conversation"):
    for window in (10, 20, 30, 44):
        name = f"{grouping}_w{window}"
        rf = f"{outputs}/window_search/rf_{name}.csv"
        if os.path.exists(rf):
            df = pd.read_csv(rf)
            rows.append(("RF", grouping, window, len(df), df.val_auc.mean(), df.val_auc.std(ddof=0), df.val_auc.min()))
        cnn = sorted(glob.glob(f"{outputs}/*_pytorch_train*/metrics/val_metrics_PrunedCNNIDS_{name}_*.csv"))
        if cnn:
            # The latest run of this variant (one file holds the 5 folds)
            df = pd.read_csv(cnn[-1])
            rows.append(("CNN", grouping, window, len(df), df.roc_auc.mean(), df.roc_auc.std(ddof=0), df.roc_auc.min()))

if not rows:
    raise SystemExit(f"no window search results in {outputs}")
table = pd.DataFrame(rows, columns=["model", "grouping", "window", "folds", "val_auc", "std", "min"])
print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
for (model, grouping), part in table.groupby(["model", "grouping"]):
    best = part.val_auc.max()
    tied = part[part.val_auc >= best - TIE]
    pick = tied.sort_values("window").iloc[0]
    print(f"{model:>3} {grouping:>12}: selected window {int(pick.window)} (val AUC {pick.val_auc:.4f} ± {pick['std']:.4f}; "
          f"{len(tied)} within {TIE} of the best)")
