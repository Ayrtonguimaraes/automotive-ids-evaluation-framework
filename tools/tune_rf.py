"""Grid search of the Random Forest hyperparameters on the SOME/IP training windows.

Only training data is used. Each candidate is trained on part of the training captures and scored
on the remaining captures (the folds of model_train_validation/splits.py with validation by capture),
so near-identical overlapping windows never sit on both sides. The test set is not touched here.

The model is the same pipeline the authors use (StandardScaler + RandomForestClassifier, see
models/sklearn_classifier.py), and the search follows Section 4.2 of the paper: the same three
hyperparameters, in the same ranges (n_estimators 50-300, max_depth and max_features 3-10). A few
candidates outside those ranges are added only to check whether a larger forest would help; they are
reported, but they are not eligible, because the paper keeps the first stage small on purpose (speed).

Selection rule, decided before looking at the results: highest mean validation ROC-AUC (it does not
depend on the 0.5 threshold, and most windows are attacks, so accuracy alone would reward a model that
calls everything an attack). Among candidates within TIE of the best, the cheapest one wins (fewest
trees, then the shallowest), since the paper's main criterion for this stage is a short detection time.

Usage (from the repository root, on the cluster; see jobs/tune_rf.sbatch):
    python tools/tune_rf.py --x <X_train ... sumX_True.npz> --y <y_train ... sumX_True.npz> \
        --groups data/someip/processed/groups_train.npz --folds 3 --out ~/outputs/someip_rf_tuning.csv
"""
import argparse
import importlib.util
import itertools
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load model_train_validation/splits.py by path: importing the package would also import torch
_spec = importlib.util.spec_from_file_location(
    "splits", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_train_validation", "splits.py"))
splits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(splits)

# Search space, fixed before running, inside the ranges of the paper. It contains the values of Table 3:
# (100, 4, 3) for AEID and (200, 10, 7) for TOW-IDS
GRID = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [4, 7, 10],
    "max_features": [3, 7, 10],
}
# Outside the paper's ranges: shown in the ranking, never selected
EXTRA = [(100, 16, 10), (100, None, 10), (100, 10, 30)]
FIXED = {"bootstrap": True, "random_state": 42}
TIE = 0.002

HEADER = "n_estimators,max_depth,max_features,in_paper_ranges,fold,val_auc,val_f1,val_acc,val_prec,val_recall,train_auc,fit_seconds"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--x", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--folds", type=int, default=3, help="how many of the 5 capture folds to use (time budget)")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    X = np.load(args.x)["arr_0"]
    y = np.load(args.y)["arr_0"].ravel().astype(int)
    folds = splits.make_folds(y, {"validation_groups_path": args.groups})[:args.folds]
    print(f"X {X.shape}, attack windows {y.mean():.4f}, using {len(folds)} capture folds")
    for f, (tr, va) in enumerate(folds):
        print(f"  fold {f}: train {len(tr)} windows ({y[tr].mean():.3f} attack), val {len(va)} ({y[va].mean():.3f} attack)")

    out = os.path.expanduser(args.out)
    with open(out, "w") as fh:
        fh.write(HEADER + "\n")

    candidates = [(c, True) for c in itertools.product(GRID["n_estimators"], GRID["max_depth"], GRID["max_features"])]
    candidates += [(c, False) for c in EXTRA]

    results = {}
    for (trees, depth, feats), in_ranges in candidates:
        for f, (tr, va) in enumerate(folds):
            model = Pipeline([("scaler", StandardScaler()),
                              ("clf", RandomForestClassifier(n_estimators=trees, max_depth=depth, max_features=feats,
                                                             n_jobs=args.jobs, **FIXED))])
            start = time.time()
            model.fit(X[tr], y[tr])
            seconds = time.time() - start

            p_val = model.predict_proba(X[va])[:, 1]
            pred = (p_val >= 0.5).astype(int)
            # Train AUC on a sample of the training windows, to see overfitting without scoring all of them
            sample = np.random.default_rng(0).choice(tr, size=min(200_000, len(tr)), replace=False)
            train_auc = roc_auc_score(y[sample], model.predict_proba(X[sample])[:, 1])

            row = [trees, depth, feats, int(in_ranges), f, roc_auc_score(y[va], p_val), f1_score(y[va], pred),
                   accuracy_score(y[va], pred), precision_score(y[va], pred, zero_division=0), recall_score(y[va], pred),
                   train_auc, seconds]
            with open(out, "a") as fh:
                fh.write(",".join(str(v) for v in row) + "\n")
            results.setdefault((trees, depth, feats, in_ranges), []).append(row)
            print(f"trees={trees} depth={depth} features={feats} fold={f}: val_auc={row[5]:.4f} val_f1={row[6]:.4f} "
                  f"train_auc={train_auc:.4f} ({seconds:.0f} s)")

    def mean_auc(rows):
        return float(np.mean([r[5] for r in rows]))

    print("\nmean over capture folds, sorted by validation AUC")
    ranking = sorted(results.items(), key=lambda item: -mean_auc(item[1]))
    for (trees, depth, feats, in_ranges), rows in ranking:
        auc = [r[5] for r in rows]
        print(f"trees={trees:>3} depth={str(depth):>4} features={feats:>2}: val_auc={np.mean(auc):.4f} ± {np.std(auc):.4f} "
              f"val_f1={np.mean([r[6] for r in rows]):.4f} train_auc={np.mean([r[10] for r in rows]):.4f}"
              f"{'' if in_ranges else '   (outside the paper ranges, not eligible)'}")

    eligible = [(key, rows) for key, rows in ranking if key[3]]
    best_auc = mean_auc(eligible[0][1])
    tied = [key for key, rows in eligible if mean_auc(rows) >= best_auc - TIE]
    trees, depth, feats, _ = min(tied, key=lambda key: (key[0], key[1], key[2]))
    print(f"\nbest eligible validation AUC: {best_auc:.4f}; {len(tied)} candidate(s) within {TIE} of it")
    print(f"selected (cheapest of those): n_estimators={trees}, max_depth={depth}, max_features={feats}")


if __name__ == "__main__":
    main()
