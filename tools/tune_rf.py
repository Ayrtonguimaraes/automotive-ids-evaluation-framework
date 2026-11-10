"""Grid search of the Random Forest hyperparameters on the SOME/IP training windows.

Only training data is used. Each candidate is trained on part of the training captures and scored
on the remaining captures (the folds of model_train_validation/splits.py with validation by capture),
so near-identical overlapping windows never sit on both sides. The test set is not touched here.

The model is the same pipeline the authors use (StandardScaler + RandomForestClassifier, see
models/sklearn_classifier.py). Candidates are ranked by the mean validation ROC-AUC, decided before
looking at the results: it does not depend on the 0.5 threshold, and 62% of the windows are attacks,
so accuracy alone would reward a model that calls everything an attack.

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

# Search space, fixed before running. The AEID/TOW values (depth 4, 3 features) are included as the baseline
GRID = {
    "max_depth": [4, 8, 16, None],
    "max_features": [3, 10, 30],
}
FIXED = {"n_estimators": 100, "bootstrap": True, "random_state": 42}

HEADER = "max_depth,max_features,fold,val_auc,val_f1,val_acc,val_prec,val_recall,train_auc,fit_seconds"


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

    results = {}
    for depth, feats in itertools.product(GRID["max_depth"], GRID["max_features"]):
        for f, (tr, va) in enumerate(folds):
            model = Pipeline([("scaler", StandardScaler()),
                              ("clf", RandomForestClassifier(max_depth=depth, max_features=feats, n_jobs=args.jobs, **FIXED))])
            start = time.time()
            model.fit(X[tr], y[tr])
            seconds = time.time() - start

            p_val = model.predict_proba(X[va])[:, 1]
            pred = (p_val >= 0.5).astype(int)
            # Train AUC on a sample of the training windows, to see overfitting without scoring all of them
            sample = np.random.default_rng(0).choice(tr, size=min(200_000, len(tr)), replace=False)
            train_auc = roc_auc_score(y[sample], model.predict_proba(X[sample])[:, 1])

            row = [depth, feats, f, roc_auc_score(y[va], p_val), f1_score(y[va], pred), accuracy_score(y[va], pred),
                   precision_score(y[va], pred, zero_division=0), recall_score(y[va], pred), train_auc, seconds]
            with open(out, "a") as fh:
                fh.write(",".join(str(v) for v in row) + "\n")
            results.setdefault((depth, feats), []).append(row)
            print(f"depth={depth} features={feats} fold={f}: val_auc={row[3]:.4f} val_f1={row[4]:.4f} "
                  f"train_auc={train_auc:.4f} ({seconds:.0f} s)")

    print("\nmean over capture folds, sorted by validation AUC (selection rule)")
    ranking = sorted(results.items(), key=lambda item: -np.mean([r[3] for r in item[1]]))
    for (depth, feats), rows in ranking:
        auc = [r[3] for r in rows]
        print(f"depth={str(depth):>4} features={feats:>2}: val_auc={np.mean(auc):.4f} ± {np.std(auc):.4f} "
              f"val_f1={np.mean([r[4] for r in rows]):.4f} train_auc={np.mean([r[8] for r in rows]):.4f}")
    (depth, feats), _ = ranking[0]
    print(f"\nselected: max_depth={depth}, max_features={feats}")


if __name__ == "__main__":
    main()
