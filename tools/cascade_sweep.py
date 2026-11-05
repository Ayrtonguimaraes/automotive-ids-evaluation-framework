"""Sweep the thresholds of a two-stage cascade (Random Forest -> CNN) on saved test probabilities.

Nothing is trained here: it reuses the per-window probabilities that the test runs already saved
(`*_test_proba_*_fold_<f>*.npz`, with the keys `proba` and `y_true`).

Cascade rule, for the RF attack probability p of a window:
    p <  t_low   -> "normal"  (final, the CNN is not run)
    p >= t_high  -> "attack"  (final, the CNN is not run)
    otherwise    -> the CNN decides
With t_high above 1 the RF can never confirm an attack alone, so every window that is not discarded
as normal goes to the CNN. With t_low = 0 and t_high > 1 every window goes to the CNN (CNN alone).

IMPORTANT: this script only describes the curve on the TEST set. The thresholds used in the report
must be chosen on the out-of-fold predictions of the training data, never here.

Usage:
    python tools/cascade_sweep.py --rf-dir ~/outputs/<rf test run>/metrics \
        --cnn-dir ~/outputs/<cnn test run>/metrics --cnn-outputs 1 --out cascade_aeid.csv
For the TOW-IDS multiclass CNN use --cnn-outputs 6 (the class is the largest output, any class but
Normal means attack, and the classes are in alphabetical order, so Normal is column 4).
"""
import argparse
import glob
import os
import re

import numpy as np

NORMAL_COLUMN_TOW = 4  # C_D, C_R, F_I, M_F, Normal, P_I (OneHotEncoder sorts alphabetically)


def find_fold_files(directory, pattern):
    """Return {fold: path} for the files in `directory` matching the glob `pattern`."""
    found = {}
    for path in glob.glob(os.path.join(os.path.expanduser(directory), pattern)):
        match = re.search(r"fold_(\d+)", os.path.basename(path))
        if match:
            found[int(match.group(1))] = path
    return found


def load_rf(path):
    data = np.load(path)
    return data["proba"][:, 1].astype(np.float64), np.asarray(data["y_true"]).ravel().astype(int)


def load_cnn(path, n_outputs):
    """Return the CNN binary decision (1 = attack) and the binary labels of each window."""
    data = np.load(path)
    proba, y_true = data["proba"], data["y_true"]
    if n_outputs == 1:
        return (proba.reshape(-1) >= 0.5).astype(int), np.asarray(y_true).reshape(-1).astype(int)
    # Multiclass: the 6 outputs are independent sigmoids, so they do not sum to 1. The class is the
    # largest output, the same rule as the confusion matrix, and any class but Normal means attack
    return (np.argmax(proba, axis=1) != NORMAL_COLUMN_TOW).astype(int), 1 - np.asarray(y_true)[:, NORMAL_COLUMN_TOW].astype(int)


def evaluate(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, fp, fn, tn, acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rf-dir", required=True, help="metrics folder of the RF test run")
    parser.add_argument("--cnn-dir", required=True, help="metrics folder of the CNN test run")
    parser.add_argument("--cnn-outputs", type=int, choices=[1, 6], default=1)
    parser.add_argument("--rf-time-us", type=float, default=107.0, help="RF time per window in microseconds (default: paper, Table 7)")
    parser.add_argument("--cnn-time-us", type=float, default=4510.0, help="CNN time per window in microseconds (default: paper, Table 7)")
    parser.add_argument("--out", default="cascade_sweep.csv")
    args = parser.parse_args()

    rf_files = find_fold_files(args.rf_dir, "*test_proba*RandomForestClassifier*.npz")
    cnn_files = find_fold_files(args.cnn_dir, "test_proba*.npz")
    folds = sorted(set(rf_files) & set(cnn_files))
    if not folds:
        raise SystemExit(f"no matching folds. RF folds: {sorted(rf_files)}, CNN folds: {sorted(cnn_files)}")

    # A shallow forest averages many trees, so its probabilities rarely get close to 0 or 1:
    # the useful thresholds are in the middle of the range, and the grid is dense there
    t_lows = [round(0.02 * i, 2) for i in range(31)]  # 0.00 to 0.60
    t_highs = [round(0.5 + 0.05 * i, 2) for i in range(10)] + [0.99, 1.01]  # 1.01 = the RF never confirms an attack alone

    rows = []
    for fold in folds:
        p_rf, y_rf = load_rf(rf_files[fold])
        cnn_attack, y_cnn = load_cnn(cnn_files[fold], args.cnn_outputs)
        if len(y_rf) != len(y_cnn) or not np.array_equal(y_rf, y_cnn):
            raise SystemExit(f"fold {fold}: RF and CNN do not describe the same windows "
                             f"({len(y_rf)} vs {len(y_cnn)} windows, labels equal: "
                             f"{len(y_rf) == len(y_cnn) and np.array_equal(y_rf, y_cnn)})")
        for t_low in t_lows:
            for t_high in t_highs:
                if t_low > t_high:
                    continue
                to_cnn = (p_rf >= t_low) & (p_rf < t_high)
                pred = np.where(to_cnn, cnn_attack, (p_rf >= t_high).astype(int))
                frac = float(to_cnn.mean())
                tp, fp, fn, tn, acc, prec, rec, f1 = evaluate(y_rf, pred)
                time_us = args.rf_time_us + frac * args.cnn_time_us
                rows.append([fold, t_low, t_high, frac, tp, fp, fn, tn, acc, prec, rec, f1, time_us])
        print(f"fold {fold}: {len(y_rf)} windows done")

    header = "fold,t_low,t_high,frac_to_cnn,tp,fp,fn,tn,acc,prec,recall,f1,est_time_us"
    np.savetxt(args.out, np.array(rows), delimiter=",", header=header, comments="", fmt="%.10g")
    print(f"saved {len(rows)} rows to {args.out}")

    # Fold-averaged summary. Columns of `table`: see `header` above
    table = np.array(rows)
    pairs = sorted({(row[1], row[2]) for row in rows})
    means = [(t_low, t_high, table[(table[:, 1] == t_low) & (table[:, 2] == t_high)].mean(axis=0)) for t_low, t_high in pairs]

    def show(label, t_low, t_high, m):
        print(f"{label:>22}: t_low={t_low:<5} t_high={t_high:<5} to_cnn={m[3]:.3f} "
              f"recall={m[10]:.4f} prec={m[9]:.4f} f1={m[11]:.4f} fp={m[5]:.0f} fn={m[6]:.0f} time={m[12]:.0f}")

    print("\nmean over folds, TEST set, descriptive only (do not choose the thresholds from this table)")
    cnn_alone = next(m for t_low, t_high, m in means if t_low == 0.0 and t_high == 1.01)
    rf_alone = next(m for t_low, t_high, m in means if t_low == 0.5 and t_high == 0.5)
    show("CNN alone", 0.0, 1.01, cnn_alone)
    show("RF alone", 0.5, 0.5, rf_alone)

    # Cheapest cascade (least sent to the CNN) that stays within `gap` of the F1 of the CNN alone
    for gap in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02]:
        candidates = [item for item in means if item[2][11] >= cnn_alone[11] - gap]
        if candidates:
            t_low, t_high, m = min(candidates, key=lambda item: item[2][3])
            show(f"F1 gap <= {gap}", t_low, t_high, m)


if __name__ == "__main__":
    main()
