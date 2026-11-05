"""Sweep the thresholds of a two-stage cascade (Random Forest -> CNN) on saved test probabilities.

Nothing is trained here: it reuses the per-window probabilities that the test runs already saved
(`*_test_proba_*_fold_<f>*.npz`, with the keys `proba` and `y_true`).

Cascade rule, for the RF attack probability p of a window:
    p <  t_low   -> "normal"  (final, the CNN is not run)
    p >= t_high  -> "attack"  (final, the CNN is not run)
    otherwise    -> the CNN decides (its attack probability >= 0.5 means attack)
With t_high above 1 the RF can never confirm an attack alone, so every window that is not discarded
as normal goes to the CNN. With t_low = 0 and t_high > 1 every window goes to the CNN (CNN alone).

IMPORTANT: this script only describes the curve on the TEST set. The thresholds used in the report
must be chosen on the out-of-fold predictions of the training data, never here.

Usage:
    python tools/cascade_sweep.py --rf-dir ~/outputs/<rf test run>/metrics \
        --cnn-dir ~/outputs/<cnn test run>/metrics --cnn-outputs 1 --out cascade_aeid.csv
For the TOW-IDS multiclass CNN use --cnn-outputs 6 (the attack probability is 1 - P(Normal), and the
classes are in alphabetical order, so Normal is column 4).
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
    data = np.load(path)
    proba, y_true = data["proba"], data["y_true"]
    if n_outputs == 1:
        return proba.reshape(-1).astype(np.float64), np.asarray(y_true).reshape(-1).astype(int)
    # Multiclass: attack probability is everything that is not "Normal"
    return 1.0 - proba[:, NORMAL_COLUMN_TOW].astype(np.float64), 1 - np.asarray(y_true)[:, NORMAL_COLUMN_TOW].astype(int)


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

    t_lows = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    t_highs = [0.5, 0.7, 0.9, 0.95, 0.99, 1.01]  # 1.01 = the RF never confirms an attack alone

    rows = []
    for fold in folds:
        p_rf, y_rf = load_rf(rf_files[fold])
        p_cnn, y_cnn = load_cnn(cnn_files[fold], args.cnn_outputs)
        if len(y_rf) != len(y_cnn) or not np.array_equal(y_rf, y_cnn):
            raise SystemExit(f"fold {fold}: RF and CNN do not describe the same windows "
                             f"({len(y_rf)} vs {len(y_cnn)} windows, labels equal: "
                             f"{len(y_rf) == len(y_cnn) and np.array_equal(y_rf, y_cnn)})")
        cnn_attack = (p_cnn >= 0.5).astype(int)
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
    np.savetxt(args.out, np.array(rows), delimiter=",", header=header, comments="", fmt="%.6g")
    print(f"saved {len(rows)} rows to {args.out}")

    # Fold-averaged summary of some reference points
    table = np.array(rows)
    print("\nmean over folds (t_low, t_high -> fraction sent to the CNN, recall, precision, F1, est. time us)")
    for t_low, t_high, label in [(0.0, 1.01, "CNN alone"), (0.5, 0.5, "RF alone"),
                                 (0.1, 1.01, "RF discards p<0.1"), (0.01, 1.01, "RF discards p<0.01"),
                                 (0.1, 0.99, "two thresholds 0.1/0.99")]:
        sel = table[(np.isclose(table[:, 1], t_low)) & (np.isclose(table[:, 2], t_high))]
        if len(sel):
            m = sel.mean(axis=0)
            print(f"{label:>26}: to_cnn={m[3]:.3f} recall={m[10]:.4f} prec={m[9]:.4f} f1={m[11]:.4f} time={m[12]:.0f}")


if __name__ == "__main__":
    main()
