"""Diagnose why the attack detector (Random Forest) misses CAN Replay windows in TOW-IDS.

Part 1 (test set, our retrained RF): the equivalent of Table 9 of the paper (false negatives per
attack), where the RF probabilities of each class sit, and what lowering the 0.5 threshold buys.
Lowering the threshold is the baseline any new feature has to beat at the same false positive rate.

Part 2 (training set only): how often the sum of a window overflows. The sum of 44 nibbles reaches
660, but the feature generator stores it as uint8, so it is kept modulo 256.

Part 3 (training set only, a first probe): the same forest as the paper (Table 3, TOW-IDS) trained on
different summaries of the 44 x 116 window image:
    sum_wrapped   the paper's feature (sum modulo 256)
    sum_true      the sum without overflow
    changes       per column: the true sum, how many of the 44 rows are not zero, and the largest value
    blocks        the sum of each quarter of the window (4 x 116), which keeps a coarse order in time
    all           changes + blocks
Validation uses contiguous blocks of windows: neighbouring windows share 43 of 44 rows, so a random
split would put near-copies on both sides. Windows closer than one window length to a block edge are
left out of the validation. The test set is not used in parts 2 and 3.

Usage: see jobs/diagnose_can_replay.sbatch
"""
import argparse
import glob
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

WINDOW = 44
COLUMNS = 116
NORMAL = "Normal"
ATTACKS = ["C_D", "C_R", "F_I", "M_F", "P_I"]
# Nibbles of the CAN ID (72-75) and of the DLC and payload (84-101), Section 6.1.1 of the paper
CAN_COLUMNS = list(range(72, 76)) + list(range(84, 102))
PAPER_RF = {"n_estimators": 200, "max_depth": 10, "bootstrap": True, "random_state": 42}  # Table 3, TOW-IDS
N_BLOCKS = 200
CHUNK = 50_000

# A class with no window in a validation fold gives an empty mean (shown as nan); numpy warns on each one
warnings.filterwarnings("ignore", category=RuntimeWarning)


def read_classes(csv_path):
    return pd.read_csv(csv_path)["Class"].astype(str).values


def recall_at_fpr(p, is_normal, mask, fpr):
    """Recall on the windows in `mask` when the threshold lets `fpr` of the normal windows through."""
    threshold = np.quantile(p[is_normal], 1 - fpr)
    return float(np.mean(p[mask] > threshold))


def part1(rf_dir, classes):
    print("\n=== PART 1: retrained RF on the TEST set, per attack (mean of the folds) ===")
    paths = {}
    for path in glob.glob(os.path.join(os.path.expanduser(rf_dir), "*test_proba*RandomForestClassifier*.npz")):
        paths[int(re.search(r"fold_(\d+)", os.path.basename(path)).group(1))] = path
    if not paths:
        raise SystemExit(f"no RF probability files in {rf_dir}")

    is_normal = classes == NORMAL
    table, quant, moves = [], [], []
    for fold in sorted(paths):
        data = np.load(paths[fold])
        p, y = data["proba"][:, 1], np.asarray(data["y_true"]).ravel().astype(int)
        if len(p) != len(classes) or not np.array_equal(y == 0, is_normal):
            raise SystemExit(f"fold {fold}: the RF windows do not match the multiclass labels")
        pred = p >= 0.5
        total_fn = int(np.sum(~pred & ~is_normal))
        table.append([[np.sum(classes == a), np.sum(~pred & (classes == a)), np.sum(pred & (classes == a)), total_fn]
                      for a in ATTACKS] + [[np.sum(is_normal), np.sum(pred & is_normal), np.sum(~pred & is_normal), 0]])
        quant.append([[np.median(p[classes == c]), np.mean(p[classes == c] < 0.1), np.mean(p[classes == c] < 0.3),
                       np.mean(p[classes == c] < 0.5)] for c in ATTACKS + [NORMAL]])
        moves.append([[np.mean(p[is_normal] >= t), np.mean(p[classes == "C_R"] >= t), np.mean(p[~is_normal] >= t),
                       f1_score(y, (p >= t).astype(int))] for t in (0.5, 0.4, 0.3, 0.2, 0.1)]
                     + [[f, recall_at_fpr(p, is_normal, classes == "C_R", f), recall_at_fpr(p, is_normal, ~is_normal, f), np.nan]
                        for f in (0.01, 0.05)])
    table, quant, moves = np.mean(table, axis=0), np.mean(quant, axis=0), np.mean(moves, axis=0)

    print(f"folds: {sorted(paths)}. Paper (Table 9): CAN Replay 62593 false negatives = 82.26% of all")
    print(f"{'class':>7} {'windows':>9} {'missed':>9} {'% of class':>10} {'% of all FN':>11} {'detected':>9}")
    for name, (n, fn, tp, total_fn) in zip(ATTACKS, table[:-1]):
        print(f"{name:>7} {n:9.0f} {fn:9.0f} {100 * fn / n:10.2f} {100 * fn / total_fn:11.2f} {tp:9.0f}")
    n, fp, tn, _ = table[-1]
    print(f"{NORMAL:>7} {n:9.0f}   false positives {fp:.0f} ({100 * fp / n:.2f}%), true negatives {tn:.0f}")

    print("\nRF attack probability p per class")
    print(f"{'class':>7} {'median p':>9} {'p<0.1':>7} {'p<0.3':>7} {'p<0.5':>7}")
    for name, (med, a, b, c) in zip(ATTACKS + [NORMAL], quant):
        print(f"{name:>7} {med:9.3f} {a:7.3f} {b:7.3f} {c:7.3f}")

    print("\nbaseline to beat: the same RF with a lower threshold")
    print(f"{'rule':>12} {'FPR':>7} {'recall C_R':>10} {'recall all':>10} {'F1':>7}")
    for label, (fpr, rec_cr, rec_all, f1) in zip(["p>=0.5", "p>=0.4", "p>=0.3", "p>=0.2", "p>=0.1", "FPR=1%", "FPR=5%"], moves):
        print(f"{label:>12} {fpr:7.4f} {rec_cr:10.4f} {rec_all:10.4f} {f1:7.4f}")


def summaries(img):
    """Per-window summaries of the image, computed in chunks to keep the memory low."""
    n = len(img)
    sum_true = np.empty((n, COLUMNS), dtype=np.uint16)
    nonzero = np.empty((n, COLUMNS), dtype=np.uint8)
    largest = np.empty((n, COLUMNS), dtype=np.uint8)
    blocks = np.empty((n, 4 * COLUMNS), dtype=np.uint8)
    for start in range(0, n, CHUNK):
        chunk = img[start:start + CHUNK]
        sum_true[start:start + CHUNK] = chunk.sum(axis=1, dtype=np.uint16)
        nonzero[start:start + CHUNK] = np.count_nonzero(chunk, axis=1)
        largest[start:start + CHUNK] = chunk.max(axis=1)
        blocks[start:start + CHUNK] = chunk.reshape(len(chunk), 4, WINDOW // 4, COLUMNS).sum(axis=2).reshape(len(chunk), -1)
    return sum_true, nonzero, largest, blocks


def part2(sum_true, classes, x_sum_path):
    print("\n=== PART 2: overflow of the window sum, TRAINING set ===")
    if x_sum_path:
        stored = np.load(x_sum_path)["arr_0"][:20000]
        same = np.array_equal(stored, (sum_true[:20000] % 256).astype(stored.dtype))
        print(f"stored RF feature == true sum modulo 256 (first 20000 windows): {same}")
    over = sum_true > 255
    can = np.zeros(COLUMNS, dtype=bool)
    can[CAN_COLUMNS] = True
    print(f"largest sum: {int(sum_true.max())} (limit without overflow: 255)")
    print(f"{'class':>7} {'windows w/ overflow':>20} {'cells overflowed':>17} {'CAN cells overflowed':>21}")
    for name in ATTACKS + [NORMAL, "all"]:
        mask = np.ones(len(classes), dtype=bool) if name == "all" else classes == name
        print(f"{name:>7} {np.mean(over[mask].any(axis=1)):20.4f} {np.mean(over[mask]):17.4f} {np.mean(over[mask][:, can]):21.4f}")
    worst = np.argsort(-over.mean(axis=0))[:10]
    print("columns that overflow most (column: fraction of windows): "
          + ", ".join(f"{c}{'*' if can[c] else ''}: {over[:, c].mean():.3f}" for c in worst) + "   (* = CAN field)")


def part3(families, classes, n_folds, jobs):
    print("\n=== PART 3: first probe of new summaries, TRAINING set, validation by contiguous blocks ===")
    n = len(classes)
    y = (classes != NORMAL).astype(int)
    block_size = int(np.ceil(n / N_BLOCKS))
    block = np.arange(n) // block_size
    # Windows next to a block edge overlap windows of the neighbouring block, so they are not scored
    position = np.arange(n) % block_size
    interior = (position >= WINDOW) & (position < block_size - WINDOW)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1).split(np.zeros(n), classes, groups=block))[:n_folds]
    print(f"{n} windows in {N_BLOCKS} blocks of {block_size}; {len(folds)} validation folds; forest: {PAPER_RF}")

    header = f"{'features':>12} {'cols':>5} {'max_feat':>8} {'AUC':>7} {'F1':>7} {'FPR':>7} " \
             + " ".join(f"{'rec ' + a:>8}" for a in ATTACKS) + f" {'C_R@1%':>7} {'C_R@5%':>7} {'fit s':>6}"
    print(header)
    for name, X in families.items():
        for max_features in (7, "sqrt"):
            rows = []
            for train_idx, val_idx in folds:
                val_idx = val_idx[interior[val_idx]]
                model = RandomForestClassifier(max_features=max_features, n_jobs=jobs, **PAPER_RF)
                start = time.time()
                model.fit(X[train_idx], y[train_idx])
                seconds = time.time() - start
                p = model.predict_proba(X[val_idx])[:, 1]
                cls, normal = classes[val_idx], classes[val_idx] == NORMAL
                rows.append([roc_auc_score(y[val_idx], p), f1_score(y[val_idx], (p >= 0.5).astype(int)), np.mean(p[normal] >= 0.5)]
                            + [np.mean(p[cls == a] >= 0.5) for a in ATTACKS]
                            + [recall_at_fpr(p, normal, cls == "C_R", 0.01), recall_at_fpr(p, normal, cls == "C_R", 0.05), seconds])
            m = np.mean(rows, axis=0)
            print(f"{name:>12} {X.shape[1]:5d} {str(max_features):>8} {m[0]:7.4f} {m[1]:7.4f} {m[2]:7.4f} "
                  + " ".join(f"{v:8.4f}" for v in m[3:8]) + f" {m[8]:7.4f} {m[9]:7.4f} {m[10]:6.0f}")
    print("rec X = recall of class X at p>=0.5; C_R@1% = recall of CAN Replay when 1% of the normal windows raise an alarm")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rf-test-dir", help="metrics folder of the RF test run (part 1)")
    parser.add_argument("--y-test", help="multiclass labels of the test windows, csv (part 1)")
    parser.add_argument("--x-train-img", required=True, help="training window images, npz")
    parser.add_argument("--y-train", required=True, help="multiclass labels of the training windows, csv")
    parser.add_argument("--x-train-sum", help="stored RF training feature, npz (optional check)")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = parser.parse_args()

    if args.rf_test_dir and args.y_test:
        part1(args.rf_test_dir, read_classes(args.y_test))

    classes = read_classes(args.y_train)
    img = np.load(args.x_train_img)["arr_0"].reshape(-1, WINDOW, COLUMNS)
    if len(img) != len(classes):
        raise SystemExit(f"{len(img)} training images but {len(classes)} labels")
    print(f"\ntraining windows per class: {dict(zip(*np.unique(classes, return_counts=True)))}")
    sum_true, nonzero, largest, blocks = summaries(img)
    del img

    part2(sum_true, classes, args.x_train_sum)

    changes = np.hstack([sum_true, nonzero, largest])
    families = {
        "sum_wrapped": (sum_true % 256).astype(np.uint8),
        "sum_true": sum_true,
        "changes": changes,
        "blocks": blocks,
        "all": np.hstack([changes, blocks]),
    }
    part3(families, classes, args.folds, args.jobs)


if __name__ == "__main__":
    main()
