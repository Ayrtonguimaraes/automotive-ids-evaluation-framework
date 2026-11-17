"""Why does the Frame Injection (F_I) recall of the TOW-IDS RF drop after the feedback-loop retraining?

Reads only saved predictions; trains nothing and selects nothing.
  Part 1 (test): where the RF attack probability of each class sits, for the paper's RF and the retrained RF.
          A small shift across 0.5 (median just below the threshold) is a different story from a collapse.
  Part 2 (test): threshold sweep of the retrained RF: F_I recall, C_R recall and false positive rate.
          Shows whether a threshold other than 0.5 would keep both attacks (diagnosis only, not a choice).
  Part 3 (training, out-of-fold): how the F_I windows split between "RF already right", "in the feedback
          set" (weight 5) and "RF wrong but not in the set", and the same for the normal windows; and how
          the validation prediction of each group changed between the paper's RF and the retrained RF.
The causal test is the pair of ablations of tools/feedback_loop.py (fbattacks_w5, fbnormals_w5).

Usage: python tools/diagnose_frame_injection.py --base-test DIR --feedback-test DIR --y-test CSV \
           --weights-dir data/tow/feedback --base-run DIR --feedback-run DIR
"""
import argparse
import glob
import re

import numpy as np
import pandas as pd

CLASSES = ["C_D", "C_R", "F_I", "M_F", "Normal", "P_I"]


def test_proba(folder):
    paths = sorted(glob.glob(f"{folder}/*test_proba*.npz"), key=lambda p: int(re.search(r"fold_(\d+)", p).group(1)))
    return np.mean([np.load(p)["proba"][:, 1] for p in paths], axis=0), [np.load(p)["proba"][:, 1] for p in paths]


def oof(run, n):
    p = np.full(n, np.nan)
    for path in glob.glob(f"{run}/metrics/oof_proba_*_fold_*.npz"):
        d = np.load(path)
        p[d["index"]] = d["proba"][:, 1]
    return p


def main():
    a = argparse.ArgumentParser()
    for k in ("base-test", "feedback-test", "y-test", "weights-dir", "base-run", "feedback-run"):
        a.add_argument(f"--{k}", required=True)
    args = a.parse_args()
    cls = pd.read_csv(args.y_test)["Class"].astype(str).values
    (_, base_folds), (_, fb_folds) = test_proba(args.base_test), test_proba(args.feedback_test)

    print("=== PART 1 (test): RF attack probability per class, mean of the 5 fold models ===")
    print(f"{'class':>7} | {'paper: median':>13} {'p<0.5':>6} {'0.3<=p<0.5':>10} | {'retrained: median':>17} {'p<0.5':>6} {'0.3<=p<0.5':>10}")
    for c in CLASSES:
        m = cls == c
        row = []
        for folds in (base_folds, fb_folds):
            row += [np.mean([np.median(p[m]) for p in folds]), np.mean([(p[m] < 0.5).mean() for p in folds]),
                    np.mean([((p[m] >= 0.3) & (p[m] < 0.5)).mean() for p in folds])]
        print(f"{c:>7} | {row[0]:13.3f} {row[1]:6.3f} {row[2]:10.3f} | {row[3]:17.3f} {row[4]:6.3f} {row[5]:10.3f}")

    print("\n=== PART 2 (test): threshold sweep of the retrained RF (diagnosis only) ===")
    print(f"{'threshold':>9} {'F_I recall':>10} {'C_R recall':>10} {'FPR':>7} {'all attacks':>11}")
    for t in (0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2):
        r = [[(p[cls == "F_I"] >= t).mean(), (p[cls == "C_R"] >= t).mean(), (p[cls == "Normal"] >= t).mean(),
              (p[cls != "Normal"] >= t).mean()] for p in fb_folds]
        print(f"{t:9.2f} " + " ".join(f"{v:{w}.4f}" for v, w in zip(np.mean(r, axis=0), (10, 10, 7, 11))))

    print("\n=== PART 3 (training, out-of-fold): who was reweighted and how their validation prediction moved ===")
    info = np.load(f"{args.weights_dir}/windows.npz")
    c, fb, wrong = info["cls"], info["feedback"], info["rf_wrong"]
    p0, p1 = oof(args.base_run, len(c)), oof(args.feedback_run, len(c))
    print(f"{'class':>7} {'group':>26} {'windows':>9} {'paper p':>8} {'retr. p':>8} {'paper det.':>10} {'retr. det.':>10}")
    for k in (2, 4, 1):
        for name, m in (("RF right (weight 1)", ~wrong), ("feedback set (weight 5)", fb), ("RF wrong, not in set", wrong & ~fb)):
            g = (c == k) & m
            if g.sum():
                print(f"{CLASSES[k]:>7} {name:>26} {g.sum():9d} {np.median(p0[g]):8.3f} {np.median(p1[g]):8.3f} "
                      f"{(p0[g] >= 0.5).mean():10.3f} {(p1[g] >= 0.5).mean():10.3f}")
    print("'det.' = share predicted as attack. For Normal, lower is better.")


if __name__ == "__main__":
    main()
