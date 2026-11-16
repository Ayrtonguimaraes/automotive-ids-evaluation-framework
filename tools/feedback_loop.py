"""Feedback loop from the attack classifier (CNN) to the attack detector (RF), TOW-IDS, offline version.

Future work of the paper (Section 7): "in cases where the attack detector stage makes a wrong detection and
the attack classifier a successful detection, the wrong detection information could be used to improve the
attack detector stage model parameters". The paper suggests online learning; the scikit-learn Random Forest
cannot be updated online, so this is the batch version: the RF is trained again, with a larger weight on the
training windows where the RF was wrong and the CNN was right.

Where the errors come from: the out-of-fold predictions saved by the training runs (each training window was
predicted by an RF and a CNN that did not train on it). The test set is not used to build the weights.

Decided before running:
  * RF wrong: RF attack probability >= 0.5 disagrees with the binary label.
  * CNN right: the largest of the 6 outputs is "attack" (any class but Normal) exactly when the window is an
    attack, as in the paper's confusion matrix. Windows where the two largest outputs tie are NOT counted
    as CNN right (fold 4 of our CNN gives all-zero outputs on CAN Replay and wins only by tie-breaking).
  * feedback_wX: weight X on those windows (X in 2, 5, 10), 1 elsewhere.
  * control_wX: the same extra weight per binary class, spread evenly over every window of that class,
    without looking at the CNN. If feedback_wX does not beat control_wX, the gain is "more weight on
    attacks", not information from the CNN.
  * Same RF as the paper for TOW-IDS (200 trees, depth 10, 7 features), same 5 folds.
  * Selection: highest mean validation F1 (threshold 0.5, the paper's metric) among feedback_wX; ties
    within 0.002 go to the smaller weight. The selected feedback variant and its control are tested once.

Subcommands:
  weights   --rf-run DIR --cnn-runs DIR x5 --out DIR       weights files (on the cluster)
  configs                                                  training and test configs (once, committed)
  summarize --weights-dir DIR [--outputs DIR]              validation table and selection
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

WEIGHTS = [2, 5, 10]
VARIANTS = [f"feedback_w{x}" for x in WEIGHTS] + [f"control_w{x}" for x in WEIGHTS]
# Second control, added after the test of feedback_w5 (declared in the report): weight 5 on EVERY window the
# RF got wrong, whether or not the CNN was right. It tells "the CNN's information" apart from plain
# reweighting of the RF's own errors (boosting-like). Not part of the selection.
EXTRA_VARIANTS = ["errors_w5"]
CLASSES = ["C_D", "C_R", "F_I", "M_F", "Normal", "P_I"]   # order of the one-hot encoder (alphabetical)
NORMAL = 4
TIE = 0.002
CONFIGS = "config_jsons/feedback"
WEIGHTS_DIR = "data/tow/feedback"


def by_index(paths, n=None):
    parts = [np.load(p) for p in paths]
    n = n or sum(len(p["index"]) for p in parts)
    y, proba = None, None
    for p in parts:
        idx, yy, pp = p["index"], np.asarray(p["y_true"]), p["proba"]
        if y is None:
            y = np.full(n, -1, dtype=np.int64)
            proba = np.full((n,) + pp.shape[1:], np.nan)
        y[idx] = yy.argmax(axis=1) if yy.ndim == 2 else yy.ravel()
        proba[idx] = pp
    if (y < 0).any():
        raise SystemExit(f"{(y < 0).sum()} windows have no out-of-fold prediction")
    return y, proba


def weights(args):
    rf_y, rf_p = by_index(sorted(glob.glob(f"{args.rf_run}/metrics/oof_proba_*_fold_*.npz")))
    cnn_paths = [glob.glob(f"{run}/metrics/oof_proba_*_fold_*.npz")[0] for run in args.cnn_runs]
    cls, cnn_p = by_index(cnn_paths, len(rf_y))
    if not np.array_equal(rf_y == 0, cls == NORMAL):
        raise SystemExit("the RF binary labels and the CNN classes do not describe the same windows")

    rf_wrong = (rf_p[:, 1] >= 0.5) != (rf_y == 1)
    top2 = np.sort(cnn_p, axis=1)[:, -2:]
    tie = np.isclose(top2[:, 0], top2[:, 1], atol=1e-6)
    cnn_attack = cnn_p.argmax(axis=1) != NORMAL
    cnn_right = (cnn_attack == (rf_y == 1)) & ~tie
    fb = rf_wrong & cnn_right

    print(f"{len(rf_y)} training windows; RF wrong {rf_wrong.sum()}, CNN right {cnn_right.sum()}, "
          f"CNN ties {tie.sum()}, RF wrong and CNN right (feedback set) {fb.sum()}")
    print(f"{'class':>7} {'windows':>9} {'RF wrong':>9} {'CNN ties':>9} {'feedback':>9}")
    for c, name in enumerate(CLASSES):
        m = cls == c
        print(f"{name:>7} {m.sum():9d} {rf_wrong[m].sum():9d} {tie[m].sum():9d} {fb[m].sum():9d}")

    os.makedirs(args.out, exist_ok=True)
    np.savez(f"{args.out}/windows.npz", cls=cls, rf_wrong=rf_wrong, cnn_right=cnn_right, tie=tie, feedback=fb)
    for x in WEIGHTS:
        w = np.where(fb, float(x), 1.0)
        np.savez(f"{args.out}/weights_feedback_w{x}.npz", weight=w)
        control = np.ones(len(w))
        for label in (0, 1):
            m = rf_y == label
            control[m] += (x - 1) * fb[m].sum() / m.sum()
        np.savez(f"{args.out}/weights_control_w{x}.npz", weight=control)
        if x == 5:
            np.savez(f"{args.out}/weights_errors_w{x}.npz", weight=np.where(rf_wrong, float(x), 1.0))
        print(f"w={x}: feedback total weight {w.sum():.0f}, control total weight {control.sum():.0f} "
              f"(control weight normal {control[rf_y == 0][0]:.4f}, attack {control[rf_y == 1][0]:.4f})")


def configs(_):
    train = json.load(open("config_jsons/model_train_validate/TOW_RandomForest.json"))
    test = json.load(open("config_jsons/model_test/TOW_RandomForest_test.json"))
    os.makedirs(CONFIGS, exist_ok=True)
    for v in VARIANTS + EXTRA_VARIANTS:
        t = json.loads(json.dumps(train))
        t["model_specs"]["sample_weights_path"] = f"{WEIGHTS_DIR}/weights_{v}.npz"
        json.dump(t, open(f"{CONFIGS}/TOW_RandomForest_{v}.json", "w"), indent=4)
        s = json.loads(json.dumps(test))
        s["model_specs"]["presaved_paths"] = {str(f): f"retrained/tow_rf_{v}/RandomForestClassifier_fold_{f}.pkl"
                                              for f in range(5)}
        json.dump(s, open(f"{CONFIGS}/TOW_RandomForest_{v}_test.json", "w"), indent=4)
    print(f"wrote {2 * len(VARIANTS + EXTRA_VARIANTS)} configs in {CONFIGS}")


def val_row(paths, cls):
    rows = []
    for p in sorted(paths):
        d = np.load(p)
        idx, y, pr = d["index"], np.asarray(d["y_true"]).ravel(), d["proba"][:, 1]
        pred = pr >= 0.5
        c = cls[idx]
        rate = lambda m: pred[m].mean() if m.any() else np.nan
        rows.append([roc_auc_score(y, pr), f1_score(y, pred), rate(y == 1), rate(y == 0), rate(c == 1), rate(c == 2)])
    return np.array(rows)


def summarize(args):
    info = np.load(f"{args.weights_dir}/windows.npz")
    cls = info["cls"]
    runs = {"baseline (paper)": args.baseline_run}
    for v in VARIANTS + EXTRA_VARIANTS:
        found = sorted(glob.glob(f"{args.outputs}/*_sklearn_train_val_{v}"))
        if found:
            runs[v] = found[-1]
    print(f"validation, mean ± std over the 5 folds (threshold 0.5)")
    print(f"{'variant':>17} {'AUC':>15} {'F1':>15} {'recall':>7} {'FPR':>7} {'C_R rec':>8} {'F_I rec':>8}")
    means = {}
    for name, run in runs.items():
        paths = glob.glob(f"{run}/metrics/oof_proba_*_fold_*.npz")
        if len(paths) != 5:
            print(f"{name:>17} unfinished ({len(paths)} folds)")
            continue
        r = val_row(paths, cls)
        m, s = r.mean(0), r.std(0)
        means[name] = m
        print(f"{name:>17} {m[0]:.4f}±{s[0]:.4f} {m[1]:.4f}±{s[1]:.4f} {m[2]:7.4f} {m[3]:7.4f} {m[4]:8.4f} {m[5]:8.4f}")
    fb = {v: means[v] for v in VARIANTS if v.startswith("feedback") and v in means}
    if len(fb) == len(WEIGHTS):
        best = max(m[1] for m in fb.values())
        pick = min((v for v, m in fb.items() if m[1] >= best - TIE), key=lambda v: int(v.split("_w")[1]))
        control = pick.replace("feedback", "control")
        print(f"\nselected: {pick} (validation F1 {fb[pick][1]:.4f}); test it once together with {control}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("weights")
    p.add_argument("--rf-run", required=True)
    p.add_argument("--cnn-runs", nargs=5, required=True)
    p.add_argument("--out", default=WEIGHTS_DIR)
    sub.add_parser("configs")
    p = sub.add_parser("summarize")
    p.add_argument("--weights-dir", default=WEIGHTS_DIR)
    p.add_argument("--baseline-run", required=True, help="the paper's RF training run (out-of-fold predictions)")
    p.add_argument("--outputs", default=os.environ.get("IDS_ARTIFACTS_DIR", os.path.expanduser("~/outputs")))
    args = parser.parse_args()
    {"weights": weights, "configs": configs, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
