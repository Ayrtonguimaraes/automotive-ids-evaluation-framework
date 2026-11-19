"""Compare the paper's window (arrival order, 44) and the per-conversation window (10) on the SAME messages.

The two windowings do not give the same windows nor the same window labels (89% attack windows with the
paper's window, 42% per conversation), so their window-level AUCs answer slightly different questions.
Here every test message gets one score: the highest attack probability among the windows whose labels
include it (the windows that would raise an alarm for it). A message that no window covers gets no
alarm (score 0). Each message is then scored against its own label, so both windowings are judged on
the same messages and the same labels. Nothing here is selected or tuned: it only re-reads the test
probabilities already saved by the one-time tests.

The same with the mean of the windows instead of the max is printed as a check that the
conclusion does not depend on that choice. Also prints a 95% bootstrap interval of each AUC and of the difference (per conversation - arrival),
resampling whole conversations inside each test file (messages of one conversation are not independent).

Usage (repository root, on the cluster; see jobs/message_level_eval.sbatch):
    python tools/message_level_eval.py
The test run folders are found automatically (newest run of each kind whose number of test windows
matches), or can be given with --arrival-rf, --arrival-cnn, --conversation-rf, --conversation-cnn.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from feature_generator import someip_csv  # noqa: E402

ARMS = {"arrival": 44, "conversation": 10}
B = 1000


def windows(csv_paths, grouping, w):
    """Message ids (global, over all files) whose labels make each window, in the order of the features."""
    covers, labels, groups, offset = [], [], [], 0
    for file_index, path in enumerate(csv_paths):
        _, lab = someip_csv.read_capture(path)
        keys = someip_csv.conversation_keys(path)
        labels.append(lab)
        groups.append(np.array([f"{file_index}:{k}" for k in keys]))
        seqs = [np.arange(len(lab))] if grouping == "arrival" else [np.flatnonzero(keys == k) for k in dict.fromkeys(keys)]
        for rows in seqs:
            # Same count as the authors' aggregation: len - 1 - w windows, window s uses the labels s..s+w-1
            for s in range(max(len(rows) - 1 - w, 0)):
                covers.append(offset + rows[s:s + w])
        offset += len(lab)
    return covers, np.concatenate(labels).astype(int), np.concatenate(groups)


def load_proba(folder):
    """Attack probability and label of each test window, one pair per fold model, from one test run."""
    paths = sorted(glob.glob(f"{folder}/*test_proba*.npz"), key=lambda p: int(re.search(r"fold_(\d+)", p).group(1)))
    out = []
    for p in paths:
        d = np.load(p)
        proba = d["proba"]
        proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()
        out.append((proba, np.asarray(d["y_true"]).ravel().astype(int)))
    return out


def find_run(outputs, kind, n_windows):
    """Latest test run of the given kind whose test set has exactly n_windows windows."""
    for run in sorted(glob.glob(f"{outputs}/*_{kind}"), reverse=True):
        files = glob.glob(f"{run}/metrics/*test_proba*fold_0*.npz")
        if files and len(np.load(files[0])["y_true"]) == n_windows:
            return f"{run}/metrics"
    raise SystemExit(f"no {kind} run with {n_windows} test windows in {outputs}")


def message_scores(covers, proba, n_messages, how="max"):
    """max: the alarm a message gets (any window over it fires); mean: average over its windows (sensitivity check)"""
    ids = np.concatenate(covers)
    values = np.repeat(proba, [len(c) for c in covers])
    if how == "max":
        score = np.zeros(n_messages)
        np.maximum.at(score, ids, values)
        return score
    total, count = np.zeros(n_messages), np.zeros(n_messages)
    np.add.at(total, ids, values)
    np.add.at(count, ids, 1)
    return np.divide(total, count, out=np.zeros(n_messages), where=count > 0)


def summary(y, s):
    """AUC, recall, specificity and balanced accuracy of message scores (threshold 0.5)."""
    pred = s >= 0.5
    rec, spec = pred[y == 1].mean(), 1 - pred[y == 0].mean()
    return roc_auc_score(y, s), rec, spec, (rec + spec) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/someip_csv/Data/Test")
    for arm in ARMS:
        for model in ("rf", "cnn"):
            parser.add_argument(f"--{arm}-{model}")
    args = parser.parse_args()
    outputs = os.environ.get("IDS_ARTIFACTS_DIR", os.path.expanduser("~/outputs"))
    csv_paths = sorted(glob.glob(f"{args.data}/*.csv"))

    covers, scores = {}, {}
    for arm, w in ARMS.items():
        covers[arm], y, groups = windows(csv_paths, arm, w)
        window_y = np.array([int(y[c].any()) for c in covers[arm]])
        covered = np.zeros(len(y), bool)
        covered[np.concatenate(covers[arm])] = True
        print(f"{arm} (window {w}): {len(covers[arm])} test windows, {covered.sum()}/{len(y)} messages covered "
              f"({(~covered & (y == 1)).sum()} of {y.sum()} attack messages without any window)")
        for model, kind in (("rf", "sklearn_test"), ("cnn", "pytorch_test")):
            folder = getattr(args, f"{arm}_{model}") or find_run(outputs, kind, len(covers[arm]))
            print(f"  {model}: {folder}")
            folds = load_proba(folder)
            # The saved window labels must be the ones rebuilt from the CSV: checks the window order
            for proba, yt in folds:
                if not np.array_equal(yt, window_y):
                    raise SystemExit(f"{folder}: saved window labels differ from the CSV, windows are not aligned")
            scores[(arm, model)] = [message_scores(covers[arm], p, len(y)) for p, _ in folds]
            scores[(arm, model, "mean")] = [message_scores(covers[arm], p, len(y), "mean") for p, _ in folds]

    print(f"\nmessage level, {len(y)} test messages ({y.mean():.1%} attack); mean ± std over the 5 fold models; "
          f"threshold 0.5; uncovered messages count as 'no alarm'")
    for how in ("max", "mean"):
        print(f" message score = {how} of its windows{' (main)' if how == 'max' else ' (sensitivity check)'}")
        for model in ("rf", "cnn"):
            for arm in ARMS:
                key = (arm, model) if how == "max" else (arm, model, "mean")
                rows = np.array([summary(y, s) for s in scores[key]])
                m, sd = rows.mean(0), rows.std(0)
                print(f"  {model.upper():>3} {arm:>12}: AUC {m[0]:.4f}±{sd[0]:.4f}  recall {m[1]:.4f}  "
                      f"specificity {m[2]:.4f}  balanced acc {m[3]:.4f}")

    # Bootstrap over conversations, inside each file
    rng = np.random.default_rng(0)
    files = np.array([g.split(":")[0] for g in groups])
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    by_file = {f: [g for g in members if g.split(":")[0] == f] for f in np.unique(files)}
    print(f"\n95% bootstrap intervals ({B} resamples of the {len(members)} test conversations, inside each file)")
    for model in ("rf", "cnn"):
        boot = {"arrival": [], "conversation": [], "diff": []}
        for _ in range(B):
            idx = np.concatenate([members[g] for f in by_file for g in rng.choice(by_file[f], len(by_file[f]))])
            if y[idx].min() == y[idx].max():
                continue
            a = np.mean([roc_auc_score(y[idx], s[idx]) for s in scores[("arrival", model)]])
            c = np.mean([roc_auc_score(y[idx], s[idx]) for s in scores[("conversation", model)]])
            boot["arrival"].append(a); boot["conversation"].append(c); boot["diff"].append(c - a)
        for key, vals in boot.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            print(f"  {model.upper():>3} {key:>12}: AUC{' difference' if key == 'diff' else ''} "
                  f"{np.mean(vals):+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
