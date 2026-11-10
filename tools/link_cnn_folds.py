"""Link the CNN fold models we trained into retrained/<name>/ so the test configs can find them.

The fold jobs write to separate run folders. The number of outputs of the saved out-of-fold
probabilities (1 = binary AEID/SOME/IP, 6 = TOW-IDS multiclass) tells which dataset a folder belongs to.
If the same fold was trained more than once, the most recent run is used.

AEID and SOME/IP both have 1 output, and tuning runs also train fold 0, so the number of outputs alone
can pick a model of the wrong dataset. The optional third argument keeps only the runs started at or
after a timestamp (the prefix of the run folder names, e.g. 2026_11_11_21_00_00): pass the time the
final training was submitted.

Usage (from the repository root): python tools/link_cnn_folds.py <name> <number_of_outputs> [<not before>]
  e.g. python tools/link_cnn_folds.py aeid_cnn 1
       python tools/link_cnn_folds.py someip_cnn 1 2026_11_11_21_00_00
"""
import glob
import os
import sys

import numpy as np

N_FOLDS = 5

name, n_outputs = sys.argv[1], int(sys.argv[2])
not_before = sys.argv[3] if len(sys.argv) > 3 else ""
outputs_dir = os.environ.get("IDS_ARTIFACTS_DIR", os.path.expanduser("~/outputs"))

found = {}
# Folder names start with a timestamp, so sorted() puts the most recent run last and it overwrites older ones
for oof_path in sorted(glob.glob(f"{outputs_dir}/*_pytorch_train_fold*/metrics/oof_proba_*_fold_*.npz")):
    if np.load(oof_path)["proba"].shape[1] != n_outputs:
        continue
    run_dir = os.path.dirname(os.path.dirname(oof_path))
    # Run folders are named <timestamp>_pytorch_train_fold<n>, so comparing the names compares the start times
    if os.path.basename(run_dir) < not_before:
        continue
    fold = int(oof_path.rsplit("_fold_", 1)[1].split(".")[0])
    # The model is saved after the out-of-fold probabilities, so a fold still running has no model yet
    model_paths = glob.glob(f"{run_dir}/models/pytorch_model_*_{fold}")
    if model_paths:
        found[fold] = model_paths[0]

missing = [fold for fold in range(N_FOLDS) if fold not in found]
if missing:
    sys.exit(f"Folds {missing} are not finished yet for {n_outputs} output(s); nothing was linked")

os.makedirs(f"retrained/{name}", exist_ok=True)
for fold, model_path in sorted(found.items()):
    link = f"retrained/{name}/pytorch_model_PrunedCNNIDS_{fold}"
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(model_path, link)
    print(link, "->", model_path)
