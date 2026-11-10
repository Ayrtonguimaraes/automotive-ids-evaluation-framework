"""Folds used by the train/validation step.

By default this is the paper's protocol: stratified 5-fold cross-validation over the windows
(StratifiedKFold, random_state=1), so the AEID and TOW-IDS results are unchanged.

If the model config has "validation_groups_path" (an npz written by tools/someip_groups.py), the folds
keep whole captures apart instead (StratifiedGroupKFold): every window of a capture lands in the same
fold, and each fold gets captures of every attack folder. Windows overlap by 43 of 44 messages, so
splitting them at random puts near-copies on both sides and makes the validation optimistic.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

N_SPLITS = 5
RANDOM_STATE = 1


def make_folds(y, model_config_dict):
    """Return the list of (train_idx, val_idx) pairs for the labels y (1-D)."""
    n_samples = len(y)
    groups_path = model_config_dict.get("validation_groups_path")

    if not groups_path:
        skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
        return list(skf.split(np.zeros(n_samples), y))

    groups = np.load(groups_path)
    capture, folder = groups["capture"], groups["folder"]
    if len(capture) != n_samples:
        raise ValueError(f"{groups_path} describes {len(capture)} windows, but the data has {n_samples}")

    print(f">> Validation by capture: {len(np.unique(capture))} captures from {groups_path}")
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    folds = list(sgkf.split(np.zeros(n_samples), folder, groups=capture))
    for fold, (_, val_idx) in enumerate(folds):
        print(f">> fold {fold}: validation has {len(np.unique(capture[val_idx]))} captures, "
              f"windows per attack folder {np.bincount(folder[val_idx], minlength=folder.max() + 1).tolist()}")
    return folds
