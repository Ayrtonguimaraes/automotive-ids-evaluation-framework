"""Folds used by the train/validation step.

By default this is the paper's protocol: stratified 5-fold cross-validation over the windows
(StratifiedKFold, random_state=1), so the AEID and TOW-IDS results are unchanged.

If the model config has "validation_groups_path", the folds keep overlapping windows apart. Windows
overlap by 43 of 44 rows, so splitting them at random puts near-copies on both sides and makes the
validation optimistic. Two kinds of groups file are understood:

  * capture groups (keys "capture" and "folder", written by tools/someip_groups.py for the SOME/IP
    pickles): whole captures go to one fold (StratifiedGroupKFold), and each fold gets captures of
    every attack folder;
  * block groups (keys "file_id" and "pos", written by the SOME/IP CSV feature generator, where there is
    one file per attack and so no capture to hold out): each file is cut into N_SPLITS contiguous
    blocks, fold k validates on block k of every file, and the training windows within PURGE
    positions of the validation block are dropped (purged k-fold), so no training window shares a
    message with a validation window.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

N_SPLITS = 5
RANDOM_STATE = 1
# Windows i and j share a message when |i - j| <= 44 (a window spans 45 messages)
PURGE = 45


def purged_block_folds(file_ids, pos):
    folds = [([], []) for _ in range(N_SPLITS)]
    for f in np.unique(file_ids):
        idx = np.flatnonzero(file_ids == f)
        idx = idx[np.argsort(pos[idx])]
        blocks = np.array_split(idx, N_SPLITS)
        for k, block in enumerate(blocks):
            lo, hi = pos[block].min(), pos[block].max()
            keep = (pos[idx] < lo - PURGE + 1) | (pos[idx] > hi + PURGE - 1)
            folds[k][0].append(idx[keep])
            folds[k][1].append(block)
    return [(np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))) for tr, va in folds]


def make_folds(y, model_config_dict):
    """Return the list of (train_idx, val_idx) pairs for the labels y (1-D)."""
    n_samples = len(y)
    groups_path = model_config_dict.get("validation_groups_path")

    if not groups_path:
        skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
        return list(skf.split(np.zeros(n_samples), y))

    groups = np.load(groups_path)

    if "pos" in groups.files:
        file_ids, pos = groups["file_id"], groups["pos"]
        if len(file_ids) != n_samples:
            raise ValueError(f"{groups_path} describes {len(file_ids)} windows, but the data has {n_samples}")
        folds = purged_block_folds(file_ids, pos)
        y = np.asarray(y).ravel()
        print(f">> Validation by purged blocks: {len(np.unique(file_ids))} files from {groups_path}")
        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f">> fold {fold}: train {len(train_idx)} windows ({y[train_idx].mean():.3f} attack), "
                  f"validation {len(val_idx)} ({y[val_idx].mean():.3f} attack), "
                  f"{n_samples - len(train_idx) - len(val_idx)} purged")
        return folds

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
