"""Write, for every SOME/IP window, the capture it came from and the attack folder of that capture.

The feature generator concatenates the windows of all captures without recording where each one came
from. This script walks the captures in the same order as the generator
(feature_generator/cnn_ids_feature_generator.py, __someip_dataset_generate_features) and counts the
windows of each one, so a validation split can keep whole captures apart (overlapping windows of the
same capture would otherwise fall on both sides of the split and make the validation optimistic).

Windows per capture: N messages give N - 1 difference rows, and the generator stops when a window of
44 rows would reach the last row, so a capture yields max(0, N - 45) windows (window 44, slide 1).
The script checks that the total matches the saved labels before writing anything.

Usage (from the repository root, on the cluster):
    python tools/someip_groups.py --split train \
        --dataset data/someip/SOME_IP_Intrusion_Detection_Dataset \
        --labels data/someip/processed/y_train_SOMEIP_dataset_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_True.npz \
        --out data/someip/processed/groups_train.npz
"""
import argparse
import glob
import importlib.util
import os
import re

import numpy as np

# Load feature_generator/safe_pickle.py by path, so the heavy feature generator package is not imported
_spec = importlib.util.spec_from_file_location(
    "safe_pickle", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feature_generator", "safe_pickle.py"))
_safe_pickle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_safe_pickle)

WINDOW_SIZE = 44


def capture_index(x_path):
    # Same rule as the generator: captures are ordered by the number in <name>_<i>_x.pickle
    return int(re.search(r'_(\d+)_x\.pickle$', x_path).group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--dataset", required=True, help="folder with one subfolder per attack")
    parser.add_argument("--labels", required=True, help="y_*.npz written by the feature generator for this split")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="must match the feature generator config")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    captures, folders, names = [], [], []
    capture_id = 0
    for folder_id, attack_dir in enumerate(sorted(glob.glob(f"{args.dataset}/*/"))):
        x_paths = sorted(glob.glob(f"{attack_dir}*_x.pickle"), key=capture_index)
        n_train = int(len(x_paths) * args.train_fraction)
        selected = x_paths[:n_train] if args.split == "train" else x_paths[n_train:]

        folder_windows = 0
        for x_path in selected:
            n_messages = len(np.asarray(_safe_pickle.safe_load(x_path.replace('_x.pickle', '_y.pickle'))).ravel())
            n_windows = max(0, n_messages - 1 - WINDOW_SIZE)
            captures.append(np.full(n_windows, capture_id, dtype=np.int32))
            folders.append(np.full(n_windows, folder_id, dtype=np.int16))
            names.append(os.path.relpath(x_path, args.dataset))
            folder_windows += n_windows
            capture_id += 1
        print(f"{os.path.basename(os.path.normpath(attack_dir))}: {len(selected)} captures, {folder_windows} windows")

    capture = np.concatenate(captures)
    folder = np.concatenate(folders)
    labels = np.load(args.labels)["arr_0"].ravel()
    if len(capture) != len(labels):
        raise SystemExit(f"window count mismatch: {len(capture)} counted here, {len(labels)} in {args.labels}. "
                         "Nothing was written.")

    np.savez(args.out, capture=capture, folder=folder, names=np.array(names))
    print(f"{args.split}: {capture_id} captures, {len(capture)} windows (matches the labels). Saved to {args.out}")


if __name__ == "__main__":
    main()
