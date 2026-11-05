"""Compare the metrics saved by the multiclass CNN test with metrics computed from the class predictions.

The test code gives the torchmetrics multiclass metrics a one-hot target of shape (N, 6). Those metrics
expect one class index per window, so this script runs both versions on the saved probabilities.

Usage: python tools/check_multiclass_metrics.py <metrics folder of the multiclass CNN test run>
"""
import glob
import os
import re
import sys

import numpy as np
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

CLASSES = ["C_D", "C_R", "F_I", "M_F", "Normal", "P_I"]  # OneHotEncoder sorts them alphabetically
NORMAL = CLASSES.index("Normal")
BATCH = 64


def metrics_as_in_the_test_code(proba, one_hot):
    """Same calls as model_test/pytorch_model_test.py: float outputs and a one-hot target, batch by batch."""
    metrics = {name: cls(num_classes=6) for name, cls in [("acc", MulticlassAccuracy), ("prec", MulticlassPrecision),
                                                          ("recall", MulticlassRecall), ("f1", MulticlassF1Score)]}
    output, target = torch.tensor(proba).float(), torch.tensor(one_hot).float()
    for start in range(0, len(output), BATCH):
        for metric in metrics.values():
            metric.update(output[start:start + BATCH], target[start:start + BATCH])
    return {name: float(metric.compute()) for name, metric in metrics.items()}


def main(directory):
    paths = {int(re.search(r"fold_(\d+)", p).group(1)): p for p in glob.glob(os.path.join(os.path.expanduser(directory), "test_proba*.npz"))}
    for fold, path in sorted(paths.items()):
        data = np.load(path)
        proba, one_hot = data["proba"], data["y_true"]
        y, pred = one_hot.argmax(axis=1), proba.argmax(axis=1)
        matrix = np.zeros((6, 6), dtype=np.int64)
        np.add.at(matrix, (y, pred), 1)

        tp = np.diag(matrix).astype(float)
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        recall = tp / support
        precision = np.divide(tp, predicted, out=np.zeros(6), where=predicted > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=precision + recall > 0)

        top_two = np.sort(proba, axis=1)[:, -2:]
        print(f"\nfold {fold}: {len(y)} windows, output range {proba.min():.4f} to {proba.max():.4f}, "
              f"mean row sum {proba.sum(axis=1).mean():.3f}")
        print("  from the class predictions: "
              f"accuracy={tp.sum() / len(y):.4f} macro_f1={f1.mean():.4f} weighted_f1={(f1 * support).sum() / len(y):.4f}")
        print("  as in the test code:       ", {k: round(v, 4) for k, v in metrics_as_in_the_test_code(proba, one_hot).items()})
        print("  recall per class:          ", {c: round(float(r), 4) for c, r in zip(CLASSES, recall)})
        print(f"  windows with a tie in the two largest outputs: {int(np.sum(top_two[:, 0] == top_two[:, 1]))}")
        is_replay = y == CLASSES.index("C_R")
        print("  mean outputs on real C_R:  ", {c: round(float(v), 3) for c, v in zip(CLASSES, proba[is_replay].mean(axis=0))})
        print(f"  binary decisions that differ (largest output vs Normal output below 0.5): "
              f"{int(np.sum((pred != NORMAL) != (proba[:, NORMAL] < 0.5)))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
