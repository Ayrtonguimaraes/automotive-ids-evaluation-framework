"""Write the configs of the SOME/IP CSV window search into config_jsons/window_search/.

The search compares two ways of building the windows, each with four window sizes (number of
differences between messages; 44 is the paper's value), decided before running:
  * arrival: the paper's method, windows over the messages in capture order;
  * conversation: windows inside each client-server conversation (improvement proposed after the
    exploratory analysis, where a window of 45 messages held 14 conversations).
The RF and CNN hyperparameters stay at the values tuned for the SOME/IP CSV with the paper's window
(RF 50 trees, depth 7, 10 features; CNN learning rate 0.0001, batch 256), so the only change is the window.
Only the training files are used; the test configs are written later, for the selected windows only.

Usage (repository root): python tools/make_window_search_configs.py
"""
import json
import os

GROUPINGS = ["arrival", "conversation"]
WINDOWS = [10, 20, 30, 44]
OUT = "config_jsons/window_search"


def variants():
    return [(g, w) for g in GROUPINGS for w in WINDOWS]


def name(grouping, window):
    return f"{grouping}_w{window}"


def feat_config(grouping, window, sum_x, split="train"):
    return {
        "window_size": window, "window_slide": 1, "number_of_bytes": 58, "multiclass": False,
        "labeling_schema": "SOMEIP_dataset", "dataset": "SOMEIP_CSV_dataset",
        "window_grouping": grouping, "split": split, "suffix": split, "sum_x": sum_x,
    }


def data_dir(grouping, window):
    return f"data/someip_csv/window_search/{name(grouping, window)}"


def features_file(kind, grouping, window, sum_x, split="train"):
    return (f"{data_dir(grouping, window)}/{kind}_{split}_SOMEIP_dataset_Wsize_{window}_Cols_116_Wslide_1"
            f"_MC_False_sumX_{sum_x}.npz")


def main():
    os.makedirs(OUT, exist_ok=True)
    for grouping, window in variants():
        n = name(grouping, window)
        for sum_x, kind in [(False, "img"), (True, "sumX")]:
            with open(f"{OUT}/{n}_feat_train_{kind}.json", "w") as fh:
                json.dump({"feature_generator": "CNNIDSFeatureGenerator",
                           "config": feat_config(grouping, window, sum_x),
                           "paths": {"dataset_path": "data/someip_csv/Data", "output_path": data_dir(grouping, window)}},
                          fh, indent=4)
        cnn = {
            "feat_gen": {"feature_generator": "CNNIDSFeatureGenerator",
                         "config": {k: v for k, v in feat_config(grouping, window, False).items()
                                    if k not in ("split", "suffix")},
                         "load_paths": {"X_path": features_file("X", grouping, window, False),
                                        "y_path": features_file("y", grouping, window, False)}},
            "model_specs": {"framework": "pytorch", "model": "PrunedCNNIDS", "model_name": f"PrunedCNNIDS_{n}",
                            "criterion": "binary-cross-entropy",
                            "hyperparameters": {"batch_size": 256, "learning_rate": 0.0001, "num_epochs": 30,
                                                "early_stopping_patience": 5, "num_outputs": 1},
                            "validation_groups_path": f"{data_dir(grouping, window)}/groups_train.npz"},
        }
        with open(f"{OUT}/{n}_cnn.json", "w") as fh:
            json.dump(cnn, fh, indent=4)
    print(f"wrote {len(variants()) * 3} configs in {OUT}")


if __name__ == "__main__":
    main()
