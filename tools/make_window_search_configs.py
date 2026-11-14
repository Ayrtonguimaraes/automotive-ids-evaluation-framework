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
       python tools/make_window_search_configs.py --test conversation_w10   (after the selection)
"""
import argparse
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


RF_PARAMS = {"n_estimators": 50, "max_depth": 7, "max_features": 10, "bootstrap": True, "random_state": 42}
CNN_HYPER = {"batch_size": 256, "learning_rate": 0.0001, "num_epochs": 30, "early_stopping_patience": 5,
             "num_outputs": 1}


def write_test_configs(variant):
    """Configs to test one selected variant once: test features, final RF training, RF and CNN tests.

    The CNN fold models are the ones trained by the window search (same config as a final training);
    the RF has to be trained again with the training step of the pipeline, because tools/tune_rf.py
    does not save models. Fold models are linked into retrained/window_<variant>_rf and _cnn.
    """
    grouping, window = variant.rsplit("_w", 1)
    window = int(window)
    if (grouping, window) not in variants():
        raise SystemExit(f"unknown variant {variant}")
    model_feat = {k: v for k, v in feat_config(grouping, window, False).items() if k not in ("split", "suffix")}
    for sum_x, kind in [(False, "img"), (True, "sumX")]:
        with open(f"{OUT}/{variant}_feat_test_{kind}.json", "w") as fh:
            json.dump({"feature_generator": "CNNIDSFeatureGenerator",
                       "config": feat_config(grouping, window, sum_x, split="test"),
                       "paths": {"dataset_path": "data/someip_csv/Data", "output_path": data_dir(grouping, window)}},
                      fh, indent=4)
    rf_feat = dict(model_feat, sum_x=True)
    rf_train = {"feat_gen": {"feature_generator": "CNNIDSFeatureGenerator", "config": rf_feat,
                             "load_paths": {"X_path": features_file("X", grouping, window, True),
                                            "y_path": features_file("y", grouping, window, True)}},
                "model_specs": {"framework": "sklearn", "model": "SklearnClassifier",
                                "model_name": "RandomForestClassifier", "model_params": RF_PARAMS,
                                "validation_groups_path": f"{data_dir(grouping, window)}/groups_train.npz"}}
    rf_test = {"feat_gen": {"feature_generator": "CNNIDSFeatureGenerator", "config": rf_feat,
                            "load_paths": {"X_path": features_file("X", grouping, window, True, "test"),
                                           "y_path": features_file("y", grouping, window, True, "test")}},
               "model_specs": {"framework": "sklearn", "model": "SklearnClassifier",
                               "model_name": "RandomForestClassifier", "model_params": RF_PARAMS,
                               "presaved_paths": {str(f): f"retrained/window_{variant}_rf/RandomForestClassifier_fold_{f}.pkl"
                                                  for f in range(5)}}}
    cnn_test = {"feat_gen": {"feature_generator": "CNNIDSFeatureGenerator", "config": model_feat,
                             "load_paths": {"X_path": features_file("X", grouping, window, False, "test"),
                                            "y_path": features_file("y", grouping, window, False, "test")}},
                "model_specs": {"framework": "pytorch", "model": "PrunedCNNIDS", "model_name": f"PrunedCNNIDS_{variant}",
                                "criterion": "binary-cross-entropy", "hyperparameters": CNN_HYPER,
                                "presaved_paths": {str(f): f"retrained/window_{variant}_cnn/pytorch_model_PrunedCNNIDS_{f}"
                                                   for f in range(5)}}}
    for suffix, cfg in [("rf_train", rf_train), ("rf_test", rf_test), ("cnn_test", cnn_test)]:
        with open(f"{OUT}/{variant}_{suffix}.json", "w") as fh:
            json.dump(cfg, fh, indent=4)
    print(f"wrote the test configs of {variant} in {OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="write the test configs of one selected variant, e.g. conversation_w10")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.test:
        write_test_configs(args.test)
        return
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
                            "hyperparameters": CNN_HYPER,
                            "validation_groups_path": f"{data_dir(grouping, window)}/groups_train.npz"},
        }
        with open(f"{OUT}/{n}_cnn.json", "w") as fh:
            json.dump(cnn, fh, indent=4)
    print(f"wrote {len(variants()) * 3} configs in {OUT}")


if __name__ == "__main__":
    main()
