#!/bin/bash
# Apuana job: test ONCE the window search variant selected on validation (tools/summarize_window_search.py).
# 1. test features; 2. final RF: 5 folds with the training step (tune_rf.py does not save models), then
# test; 3. CNN: links the 5 fold models of the window search run of this variant, then test;
# 4. balanced metrics (tools/binary_report.py) of both tests.
# Write the configs first (python tools/make_window_search_configs.py --test <variant>), then submit
# from the repository root:
#   sbatch jobs/window_search_test.sbatch conversation_w10
#SBATCH --job-name=window_search_test
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

V="$1"
if [ -z "$V" ]; then echo "usage: sbatch jobs/window_search_test.sbatch <variant, e.g. conversation_w10>" >&2; exit 1; fi

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
export PYTHONUNBUFFERED=1
set -e
C="config_jsons/window_search/$V"

# Newest run folder of a kind (<timestamp>_<kind>) created after START
newest_after() {
    for dir in $(ls -d "$IDS_ARTIFACTS_DIR"/*_"$1" 2>/dev/null | sort -r); do
        if [[ "$(basename "$dir")" > "$START" || "$(basename "$dir")" == "$START"* ]]; then echo "$dir"; return; fi
    done
}

echo "=== $V: test features ==="
$PYTHON execute_feature_generator.py --feat_gen_config "${C}_feat_test_img.json"
$PYTHON execute_feature_generator.py --feat_gen_config "${C}_feat_test_sumX.json"

echo "=== final RF: training (5 folds) ==="
START=$(date +%Y_%m_%d_%H_%M_%S)
$PYTHON execute_model_train_validation.py --model_train_valid_config "${C}_rf_train.json"
RUN=$(newest_after sklearn_train_val)
if [ -z "$RUN" ]; then echo "error: no RF training run after $START" >&2; exit 1; fi
cat "$RUN"/metrics/train_val_metrics_RandomForestClassifier.csv
mkdir -p "retrained/window_${V}_rf"
for f in 0 1 2 3 4; do
    ln -sfn "$RUN/models/RandomForestClassifier_fold_$f.pkl" "retrained/window_${V}_rf/RandomForestClassifier_fold_$f.pkl"
done

echo "=== CNN: fold models of the window search ==="
METRICS=$(ls -t "$IDS_ARTIFACTS_DIR"/*_pytorch_train*/metrics/val_metrics_PrunedCNNIDS_${V}_*.csv | head -1)
CNN_RUN=$(dirname "$(dirname "$METRICS")")
echo "search run: $CNN_RUN"; cat "$METRICS"
mkdir -p "retrained/window_${V}_cnn"
for f in 0 1 2 3 4; do
    test -f "$CNN_RUN/models/pytorch_model_PrunedCNNIDS_${V}_$f"
    ln -sfn "$CNN_RUN/models/pytorch_model_PrunedCNNIDS_${V}_$f" "retrained/window_${V}_cnn/pytorch_model_PrunedCNNIDS_$f"
done
ls -l "retrained/window_${V}_rf" "retrained/window_${V}_cnn"

echo "=== RF test (once) ==="
START=$(date +%Y_%m_%d_%H_%M_%S)
$PYTHON execute_model_test.py --model_test_config "${C}_rf_test.json"
RF_TEST=$(newest_after sklearn_test)
echo "=== CNN test (once) ==="
START=$(date +%Y_%m_%d_%H_%M_%S)
$PYTHON execute_model_test.py --model_test_config "${C}_cnn_test.json"
CNN_TEST=$(newest_after pytorch_test)

echo "=== balanced metrics: RF ($RF_TEST) ==="
$PYTHON tools/binary_report.py "$RF_TEST/metrics"
echo "=== balanced metrics: CNN ($CNN_TEST) ==="
$PYTHON tools/binary_report.py "$CNN_TEST/metrics"
