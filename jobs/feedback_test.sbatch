#!/bin/bash
# Apuana job: test ONCE a feedback-loop variant of the TOW-IDS RF selected on validation
# (tools/feedback_loop.py summarize). Links the 5 fold models of its training run into retrained/tow_rf_<v>,
# runs the test and the per-attack report (part 1 of tools/diagnose_can_replay.py). Submit with the
# selected variant and its control, e.g.:
#   sbatch jobs/feedback_test.sbatch feedback_w5 && sbatch jobs/feedback_test.sbatch control_w5
#SBATCH --job-name=feedback_test
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

V="$1"
if [ -z "$V" ]; then echo "usage: sbatch jobs/feedback_test.sbatch <variant>" >&2; exit 1; fi
PYTHON="$HOME/.conda/envs/ids-env/bin/python"
cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
export PYTHONUNBUFFERED=1
set -e
RUN=$(ls -d "$IDS_ARTIFACTS_DIR"/*_sklearn_train_val_$V | sort | tail -1)
echo "training run: $RUN"
mkdir -p "retrained/tow_rf_$V"
for f in 0 1 2 3 4; do
    ln -sfn "$RUN/models/RandomForestClassifier_fold_$f.pkl" "retrained/tow_rf_$V/RandomForestClassifier_fold_$f.pkl"
done
echo "=== test (once) ==="
# The test folder gets the variant in its name, so two tests started in the same second never share it
export IDS_RUN_TAG="$V"
$PYTHON execute_model_test.py --model_test_config "config_jsons/feedback/TOW_RandomForest_${V}_test.json"
TEST=$(ls -d "$IDS_ARTIFACTS_DIR"/*_sklearn_test_$V | sort | tail -1)
echo "test run: $TEST"
cat "$TEST"/metrics/*test_metrics*.csv
echo "=== per attack (Table 9) ==="
$PYTHON tools/diagnose_can_replay.py --rf-test-dir "$TEST/metrics" \
    --y-test data/tow/processed/y_test_TOW_IDS_dataset_multi_class_Wsize_44_Cols_116_Wslide_1_MC_True_sumX_False.csv
