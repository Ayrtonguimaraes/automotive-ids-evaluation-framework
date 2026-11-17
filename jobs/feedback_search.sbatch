#!/bin/bash
# Apuana job array: feedback-loop retraining of the TOW-IDS RF, one variant per task (5 folds each).
# Tasks 0-2: feedback_w2, w5, w10 (weight on windows where the RF was wrong and the CNN right);
# tasks 3-5: control_w2, w5, w10 (same extra weight per class, spread evenly). Training data only.
# Needs the weights (jobs/feedback_weights.sbatch). Submit from the repository root:
#   sbatch --array=0-5%4 jobs/feedback_search.sbatch
# Then: python tools/feedback_loop.py summarize --baseline-run ~/outputs/2026_11_04_18_02_35_sklearn_train_val
#SBATCH --job-name=feedback_search
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "error: submit as an array, e.g. sbatch --array=0-5%4 jobs/feedback_search.sbatch" >&2
    exit 1
fi
PYTHON="$HOME/.conda/envs/ids-env/bin/python"
cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
export PYTHONUNBUFFERED=1
set -e
# Task 6: errors_w5, the second control (weight on every RF error, without the CNN); run it alone with --array=6
# Tasks 7 and 8: ablations for the Frame Injection diagnosis (weight only on the attack half or only on the
# normal half of the feedback set); run with --array=7-8
VARIANTS=(feedback_w2 feedback_w5 feedback_w10 control_w2 control_w5 control_w10 errors_w5 fbattacks_w5 fbnormals_w5)
V="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
# The run folder gets the variant in its name (<timestamp>_sklearn_train_val_<variant>)
export IDS_RUN_TAG="$V"
echo "=== variant $V ==="
/usr/bin/time -v $PYTHON execute_model_train_validation.py \
    --model_train_valid_config config_jsons/feedback/TOW_RandomForest_$V.json
