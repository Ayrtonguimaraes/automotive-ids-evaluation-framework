#!/bin/bash
# Apuana job array: SOME/IP CSV window search, one variant per task (tools/make_window_search_configs.py).
# Tasks 0-3: windows over the capture order (arrival) of 10, 20, 30 and 44 differences;
# tasks 4-7: the same sizes inside each client-server conversation.
# Each task generates the training features, scores the tuned RF (50, 7, 10) and the tuned CNN
# (lr 0.0001, batch 256) on the 5 validation folds. Only training files are used.
# Submit from the repository root (2 tasks at a time, the GPU limit per user):
#   sbatch --array=0-7%2 jobs/window_search.sbatch
# Then: python tools/summarize_window_search.py
#SBATCH --job-name=window_search
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "error: submit as an array, e.g. sbatch --array=0-7%2 jobs/window_search.sbatch" >&2
    exit 1
fi

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
export PYTHONUNBUFFERED=1
set -e

GROUPINGS=(arrival arrival arrival arrival conversation conversation conversation conversation)
WINDOWS=(10 20 30 44 10 20 30 44)
G="${GROUPINGS[$SLURM_ARRAY_TASK_ID]}"
W="${WINDOWS[$SLURM_ARRAY_TASK_ID]}"
NAME="${G}_w${W}"
DATA="data/someip_csv/window_search/$NAME"
CONFIGS="config_jsons/window_search/$NAME"
mkdir -p "$DATA" "$IDS_ARTIFACTS_DIR/window_search"
echo "=== variant $NAME ==="

echo "=== features ==="
$PYTHON execute_feature_generator.py --feat_gen_config "${CONFIGS}_feat_train_img.json"
$PYTHON execute_feature_generator.py --feat_gen_config "${CONFIGS}_feat_train_sumX.json"

echo "=== RF (50, 7, 10), 5 folds ==="
$PYTHON tools/tune_rf.py --only 50,7,10 --folds 5 --jobs 8 \
    --x "$DATA/X_train_SOMEIP_dataset_Wsize_${W}_Cols_116_Wslide_1_MC_False_sumX_True.npz" \
    --y "$DATA/y_train_SOMEIP_dataset_Wsize_${W}_Cols_116_Wslide_1_MC_False_sumX_True.npz" \
    --groups "$DATA/groups_train.npz" --out "$IDS_ARTIFACTS_DIR/window_search/rf_${NAME}.csv"

echo "=== CNN (lr 0.0001, batch 256), 5 folds ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
/usr/bin/time -v $PYTHON execute_model_train_validation.py --model_train_valid_config "${CONFIGS}_cnn.json"
