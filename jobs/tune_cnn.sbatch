#!/bin/bash
# Apuana job array: one SOME/IP CNN candidate per array task, all trained on fold 0 of the
# validation by capture (config key "validation_groups_path"). Only training data is used.
# Submit from the repository root (at most 2 tasks at a time, the GPU limit per user):
#   sbatch --array=0-5%2 jobs/tune_cnn.sbatch
# Task i trains the i-th file, in alphabetical order, of:
#   config_jsons/model_train_validate/SOMEIP_PrunedCNNIDS_tune_*.json
#SBATCH --job-name=tune_cnn
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "error: submit as an array, e.g. sbatch --array=0-5%2 jobs/tune_cnn.sbatch" >&2
    exit 1
fi

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
# Without this, python buffers its output when writing to a file and the log stays empty until the end
export PYTHONUNBUFFERED=1
# Every candidate is scored on the same validation captures (fold 0)
export IDS_FOLD=0

CONFIGS=(config_jsons/model_train_validate/SOMEIP_PrunedCNNIDS_tune_*.json)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$CONFIG" ]; then
    echo "error: no tuning config number $SLURM_ARRAY_TASK_ID (there are ${#CONFIGS[@]})" >&2
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "config: $CONFIG | fold: $IDS_FOLD"
/usr/bin/time -v $PYTHON execute_model_train_validation.py --model_train_valid_config "$CONFIG"
