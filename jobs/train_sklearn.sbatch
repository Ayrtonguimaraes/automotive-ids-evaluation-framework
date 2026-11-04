#!/bin/bash
# Apuana job: train and validate a scikit-learn model (5 folds) with the config given as argument.
# Submit from the repository root:
#   sbatch jobs/train_sklearn.sbatch config_jsons/model_train_validate/AEID_RandomForest.json
#SBATCH --job-name=train_sklearn
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "usage: sbatch jobs/train_sklearn.sbatch <train config>" >&2
    exit 1
fi

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

# Config paths are relative to the repository root
cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"

echo "config: $CONFIG"
/usr/bin/time -v $PYTHON execute_model_train_validation.py --model_train_valid_config "$CONFIG"
