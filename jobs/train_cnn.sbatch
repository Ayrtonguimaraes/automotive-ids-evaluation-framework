#!/bin/bash
# Apuana job array: train and validate one CNN fold per array task (tasks 0 to 4 = folds 0 to 4).
# Submit from the repository root:
#   sbatch --array=0-4 jobs/train_cnn.sbatch config_jsons/model_train_validate/AEID_PrunedCNNIDS.json
# The 5 folds run as independent jobs; each one is saved as soon as it finishes, so a
# preempted fold can be resubmitted alone with: sbatch --array=<fold> jobs/train_cnn.sbatch <config>
#SBATCH --job-name=train_cnn
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "usage: sbatch --array=0-4 jobs/train_cnn.sbatch <train config>" >&2
    exit 1
fi

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

# Config paths are relative to the repository root
cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
export IDS_FOLD="$SLURM_ARRAY_TASK_ID"

# Record which GPU the fold ran on
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "config: $CONFIG | fold: $IDS_FOLD"
/usr/bin/time -v $PYTHON execute_model_train_validation.py --model_train_valid_config "$CONFIG"
