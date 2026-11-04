#!/bin/bash
# Apuana job: test the models of a config on the test features.
# Submit from the repository root. The Random Forest needs no GPU; for a CNN add --gpus=1:
#   sbatch jobs/test_model.sbatch config_jsons/model_test/AEID_RandomForest_test.json
#   sbatch --gpus=1 jobs/test_model.sbatch config_jsons/model_test/AEID_PrunedCNNIDS_test.json
#SBATCH --job-name=test_model
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "usage: sbatch [--gpus=1] jobs/test_model.sbatch <test config>" >&2
    exit 1
fi

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

# Config paths are relative to the repository root
cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
# Without this, python buffers its output when writing to a file and the log stays empty until the end
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU in this job"
echo "config: $CONFIG"
/usr/bin/time -v $PYTHON execute_model_test.py --model_test_config "$CONFIG"
