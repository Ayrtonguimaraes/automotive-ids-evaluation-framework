#!/bin/bash
# Apuana job: test the AVTP Random Forest (first stage) on the small sample inputs.
# Submit from the repository root:  mkdir -p logs && sbatch jobs/test_rf_sample.sbatch
#SBATCH --job-name=rf_sample_test
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Compute nodes do not have the miniforge3 module (only the login nodes do),
# so call the environment's python directly; $HOME is shared by all nodes
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

# Config paths are relative to the repository root
cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"

$PYTHON execute_model_test.py --model_test_config config_jsons/test_detection_time/AVTP_RandomForest_detection_time.json
