#!/bin/bash
# Apuana job: test the AVTP Pruned CNN (second stage) on the small sample inputs.
# The pytorch test always moves batches to cuda:0, so a GPU is required.
# Submit from the repository root:  mkdir -p logs && sbatch jobs/test_cnn_sample.sbatch
#SBATCH --job-name=cnn_sample_test
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Compute nodes do not have the miniforge3 module (only the login nodes do),
# so call the environment's python directly; $HOME is shared by all nodes
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

# Config paths are relative to the repository root
cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"

# Record which GPU the job received (useful when reporting detection times)
nvidia-smi --query-gpu=name,memory.total --format=csv

$PYTHON execute_model_test.py --model_test_config config_jsons/test_detection_time/AVTP_PrunedCNNIDS_detection_time.json
