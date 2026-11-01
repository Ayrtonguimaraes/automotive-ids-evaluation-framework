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

# Batch jobs start with a clean shell: load conda and the project environment again
module load miniforge3/2025.07
eval "$(conda shell.bash hook)"
conda activate ids-env

# Config paths are relative to the repository root
cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"

python execute_model_test.py --model_test_config config_jsons/test_detection_time/AVTP_RandomForest_detection_time.json
