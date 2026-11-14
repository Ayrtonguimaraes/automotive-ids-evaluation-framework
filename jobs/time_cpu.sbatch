#!/bin/bash
# Apuana job: measure the detection time of the retrained RF and CNN models on one CPU, as in Table 10 of the paper.
# All eight models (AEID, TOW-IDS and the two SOME/IP CSV windows) run in the same job, so they are timed on the same machine. The timing uses a dummy batch of 64
# windows; the sample inputs only let the test code run, so ignore every metric of these runs except inference_time.
# Submit from the repository root: sbatch jobs/time_cpu.sbatch
#SBATCH --job-name=time_cpu
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
export PYTHONUNBUFFERED=1
export IDS_DEVICE=cpu
# Use exactly the cores Slurm gave the job, so the timing is not disturbed by extra threads
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# The CPU model and the core count go in the report next to the times
hostname
lscpu | grep -E "Model name|^CPU\(s\)|MHz"
echo "cores given by Slurm: $SLURM_CPUS_PER_TASK"

# SOME/IP CSV: the paper's window (arrival, 44) and the per-conversation window of 10 differences
for NAME in AEID_RandomForest AEID_PrunedCNNIDS TOW_RandomForest TOW_PrunedCNNIDS \
            SOMEIPCSV_RandomForest SOMEIPCSV_PrunedCNNIDS \
            SOMEIPCSV_conversation_w10_RandomForest SOMEIPCSV_conversation_w10_PrunedCNNIDS; do
    echo "=== $NAME"
    $PYTHON execute_model_test.py --model_test_config "config_jsons/test_detection_time/${NAME}_retrained_cpu.json"
done
