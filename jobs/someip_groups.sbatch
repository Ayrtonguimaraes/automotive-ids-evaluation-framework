#!/bin/bash
# Apuana job: write the capture of every SOME/IP window (train and test), needed for the
# validation by capture (tools/someip_groups.py). Takes a few minutes, CPU only.
# Submit from the repository root:
#   sbatch jobs/someip_groups.sbatch
#SBATCH --job-name=someip_groups
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
export PYTHONUNBUFFERED=1
# Stop at the first failure, so a dependent job never starts on missing groups
set -e

DATA=data/someip/processed
for SPLIT in train test; do
    $PYTHON tools/someip_groups.py --split $SPLIT \
        --dataset data/someip/SOME_IP_Intrusion_Detection_Dataset \
        --labels $DATA/y_${SPLIT}_SOMEIP_dataset_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_True.npz \
        --out $DATA/groups_${SPLIT}.npz
done
