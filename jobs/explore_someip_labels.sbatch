#!/bin/bash
# Apuana job: second exploratory pass (do the labels match the bytes?) (tools/explore_someip_labels.py). CPU only.
# Submit from the repository root:
#   sbatch jobs/explore_someip_labels.sbatch
#SBATCH --job-name=explore_someip_labels
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
$PYTHON tools/explore_someip_labels.py --dataset data/someip/SOME_IP_Intrusion_Detection_Dataset
