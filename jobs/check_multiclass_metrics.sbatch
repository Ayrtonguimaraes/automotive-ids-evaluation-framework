#!/bin/bash
# Apuana job: recompute the multiclass metrics from the saved probabilities (CPU only).
# Submit from the repository root: sbatch jobs/check_multiclass_metrics.sbatch <CNN test metrics dir>
#SBATCH --job-name=check_metrics
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

if [ "$#" -ne 1 ]; then
    echo "usage: sbatch jobs/check_multiclass_metrics.sbatch <CNN test metrics dir>" >&2
    exit 1
fi

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
export PYTHONUNBUFFERED=1

$PYTHON tools/check_multiclass_metrics.py "$1"
