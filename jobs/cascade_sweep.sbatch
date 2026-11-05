#!/bin/bash
# Apuana job: sweep the cascade thresholds on the saved test probabilities (CPU only, nothing is trained).
# Submit from the repository root:
#   sbatch jobs/cascade_sweep.sbatch <RF test metrics dir> <CNN test metrics dir> <1 or 6 CNN outputs> <output csv>
#SBATCH --job-name=cascade_sweep
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

if [ "$#" -ne 4 ]; then
    echo "usage: sbatch jobs/cascade_sweep.sbatch <RF metrics dir> <CNN metrics dir> <1 or 6> <output csv>" >&2
    exit 1
fi

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
export PYTHONUNBUFFERED=1

/usr/bin/time -v $PYTHON tools/cascade_sweep.py --rf-dir "$1" --cnn-dir "$2" --cnn-outputs "$3" --out "$4"
