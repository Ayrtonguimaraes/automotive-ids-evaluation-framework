#!/bin/bash
# Apuana job: grid search of the SOME/IP Random Forest with validation by capture (tools/tune_rf.py).
# Submit from the repository root, after tools/someip_groups.py has written groups_train.npz:
#   sbatch jobs/tune_rf.sbatch
#SBATCH --job-name=tune_rf
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
# Without this, python buffers its output when writing to a file and the log stays empty until the end
export PYTHONUNBUFFERED=1

DATA=data/someip/processed
/usr/bin/time -v $PYTHON tools/tune_rf.py \
    --x $DATA/X_train_SOMEIP_dataset_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_True.npz \
    --y $DATA/y_train_SOMEIP_dataset_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_True.npz \
    --groups $DATA/groups_train.npz --folds 3 --jobs 16 \
    --out "$HOME/outputs/someip_rf_tuning.csv"
