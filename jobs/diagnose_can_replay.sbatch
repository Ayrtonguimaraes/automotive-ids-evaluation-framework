#!/bin/bash
# Apuana job: diagnose the CAN Replay misses of the TOW-IDS Random Forest (tools/diagnose_can_replay.py).
# CPU only. Submit from the repository root, optionally giving the metrics folder of the RF test run:
#   sbatch jobs/diagnose_can_replay.sbatch [<RF test metrics dir>]
#SBATCH --job-name=diag_can_replay
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

RF_TEST_DIR="${1:-$HOME/outputs/2026_11_04_19_13_31_sklearn_test/metrics}"

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
# Without this, python buffers its output when writing to a file and the log stays empty until the end
export PYTHONUNBUFFERED=1

DATA=data/tow/processed
MULTI=TOW_IDS_dataset_multi_class_Wsize_44_Cols_116_Wslide_1_MC_True_sumX_False
BINARY=TOW_IDS_dataset_one_class_Wsize_44_Cols_116_Wslide_1_MC_False_sumX_True
echo "RF test run: $RF_TEST_DIR"
/usr/bin/time -v $PYTHON tools/diagnose_can_replay.py \
    --rf-test-dir "$RF_TEST_DIR" --y-test $DATA/y_test_$MULTI.csv \
    --x-train-img $DATA/X_train_$MULTI.npz --y-train $DATA/y_train_$MULTI.csv \
    --x-train-sum $DATA/X_train_$BINARY.npz --folds 2 --jobs 16
