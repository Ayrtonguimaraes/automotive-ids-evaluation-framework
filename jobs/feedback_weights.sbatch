#!/bin/bash
# Apuana job: weights of the feedback-loop retraining of the TOW-IDS RF (tools/feedback_loop.py weights).
# Uses the out-of-fold predictions of our TOW-IDS training runs (identified with tools/list_oof.py):
# RF 2026_11_04_18_02_35 and the 5 CNN folds 2026_11_04_19_25_04 ... 21_09_25. Training data only.
# Submit from the repository root: sbatch jobs/feedback_weights.sbatch
#SBATCH --job-name=feedback_weights
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

PYTHON="$HOME/.conda/envs/ids-env/bin/python"
cd "$HOME/automotive-ids"
export PYTHONUNBUFFERED=1
O="$HOME/outputs"
$PYTHON tools/feedback_loop.py weights --rf-run $O/2026_11_04_18_02_35_sklearn_train_val \
    --cnn-runs $O/2026_11_04_19_25_04_pytorch_train_fold0 $O/2026_11_04_19_44_21_pytorch_train_fold1 \
               $O/2026_11_04_20_19_20_pytorch_train_fold2 $O/2026_11_04_20_40_52_pytorch_train_fold3 \
               $O/2026_11_04_21_09_25_pytorch_train_fold4
