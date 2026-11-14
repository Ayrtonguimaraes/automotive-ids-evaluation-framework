#!/bin/bash
# Apuana job: message-level comparison of the SOME/IP CSV windowings (tools/message_level_eval.py).
# Re-reads the saved test probabilities of the one-time tests; trains and selects nothing. CPU only.
# Submit from the repository root: sbatch jobs/message_level_eval.sbatch
#SBATCH --job-name=message_level_eval
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

PYTHON="$HOME/.conda/envs/ids-env/bin/python"
cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
export PYTHONUNBUFFERED=1
$PYTHON tools/message_level_eval.py
