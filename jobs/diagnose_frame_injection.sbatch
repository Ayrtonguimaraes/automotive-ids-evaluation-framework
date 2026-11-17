#!/bin/bash
# Apuana job: why Frame Injection drops after the feedback-loop retraining (tools/diagnose_frame_injection.py).
# Reads saved predictions only. Then prints the validation table with the two ablations, if they have run
# (sbatch --array=7-8 jobs/feedback_search.sbatch). Submit: sbatch jobs/diagnose_frame_injection.sbatch
#SBATCH --job-name=diagnose_fi
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

PYTHON="$HOME/.conda/envs/ids-env/bin/python"
cd "$HOME/automotive-ids"
export PYTHONUNBUFFERED=1
O="$HOME/outputs"
FB_TEST=$(ls -d $O/*_sklearn_test_feedback_w5 | sort | tail -1)
FB_RUN=$(ls -d $O/*_sklearn_train_val_feedback_w5 | sort | tail -1)
$PYTHON tools/diagnose_frame_injection.py \
    --base-test $O/2026_11_04_19_13_31_sklearn_test/metrics --feedback-test $FB_TEST/metrics \
    --y-test data/tow/processed/y_test_TOW_IDS_dataset_multi_class_Wsize_44_Cols_116_Wslide_1_MC_True_sumX_False.csv \
    --weights-dir data/tow/feedback --base-run $O/2026_11_04_18_02_35_sklearn_train_val --feedback-run $FB_RUN
echo
echo "=== validation with the ablations ==="
$PYTHON tools/feedback_loop.py summarize --baseline-run $O/2026_11_04_18_02_35_sklearn_train_val
