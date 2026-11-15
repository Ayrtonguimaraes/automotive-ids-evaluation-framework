#!/bin/bash
# Apuana job: figures and table rows of the report (tools/report_figures.py), from saved test predictions.
# Finds the feedback_w5 and control_w5 test runs in the logs of jobs/feedback_test.sbatch.
# Submit from the repository root: sbatch jobs/report_figures.sbatch
# Then copy ~/outputs/report_figures to the Mac (rsync).
#SBATCH --job-name=report_figures
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

PYTHON="$HOME/.conda/envs/ids-env/bin/python"
cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
export PYTHONUNBUFFERED=1

test_run() {
    local log
    log=$(grep -l "sklearn_train_val_$1" logs/feedback_test_*.out 2>/dev/null | tail -1)
    [ -n "$log" ] && grep "^test run:" "$log" | awk '{print $3 "/metrics"}'
}
FB=$(test_run feedback_w5)
CT=$(test_run control_w5)
echo "feedback test: $FB | control test: $CT"
ARGS=""
if [ -n "$FB" ] && [ -n "$CT" ]; then ARGS="--feedback-test $FB --control-test $CT"; fi
$PYTHON tools/report_figures.py --out "$IDS_ARTIFACTS_DIR/report_figures" $ARGS
