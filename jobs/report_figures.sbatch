#!/bin/bash
# Apuana job: figures and table rows of the report (tools/report_figures.py), from saved test predictions.
# Finds the feedback_w5 and control_w5 test runs by their folder names (<timestamp>_sklearn_test_<variant>).
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
    # Test folders of the feedback variants carry the variant name (<timestamp>_sklearn_test_<variant>)
    local dir
    dir=$(ls -d "$IDS_ARTIFACTS_DIR"/*_sklearn_test_$1 2>/dev/null | sort | tail -1)
    [ -n "$dir" ] && echo "$dir/metrics"
}
FB=$(test_run feedback_w5)
CT=$(test_run control_w5)
echo "feedback test: $FB | control test: $CT"
ARGS=""
if [ -n "$FB" ] && [ -n "$CT" ]; then ARGS="--feedback-test $FB --control-test $CT"; fi
$PYTHON tools/report_figures.py --out "$IDS_ARTIFACTS_DIR/report_figures" $ARGS
