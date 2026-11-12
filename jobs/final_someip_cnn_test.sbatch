#!/bin/bash
# Apuana job: test the final SOME/IP CNN. Links the 5 fold models trained after <not before>
# (the time the final training was submitted) into retrained/<link>/ and runs the test once.
# Submit it to start after the training array, from the repository root:
#   START=$(date +%Y_%m_%d_%H_%M_%S)
#   TRAIN=$(sbatch --parsable --array=0-4 jobs/train_cnn.sbatch config_jsons/model_train_validate/SOMEIP_PrunedCNNIDS.json)
#   sbatch --dependency=afterok:$TRAIN jobs/final_someip_cnn_test.sbatch $START
# For the SOME/IP CSV version add the config name and the link name:
#   ... jobs/final_someip_cnn_test.sbatch $START SOMEIPCSV_PrunedCNNIDS someipcsv_cnn
#SBATCH --job-name=final_someip_cnn_test
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

NOT_BEFORE="$1"
NAME="${2:-SOMEIP_PrunedCNNIDS}"
LINK="${3:-someip_cnn}"
if [ -z "$NOT_BEFORE" ]; then
    echo "usage: sbatch jobs/final_someip_cnn_test.sbatch <not before, e.g. 2026_11_11_21_00_00>" >&2
    exit 1
fi

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
export PYTHONUNBUFFERED=1
set -e

echo "=== validation of each fold ==="
for f in "$IDS_ARTIFACTS_DIR"/*_pytorch_train_fold*/metrics/val_metrics_PrunedCNNIDS_*.csv; do
    run=$(basename "$(dirname "$(dirname "$f")")")
    if [[ "$run" > "$NOT_BEFORE" ]]; then echo "$run"; tail -n 1 "$f"; fi
done

echo "=== linking the fold models ==="
$PYTHON tools/link_cnn_folds.py "$LINK" 1 "$NOT_BEFORE"

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== test ==="
/usr/bin/time -v $PYTHON execute_model_test.py --model_test_config config_jsons/model_test/${NAME}_test.json
