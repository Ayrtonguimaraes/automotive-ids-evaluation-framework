#!/bin/bash
# Apuana job: final SOME/IP Random Forest. Trains the 5 folds with the hyperparameters chosen by
# tools/tune_rf.py (validation by capture or purged blocks, as the config says), links the fold models
# into retrained/<link>/ and runs the test once. CPU only. Submit from the repository root:
#   sbatch jobs/final_someip_rf.sbatch                                       (SOME/IP pickles)
#   sbatch jobs/final_someip_rf.sbatch SOMEIPCSV_RandomForest someipcsv_rf   (SOME/IP CSV)
# The first argument names config_jsons/model_train_validate/<name>.json and model_test/<name>_test.json.
#SBATCH --job-name=final_someip_rf
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Compute nodes do not have the miniforge3 module, so call the environment's python directly
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

cd "$HOME/automotive-ids"
export IDS_ARTIFACTS_DIR="$HOME/outputs"
export PYTHONUNBUFFERED=1
set -e

NAME="${1:-SOMEIP_RandomForest}"
LINK="${2:-someip_rf}"
echo "configs: $NAME | models linked into retrained/$LINK"

# Run folders are named <timestamp>_sklearn_train_val, so this one is the first named after START
START=$(date +%Y_%m_%d_%H_%M_%S)
echo "=== training (5 folds) ==="
/usr/bin/time -v $PYTHON execute_model_train_validation.py \
    --model_train_valid_config config_jsons/model_train_validate/$NAME.json

RUN=""
for dir in $(ls -d "$IDS_ARTIFACTS_DIR"/*_sklearn_train_val | sort); do
    if [[ "$(basename "$dir")" > "$START" || "$(basename "$dir")" == "$START"* ]]; then RUN="$dir"; break; fi
done
if [ -z "$RUN" ]; then echo "error: no training run found after $START" >&2; exit 1; fi
echo "training run: $RUN"
cat "$RUN"/metrics/train_val_metrics_RandomForestClassifier.csv

mkdir -p retrained/$LINK
for fold in 0 1 2 3 4; do
    ln -sfn "$RUN/models/RandomForestClassifier_fold_$fold.pkl" "retrained/$LINK/RandomForestClassifier_fold_$fold.pkl"
done
ls -l retrained/$LINK

echo "=== test ==="
/usr/bin/time -v $PYTHON execute_model_test.py --model_test_config config_jsons/model_test/${NAME}_test.json
