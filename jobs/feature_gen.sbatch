#!/bin/bash
# Apuana job: generate features from raw pcaps with the config given as argument.
# Submit from the repository root:
#   sbatch jobs/feature_gen.sbatch config_jsons/feat_generator/AEID_smoke_sumX.json
# Needs the data link in the repository root: ln -s ~/data data
#SBATCH --job-name=feature_gen
#SBATCH -p short-simple
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "usage: sbatch jobs/feature_gen.sbatch <feature generator config>" >&2
    exit 1
fi

# Compute nodes do not have the miniforge3 module (only the login nodes do),
# so call the environment's python directly; $HOME is shared by all nodes
PYTHON="$HOME/.conda/envs/ids-env/bin/python"

# Config paths are relative to the repository root
cd "$HOME/automotive-ids"

# The generator saves into output_path but does not create it
OUTPUT_PATH=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1]))['paths']['output_path'])" "$CONFIG")
mkdir -p "$OUTPUT_PATH"

echo "config: $CONFIG"
# /usr/bin/time -v reports elapsed time and peak memory (Maximum resident set size)
/usr/bin/time -v $PYTHON execute_feature_generator.py --feat_gen_config "$CONFIG"
