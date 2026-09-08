#!/bin/bash
#SBATCH --job-name=o7_alltap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=h100-reserved
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/o7-%j-%x.out
set -euo pipefail
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
"$SV/.pixi/envs/default/bin/python" -u -m "$@"
echo "done"
