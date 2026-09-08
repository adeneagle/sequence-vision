#!/bin/bash
#SBATCH --job-name=cleandift
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=192G
#SBATCH --qos=mid
#SBATCH --time=08:00:00
#SBATCH --output=logs/cleandift-%j-%x.out
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
# Positional "$@", never --export: sbatch --export silently truncates values that
# contain a comma, which has already corrupted one sweep in this project.
"$SV/.pixi/envs/default/bin/python" -u -m "$@"
echo "done"
