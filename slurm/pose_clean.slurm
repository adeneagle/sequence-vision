#!/bin/bash
#SBATCH --job-name=pose_clean
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=128G
#SBATCH --qos=mid
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
export PYTHONPATH="$SV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pixi run python probes/pose_invariance_clean.py --limit "${LIMIT:-8}" --n-res "${NRES:-250}" --n-rot "${NROT:-3}" --batch-size 16 --out results/pose_clean_margin48.json
