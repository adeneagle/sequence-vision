#!/bin/bash
#SBATCH --job-name=cryofm_localframe
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
pixi run python probes/local_frame_stability.py --limit "${LIMIT:-6}" --n-res "${NRES:-300}" --n-rot "${NROT:-3}" --batch-size 16
