#!/bin/bash
#SBATCH --job-name=align_targets
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=128G
#SBATCH --qos=mid
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SV"
pixi run python -u ${SCRIPT:-probes/extract_density_targets.py} --batch-size 64 ${EXTRA:-}
echo "done"
