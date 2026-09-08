#!/bin/bash
#SBATCH --job-name=tsw_pose_c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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
echo "=== COUPLED pose sweep t=${T} (noise_level = t/1000, the trained pairing) ==="
pixi run python -u probes/pose_invariance_clean.py --couple \
  --timestep "${T}" --limit 8 --n-res 250 --n-rot 3 --batch-size 16 \
  --out "results/tsweep_posec_t${T}.json"
echo "done coupled t=${T}"
