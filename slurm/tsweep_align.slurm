#!/bin/bash
#SBATCH --job-name=tsw_align
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=460G
#SBATCH --qos=mid
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
export PYTHONPATH="$SV"
echo "=== alignment R2 at t=${T} (targets: data/density_targets_t${T}) ==="
pixi run python -u probes/align_esmc_density.py --mlp \
  --target-dir "data/density_targets_t${T}" \
  --out "results/tsweep_align_t${T}.json"
echo "done align t=${T}"
