#!/bin/bash
#SBATCH --job-name=tsw_ceil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=128G
#SBATCH --qos=mid
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
export PYTHONPATH="$SV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== ceiling sweep t=${T}: sequence-trackable fraction of the per-residue target ==="
pixi run python -u probes/homolog_diagnostic_residue.py \
  --timestep "${T}" --rungs local frameavg --batch-size 32 \
  --out "results/tsweep_ceiling_t${T}.json"
echo "done t=${T}"
