#!/bin/bash
#SBATCH --job-name=tsw_rel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=128G
#SBATCH --qos=mid
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
export PYTHONPATH="$SV" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== relational sweep t=${T} (GLOBAL framing, frame-free route) ==="
pixi run python -u probes/layer_relational_invariance.py \
  --timestep "${T}" --limit 24 --n-rot 3 --n-res 60 --batch-size 16 \
  --out "results/tsweep_rel_t${T}.json"
echo "done t=${T}"
