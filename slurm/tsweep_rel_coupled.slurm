#!/bin/bash
#SBATCH --job-name=tsw_rel_c
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
echo "=== COUPLED relational t=${T}: is FLAT-in-t true on the coupled arm too? ==="
pixi run python -u probes/layer_relational_invariance.py --couple \
  --timestep "${T}" --limit 24 --n-rot 3 --n-res 60 --batch-size 16 \
  --out "results/tsweep_relc_t${T}.json"
echo "done coupled rel t=${T}"
