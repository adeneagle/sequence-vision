#!/bin/bash
#SBATCH --job-name=tsw_rel_n
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
echo "=== relational at n_res=240, t=${T} (robustness of the FLAT-in-t result) ==="
pixi run python -u probes/layer_relational_invariance.py \
  --timestep "${T}" --limit 16 --n-rot 2 --n-res 240 --batch-size 16 \
  --out "results/tsweep_rel_n240_t${T}.json"
echo "done n240 t=${T}"
