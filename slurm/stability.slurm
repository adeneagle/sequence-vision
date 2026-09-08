#!/bin/bash
#SBATCH --job-name=cryofm_stability
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=128G
#SBATCH --qos=mid
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out

# Phase 0a gate: is a CryoFM2 activation pose-stable enough to be a prediction
# target for a pose-free sequence model? See PLAN.md section 3.
#
# 1 GPU is enough -- this is inference over a few dozen small maps, not training.
# Many h100 nodes sit in 'mix' state with free GPUs, so a 1-GPU request schedules
# far sooner than the 1300-deep queue of multi-node jobs suggests.

set -euo pipefail

SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"
mkdir -p logs results

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SV"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

pixi run python probes/stability.py \
  --manifest data/manifest.csv \
  --limit "${LIMIT:-25}" \
  --timestep "${TIMESTEP:-10}" \
  --n-rot "${NROT:-8}" \
  --n-noise "${NNOISE:-3}" \
  --batch-size "${BS:-16}" \
  --out "results/stability_t${TIMESTEP:-10}_aligned.json"

echo "done"
