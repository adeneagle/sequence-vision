#!/bin/bash
#SBATCH --job-name=homolog_diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=128G
#SBATCH --qos=mid
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out

# Homolog diagnostic: does a density descriptor vary smoothly with sequence?
# Descriptor ladder over SIMULATED density; see probes/homolog_diagnostic.py.
set -euo pipefail
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SV"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader

pixi run python -u probes/homolog_diagnostic.py \
  --limit "${LIMIT:-80}" \
  --out "results/${OUT:-homolog_diagnostic}.json" \
  ${EXTRA:-}
echo "done"
