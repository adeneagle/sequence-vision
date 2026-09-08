#!/bin/bash
#SBATCH --job-name=homolog_res
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SV"
pixi run python -u probes/homolog_diagnostic_residue.py \
  --limit "${LIMIT:-60}" --n-res "${NRES:-40}" \
  --out "results/${OUT:-homolog_residue}.json" ${EXTRA:-}
echo "done"
