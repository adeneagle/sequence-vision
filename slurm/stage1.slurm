#!/bin/bash
#SBATCH --job-name=stage1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=128G
#SBATCH --qos=mid
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage1-%j-%x.out
set -euo pipefail
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
# This project's OWN env: torch is here, and we need no ESM SDK for Stage 1.
"$SV/.pixi/envs/default/bin/python" -u -m probes.stage1_pair_head \
  --pair "$PAIR" --n-tri "$NTRI" --tag "$TAG" \
  --out "results/stage1_${TAG}.json" ${EXTRA:-}
echo "done $TAG"
