#!/bin/bash
#SBATCH --job-name=align_run
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
pixi run python -u ${SCRIPT:-probes/align_esmc_density.py} ${EXTRA:---mlp}
echo "done"
