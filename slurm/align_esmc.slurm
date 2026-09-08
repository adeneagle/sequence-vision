#!/bin/bash
#SBATCH --job-name=align_esmc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=128G
#SBATCH --qos=mid
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs
# MAIN LAB ENV -- esm lives there, not in this project's env (numpy<2.0 pin).
/mnt/main0/projects/et-foundation-vision-model/aden/aden/.pixi/envs/default/bin/python3.10 \
  -u ${SCRIPT:-data/extract_esmc_chains.py} ${EXTRA:-}
echo "done"
