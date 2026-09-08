#!/bin/bash
#SBATCH --job-name=esmc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=128G
#SBATCH --qos=mid
#SBATCH --time=08:00:00
#SBATCH --output=logs/esmc-%j-%x.out
set -euo pipefail
# The MAIN LAB env, not this project's: `esm` is not installed here and adding it
# would risk CryoFM's numpy<2.0 pin. Same rule as extract_esmc_chains.py.
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
LAB=/mnt/main0/projects/et-foundation-vision-model/aden/aden/.pixi/envs/default/bin/python3.10
cd "$SV"; mkdir -p logs results
"$LAB" -u "$@"
echo "done"
