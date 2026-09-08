#!/bin/bash
#SBATCH --job-name=cdift_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cleandift-%j-%x.out
set -euo pipefail
# Volume caching is pure CPU (mrcfile read + Fourier resample + numpy normalise),
# so it does not hold an H100 for an hour.
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
"$SV/.pixi/envs/default/bin/python" -u -m "$@"
echo "done"
