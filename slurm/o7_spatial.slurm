#!/bin/bash
#SBATCH --job-name=o7_spatial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=cpu
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=logs/o7-%j-%x.out
set -euo pipefail
# Pure CPU: pyramid smoothing + ridge solves. No GPU, no CryoFM forward -- the
# latents are already extracted and the coordinates are recovered by rng replay.
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
"$SV/.pixi/envs/default/bin/python" -u -m "$@"
echo "done"
