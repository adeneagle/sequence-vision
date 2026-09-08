#!/bin/bash
#SBATCH --job-name=o1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-reserved
#SBATCH --mem=192G
#SBATCH --qos=mid
#SBATCH --time=10:00:00
#SBATCH --output=logs/o1-%j-%x.out
set -euo pipefail
SV=/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision
cd "$SV"; mkdir -p logs results
# Args come in POSITIONALLY ("$@"), not via EXTRA. Reason: `sbatch --export`
# parses its value as a comma-separated VAR=VALUE list, so any argument
# containing a comma (e.g. --settings 10:d,100:d,261:d) is silently TRUNCATED at
# the first comma and everything after it is dropped -- which quietly reverted a
# t-sweep to a single setting AND dropped --cache, making the job reload an old
# cache and re-run the wrong thing. Positional args are not re-parsed.
"$SV/.pixi/envs/default/bin/python" -u -m probes.o1_cryofm_benchmark "$@" ${EXTRA:-}
echo "done"
