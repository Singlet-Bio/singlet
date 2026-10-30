#!/bin/bash
# Per-dataset stages, one dataset per array task. Submit from bench/:
#   N=$(python -c "import yaml; print(len(yaml.safe_load(open('../datasets.yaml'))['datasets']))")
#   sbatch --array=0-$((N-1)) slurm/array.sbatch
# then, after the array finishes, slurm/serial.sbatch (minibatch, eco, report).
# Timings need a whole node to themselves: keep --exclusive. Edit partition/account for your cluster.
#SBATCH --job-name=s1bench
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
DS=$(python -c "import yaml,sys; print(yaml.safe_load(open('../datasets.yaml'))['datasets'][int(sys.argv[1])]['id'])" "$SLURM_ARRAY_TASK_ID")
echo "dataset $DS on $(hostname)"
python -m s1bench env
for stage in fetch canon write decode ladder floor second; do
  python -m s1bench "$stage" --datasets "$DS"
done
python -m s1bench decode --datasets "$DS" --sweep --formats s1,h5ad_gzip,csc_i32
