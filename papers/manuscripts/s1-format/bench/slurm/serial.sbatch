#!/bin/bash
# Stages that need every dataset (minibatch over the loader panel, ecosystem calls) and the report. Submit from bench/
# after slurm/array.sbatch:  sbatch --dependency=afterok:<array job id> slurm/serial.sbatch
#SBATCH --job-name=s1bench-serial
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
EXT=$(python -c "import yaml; print(','.join(d['id'] for d in yaml.safe_load(open('../datasets.yaml'))['extensions']))")
for stage in fetch canon write floor; do
  python -m s1bench "$stage" --extensions --datasets "$EXT"           # out-of-domain count data (Q11)
done
python -m s1bench minibatch
python -m s1bench eco
python -m s1bench report
