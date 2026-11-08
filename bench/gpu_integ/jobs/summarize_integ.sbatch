#!/bin/bash
#SBATCH --job-name=gi-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# gpu/bold-integ summarizer (CPU). Parameters (--export): JOBSPEC "build=ID;ident=ID;paths=ID;tp=ID".
# Writes $W/jobs/gpu-integ/results/summary.md. Reads only logs and result files.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/gpu-integ; OUT=$J/results; mkdir -p $OUT
IR=$(ls $W/src/singlet-agents/*/hpc/clipper/in_runtime.sh | head -1); source $IR
in_runtime python3 $J/integ_summary.py "${JOBSPEC:?}" $OUT
ls -la $OUT
