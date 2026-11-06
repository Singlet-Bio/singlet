#!/bin/bash
#SBATCH --job-name=gk2-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# gpu/bold-kern2 summarizer (CPU). Parameters (--export): JOBSPEC "kind=jobid=label;..." (kern2_summary.py), OUT (default results/overnight).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/gpu-kern2; OUT=${OUT:-$J/results/overnight}; mkdir -p $OUT
IR=$(ls $W/src/singlet-agents/*/hpc/clipper/in_runtime.sh | head -1); source $IR
in_runtime python3 $J/kern2_summary.py summary "${JOBSPEC:?}" $OUT
ls -la $OUT
