#!/bin/bash
#SBATCH --job-name=ghc-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# gpu/bold-hostcut summarizer (CPU). Parameters (--export): JOBSPEC "kind=jobid;..." (summarize_hostcut.py), OUT (default $W/jobs/gpu-hostcut/results).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/gpu-hostcut; OUT=${OUT:-$J/results}; mkdir -p $OUT
IR=$(ls $W/src/singlet-agents/*/hpc/clipper/in_runtime.sh | head -1); source $IR
in_runtime python3 $J/summarize_hostcut.py "${JOBSPEC:?}" $OUT
ls -la $OUT
