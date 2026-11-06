#!/bin/bash
#SBATCH --job-name=gbold-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Summarizer of the gpu/bold overnight campaign. CPU only. Parameters (--export): JOBSPEC "kind=jobid=label;..." (see overnight_summary.py), OUT (default results/overnight).
set -uo pipefail
[ -n "${JOBSPEC_FILE:-}" ] && JOBSPEC=$(cat "$JOBSPEC_FILE")
W=/mnt/projects/debruinz_project/debruinz/singlet; OUT=${OUT:-$W/jobs/gpu-bold/results/overnight}; mkdir -p $OUT
IR=$(ls $W/src/singlet-agents/*/hpc/clipper/in_runtime.sh | head -1); echo "in_runtime: $IR"; source $IR
: > $OUT/states.txt
for id in $(echo "$JOBSPEC" | tr ';' '\n' | cut -d= -f2); do sacct -j $id -n -X -P -o JobID,State,ExitCode,Elapsed,NodeList >> $OUT/states.txt; done
in_runtime python3 $W/jobs/gpu-bold/jobs/overnight_summary.py "$JOBSPEC" "$OUT"
ls -la $OUT; head -60 $OUT/summary.md | cut -c1-240
