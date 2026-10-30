#!/bin/bash
#SBATCH --job-name=w11-m2test
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Smoke test of the decode -> distinct_fraction.py pipe on a 2 M-read sample (read-only on the data).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
BIN=$W/pipeline-benchmark/bin/singlet-w3
IMG=$W/images/python-3.12-bookworm.sif; PY=$W/venvs/singlet/bin/python
F=$W/data/atlas-rerun/perf/GSM6996274/sample.1fq
$BIN decode $F --r2 /dev/stdout --threads 1 2>/tmp/dec.$SLURM_JOB_ID.err | head -8 | cut -c1-120
cat /tmp/dec.$SLURM_JOB_ID.err | head -8
time ($BIN decode $F --r2 /dev/stdout --threads 1 2>/dev/null | apptainer exec -B $W $IMG $PY -I $W/jobs/w11/distinct_fraction.py - --len 90)
