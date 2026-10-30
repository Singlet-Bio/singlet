#!/bin/bash
#SBATCH --job-name=w11-ing-nvcomp
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only: pip-install the nvCOMP wheel into a job-local venv, keep include/ and lib64/ under our job dir (libraries, no data).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
D=$W/jobs/w11/ingest/nvcomp; mkdir -p $D
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
IMG=$W/images/python-3.12-bookworm.sif
apptainer exec -B $W $IMG bash -c "python -m venv $S/v && $S/v/bin/pip install -q nvidia-libnvcomp-cu12 2>&1 | tail -2; $S/v/bin/pip list 2>/dev/null | grep -i nvcomp"
cp -r $S/v/lib/python3.12/site-packages/nvidia/libnvcomp/include $S/v/lib/python3.12/site-packages/nvidia/libnvcomp/lib64 $D/
ls -la $D/lib64; du -sh $D
cat $D/include/nvcomp/version.h | grep -E "define NVCOMP_(VER|MAJOR|MINOR|PATCH)" || true
