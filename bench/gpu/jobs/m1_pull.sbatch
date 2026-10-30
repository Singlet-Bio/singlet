#!/bin/bash
#SBATCH --job-name=w11-pull
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:28:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Pull an NVIDIA CUDA devel image (no --fakeroot). Cache and tmp in node-local /tmp, image into $WORK/images.
set -eu
WORK=/mnt/projects/debruinz_project/debruinz/singlet
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
export APPTAINER_CACHEDIR=$S/cache APPTAINER_TMPDIR=$S/tmp; mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
IMG=$WORK/images/cuda-12.4.1-devel-ubuntu22.04.sif
[ -s $IMG ] || apptainer pull $IMG docker://nvidia/cuda:12.4.1-devel-ubuntu22.04
ls -l $IMG
