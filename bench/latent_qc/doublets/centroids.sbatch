#!/bin/bash
#SBATCH --job-name=lqc-cent
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/study/latent-qc:refs/heads/study/latent-qc || true
git --git-dir=$W/repos/singlet-py.git show refs/heads/study/latent-qc:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh
source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
B=$SRC/bench/latent_qc/doublets; export PYTHONPATH=$B
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap "rm -rf $SCRATCH" EXIT; export TMPDIR=$SCRATCH
rt python $B/atlas_centroids.py
