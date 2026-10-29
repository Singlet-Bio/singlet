#!/bin/bash
# Build doublet datasets:  sbatch --export=ALL,MODES="xili kang hgmm homsim" prep.sbatch
#SBATCH --job-name=lqc-prep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/study/latent-qc:refs/heads/study/latent-qc || true
git --git-dir=$W/repos/singlet-py.git show refs/heads/study/latent-qc:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh
source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
B=$SRC/bench/latent_qc/doublets; export PYTHONPATH=$B
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap "rm -rf $SCRATCH" EXIT; export TMPDIR=$SCRATCH
for m in ${MODES:-xili kang hgmm homsim}; do rt env PYTHONPATH=$B python $B/prep.py $m; done
ls $LQC/dbl/data/*/ds.npz | xargs -n1 dirname | xargs -n1 basename > $LQC/dbl/datasets.txt; cat $LQC/dbl/datasets.txt
