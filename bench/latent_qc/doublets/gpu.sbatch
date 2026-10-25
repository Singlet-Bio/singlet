#!/bin/bash
# GPU latent doublet scores per dataset (array over $LQC/dbl/datasets.txt): Solo-style VAE, atlas cell type network encoder.
#SBATCH --job-name=lqc-gpu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/study/latent-qc:refs/heads/study/latent-qc
git --git-dir=$W/repos/singlet-py.git show refs/heads/study/latent-qc:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh
source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
B=$SRC/bench/latent_qc/doublets; export PYTHONPATH=$B
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap "rm -rf $SCRATCH" EXIT; export TMPDIR=$SCRATCH
N=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $LQC/dbl/datasets.txt); [ -n "$N" ] || exit 0; echo "dataset $N"
nvidia-smi -L || true
rtt env PYTHONPATH=$B:$TORCH:$WORK/src/singlet-py/claude__celltype-nn/python python $B/latent_gpu.py $N ${METHODS:-solo atlas}
