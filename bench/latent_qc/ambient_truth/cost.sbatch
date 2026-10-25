#!/bin/bash
#SBATCH --job-name=at-cost
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Cost: one process per method, ONE CPU thread, wall time and peak RSS (includes loading the library's trimmed raw matrix).
set -euo pipefail
export STUDY_BRANCH=study/latent-qc-ambient-truth; W=/mnt/projects/debruinz_project/debruinz/singlet; git --git-dir=$W/repos/singlet-py.git fetch -q --refmap= origin +refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH; git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh; source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
T=$SRC/bench/latent_qc/ambient_truth; P="tau=10000:kappa=1000:hier=1"
for L in hgmm5k_v3 hgmm10k_v3; do
  for S in "knn:k=30:$P" "knn:k=30:tau=1000:kappa=1000:hier=0" "cluster:tau=100:kappa=1000" "marker:m=100"; do
    rt python -I $T/run_method.py $L full full "$S" --tag cost
  done
done
