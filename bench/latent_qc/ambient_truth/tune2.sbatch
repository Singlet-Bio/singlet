#!/bin/bash
#SBATCH --job-name=at-tune2
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Hyperparameter grid for the latent-kNN / cluster mixture estimator on the TUNING library only (hgmm1k_v3_singlet).
set -euo pipefail
export STUDY_BRANCH=study/latent-qc-ambient-truth; W=/mnt/projects/debruinz_project/debruinz/singlet; git --git-dir=$W/repos/singlet-py.git fetch -q --refmap= origin +refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH; git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh; source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
T=$SRC/bench/latent_qc/ambient_truth; L=hgmm1k_v3_singlet
SPECS=()
for k in 30 50; do for tau in 100 1000 10000; do for h in 0 1; do for kap in 0 1000; do SPECS+=("knn:k=$k:tau=$tau:kappa=$kap:hier=$h"); done; done; done; done
for tau in 100 1000 10000; do for kap in 0 1000; do SPECS+=("cluster:tau=$tau:kappa=$kap"); done; done
rt python -I $T/run_method.py $L full full "${SPECS[@]}" --tag tune
rt python -I $T/run_method.py $L A B "${SPECS[@]}" --tag tune
rt python -I $T/run_method.py $L B A "${SPECS[@]}" --tag tune
