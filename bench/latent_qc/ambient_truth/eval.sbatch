#!/bin/bash
#SBATCH --job-name=at-eval
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-6%4
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Fixed method set (hyperparameters chosen on hgmm1k_v3_singlet only) on every library: full-gene truth, and gene-split A->B / B->A.
set -euo pipefail; sleep $((SLURM_ARRAY_TASK_ID*7))
export STUDY_BRANCH=study/latent-qc-ambient-truth; W=/mnt/projects/debruinz_project/debruinz/singlet; git --git-dir=$W/repos/singlet-py.git fetch -q --refmap= origin +refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH; git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh; source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
LIBS=(hgmm1k_v3_singlet hgmm1k_v3 hgmm5k_v3 hgmm10k_v3 hgmm1k_v2 hgmm6k_v2 hgmm12k_v2)
L=${LIBS[$SLURM_ARRAY_TASK_ID]}; T=$SRC/bench/latent_qc/ambient_truth
P="tau=10000:kappa=1000:hier=1"
SPECS=("knn:k=30:$P" "knn:k=15:$P" "knn:k=50:$P" "knn:k=30:tau=1000:kappa=1000:hier=0" "cluster:tau=100:kappa=1000" "knn:k=30:$P:latent=vae"
       "marker:m=10" "marker:m=30" "marker:m=100" "marker:m=300" "marker:m=1000" "marker:m=all")
rt python -I $T/run_method.py $L full full "${SPECS[@]}" --tag eval
rt python -I $T/run_method.py $L A B "${SPECS[@]}" --tag eval
rt python -I $T/run_method.py $L B A "${SPECS[@]}" --tag eval
