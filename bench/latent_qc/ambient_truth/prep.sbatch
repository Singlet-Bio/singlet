#!/bin/bash
#SBATCH --job-name=at-prep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --array=0,2,4,5,6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail; sleep $((SLURM_ARRAY_TASK_ID*10))
export STUDY_BRANCH=study/latent-qc-ambient-truth; W=/mnt/projects/debruinz_project/debruinz/singlet; git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH; git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh; source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
LIBS=(hgmm1k_v3_singlet hgmm1k_v3 hgmm5k_v3 hgmm10k_v3 hgmm1k_v2 hgmm6k_v2 hgmm12k_v2)
L=${LIBS[$SLURM_ARRAY_TASK_ID]}
mkdir -p $LQC/ambient_truth/data $LQC/ambient_truth/out
rt python -I $SRC/bench/latent_qc/ambient_truth/prep_lib.py $L
