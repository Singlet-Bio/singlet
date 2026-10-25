#!/bin/bash
#SBATCH --job-name=at-score
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --array=0-6%4
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Score production + DecontX outputs (full-full, A->B, B->A), then the noise ceilings.
set -euo pipefail; sleep $((SLURM_ARRAY_TASK_ID*7))
export STUDY_BRANCH=study/latent-qc-ambient-truth; W=/mnt/projects/debruinz_project/debruinz/singlet; git --git-dir=$W/repos/singlet-py.git fetch -q --refmap= origin +refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH; git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh; source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
LIBS=(hgmm1k_v3_singlet hgmm1k_v3 hgmm5k_v3 hgmm10k_v3 hgmm1k_v2 hgmm6k_v2 hgmm12k_v2)
L=${LIBS[$SLURM_ARRAY_TASK_ID]}; T=$SRC/bench/latent_qc/ambient_truth
O=$LQC/ambient_truth/out/$L; rm -f $O/ext_*.csv $O/extcorr_*.csv
rt python -I $T/score_ext.py $L full full
rt python -I $T/score_ext.py $L A B
rt python -I $T/score_ext.py $L B A
