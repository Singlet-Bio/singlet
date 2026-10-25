#!/bin/bash
#SBATCH --job-name=at-vae
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-6%3
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Per-sample scVI-style VAE latent for every library and gene view.
set -euo pipefail; sleep $((SLURM_ARRAY_TASK_ID*7))
export STUDY_BRANCH=study/latent-qc-ambient-truth; W=/mnt/projects/debruinz_project/debruinz/singlet; git --git-dir=$W/repos/singlet-py.git fetch -q --refmap= origin +refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH; git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh; source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
LIBS=(hgmm1k_v3_singlet hgmm1k_v3 hgmm5k_v3 hgmm10k_v3 hgmm1k_v2 hgmm6k_v2 hgmm12k_v2)
L=${LIBS[$SLURM_ARRAY_TASK_ID]}; T=$SRC/bench/latent_qc/ambient_truth
for V in full A B; do rtt python $T/vae_latent.py $L $V; done
