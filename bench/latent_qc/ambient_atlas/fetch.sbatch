#!/bin/bash
#SBATCH --job-name=aa-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=03:30:00
#SBATCH --array=0-39%10
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
export STUDY_BRANCH=study/latent-qc-ambient-atlas; W=/mnt/projects/debruinz_project/debruinz/singlet; git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH; git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh; source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
AA=$SRC/bench/latent_qc/ambient_atlas
OUT=$LQC/ambient_atlas/npz; mkdir -p $OUT
rt python -P $AA/fetch_driver.py $SLURM_ARRAY_TASK_ID 40 $OUT $EDROP $AA/samples.tsv
