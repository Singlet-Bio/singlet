#!/bin/bash
#SBATCH --job-name=aa-analyze
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# usage: sbatch analyze.sbatch STEP   (STEP = a | mouse | prior | rho)
export STUDY_BRANCH=study/latent-qc-ambient-atlas; W=/mnt/projects/debruinz_project/debruinz/singlet; git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH; git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh; source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
AA=$SRC/bench/latent_qc/ambient_atlas; O=$LQC/ambient_atlas; T=$AA/samples_full.tsv
NUC=${NUC:-0.45}
case "$1" in
  a)     rt python $AA/analyze_a.py $O/rec $T $O/res_human $NUC ;;
  mouse) AA_ORG=mouse rt python $AA/analyze_a.py $O/rec $T $O/res_mouse $NUC ;;
  prior) rt python $AA/prior_test.py $O/rec $T $O/res_prior 20 ;;
  rho)   mkdir -p $O/res_rho; rt python $AA/rho_summary.py $O/rho $O/res_human/sample_summary.csv $O/res_rho ;;
esac
