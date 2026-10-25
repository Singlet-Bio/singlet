#!/bin/bash
#SBATCH --job-name=aa-extract
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-19%10
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
export STUDY_BRANCH=study/latent-qc-ambient-atlas; W=/mnt/projects/debruinz_project/debruinz/singlet; git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH; git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh; source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
AA=$SRC/bench/latent_qc/ambient_atlas
OUT=$LQC/ambient_atlas; REC=$OUT/rec; RHO=$OUT/rho; mkdir -p $REC $RHO
DUMP=/tmp/aa-dump-$SLURM_JOB_ID; mkdir -p $DUMP; trap 'rm -rf $DUMP' EXIT
NCH=20
rt python -P $AA/extract.py $REC $DUMP $SLURM_ARRAY_TASK_ID $NCH $OUT/npz $WORK/data/emptydrop
# production estimator (compiled unchanged from prod/modality-fixes headers)
$CXX -std=c++20 -O2 -pthread -I$PROD/include $AA/harness_split.cpp -o $DUMP/harness 2>&1 | tail -5
export LD_LIBRARY_PATH=$WORK/envs/pbench/lib:${LD_LIBRARY_PATH:-}
for d in $DUMP/GSM*; do
  g=$(basename $d)
  lower=$(awk '{print $6}' $d/dims.txt)
  for v in prod rule1; do
    L=0; [ $v = rule1 ] && L=$lower
    [ -s $RHO/${g}_$v.json ] || $DUMP/harness $d $RHO/${g}_$v.json $SLURM_CPUS_PER_TASK $L 2> $RHO/${g}_$v.err || echo "harness $g $v failed"
  done
  rm -rf $d
done
echo done
