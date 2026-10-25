#!/bin/bash
#SBATCH --job-name=at-ext
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-20%6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Production harness (compiled from $PROD headers, default config) and DecontX, per (library, gene view).
set -euo pipefail; sleep $((SLURM_ARRAY_TASK_ID*7))
export STUDY_BRANCH=study/latent-qc-ambient-truth; W=/mnt/projects/debruinz_project/debruinz/singlet; git --git-dir=$W/repos/singlet-py.git fetch -q --refmap= origin +refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH; git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh; source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
LIBS=(hgmm1k_v3_singlet hgmm1k_v3 hgmm5k_v3 hgmm10k_v3 hgmm1k_v2 hgmm6k_v2 hgmm12k_v2); VIEWS=(full A B)
L=${LIBS[$((SLURM_ARRAY_TASK_ID/3))]}; V=${VIEWS[$((SLURM_ARRAY_TASK_ID%3))]}
T=$SRC/bench/latent_qc/ambient_truth; D=$LQC/ambient_truth/work/${L}_$V
rt python -I $T/dump_inputs.py $L $V
BIN=$LQC/ambient_truth/work/harness_$SLURM_JOB_ID
$CXX -std=c++20 -O2 -pthread -I$PROD/include $T/harness.cpp -o $BIN
export LD_LIBRARY_PATH=$W/envs/pbench/lib:${LD_LIBRARY_PATH:-}
for th in 1 16; do
  if [ "$V" != full ] && [ $th = 16 ]; then continue; fi
  mkdir -p $D/prod_t$th
  /usr/bin/time -f "%e %M" -o $D/prod_t$th/time.txt $BIN $D/in $D/prod_t$th $th 2>&1 | tail -n 4
done
rm -f $BIN
R $T/decontx.R $D
