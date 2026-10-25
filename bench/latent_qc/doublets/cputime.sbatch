#!/bin/bash
# CPU cost of the atlas encoder scores (atlas_knn, ct_*, atlas_nnls, atlas_clf) on Kang lane 2.1, written as <method>_cpu
#SBATCH --job-name=lqc-cputime
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/study/latent-qc:refs/heads/study/latent-qc || true
git --git-dir=$W/repos/singlet-py.git show refs/heads/study/latent-qc:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh
source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
B=$SRC/bench/latent_qc/doublets
for n in kang21 pbmc-ch; do
  rtt env PYTHONPATH=$B:$TORCH:$WORK/src/singlet-py/claude__celltype-nn/python SCORE_SUFFIX=_cpu FORCE=1 CUDA_VISIBLE_DEVICES= python $B/latent_gpu.py $n atlas
done
grep -h seconds $LQC/dbl/scores/{kang21,pbmc-ch}/*_cpu.json
