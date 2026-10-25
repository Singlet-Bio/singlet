#!/bin/bash
# Doublet evaluation -> $LQC/dbl/eval (small CSV / markdown, pulled back to bench/latent_qc/doublets/results)
#SBATCH --job-name=lqc-eval
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/study/latent-qc:refs/heads/study/latent-qc || true
git --git-dir=$W/repos/singlet-py.git show refs/heads/study/latent-qc:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh
source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
B=$SRC/bench/latent_qc/doublets; export PYTHONPATH=$B
L=""; for n in kang21 pbmc-1A-dm pbmc-1B-dm pbmc-1C-dm pbmc-2ctrl-dm pbmc-2stim-dm pbmc-ch; do [ -f $LQC/dbl/scores/$n/lineage_prob.npy ] || L="$L $n"; done
[ -z "$L" ] || rtt env PYTHONPATH=$B:$TORCH:$WORK/src/singlet-py/claude__celltype-nn/python python $B/ct_lineage.py $L
rt env PYTHONPATH=$B python $B/evaluate.py $LQC/dbl/eval
