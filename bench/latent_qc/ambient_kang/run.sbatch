#!/bin/bash
# Kang 2.1 genotype ambient truth vs the production estimator, full input and truncated inputs (legacy-atlas-like).
#SBATCH --job-name=lqc-kamb
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/study/latent-qc:refs/heads/study/latent-qc || true
git --git-dir=$W/repos/singlet-py.git show refs/heads/study/latent-qc:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh
source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
B=$SRC/bench/latent_qc/ambient_kang; O=$LQC/ambient_kang; mkdir -p $O
[ -f $O/in_T50/dims.txt ] || rt python $B/kang_ambient.py dump $O
$CXX -std=c++20 -O2 -pthread -I$PROD/include $B/harness_cells.cpp -o $O/harness_cells
for T in 0 5 10 20 30 50; do $O/harness_cells $O/in_T$T $O/run_T$T.json $O/rho_T$T.f64 8; cat $O/run_T$T.json; done
rt python $B/kang_ambient.py score $O
