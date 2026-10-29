#!/bin/bash
# Harness vs production run on Kang 2.1: same estimator, the run's own cell set (17,004) instead of cell_calls.tsv (14,407).
#SBATCH --job-name=lqc-kamb2
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
git --git-dir=$W/repos/singlet-py.git fetch -q origin +refs/heads/study/latent-qc:refs/heads/study/latent-qc || true
git --git-dir=$W/repos/singlet-py.git show refs/heads/study/latent-qc:bench/latent_qc/common.sh > /tmp/lqc_common_$SLURM_JOB_ID.sh
source /tmp/lqc_common_$SLURM_JOB_ID.sh; lqc_sync
B=$SRC/bench/latent_qc/ambient_kang; O=$LQC/ambient_kang
for T in 0 20; do
  rt env CELLS=run THRESH=$T TAG=_runcells python $B/kang_ambient.py dump $O
  $O/harness_cells $O/in_T${T}_runcells $O/run_T${T}_runcells.json $O/rho_T${T}_runcells.f64 8; cat $O/run_T${T}_runcells.json
done
head -3 $W/modality-bench/genotype_pool/singlet_modfix/ambient_contamination.tsv
grep -E "\[ambient\]" $W/modality-bench/genotype_pool/singlet_modfix/run.log | head -3
