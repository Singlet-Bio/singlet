#!/bin/bash
#SBATCH --job-name=dmx2-qc
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Per-sample QC of the single-donor runs: pseudobulk alt-fraction histogram, covered and informative SNPs per cell.
# Output is aggregates only: $J/results/qc/qc_<id>.json
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/demux-truth2
mkdir -p $J/results/qc
for id in S1 S2 S5 S6 S3 S4; do
  $J/bin/demux_mix --samples $J/samples.tsv --qc $id --out $J/results/qc 2>&1 | grep -E '^\[qc\]|rror|mismatch' || echo "qc failed for $id"
done
ls $J/results/qc
