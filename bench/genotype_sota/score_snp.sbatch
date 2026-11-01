#!/bin/bash
#SBATCH --job-name=gsota-score
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch score_snp.sbatch <run-suffix e.g. mc99cpu_binned> <pool> [<pool> ...]
# Counts-only concordance tables (3x3 per tool and donor) for Singlet variant runs plus vireo/souporcell/cellsnp-pileup.
set -u
SUF=$1; shift
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
L=$W/bench-public/code_gsota/public/genotype/lib
A=$W/bench-public/artifacts/genotype-sota; mkdir -p $A
SINGLET_RUN=singlet_$SUF python -I $L/snp_concordance.py $W/bench-public/runs $W/bench-public/data/kang_truth $A/snps_$SUF.tsv "$@"
