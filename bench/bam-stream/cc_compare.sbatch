#!/bin/bash
#SBATCH --job-name=bs-ccmp
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch cc_compare.sbatch        (~0.5 core-h per sample; after the runs_cc runs)
# Cell calls, stream (r1) vs sorted (r2): GSM4006843 with celltypist Immune_All_High labels, Kang batch 2 with its
# published demuxlet/cell-type table as truth. Writes $W/bam-stream/results/cc_<set>.json.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
R=$W/bam-stream/runs_cc; mkdir -p $W/bam-stream/results
export CELLTYPIST_FOLDER=$W/data/celltypist
in_runtime python -I $W/bam-stream/cc_compare.py $R/GSM4006843/t4_r1 $R/GSM4006843/t4_r2 $W/bam-stream/results/cc_GSM4006843.json \
  --celltypist $W/data/celltypist/data/models/Immune_All_High.pkl
in_runtime python -I $W/bam-stream/cc_compare.py $R/kang21/t4_r1 $R/kang21/t4_r2 $W/bam-stream/results/cc_kang21.json \
  --kang $W/modality-bench/genotype_pool/truth/GSE96583_batch2.total.tsne.df.tsv.gz
