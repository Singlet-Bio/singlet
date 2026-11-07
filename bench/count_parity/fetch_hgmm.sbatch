#!/bin/bash
#SBATCH --job-name=cp-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Fetch 10x barnyard datasets (public, 10x Genomics CDN) into pipeline-benchmark/data/<ds>/ in the layout of hgmm1k_v3:
# FASTQ tar extracted, gex_fastqs.txt (R1/R2 list), the published Cell Ranger 3.0.0 filtered h5.
set -euo pipefail
PB=/mnt/projects/debruinz_project/debruinz/singlet/pipeline-benchmark
fetch() {  # ds 10x_name
  local D=$PB/data/$1 N=$2 U=https://cf.10xgenomics.com/samples/cell-exp/3.0.0/$2
  mkdir -p $D; cd $D
  [ -e .extracted ] && { echo "$1 present"; return; }
  curl -fsSL --retry 5 -o ${N}_fastqs.tar $U/${N}_fastqs.tar
  tar -xf ${N}_fastqs.tar && rm -f ${N}_fastqs.tar
  curl -fsSL --retry 5 -o cellranger_${N}_filtered_feature_bc_matrix.h5 $U/${N}_filtered_feature_bc_matrix.h5
  find $D -name "*.fastq.gz" | sort > all_fastqs.txt
  grep -E "_R[12]_" all_fastqs.txt > gex_fastqs.txt; : > adt_fastqs.txt
  touch .extracted; du -sh $D; wc -l gex_fastqs.txt
}
fetch hgmm5k_v3 hgmm_5k_v3
fetch hgmm1k_v2 hgmm_1k_v2
