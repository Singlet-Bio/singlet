#!/bin/bash
#SBATCH --job-name=vdjc-dl-a375
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# 10x A375 CRISPR 10k (cohort tenx_crispr_a375_10k): FASTQ tar + Cell Ranger 4.0.0 outputs (CC BY 4.0).
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
o=$WORK/data/vdjc/a375; mkdir -p $o
N=SC3_v3_NextGem_DI_CRISPR_10K
X=https://cf.10xgenomics.com/samples/cell-exp/4.0.0/$N
get() { [ -s "$2" ] || { curl -fsSL --retry 5 -o "$2.part" "$1" && mv "$2.part" "$2"; } || { rm -f "$2.part"; echo "missing $1"; }; }
for s in crispr_analysis.tar.gz feature_reference.csv filtered_feature_bc_matrix.h5 raw_feature_bc_matrix.h5 metrics_summary.csv; do get $X/${N}_$s $o/$s; done
ls -la $o
[ -d $o/crispr_analysis ] || { [ -s $o/crispr_analysis.tar.gz ] && mkdir -p $o/crispr_analysis && tar -xzf $o/crispr_analysis.tar.gz -C $o/crispr_analysis; }
if [ ! -d $o/fastqs ]; then
  get https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/4.0.0/$N/${N}_fastqs.tar $o/fastqs.tar
  mkdir -p $o/fastqs; tar -xf $o/fastqs.tar -C $o/fastqs --wildcards "*rispr*" "*RISPR*" 2>/dev/null || tar -xf $o/fastqs.tar -C $o/fastqs
  rm -f $o/fastqs.tar
fi
find $o -maxdepth 4 | grep -v "fastq.gz$" | head -40; find $o/fastqs -name "*.fastq.gz" | head -20
