#!/bin/bash
#SBATCH --job-name=vdjc-dl-crispr
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CRISPR benchmark inputs -> $WORK/data/vdjc/<dataset>/: 10x A549 5K CRISPR (3' v3.1, Cell Ranger 6.0.0 calls; CC BY 4.0),
# Replogle 2022 K562 essential gemgroup 41 guide runs (SRA) + the published per-cell calls (figshare h5ad, CC BY 4.0).
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
export PATH=$ENV/bin:$PATH HOME=$WORK/cache/condahome
D=$WORK/data/vdjc
get() { [ -s "$2" ] || curl -fsSL --retry 5 -C - -o "$2.part" "$1" && mv "$2.part" "$2" 2>/dev/null || [ -s "$2" ]; }
X=https://cf.10xgenomics.com/samples/cell-exp/6.0.0
A=SC3_v3_NextGem_DI_CRISPR_A549_5K_SC3_v3_NextGem_DI_CRISPR_A549_5K
o=$D/a549; mkdir -p $o
for s in count_feature_reference.csv count_crispr_analysis.tar.gz count_sample_feature_bc_matrix.h5 metrics_summary.csv; do get $X/$A/${A}_$s $o/$s; done
get $X/SC3_v3_NextGem_DI_CRISPR_A549_5K_Multiplex/SC3_v3_NextGem_DI_CRISPR_A549_5K_Multiplex_count_raw_feature_bc_matrix.h5 $o/raw_feature_bc_matrix.h5
[ -d $o/crispr_analysis ] || (mkdir -p $o/crispr_analysis && tar -xzf $o/count_crispr_analysis.tar.gz -C $o/crispr_analysis)
if [ ! -d $o/fastqs ]; then
  get $X/SC3_v3_NextGem_DI_CRISPR_A549_5K_Multiplex/SC3_v3_NextGem_DI_CRISPR_A549_5K_Multiplex_fastqs.tar $o/fastqs.tar
  mkdir -p $o/fastqs; tar -xf $o/fastqs.tar -C $o/fastqs; rm $o/fastqs.tar
fi
find $o -maxdepth 4 | head -60
# Replogle 2022 K562 essential, gemgroup 41
o=$D/replogle_k562_gg41; mkdir -p $o/fastqs
get https://ndownloader.figshare.com/files/35773886 $o/KD6_raw_files.csv
for run in SRR19331011 SRR19331013 SRR19331010 SRR19331014; do
  if [ ! -s $o/fastqs/${run}_S1_L001_R2_001.fastq.gz ]; then
    T=/tmp/$USER-$SLURM_JOB_ID/$run; mkdir -p $T
    prefetch -q -O $T $run && fasterq-dump -q --split-files --include-technical -e 8 -O $T $T/$run
    ls -la $T
    pigz -p 8 -c $T/${run}_1.fastq > $o/fastqs/${run}_S1_L001_R1_001.fastq.gz
    pigz -p 8 -c $T/${run}_2.fastq > $o/fastqs/${run}_S1_L001_R2_001.fastq.gz
    rm -rf $T
  fi
done
get https://ndownloader.figshare.com/files/35773219 $o/K562_essential_raw_singlecell_01.h5ad
du -sh $D/*
