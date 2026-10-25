#!/bin/bash
#SBATCH --job-name=vdjc-dl-vdj
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# V(D)J benchmark inputs -> $WORK/data/vdjc/<dataset>/: 10x FASTQ tars + Cell Ranger outputs (CC BY 4.0, no login),
# GSE145370 TCR runs from SRA (prefetch + fasterq-dump, ~100x faster than ENA HTTPS from Clipper) + the authors' contigs.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
export PATH=$ENV/bin:$PATH HOME=$WORK/cache/condahome
D=$WORK/data/vdjc
get() { [ -s "$2" ] || curl -fsSL --retry 5 -C - -o "$2.part" "$1" && mv "$2.part" "$2" 2>/dev/null || [ -s "$2" ]; }
X=https://cf.10xgenomics.com/samples/cell-vdj
# 1. pbmc2: 5' v1, Cell Ranger 3.0.0
for lib in t b; do
  o=$D/pbmc2_$lib; mkdir -p $o; n=vdj_v1_hs_pbmc2_$lib
  for s in filtered_contig_annotations.csv all_contig_annotations.csv clonotypes.csv; do get $X/3.0.0/$n/${n}_$s $o/$s; done
  if [ ! -d $o/fastqs ]; then get $X/3.0.0/$n/${n}_fastqs.tar $o/fastqs.tar; mkdir -p $o/fastqs; tar -xf $o/fastqs.tar -C $o/fastqs; rm $o/fastqs.tar; fi
done
get $X/3.0.0/vdj_v1_hs_pbmc2_5gex_protein/vdj_v1_hs_pbmc2_5gex_protein_filtered_feature_bc_matrix.h5 $D/pbmc2_t/gex_filtered.h5
cp -n $D/pbmc2_t/gex_filtered.h5 $D/pbmc2_b/gex_filtered.h5
# 2. sc5p_v2_hs_PBMC_10k: 5' v2, 26x90, Cell Ranger 5.0.0
for lib in t b; do
  o=$D/pbmc10k_$lib; mkdir -p $o; n=sc5p_v2_hs_PBMC_10k
  for s in filtered_contig_annotations.csv all_contig_annotations.csv clonotypes.csv; do get $X/5.0.0/$n/${n}_${lib}_$s $o/$s; done
  if [ ! -d $o/fastqs ]; then get $X/5.0.0/$n/${n}_${lib}_fastqs.tar $o/fastqs.tar; mkdir -p $o/fastqs; tar -xf $o/fastqs.tar -C $o/fastqs; rm $o/fastqs.tar; fi
  get $X/5.0.0/$n/${n}_filtered_feature_bc_matrix.h5 $o/gex_filtered.h5
done
# 3. GSE145370 (ESCC, TCR, authors' Cell Ranger 3.0.1): four libraries
declare -A GSM=( [SRR11094265]="GSM4317428 S158A" [SRR11094266]="GSM4317429 S158B" [SRR11094260]="GSM4317423 S133A" [SRR11094261]="GSM4317424 S133B" )
for run in "${!GSM[@]}"; do
  read gsm smp <<< "${GSM[$run]}"
  o=$D/gse145370_$smp; mkdir -p $o/fastqs
  sfx=$(echo $gsm | sed 's/...$/nnn/')
  get https://ftp.ncbi.nlm.nih.gov/geo/samples/$sfx/$gsm/suppl/${gsm}_${smp}_filtered_contig_annotations.csv.gz $o/filtered_contig_annotations.csv.gz \
    && gunzip -f $o/filtered_contig_annotations.csv.gz || echo "no contigs for $gsm"
  if [ ! -s $o/fastqs/${run}_S1_L001_R2_001.fastq.gz ]; then
    T=/tmp/$USER-$SLURM_JOB_ID/$run; mkdir -p $T
    prefetch -q -O $T $run && fasterq-dump -q --split-files --include-technical -e 8 -O $T $T/$run
    ls -la $T
    pigz -p 8 -c $T/${run}_1.fastq > $o/fastqs/${run}_S1_L001_R1_001.fastq.gz
    pigz -p 8 -c $T/${run}_2.fastq > $o/fastqs/${run}_S1_L001_R2_001.fastq.gz
    rm -rf $T
  fi
done
du -sh $D/*; find $D -name "*.csv" | head -20
