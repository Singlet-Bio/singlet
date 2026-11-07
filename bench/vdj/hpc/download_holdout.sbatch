#!/bin/bash
#SBATCH --job-name=vdjc-dl-holdout
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Held-out V(D)J libraries, downloaded only after the assembler was frozen: GSE145370 TCR libraries not used before
# (GEO file names and SRA runs discovered here) and 10x's 10k PBMC 5' v2 (Chromium Controller, Cell Ranger 6.1.0) T/B.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$WORK/envs/vdjcrispr/bin:$PATH HOME=$WORK/cache/condahome
D=$WORK/data/vdjc
get() { [ -s "$2" ] || { curl -fsSL --retry 5 -o "$2.part" "$1" && mv "$2.part" "$2"; } || { rm -f "$2.part"; echo "missing $1"; }; }
curl -fsSL "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA606979&result=read_run&fields=run_accession,sample_alias,sample_title,experiment_title&format=tsv" > $D/gse145370_runs.tsv
n=0
for gsm in GSM4317425 GSM4317426 GSM4317427 GSM4317430 GSM4317431 GSM4317432 GSM4317433 GSM4317434 GSM4317435 GSM4317436; do
  [ $n -ge 4 ] && break
  sfx=GSM4317nnn
  f=$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/geo/samples/$sfx/$gsm/suppl/ | grep -o "${gsm}_[A-Za-z0-9]*_filtered_contig_annotations.csv.gz" | head -1 || true)
  run=$(grep -w "$gsm" $D/gse145370_runs.tsv | cut -f1 | head -1 || true)
  [ -z "$f" ] || [ -z "$run" ] && { echo "skip $gsm ($f $run)"; continue; }
  smp=$(echo $f | sed "s/${gsm}_//; s/_filtered.*//")
  o=$D/gse145370_$smp; mkdir -p $o/fastqs
  get https://ftp.ncbi.nlm.nih.gov/geo/samples/$sfx/$gsm/suppl/$f $o/filtered_contig_annotations.csv.gz && gunzip -f $o/filtered_contig_annotations.csv.gz
  if [ ! -s $o/fastqs/${run}_S1_L001_R2_001.fastq.gz ]; then
    T=/tmp/$USER-$SLURM_JOB_ID/$run; mkdir -p $T
    prefetch -q -O $T $run && fasterq-dump -q --split-files --include-technical -e 8 -O $T $T/$run
    pigz -p 8 -c $T/${run}_1.fastq > $o/fastqs/${run}_S1_L001_R1_001.fastq.gz
    pigz -p 8 -c $T/${run}_2.fastq > $o/fastqs/${run}_S1_L001_R2_001.fastq.gz
    rm -rf $T
  fi
  echo "holdout $gsm $smp $run"; n=$((n+1))
done
# 10x 10k PBMC 5' v2 (Chromium Controller): TCR / BCR outputs and the multiplex FASTQ tar (V(D)J libraries only)
X=https://cf.10xgenomics.com/samples/cell-vdj/6.1.0
C=10k_PBMC_5pv2_nextgem_Chromium_Controller_10k_PBMC_5pv2_nextgem_Chromium_Controller
for lib in t b; do
  o=$D/ctrl10k_$lib; mkdir -p $o
  get $X/$C/${C}_vdj_${lib}_filtered_contig_annotations.csv $o/filtered_contig_annotations.csv
done
get $X/$C/${C}_count_sample_feature_bc_matrix.h5 $D/ctrl10k_t/gex_filtered.h5; cp -n $D/ctrl10k_t/gex_filtered.h5 $D/ctrl10k_b/ 2>/dev/null || true
if [ ! -d $D/ctrl10k_t/fastqs ]; then
  get $X/10k_PBMC_5pv2_nextgem_Chromium_Controller_Multiplex/10k_PBMC_5pv2_nextgem_Chromium_Controller_Multiplex_fastqs.tar $D/ctrl10k.tar
  mkdir -p $D/ctrl10k_x; tar -tf $D/ctrl10k.tar | head -50
  tar -xf $D/ctrl10k.tar -C $D/ctrl10k_x --wildcards "*vdj*" "*VDJ*" "*TCR*" "*BCR*" 2>/dev/null || tar -xf $D/ctrl10k.tar -C $D/ctrl10k_x
  find $D/ctrl10k_x -name "*.fastq.gz" | head -40
fi
du -sh $D/gse145370_* $D/ctrl10k_* 2>/dev/null
