#!/bin/bash
#SBATCH --job-name=vdjc-dl-norman
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Norman 2019 (GSE133344) gem group 1 guide-barcode library from SRA with technical reads (ENA lacks the barcode read),
# the published cell identities and the GEO matrix (for the induced-target check).
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$WORK/envs/vdjcrispr/bin:$PATH HOME=$WORK/cache/condahome
o=$WORK/data/vdjc/norman2019_gem1; mkdir -p $o/fastqs
G=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl
get() { [ -s "$2" ] || { curl -fsSL --retry 5 -o "$2.part" "$1" && mv "$2.part" "$2"; } || { rm -f "$2.part"; echo "missing $1"; }; }
for f in GSE133344_filtered_cell_identities.csv.gz GSE133344_raw_cell_identities.csv.gz GSE133344_filtered_barcodes.tsv.gz GSE133344_filtered_genes.tsv.gz GSE133344_filtered_matrix.mtx.gz; do get $G/$f $o/$f; done
for run in SRR9602551 SRR9602552; do
  if [ ! -s $o/fastqs/${run}_S1_L001_R2_001.fastq.gz ]; then
    T=/tmp/$USER-$SLURM_JOB_ID/$run; mkdir -p $T
    prefetch -q -O $T $run && fasterq-dump -q --split-files --include-technical -e 8 -O $T $T/$run
    ls -la $T; for f in $T/*.fastq; do echo $f; head -2 $f | tail -1 | wc -c; done
    n=$(ls $T/*.fastq | wc -l)
    if [ "$n" -ge 2 ]; then
      pigz -p 8 -c $T/${run}_1.fastq > $o/fastqs/${run}_S1_L001_R1_001.fastq.gz
      pigz -p 8 -c $T/${run}_2.fastq > $o/fastqs/${run}_S1_L001_R2_001.fastq.gz
    fi
    rm -rf $T
  fi
done
ls -la $o $o/fastqs; zcat $o/GSE133344_filtered_cell_identities.csv.gz | head -3
