#!/bin/bash
#SBATCH --job-name=vdjc-prep-rep
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Replogle 2022 guide table: hg19 (UCSC) protospacers from the sgRNA ids, checked against the reads.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
H=$WORK/data/vdjc/hg19; mkdir -p $H
[ -s $H/hg19.fa.gz ] || { curl -fsSL --retry 5 -o $H/hg19.fa.gz.part https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz && mv $H/hg19.fa.gz.part $H/hg19.fa.gz; }
python -P $SRC/bench/crispr/prep_replogle.py $WORK/data/vdjc/replogle_k562_gg41 $H/hg19.fa.gz \
  $WORK/data/references/junctions/GRCh38-2024-A/gencode.v44.primary_assembly.annotation.gtf.gz --gemgroup 41
