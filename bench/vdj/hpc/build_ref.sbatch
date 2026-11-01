#!/bin/bash
#SBATCH --job-name=vdjc-ref
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# V(D)J segment references: GENCODE primary assembly (the pipeline's 2024-A reference) + the alternate-haplotype loci
# (GENCODE "chr_patch_hapl_scaff" annotation on the full GRCh38.p14 / GRCm39 genome), like Cell Ranger's -alts reference.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
R=$WORK/data/references/junctions; A=$WORK/data/vdjc/alts; mkdir -p $A $WORK/state/vdjc/ref
G=https://ftp.ebi.ac.uk/pub/databases/gencode
get() { [ -s "$2" ] || { curl -fsSL --retry 5 -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
get $G/Gencode_human/release_44/GRCh38.p14.genome.fa.gz $A/GRCh38.p14.genome.fa.gz
get $G/Gencode_human/release_44/gencode.v44.chr_patch_hapl_scaff.annotation.gtf.gz $A/gencode.v44.chr_patch_hapl_scaff.annotation.gtf.gz
get $G/Gencode_mouse/release_M33/GRCm39.genome.fa.gz $A/GRCm39.genome.fa.gz
get $G/Gencode_mouse/release_M33/gencode.vM33.chr_patch_hapl_scaff.annotation.gtf.gz $A/gencode.vM33.chr_patch_hapl_scaff.annotation.gtf.gz
# NCBI RefSeq (public domain): IG / TR segments GENCODE lacks (TRBV4-3, TRBV6-3, TRAJ15, IGHV1-8 ...)
N=https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF
get $N/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.gff.gz $A/refseq_grch38.gff.gz
get $N/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz $A/refseq_grch38.fna.gz
get $N/000/001/635/GCF_000001635.27_GRCm39/GCF_000001635.27_GRCm39_genomic.gff.gz $A/refseq_grcm39.gff.gz
get $N/000/001/635/GCF_000001635.27_GRCm39/GCF_000001635.27_GRCm39_genomic.fna.gz $A/refseq_grcm39.fna.gz
python -P -m singlet.vdj.reference $R/GRCh38-2024-A/GRCh38.primary_assembly.genome.fa.gz \
  $R/GRCh38-2024-A/gencode.v44.primary_assembly.annotation.gtf.gz $WORK/state/vdjc/ref/vdj_human.tsv.gz \
  $A/GRCh38.p14.genome.fa.gz $A/gencode.v44.chr_patch_hapl_scaff.annotation.gtf.gz \
  $A/refseq_grch38.fna.gz $A/refseq_grch38.gff.gz
python -P -m singlet.vdj.reference $R/GRCm39-2024-A/GRCm39.primary_assembly.genome.fa.gz \
  $R/GRCm39-2024-A/gencode.vM33.primary_assembly.annotation.gtf.gz $WORK/state/vdjc/ref/vdj_mouse.tsv.gz \
  $A/GRCm39.genome.fa.gz $A/gencode.vM33.chr_patch_hapl_scaff.annotation.gtf.gz \
  $A/refseq_grcm39.fna.gz $A/refseq_grcm39.gff.gz
zcat $WORK/state/vdjc/ref/vdj_human.tsv.gz | cut -f1,7 | grep -E "^(TRBV(11-2|21-1|4-3|6-3|7-8|5-8|6-9)|TRAJ(8|15)|IGHV(1-8|4-30-4))\s" || true
