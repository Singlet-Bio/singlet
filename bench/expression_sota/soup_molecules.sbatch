#!/bin/bash
#SBATCH -J soup-mol
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:30:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota; REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-GRCm39-barnyard
mkdir -p $X/ref; [ -e $X/ref/genome.fa ] || ln -s $REF/fasta/genome.fa $X/ref/genome.fa
PY="$W/envs/pbench/bin/python3 -I"; export PYTHONPATH=$W/envs/pylib-pysam
$W/envs/pbench/bin/python3 -c "import sys; sys.path.insert(0,'$W/envs/pylib-pysam'); import pysam; pysam.faidx('$X/ref/genome.fa'); print('faidx ok')"
$W/envs/pbench/bin/python3 $X/code/soup_molecules.py $X/hgmm_bam $X/hgmm_bam/possorted.bam $X/ref/genome.fa $REF/genes/genes.gtf $X/nuclei/hgmm_soup
$W/envs/pbench/bin/python3 $X/code/soup_analyze.py $X/nuclei/hgmm_soup.reads.tsv.gz $X/nuclei/hgmm_soup.json
