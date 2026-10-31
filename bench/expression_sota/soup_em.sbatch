#!/bin/bash
#SBATCH -J soup-em
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:30:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota; REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
PY="$W/envs/pbench/bin/python3"; export PYTHONPATH=$W/envs/pylib-pysam
$PY $X/code/soup_kang_reads.py $W/bench-public/runs/kang2018_ctrl/singlet/possorted.bam $REF/genes/genes.gtf $X/ambient/ambient-rna/kang2018_ctrl_tissue.tsv.gz 2000 $X/nuclei/kang_soup.reads.tsv.gz
unset PYTHONPATH
$PY $X/code/soup_em.py $X/nuclei/hgmm_soup.reads.tsv.gz $X/nuclei/kang_soup.reads.tsv.gz $X/ambient/ambient-rna/kang2018_ctrl_tissue.tsv.gz $X/nuclei/soup_em_kang.json
