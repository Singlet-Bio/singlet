#!/bin/bash
#SBATCH -J nh-singlet
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 16
#SBATCH --mem=150G
#SBATCH -t 03:00:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Singlet (build 92095e29: is_cell = called_v2, knee = called_knee) on the Gaublomme 2019 mouse nucleus-hashing RNA library (10x 3' v2, GRCm39).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/expression-sota/data/nucleus_hash; RG=$W/release-gate; BIN=$RG/bin/${SHA8:-92095e29}/singlet-cpu
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCm39-2024-A; O=$W/expression-sota/nucleus_hash/${OUTNAME:-singlet}; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export TMPDIR=$SCR SINGLET_STALL_TIMEOUT=10800
$BIN encode --reads $D/SRR8703773_2.fastq.gz $D/SRR8703773_3.fastq.gz -o $SCR/s.1fq > $O/encode.log 2>&1; tail -3 $O/encode.log
cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?"
tail -5 $O/run.log; grep -E "cell_calling|protocol|chemistry|whitelist" $O/run.log | head -8; wc -l $O/cell_calls.tsv
