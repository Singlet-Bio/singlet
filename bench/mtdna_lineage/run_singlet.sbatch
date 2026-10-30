#!/bin/bash
#SBATCH --job-name=mtl-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch run_singlet.sbatch <name>      (name = k562_3p | bpdcn_3p)
# The production pipeline (singlet-modfix = prod/modality-fixes mt caller) on a standard 10x 3' v3 library, as the atlas
# runs it, plus a tagged BAM (CB/UB) so the reference tools see the same alignments.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
NAME=$1; D=$W/data/maester/$NAME; O=$W/out/mtdna_lineage/$NAME
BIN=$W/pipeline-benchmark/bin/singlet-modfix
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $D/${NAME}_R1.fastq.gz $D/${NAME}_R2.fastq.gz -o $SCR/s.1fq -v > $O/encode.log 2>&1
tail -5 $O/encode.log; ls -la $SCR/s.1fq
export SINGLET_STALL_TIMEOUT=10800
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 32 --pipeline \
  --tagged-bam $O/possorted.bam --bam-sort-mode disk --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
tail -20 $O/run.log
grep -E "Elapsed|Maximum resident" $O/time.txt
samtools index -@ 16 $O/possorted.bam
ls -la $O $O/private
du -sh $O/*
