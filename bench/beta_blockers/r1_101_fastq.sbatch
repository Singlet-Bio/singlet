#!/bin/bash
#SBATCH --job-name=bb-r1-101q
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch r1_101_fastq.sbatch <SRR> <reads> <bin> [bin ...]
# A real 10x v3 library whose barcode read R1 is 101 nt (GSE230239 / GSM7208744 / SRR24236331), as FASTQ: the
# `singlet encode --reads R1 R2` path (the SRA path trims R1 itself). First <reads> spots; for each binary: encode,
# then run the pipeline on the .1fq and report reads in the archive, barcodes found, reads counted, cells called.
set -u
SRR=$1; NREADS=$2; shift 2
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; ENV=$W/envs/pbench
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
prefetch --max-size u -O $SCR/sra $SRR > $SCR/pf.log 2>&1 || { tail $SCR/pf.log; exit 1; }
SRA=$(ls $SCR/sra/$SRR/*.sra 2>/dev/null || ls $SCR/sra/*.sra)
mkdir -p $SCR/fq && fastq-dump -X $NREADS --split-files --gzip -O $SCR/fq $SRA > $SCR/fd.log 2>&1
ls -la $SCR/fq
for f in $SCR/fq/*_1.fastq.gz $SCR/fq/*_2.fastq.gz; do echo "$f: $(zcat $f | awk 'NR%4==2{print length($0)}' | sort -n | uniq -c | sort -rn | head -3 | tr '\n' ' ')"; done
R1=$(ls $SCR/fq/*_1.fastq.gz); R2=$(ls $SCR/fq/*_2.fastq.gz)
for b in "$@"; do
  echo "=== $b: encode (R1=$(basename $R1) as the barcode read)"
  $PB/bin/$b encode --reads $R1 $R2 -o $SCR/$b.1fq > $SCR/$b.enc.log 2>&1; echo "encode rc=$?"
  tail -8 $SCR/$b.enc.log | cut -c1-200
  mkdir -p $SCR/$b/out
  SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_REF_BASE=$REF $PB/bin/$b $SCR/$b.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
    --exons $G/genes/genes.gtf --out-prefix $SCR/$b/out/ --threads 8 --pipeline --output-format mtx --raw-matrix \
    --bam-sort-mode disk --genome-shared off > $SCR/$b.run.log 2>&1; echo "pipeline rc=$?"
  grep -E "\.1fq: [0-9]+ reads|Discovered|Pileup: [0-9]+ reads|R1 length|Trimming|called|ERROR|WARNING.*R1|UMI truncated" $SCR/$b.run.log | cut -c1-220 | head -14
  grep -E "Uniquely mapped reads %|Number of input reads" $SCR/$b/out/star_Log.final.out 2>/dev/null
done
