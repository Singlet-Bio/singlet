#!/bin/bash
#SBATCH --job-name=bpg-singlet-fq
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch run_singlet_fq.sbatch <pool> <R1.fastq.gz> <R2.fastq.gz> [K]      K omitted or -1 = auto
# Same as run_singlet.sbatch but starts from FASTQs (encode first). Singlet 1.0.0 (singlet-campaign, 02aa9b9).
set -u
POOL=$1; R1=$2; R2=$3; K=${4:--1}
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
BIN=$W/pipeline-benchmark/bin/singlet-campaign
echo "sha256 $(sha256sum $BIN | cut -d' ' -f1)"
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
if [ "$K" = "-1" ]; then TAG=auto; else TAG=k$K; fi
O=$W/bench-public/runs/$POOL/singlet_$TAG; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $R1 $R2 -o $SCR/s.1fq -v > $O/encode.log 2>&1
tail -3 $O/encode.log
export SINGLET_STALL_TIMEOUT=10800
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 32 --pipeline \
  --n-donors $K --tagged-bam $O/possorted.bam --bam-sort-mode disk --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum resident" $O/time.txt; grep -E "auto-K|\[demux\]" $O/run.log | head -40
