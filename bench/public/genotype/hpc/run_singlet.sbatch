#!/bin/bash
#SBATCH --job-name=bpg-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch run_singlet.sbatch <pool> <input.1fq> [K]      K omitted or -1 = auto (the headline mode); K > 0 = told
# Singlet 1.0.0 (singlet-campaign, 02aa9b9) in pipeline mode with its default 1000G SNP panel, one pool.
# Output: $WORK/bench-public/runs/<pool>/singlet_<auto|kN>/ ; private outputs in .../private/ (never leave Clipper).
set -u
POOL=$1; IN=$2; K=${3:--1}
W=/mnt/projects/debruinz_project/debruinz/singlet
BIN=$W/pipeline-benchmark/bin/singlet-campaign
echo "sha256 $(sha256sum $BIN | cut -d' ' -f1)"
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
if [ "$K" = "-1" ]; then TAG=auto; else TAG=k$K; fi
O=$W/bench-public/runs/$POOL/singlet_$TAG; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cp $IN $SCR/s.1fq; cd $SCR
export SINGLET_STALL_TIMEOUT=10800
# default SNP panel (1000G, AF>=5%) is chosen by --pipeline itself; no --snps flag on purpose
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline \
  --n-donors $K --tagged-bam $O/possorted.bam --bam-sort-mode disk --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum resident" $O/time.txt
ls $O | head -80; grep -E "auto-K|\[demux\]" $O/run.log | head -40
