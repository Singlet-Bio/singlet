#!/bin/bash
#SBATCH --job-name=dmx2-run
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:59:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Pipeline run of one staged sample, the same call as jobs/fq-read-source/run.sbatch with the singlet-campaign binary
# (production path, --pipeline). private/ (per-SNP counts, genotypes) is kept apart from out/ and stays on Clipper.
set -euo pipefail
SAMPLES=(GSM7873657 GSM7873658 GSM5102900 GSM5102901)
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; J=$W/jobs/demux-truth2
SET=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
BIN=$PB/bin/singlet-campaign
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
D=$W/data/demux-truth2/$SET
INPUT=$(ls $D/none/*.1fq | sort | paste -sd,)
O=$J/runs/$SET; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-dmx2run-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF TMPDIR=$SCR
echo "node=$(hostname) set=$SET input=$INPUT $(date -Is)"; sha256sum $BIN; cat ${BIN}.commit 2>/dev/null || true
cd $SCR
if [[ $INPUT == *,* ]]; then INARG=(--1fq $INPUT); else INARG=($INPUT); fi
/usr/bin/time -v -o $O/time.txt $BIN "${INARG[@]}" --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $SCR/out/ --threads 16 --pipeline --output-format mtx --bam-sort-mode disk --genome-shared off > $O/run.log 2>&1 \
  || { echo "pipeline exit $?"; tail -20 $O/run.log; exit 1; }
rsync -a --exclude '*.bam' --exclude '*.bai' --exclude '*Unmapped.out.mate*' --exclude '*.fq' --exclude '*.fastq' \
  --exclude '*_STARtmp*' --exclude 'star_tmp*' --exclude private $SCR/out/ $O/out/
rsync -a $SCR/out/private/ $O/private/
ls $O/private | head -20; grep -E "Elapsed|Maximum resident" $O/time.txt; tail -3 $O/run.log
rm -rf $D   # reads are temporary
