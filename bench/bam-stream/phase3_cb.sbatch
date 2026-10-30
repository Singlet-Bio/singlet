#!/bin/bash
#SBATCH --job-name=bs-cb
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=<id>[,BIN=singlet-bs-dev] phase3_cb.sbatch      (~2 core-h per 67 M reads)
# Phase 3 step 1: how much does STAR's CB correction (1MM_multi_Nbase_pseudocounts) change CB vs the raw CR in the sorted BAM?
# STAR's sorted BAM is kept in job /tmp (SINGLET_KEEP_STAR_BAM=1), tags CR CY CB are read with samtools, tallied, and deleted.
# Only the tallies are written to $W/bam-stream/p3/<SET>/.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/${BIN:-singlet-bs-dev}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A; [ "$SET" = GSM5411066 ] && G=$REF/GRCm39-2024-A
D=$W/data/bam-stream/$SET; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$SET
O=$W/bam-stream/p3/$SET; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-bscb-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF SINGLET_KEEP_STAR_BAM=1 SINGLET_EXTRA_SAM_ATTRS="CY"
sha256sum $BIN > $O/binary.txt; $BIN --help 2>&1 | head -0
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
cd $SCR
$BIN $D/sample.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 4 \
  --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off "${SCREEN[@]}" > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
BAM=$(ls $SCR/out/*Aligned.sortedByCoord.out.bam)
ls -l $BAM | tee $O/bam.txt
cp $SCR/out/star_Solo.out/Barcodes.stats $O/ 2>/dev/null || true
export PATH=$W/envs/pbench/bin:$PATH
samtools view -@ 2 $BAM | awk -F'	' -v out=$O -f $W/bam-stream/cb_tally.awk | tee $O/tally.txt
