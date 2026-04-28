#!/bin/bash
source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
ulimit -n 10240

SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
VALDIR=/mnt/projects/debruinz_project/singlify_validation
REFDIR=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
WL=/mnt/home/debruinz/Singlet-AI/singlify/whitelists/gex_737K-arc-v1.txt
OUTDIR=$VALDIR/starsolo/SRR32855204_matched

rm -rf $OUTDIR; mkdir -p $OUTDIR

echo "=== Decoding C01 .1fq ===" >&2
$SINGLIFY decode $VALDIR/1fq/SRR32855204.1fq \
  --r1 /dev/shm/c01_R1.fastq --r2 /dev/shm/c01_R2.fastq --threads 16 2>&1 | tail -2

echo "=== Checking decoded files ===" >&2
ls -lh /dev/shm/c01_R*.fastq || { echo "FATAL: decode files missing"; exit 1; }

echo "=== Running STARsolo ===" >&2
STAR --runMode alignReads \
  --genomeDir $REFDIR/star_2.7.11b \
  --readFilesIn /dev/shm/c01_R2.fastq /dev/shm/c01_R1.fastq \
  --soloType CB_UMI_Simple \
  --soloCBwhitelist $WL \
  --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 \
  --outSAMtype BAM SortedByCoordinate \
  --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM \
  --soloFeatures Gene GeneFull \
  --clip3pAdapterSeq AAGCAGTGGTATCAACGCAGAGTACATGGG \
  --clip3pAdapterMMp 0.1 \
  --outFilterScoreMin 30 \
  --outFileNamePrefix $OUTDIR/ \
  --runThreadN 20 2>&1 | tail -5

echo "=== MAPPING ==="
grep "Uniquely mapped reads %" $OUTDIR/Log.final.out
echo "=== CELLS (Gene filtered) ==="
wc -l < $OUTDIR/Solo.out/Gene/filtered/barcodes.tsv

rm -f /dev/shm/c01_R1.fastq /dev/shm/c01_R2.fastq
