#!/bin/bash
#SBATCH --job-name=sp-stream
#SBATCH --partition=cpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/projects/debruinz_project/cellarium/pipeline/logs/stream_pileup_%j.out
set -euo pipefail

# ── Streaming pipeline: STARsolo → singlet-pileup (zero-disk BAM) ──
#
# Key innovation: STARsolo outputs unsorted BAM to stdout, piped directly
# to singlet-pileup. No intermediate BAM touches disk.
#
# Thread allocation:
#   STAR alignment:     8 threads
#   BAM decompression:  4 threads (singlet-pileup side)
#   Total:             12 threads
#
# Usage:
#   sbatch stream_pipeline.sh <SRR_ID> <OUT_DIR> [R1.fastq.gz R2.fastq.gz]

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <SRR_ID> <OUT_DIR> [R1.fastq.gz R2.fastq.gz]"
    exit 1
fi

SRR="$1"
OUT_DIR="$2"
R1="${3:-}"
R2="${4:-}"

module load miniconda3/25.5.1
source /opt/gvsu/clipper/2025.12/spack/apps/linux-cascadelake/miniconda3-25.5.1-xe7kyofwhfxilia75rj5t63zf6wpzzcr/etc/profile.d/conda.sh
set +u; conda activate cellarium; set -u

# ── Paths ──
REF_DIR="/mnt/projects/debruinz_project/cellarium/reference"
STAR_INDEX="$REF_DIR/GRCh38-2024-A/star_2.7.11b"
INDICES_DIR="$REF_DIR/pileup_indices"
BARCODE_WL="/mnt/projects/debruinz_project/irungum_1000G/scRNAseq/software/cellranger-10.0.0/lib/python/cellranger/barcodes/3M-february-2018_TRU.txt.gz"
SNP_TSV="$INDICES_DIR/common_snps_hg38.tsv.gz"
EXON_BED="$INDICES_DIR/exons_hg38.bed.gz"
PILEUP_BIN="/mnt/home/debruinz/Singlet-AI/singlet-pileup/build/singlet-pileup"

mkdir -p "$OUT_DIR/starsolo" "$OUT_DIR/pileup"

# ── Download FASTQs if not provided ──
if [[ -z "$R1" ]]; then
    R1="$OUT_DIR/${SRR}_1.fastq.gz"
    R2="$OUT_DIR/${SRR}_2.fastq.gz"
    if [[ ! -f "$R2" ]]; then
        echo "Downloading FASTQs for $SRR..."
        # Try ENA first (faster)
        PREFIX="${SRR:0:6}"
        SUFFIX="${SRR: -3}"
        wget -q "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/${PREFIX}/0${SUFFIX}/${SRR}/${SRR}_1.fastq.gz" -O "$R1" &
        wget -q "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/${PREFIX}/0${SUFFIX}/${SRR}/${SRR}_2.fastq.gz" -O "$R2" &
        wait
        echo "Downloaded: $(ls -lh "$R1" "$R2" 2>/dev/null)"
    fi
fi

echo "=== Streaming Pipeline ==="
echo "SRR:     $SRR"
echo "R1:      $R1 ($(du -h "$R1" | cut -f1))"
echo "R2:      $R2 ($(du -h "$R2" | cut -f1))"
echo "Output:  $OUT_DIR"
echo ""

# Decompress barcode whitelist
BC_PLAIN="/tmp/barcodes_$$.txt"
zcat "$BARCODE_WL" > "$BC_PLAIN"
echo "Barcode whitelist: $(wc -l < "$BC_PLAIN") barcodes"

# ── Phase 1: Streaming STARsolo → singlet-pileup ──
# STARsolo outputs BAM to stdout (--outStd BAM), piped to pileup engine
#
# For the first pass, we ALSO write sorted BAM for cellSNP/vireo validation.
# In production, we'd skip the sorted BAM entirely.

echo ""
echo "=== Phase 1: STARsolo + singlet-pileup (streaming) ==="
START_TIME=$(date +%s)

# Run STARsolo → tee → (sorted BAM on disk + singlet-pileup)
# Use process substitution to fork the BAM stream
STAR --runMode alignReads \
    --genomeDir "$STAR_INDEX" \
    --readFilesIn "$R2" "$R1" \
    --readFilesCommand zcat \
    --soloType CB_UMI_Simple \
    --soloCBwhitelist "$BC_PLAIN" \
    --soloCBstart 1 --soloCBlen 16 \
    --soloUMIstart 17 --soloUMIlen 12 \
    --soloFeatures Gene \
    --outSAMattributes NH HI nM AS CR UR CB UB GX GN \
    --outSAMtype BAM Unsorted \
    --outStd BAM \
    --runThreadN 8 \
    --outFileNamePrefix "$OUT_DIR/starsolo/" \
    | tee >(samtools sort -@ 4 -o "$OUT_DIR/starsolo/Aligned.sortedByCoord.out.bam" -) \
    | "$PILEUP_BIN" \
        --barcodes "$BC_PLAIN" \
        --snps "$SNP_TSV" \
        --exons "$EXON_BED" \
        --out-prefix "$OUT_DIR/pileup" \
        --out-chrm "$OUT_DIR/pileup/chrm.bam" \
        --threads 4 \
        -

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=== Pipeline complete in ${ELAPSED}s ==="

# Index sorted BAM for validation
echo "Indexing sorted BAM..."
samtools index -@ 4 "$OUT_DIR/starsolo/Aligned.sortedByCoord.out.bam"

# ── Phase 2: Extract filtered barcodes from STARsolo ──
echo ""
echo "=== Phase 2: Extract filtered barcodes ==="
FILTERED_BC="$OUT_DIR/starsolo/Solo.out/Gene/filtered/barcodes.tsv"
if [[ -f "$FILTERED_BC" ]]; then
    echo "STARsolo filtered barcodes: $(wc -l < "$FILTERED_BC")"
else
    echo "WARNING: STARsolo filtered barcodes not found"
fi

# ── Phase 3: Stats ──
echo ""
echo "=== Results ==="
cat "$OUT_DIR/pileup/pileup_stats.json"
echo ""
echo "Output files:"
ls -lh "$OUT_DIR/pileup/"
echo ""
echo "BAM:"
samtools flagstat "$OUT_DIR/starsolo/Aligned.sortedByCoord.out.bam" | head -5

# Cleanup
rm -f "$BC_PLAIN"

echo ""
echo "=== Streaming pipeline complete in ${ELAPSED}s ==="
