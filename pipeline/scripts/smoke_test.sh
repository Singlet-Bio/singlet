#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=cpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/projects/debruinz_project/cellarium/pipeline/logs/smoke_test_%j.out
set -euo pipefail

# ── End-to-end smoke test for singlet-pileup prototype ──
# Test sample: GSM8313394 (GSE269338), SRR29320040
# Human 10xv3, 38M reads, 2013 cells, 92.2% mapping rate

module load miniconda3/25.5.1
source /opt/gvsu/clipper/2025.12/spack/apps/linux-cascadelake/miniconda3-25.5.1-xe7kyofwhfxilia75rj5t63zf6wpzzcr/etc/profile.d/conda.sh
set +u; conda activate cellarium; set -u

# ── Paths ──
WORK="/mnt/projects/debruinz_project/cellarium/pipeline/smoke_test/GSM8313394"
REF_DIR="/mnt/projects/debruinz_project/cellarium/reference"
INDICES_DIR="$REF_DIR/pileup_indices"
STAR_INDEX="$REF_DIR/GRCh38-2024-A/star_2.7.11b"
GENOME_FA="$REF_DIR/GRCh38-2024-A/fasta/genome.fa"
GTF="$REF_DIR/GRCh38-2024-A/genes/genes.gtf.gz"
SNP_VCF="$INDICES_DIR/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.vcf.gz"
BARCODE_WL="/mnt/projects/debruinz_project/irungum_1000G/scRNAseq/software/cellranger-10.0.0/lib/python/cellranger/barcodes/3M-february-2018_TRU.txt.gz"

SRR="SRR29320040"
mkdir -p "$WORK/fastq" "$WORK/starsolo" "$WORK/cellsnp" "$WORK/pileup"

echo "=== Phase 1: Download FASTQs ==="
if [[ ! -f "$WORK/fastq/${SRR}_2.fastq.gz" ]]; then
    echo "Downloading from ENA..."
    cd "$WORK/fastq"
    wget -q "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR293/040/${SRR}/${SRR}_1.fastq.gz" &
    wget -q "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR293/040/${SRR}/${SRR}_2.fastq.gz" &
    wait
    echo "Downloaded:"
    ls -lh "$WORK/fastq/"
else
    echo "FASTQs already present"
    ls -lh "$WORK/fastq/"
fi

echo ""
echo "=== Phase 2: STARsolo alignment ==="
if [[ ! -f "$WORK/starsolo/Aligned.sortedByCoord.out.bam" ]]; then
    echo "Running STARsolo..."

    # Check that STAR index exists
    if [[ ! -d "$STAR_INDEX" || ! -f "$STAR_INDEX/SA" ]]; then
        echo "ERROR: STAR index not found at $STAR_INDEX"
        echo "Run download_references.sh first"
        exit 1
    fi

    # Check barcode whitelist
    if [[ ! -f "$BARCODE_WL" ]]; then
        echo "Downloading 10x v3 barcode whitelist..."
        mkdir -p "$(dirname "$BARCODE_WL")"
        wget -q "https://github.com/cellranger-atac/cellranger-atac/raw/master/lib/python/atac/barcodes/737K-arc-v1.txt.gz" \
            -O "$BARCODE_WL" || {
            # Fallback: extract from STAR's built-in list
            echo "Using STAR whitelist..."
            BARCODE_WL=$(find "$STAR_INDEX" -name "*.txt" -o -name "*.txt.gz" | head -1)
        }
    fi

    # Decompress barcode whitelist for STAR (avoids process substitution issues)
    WL_PLAIN="/tmp/barcode_whitelist_$$"
    zcat "$BARCODE_WL" > "$WL_PLAIN"
    echo "Barcode whitelist: $(wc -l < "$WL_PLAIN") barcodes"

    # For 10x Chromium 3' v3: R1=28bp barcode+UMI, R2=read
    STAR --runMode alignReads \
        --genomeDir "$STAR_INDEX" \
        --readFilesIn "$WORK/fastq/${SRR}_2.fastq.gz" "$WORK/fastq/${SRR}_1.fastq.gz" \
        --readFilesCommand zcat \
        --soloType CB_UMI_Simple \
        --soloCBwhitelist "$WL_PLAIN" \
        --soloCBstart 1 --soloCBlen 16 \
        --soloUMIstart 17 --soloUMIlen 12 \
        --soloFeatures Gene \
        --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM \
        --outSAMtype BAM SortedByCoordinate \
        --outBAMsortingBinsN 50 \
        --outBAMsortingThreadN 4 \
        --runThreadN 8 \
        --outFileNamePrefix "$WORK/starsolo/" \
        --outTmpDir "/tmp/STARsolo_smoke_$$" \
        --limitBAMsortRAM 50000000000

    echo "STARsolo complete"
    rm -f "$WL_PLAIN"  # cleanup temp whitelist
    ls -lh "$WORK/starsolo/"

    # Index the BAM
    echo "Indexing BAM..."
    samtools index -@ 4 "$WORK/starsolo/Aligned.sortedByCoord.out.bam"
else
    echo "STARsolo BAM already exists"
    ls -lh "$WORK/starsolo/Aligned.sortedByCoord.out.bam"
fi

echo ""
echo "=== Phase 3: Extract filtered barcodes ==="
BARCODES="$WORK/starsolo/Solo.out/Gene/filtered/barcodes.tsv"
if [[ ! -f "$BARCODES" && ! -f "$WORK/starsolo/Solo.out/Gene/filtered/barcodes.tsv" ]]; then
    # If STARsolo didn't produce filtered barcodes, create from existing quant output
    EXISTING_QUANT="/mnt/projects/debruinz_project/cellarium/pipeline/quant/GSE269338/GSM8313394"
    if [[ -f "$EXISTING_QUANT/counts.1pz" ]]; then
        echo "Extracting barcodes from existing quant output..."
        python3 -c "
import json, os
mf = '$EXISTING_QUANT/sample_manifest.json'
with open(mf) as f:
    manifest = json.load(f)
# Look for barcode file in quant dir
import glob
bc_files = glob.glob('$EXISTING_QUANT/cell_metadata*')
if bc_files:
    import pandas as pd
    df = pd.read_parquet(bc_files[0])
    if 'barcode' in df.columns:
        df['barcode'].to_csv('$WORK/filtered_barcodes.tsv', index=False, header=False)
        print(f'Extracted {len(df)} barcodes')
"
        BARCODES="$WORK/filtered_barcodes.tsv"
    fi
fi

if [[ -f "$BARCODES" ]]; then
    echo "Filtered barcodes: $(wc -l < "$BARCODES")"
else
    echo "WARNING: No filtered barcodes found, using STARsolo raw barcodes"
    BARCODES="$WORK/starsolo/Solo.out/Gene/raw/barcodes.tsv"
fi

echo ""
echo "=== Phase 4: cellSNP-lite baseline ==="
BAM="$WORK/starsolo/Aligned.sortedByCoord.out.bam"
CELLSNP_OUT="$WORK/cellsnp"

if [[ ! -f "$CELLSNP_OUT/cellSNP.base.vcf.gz" ]]; then
    echo "Running cellSNP-lite (mode 1, known SNPs)..."
    cellsnp-lite \
        -s "$BAM" \
        -b "$BARCODES" \
        -R "$SNP_VCF" \
        -O "$CELLSNP_OUT" \
        --minMAF 0 \
        --minCOUNT 1 \
        --gzip \
        -p 8

    echo "cellSNP-lite complete"
    ls -lh "$CELLSNP_OUT/"
else
    echo "cellSNP output exists"
    ls -lh "$CELLSNP_OUT/"
fi

echo ""
echo "=== Phase 5: Summary ==="
echo "Input: SRR29320040 (38M reads, Human 10xv3, GSM8313394)"
echo ""
echo "STARsolo BAM:"
if [[ -f "$BAM" ]]; then
    samtools flagstat "$BAM" | head -5
    echo "BAM size: $(du -h "$BAM" | cut -f1)"
fi
echo ""
echo "cellSNP output:"
if [[ -f "$CELLSNP_OUT/cellSNP.base.vcf.gz" ]]; then
    echo "  Sites: $(zcat "$CELLSNP_OUT/cellSNP.base.vcf.gz" | grep -v '^#' | wc -l)"
    echo "  Matrix files:"
    ls -lh "$CELLSNP_OUT/"*.mtx* 2>/dev/null || echo "  (no mtx files)"
fi

echo ""
echo "=== Smoke test pipeline complete ==="
echo "Next: Build singlet-pileup and compare against cellSNP-lite output"
