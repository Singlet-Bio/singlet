#!/usr/bin/env bash
#SBATCH --job-name=groundtruth-SRR32855204
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/projects/debruinz_project/singlify_validation/profiles/SRR32855204_groundtruth.log
set -euo pipefail
export TMPDIR=/dev/shm

CORPUS=/mnt/projects/debruinz_project/singlify_validation/corpus
STARSOLO=/mnt/projects/debruinz_project/singlify_validation/starsolo
PROFILES=/mnt/projects/debruinz_project/singlify_validation/profiles
OFQ=/mnt/home/debruinz/Singlet-AI/singlify/build/1fq
STAR_BIN=/mnt/home/debruinz/Singlet-AI/STAR/source/STAR_stock_baseline
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
GTF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf
WL=/mnt/home/debruinz/Singlet-AI/singlify/whitelists/gex_737K-arc-v1.txt

SRR=SRR32855204
echo "[$(date)] === Ground truth pipeline for $SRR ==="
echo "Host: $(hostname), CPUs: $(nproc)"

# ── Step 1: Decode .1fq → FASTQ ──
R1="$CORPUS/${SRR}_decoded_R1.fastq"
R2="$CORPUS/${SRR}_decoded_R2.fastq"

if [[ ! -f "$R2" ]] || [[ $(wc -l < "$R2") -lt 4 ]]; then
    echo ""
    echo "[$(date)] Step 1: Decoding .1fq → FASTQ..."
    /usr/bin/time -v "$OFQ" decode "$CORPUS/${SRR}.1fq" -o "$R1" "$R2" 2>&1
    echo "  R1: $(wc -l < "$R1" | awk '{print $1/4}') reads, $(ls -lh "$R1" | awk '{print $5}')"
    echo "  R2: $(wc -l < "$R2" | awk '{print $1/4}') reads, $(ls -lh "$R2" | awk '{print $5}')"
else
    echo "[$(date)] Step 1: FASTQs already decoded"
fi

# ── Step 2: Run STARsolo ──
OUTDIR="$STARSOLO/$SRR"
if [[ ! -f "$OUTDIR/Log.final.out" ]]; then
    mkdir -p "$OUTDIR"
    echo ""
    echo "[$(date)] Step 2: Running STARsolo (16 threads)..."
    echo "  Genome: $GENOME"
    echo "  GTF: $GTF"
    echo "  Whitelist: $WL"
    echo "  R1 (barcode): $R1"
    echo "  R2 (cDNA): $R2"

    /usr/bin/time -v "$STAR_BIN" \
        --runThreadN 16 \
        --genomeDir "$GENOME" \
        --readFilesIn "$R2" "$R1" \
        --soloType CB_UMI_Simple \
        --soloCBwhitelist "$WL" \
        --soloCBstart 1 --soloCBlen 16 \
        --soloUMIstart 17 --soloUMIlen 12 \
        --outSAMtype BAM SortedByCoordinate \
        --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM \
        --soloFeatures Gene GeneFull SJ \
        --outFileNamePrefix "${OUTDIR}/" \
        2>&1

    echo ""
    echo "[$(date)] STARsolo complete"
    echo "  Output:"
    ls -lh "$OUTDIR"/ | head -20
    echo "  Log.final.out highlights:"
    grep -E "Number of input reads|Uniquely mapped|Average|Total" "$OUTDIR/Log.final.out" 2>/dev/null
else
    echo "[$(date)] Step 2: STARsolo already ran"
fi

# ── Step 3: Run salmon/simpleaf (alternative quantification) ──
SALMON_DIR=/mnt/projects/debruinz_project/singlify_validation/salmon/$SRR
SIMPLEAF=/mnt/home/debruinz/.conda/envs/cellarium/bin/simpleaf
SALMON=/mnt/home/debruinz/.conda/envs/cellarium/bin/salmon
ALEVIN_FRY=/mnt/home/debruinz/.conda/envs/cellarium/bin/alevin-fry

# Build salmon index if not exists
SAL_IDX=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/salmon_idx
if [[ ! -d "$SAL_IDX" ]]; then
    echo ""
    echo "[$(date)] Step 3a: Building salmon/simpleaf index..."
    mkdir -p "$SAL_IDX"
    export ALEVIN_FRY_HOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/af_home
    mkdir -p "$ALEVIN_FRY_HOME"
    
    /usr/bin/time -v "$SIMPLEAF" index \
        --output "$SAL_IDX" \
        --fasta /mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/fasta/genome.fa \
        --gtf "$GTF" \
        --threads 16 \
        2>&1 || echo "WARNING: simpleaf index failed, skipping salmon"
fi

if [[ -d "$SAL_IDX" ]] && [[ ! -d "$SALMON_DIR" ]]; then
    echo ""
    echo "[$(date)] Step 3b: Running simpleaf quant..."
    mkdir -p "$SALMON_DIR"
    export ALEVIN_FRY_HOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/af_home
    
    /usr/bin/time -v "$SIMPLEAF" quant \
        --reads1 "$R1" \
        --reads2 "$R2" \
        --index "$SAL_IDX/index" \
        --chemistry 10xv3 \
        --resolution cr-like \
        --threads 16 \
        --output "$SALMON_DIR" \
        --expected-ori fw \
        2>&1 || echo "WARNING: simpleaf quant failed"
fi

# ── Step 4: Run singlify pipeline ──
SINGLIFY_OUT=/mnt/projects/debruinz_project/singlify_validation/singlify_out/$SRR
SINGLIFY_BIN=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
SNP_VCF=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.vcf.gz

if [[ ! -d "$SINGLIFY_OUT" ]]; then
    echo ""
    echo "[$(date)] Step 4: Running singlify..."
    mkdir -p "$SINGLIFY_OUT"

    /usr/bin/time -v "$SINGLIFY_BIN" \
        "$CORPUS/${SRR}.1fq" \
        --genome-dir "$GENOME" \
        --whitelist "$WL" \
        --exons "$GTF" \
        --snps "$SNP_VCF" \
        --out-prefix "$SINGLIFY_OUT/" \
        --threads 16 \
        2>&1

    echo ""
    echo "[$(date)] singlify complete"
    ls -lh "$SINGLIFY_OUT"/ 2>/dev/null
fi

echo ""
echo "[$(date)] === All ground truth complete for $SRR ==="
echo "  STARsolo: $STARSOLO/$SRR"
echo "  Salmon:   $SALMON_DIR"
echo "  Singlify: $SINGLIFY_OUT"
