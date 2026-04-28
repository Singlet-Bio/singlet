#!/usr/bin/env bash
# starsolo_groundtruth.sh — Run STARsolo on one sample for ground truth
# Submitted as SLURM job: sbatch --export=SRR=SRR32855204,SPECIES=human,PROTOCOL=10x-arc-gex starsolo_groundtruth.sh
set -euo pipefail

CORPUS=/mnt/projects/debruinz_project/singlify_validation/corpus
STARSOLO_DIR=/mnt/projects/debruinz_project/singlify_validation/starsolo
PROFILES=/mnt/projects/debruinz_project/singlify_validation/profiles
STAR_BIN=/mnt/home/debruinz/Singlet-AI/STAR/source/STAR_stock_baseline
SAMTOOLS=/mnt/home/debruinz/.conda/envs/cellarium/bin/samtools

GRCH38_STAR=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
GRCM39_STAR=/mnt/projects/debruinz_project/cellarium/reference/GRCm39-2024-A/star_2.7.11b

WL_10X_V3=/mnt/home/debruinz/Singlet-AI/singlify/whitelists/3M-february-2018.txt
WL_10X_V2=/mnt/home/debruinz/Singlet-AI/singlify/whitelists/737K-august-2016.txt
WL_ARC=/mnt/home/debruinz/Singlet-AI/singlify/whitelists/gex_737K-arc-v1.txt

export TMPDIR=/dev/shm
THREADS=16

echo "[$(date)] STARsolo ground truth: $SRR ($PROTOCOL, $SPECIES)"

# Select genome
if [[ "$SPECIES" == "human" ]]; then
    GENOME=$GRCH38_STAR
elif [[ "$SPECIES" == "mouse" ]]; then
    GENOME=$GRCM39_STAR
else
    echo "ERROR: Unknown species $SPECIES"
    exit 1
fi

# Check genome exists
if [[ ! -f "$GENOME/SA" ]]; then
    echo "ERROR: STAR index not found at $GENOME"
    exit 1
fi

# Find FASTQ files
FASTQ_DIR="$CORPUS/${SRR}_fastq"
FASTQS=($(ls "$FASTQ_DIR"/*.fastq 2>/dev/null | sort))
N_FILES=${#FASTQS[@]}

echo "  Found $N_FILES FASTQ files"

# Determine READ1 (barcode) and READ2 (cDNA) based on read lengths
# For standard 10x: R1 is short (26-28bp), R2 is long (90-150bp)
declare -A FILE_LENS
for f in "${FASTQS[@]}"; do
    len=$(head -2 "$f" | tail -1 | wc -c)
    FILE_LENS["$f"]=$((len - 1))  # subtract newline
    echo "  $(basename $f): ${FILE_LENS[$f]}bp"
done

# Sort files by read length to identify barcode (short) vs cDNA (long)
SHORT_FILE=""
LONG_FILE=""
for f in "${FASTQS[@]}"; do
    len=${FILE_LENS[$f]}
    if [[ $len -le 30 ]]; then
        SHORT_FILE="$f"
    elif [[ $len -ge 50 ]]; then
        LONG_FILE="$f"
    fi
done

# For multiome with 4 files, we need to pick correctly
# _1=I1(8bp), _2=R1(28bp BC+UMI), _3=I2(8bp), _4=R2(90bp cDNA)
if [[ $N_FILES -eq 4 ]]; then
    # Pick the 28bp file as R1 and the 90bp as R2
    for f in "${FASTQS[@]}"; do
        len=${FILE_LENS[$f]}
        if [[ $len -eq 28 ]]; then SHORT_FILE="$f"; fi
        if [[ $len -eq 90 || $len -eq 91 || $len -ge 85 ]]; then LONG_FILE="$f"; fi
    done
fi

if [[ -z "$SHORT_FILE" ]] || [[ -z "$LONG_FILE" ]]; then
    # Fallback: use _1 as R1 and _2 as R2 for standard paired-end
    if [[ $N_FILES -ge 2 ]]; then
        SHORT_FILE="${FASTQS[0]}"
        LONG_FILE="${FASTQS[1]}"
        echo "  WARNING: Could not identify BC/cDNA by length, using positional: R1=$(basename $SHORT_FILE), R2=$(basename $LONG_FILE)"
    else
        echo "  ERROR: Need at least 2 FASTQ files"
        exit 1
    fi
fi

echo "  R1 (barcode): $(basename $SHORT_FILE) (${FILE_LENS[$SHORT_FILE]}bp)"
echo "  R2 (cDNA):    $(basename $LONG_FILE) (${FILE_LENS[$LONG_FILE]}bp)"

# Determine STARsolo parameters
SOLO_TYPE="CB_UMI_Simple"
CB_START=1
CB_LEN=16
UMI_START=17
UMI_LEN=12
WL_FILE=""
EXTRA_ARGS=""

case "$PROTOCOL" in
    10x-v2)
        UMI_LEN=10; WL_FILE=$WL_10X_V2
        ;;
    10x-v3|10x-v4)
        UMI_LEN=12; WL_FILE=$WL_10X_V3
        ;;
    10x-5p)
        UMI_LEN=12; WL_FILE=$WL_10X_V3
        # 5' data: barcode is in the SHORT file (R1), cDNA in LONG (R2) — same as 3'
        ;;
    10x-arc-gex)
        UMI_LEN=12; WL_FILE=$WL_ARC
        ;;
    Drop-seq)
        CB_LEN=12; UMI_START=13; UMI_LEN=8
        WL_FILE="None"
        EXTRA_ARGS="--soloCBmatchWLtype 1MM"
        ;;
    *)
        echo "WARNING: Unknown protocol $PROTOCOL, using default 10x-v3 params"
        WL_FILE=$WL_10X_V3
        ;;
esac

OUTDIR="$STARSOLO_DIR/$SRR"
mkdir -p "$OUTDIR"

echo ""
echo "[$(date)] Running STARsolo..."
echo "  Genome: $GENOME"
echo "  Whitelist: $(basename ${WL_FILE:-NONE})"
echo "  soloType=$SOLO_TYPE CBlen=$CB_LEN UMIstart=$UMI_START UMIlen=$UMI_LEN"

/usr/bin/time -v $STAR_BIN \
    --runThreadN "$THREADS" \
    --genomeDir "$GENOME" \
    --readFilesIn "$LONG_FILE" "$SHORT_FILE" \
    --soloType "$SOLO_TYPE" \
    --soloCBwhitelist "$WL_FILE" \
    --soloCBstart "$CB_START" --soloCBlen "$CB_LEN" \
    --soloUMIstart "$UMI_START" --soloUMIlen "$UMI_LEN" \
    --outSAMtype BAM SortedByCoordinate \
    --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM \
    --soloFeatures Gene GeneFull SJ \
    --outFileNamePrefix "${OUTDIR}/" \
    $EXTRA_ARGS \
    2>&1 | tee "$PROFILES/${SRR}_starsolo.log"

echo ""
echo "[$(date)] STARsolo complete: $SRR"
echo "  Output dir: $OUTDIR"
ls -lh "$OUTDIR"/Solo.out/Gene/filtered/ 2>/dev/null || echo "  No filtered output (may need more cells)"
echo "  Log.final.out:"
cat "$OUTDIR/Log.final.out" 2>/dev/null | grep -E "Number of input reads|Uniquely mapped|Average"
