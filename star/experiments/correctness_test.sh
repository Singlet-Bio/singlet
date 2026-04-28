#!/bin/bash
#
# Quick correctness validation: verify that STAR_prefetch_sort produces
# identical alignment statistics to STAR_stock (sorting within chunks
# must not change alignment results, only read order in output).
#
set -euo pipefail

RESULTS=/mnt/home/debruinz/Singlet-AI/STAR/experiments/learned_cache/correctness_test
rm -rf "$RESULTS" && mkdir -p "$RESULTS"
echo "=== Correctness Validation ==="

R1=/mnt/projects/debruinz_project/cellarium/pipeline/smoke_test/GSM8313394/fastq/SRR29320040_1.fastq.gz
R2=/mnt/projects/debruinz_project/cellarium/pipeline/smoke_test/GSM8313394/fastq/SRR29320040_2.fastq.gz
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
WL_GZ=/mnt/projects/debruinz_project/irungum_1000G/scRNAseq/software/cellranger-10.0.0/lib/python/cellranger/barcodes/3M-february-2018_TRU.txt.gz
WL="$RESULTS/whitelist.txt"

STAR_A=/mnt/home/debruinz/Singlet-AI/STAR/source/STAR_stock
STAR_B=/mnt/home/debruinz/Singlet-AI/STAR/source/STAR_prefetch_sort

zcat "$WL_GZ" > "$WL"

STAR_ARGS="--runThreadN 8 --genomeDir $GENOME \
  --readFilesCommand zcat \
  --soloType CB_UMI_Simple --soloCBwhitelist $WL \
  --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 \
  --outSAMtype BAM Unsorted \
  --outSAMattributes NH nM AS CR UR \
  --clipAdapterType CellRanger4 --outFilterScoreMin 30 \
  --genomeLoad NoSharedMemory"

# Use a small subset (500K reads) for quick test
NREADS=500000
NLINES=$((NREADS * 4))

echo "  Creating 500K read subset..."
SUB_R1="$RESULTS/sub_R1.fastq.gz"
SUB_R2="$RESULTS/sub_R2.fastq.gz"
zcat "$R1" | head -$NLINES | gzip -1 > "$SUB_R1" &
zcat "$R2" | head -$NLINES | gzip -1 > "$SUB_R2" &
wait

echo "  Running STAR_stock..."
mkdir -p "$RESULTS/stock"
$STAR_A $STAR_ARGS \
    --readFilesIn "$SUB_R2" "$SUB_R1" \
    --outFileNamePrefix "$RESULTS/stock/" \
    > /dev/null 2>&1

echo "  Running STAR_prefetch_sort..."
mkdir -p "$RESULTS/prefetch_sort"
$STAR_B $STAR_ARGS \
    --readFilesIn "$SUB_R2" "$SUB_R1" \
    --outFileNamePrefix "$RESULTS/prefetch_sort/" \
    > /dev/null 2>&1

echo ""
echo "=== Comparing Log.final.out statistics ==="
echo "--- Stock ---"
grep -E "Number of input reads|Uniquely mapped|mapped to multiple|mapped to too many|Unmapped|Mismatch rate" "$RESULTS/stock/Log.final.out"
echo ""
echo "--- Prefetch+Sort ---"
grep -E "Number of input reads|Uniquely mapped|mapped to multiple|mapped to too many|Unmapped|Mismatch rate" "$RESULTS/prefetch_sort/Log.final.out"

echo ""
echo "=== Diff of key statistics ==="
paste <(grep -oP '\d+\.?\d*%?' "$RESULTS/stock/Log.final.out") \
      <(grep -oP '\d+\.?\d*%?' "$RESULTS/prefetch_sort/Log.final.out") | \
    awk '{if ($1 != $2) print NR": "$1" vs "$2" ***DIFF***"; else print NR": "$1" = "$2" OK"}'

echo ""
echo "=== Solo output comparison ==="
# Compare STARsolo gene expression matrices
if [[ -f "$RESULTS/stock/Solo.out/Gene/raw/matrix.mtx" && -f "$RESULTS/prefetch_sort/Solo.out/Gene/raw/matrix.mtx" ]]; then
    diff <(sort "$RESULTS/stock/Solo.out/Gene/raw/matrix.mtx") \
         <(sort "$RESULTS/prefetch_sort/Solo.out/Gene/raw/matrix.mtx") > /dev/null 2>&1 \
        && echo "Gene matrix: IDENTICAL" \
        || echo "Gene matrix: DIFFERENT ***CHECK***"
else
    echo "(Solo output not generated or path is different)"
    ls -la "$RESULTS/stock/Solo.out/" 2>/dev/null || echo "No Solo output for stock"
    ls -la "$RESULTS/prefetch_sort/Solo.out/" 2>/dev/null || echo "No Solo output for prefetch_sort"
fi

echo ""
echo "=== Correctness test complete ==="
