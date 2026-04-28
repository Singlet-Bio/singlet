#!/bin/bash
#
# Comprehensive 3-way A/B/C benchmark:
#   A = STAR_stock       (baseline)
#   B = STAR_prefetch    (SA binary search child prefetch)
#   C = STAR_prefetch_sort (prefetch + per-chunk 14-mer prefix sort)
#
# 9 trials interleaved, warm page cache.
#
set -euo pipefail

RESULTS=/mnt/home/debruinz/Singlet-AI/STAR/experiments/learned_cache/bench_3way_results
mkdir -p "$RESULTS"
echo "=== 3-Way STAR Optimization Benchmark ==="
echo "Node: $(hostname), Date: $(date)"
echo "Results: $RESULTS"

R1=/mnt/projects/debruinz_project/cellarium/pipeline/smoke_test/GSM8313394/fastq/SRR29320040_1.fastq.gz
R2=/mnt/projects/debruinz_project/cellarium/pipeline/smoke_test/GSM8313394/fastq/SRR29320040_2.fastq.gz
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
WL_GZ=/mnt/projects/debruinz_project/irungum_1000G/scRNAseq/software/cellranger-10.0.0/lib/python/cellranger/barcodes/3M-february-2018_TRU.txt.gz
WL="$RESULTS/whitelist.txt"

STAR_A=/mnt/home/debruinz/Singlet-AI/STAR/source/STAR_stock
STAR_B=/mnt/home/debruinz/Singlet-AI/STAR/source/STAR_prefetch
STAR_C=/mnt/home/debruinz/Singlet-AI/STAR/source/STAR_prefetch_sort

zcat "$WL_GZ" > "$WL"

STAR_ARGS="--runThreadN 8 --genomeDir $GENOME \
  --readFilesCommand zcat \
  --soloType CB_UMI_Simple --soloCBwhitelist $WL \
  --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 \
  --outSAMtype BAM Unsorted \
  --outSAMattributes NH nM AS CR UR \
  --clipAdapterType CellRanger4 --outFilterScoreMin 30 \
  --genomeLoad NoSharedMemory"

NREADS=5000000
NLINES=$((NREADS * 4))

echo "=== Creating 5M read subset ==="
SUB_R1="$RESULTS/sub_R1.fastq.gz"
SUB_R2="$RESULTS/sub_R2.fastq.gz"
zcat "$R1" | head -$NLINES | gzip -1 > "$SUB_R1" &
zcat "$R2" | head -$NLINES | gzip -1 > "$SUB_R2" &
wait
echo "  Done"

echo "=== Warming page cache ==="
for w in 1 2 3; do
    echo "  Warmup $w..."
    rm -rf "$RESULTS/warmup" && mkdir "$RESULTS/warmup"
    $STAR_A $STAR_ARGS \
        --readFilesIn "$SUB_R2" "$SUB_R1" \
        --outFileNamePrefix "$RESULTS/warmup/" \
        > /dev/null 2>&1
done
echo "  Page cache warm"

run_star() {
    local label=$1 binary=$2 trial=$3
    local d="$RESULTS/bench_${label}_t${trial}"
    rm -rf "$d" && mkdir "$d"
    local t0=$(date +%s%N)
    $binary $STAR_ARGS \
        --readFilesIn "$SUB_R2" "$SUB_R1" \
        --outFileNamePrefix "$d/" \
        > /dev/null 2>&1
    local t1=$(date +%s%N)
    local ms=$(( (t1 - t0) / 1000000 ))
    local mapped=$(grep "Uniquely mapped" "$d/Log.final.out" 2>/dev/null | head -1 | awk '{print $NF}' || echo "?")
    echo "${label} trial${trial}: ${ms}ms (unique=${mapped})"
    rm -rf "$d"
}

echo ""
echo "=== 9-trial interleaved A/B/C benchmark ==="
echo "  A = stock (baseline)"
echo "  B = prefetch (SA child prefetch)"
echo "  C = prefetch_sort (prefetch + per-chunk 14-mer sort)"
echo ""

for trial in 1 2 3 4 5 6 7 8 9; do
    echo "--- Trial $trial ---"
    run_star "stock"         "$STAR_A" "$trial"
    run_star "prefetch"      "$STAR_B" "$trial"
    run_star "prefetch_sort" "$STAR_C" "$trial"
done

echo ""
echo "=== 3-way benchmark complete ==="
echo "Results in: $RESULTS"
