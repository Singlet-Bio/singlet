#!/usr/bin/env bash
set -euo pipefail

# Benchmark: STAR txFirst vs baseline
# Run on compute node with 20 threads, 5M reads, warm cache

STAR_DIR=/mnt/home/debruinz/Singlet-AI/STAR/source
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
R1=/mnt/projects/debruinz_project/cellarium/pipeline/smoke_test/GSM8313394/fastq/SRR29320040_1.fastq.gz
R2=/mnt/projects/debruinz_project/cellarium/pipeline/smoke_test/GSM8313394/fastq/SRR29320040_2.fastq.gz
WL=/mnt/home/debruinz/Singlet-AI/STAR/experiments/learned_cache/correctness_test/whitelist.txt
OUTBASE=/mnt/home/debruinz/Singlet-AI/STAR/experiments/bench_txfirst_out
THREADS=20
NREADS=5000000

echo "=== TxFirst Benchmark ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Threads: $THREADS, Reads: $NREADS"
echo ""

# Warm the file cache
echo "Warming cache..."
cat "$GENOME/Genome" "$GENOME/SA" "$GENOME/SAindex" > /dev/null 2>&1
if [[ -f "$GENOME/SA_tx" ]]; then
    cat "$GENOME/SA_tx" "$GENOME/SAindex_tx" > /dev/null 2>&1
fi

COMMON_ARGS="--runThreadN $THREADS --genomeDir $GENOME \
    --readFilesIn $R2 $R1 --readFilesCommand zcat \
    --soloType CB_UMI_Simple --soloCBwhitelist $WL \
    --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 \
    --outSAMtype None \
    --readMapNumber $NREADS"

for trial in 1 2 3; do
    echo ""
    echo "=== Trial $trial ==="

    # --- Baseline ---
    rm -rf ${OUTBASE}_baseline; mkdir -p ${OUTBASE}_baseline
    echo "--- Baseline (no TX_FIRST) ---"
    /usr/bin/time -v ${STAR_DIR}/STAR_baseline $COMMON_ARGS --outFileNamePrefix ${OUTBASE}_baseline/ 2>&1 | tee ${OUTBASE}_baseline/timing.txt
    echo ""
    grep "Uniquely mapped" ${OUTBASE}_baseline/Log.final.out || true
    echo ""

    # --- TxFirst ---
    rm -rf ${OUTBASE}_txfirst; mkdir -p ${OUTBASE}_txfirst
    echo "--- TxFirst ---"
    /usr/bin/time -v ${STAR_DIR}/STAR_txfirst $COMMON_ARGS --outFileNamePrefix ${OUTBASE}_txfirst/ 2>&1 | tee ${OUTBASE}_txfirst/timing.txt
    echo ""
    grep "Uniquely mapped" ${OUTBASE}_txfirst/Log.final.out || true
    echo ""
done

echo "=== Done ==="
