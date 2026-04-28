#!/bin/bash
#SBATCH --job-name=gt-continue
#SBATCH --output=/mnt/projects/debruinz_project/singlify_validation/profiles/gt_continuation.log
#SBATCH --error=/mnt/projects/debruinz_project/singlify_validation/profiles/gt_continuation.log
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu

set -euo pipefail

SRR=SRR32855204
VALDIR=/mnt/projects/debruinz_project/singlify_validation
REFDIR=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
CONDA=/mnt/home/debruinz/.conda/envs/cellarium/bin
SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
ONEFQ=/mnt/home/debruinz/Singlet-AI/singlify/build/1fq
WL=/mnt/home/debruinz/Singlet-AI/singlify/whitelists

R1=${VALDIR}/corpus/${SRR}_decoded_R1.fastq
R2=${VALDIR}/corpus/${SRR}_decoded_R2.fastq
ONEFQ_FILE=${VALDIR}/corpus/${SRR}.1fq

echo "[$(date)] === Continuation pipeline for ${SRR} ==="
echo "Host: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}"

# ---- Step 3: Salmon/simpleaf ----
echo ""
echo "[$(date)] Step 3a: Building salmon/simpleaf index..."
export ALEVIN_FRY_HOME=${REFDIR}/af_home
export PATH="${CONDA}:${PATH}"

SALMON_IDX=${REFDIR}/salmon_idx
if [[ ! -d "${SALMON_IDX}/index" ]]; then
    mkdir -p "${SALMON_IDX}"
    /usr/bin/time -v ${CONDA}/simpleaf index \
        --output "${SALMON_IDX}" \
        --fasta "${REFDIR}/fasta/genome.fa" \
        --gtf "${REFDIR}/genes/genes.gtf" \
        --threads 16 \
        2>&1
    echo "[$(date)] Salmon index built"
else
    echo "[$(date)] Salmon index already exists"
fi

echo ""
echo "[$(date)] Step 3b: Running simpleaf quant..."
SALMON_OUT=${VALDIR}/salmon/${SRR}
mkdir -p "${SALMON_OUT}"

# 10x-arc-gex: CB=16bp (pos 1-16), UMI=12bp (pos 17-28) in R1
# Using chromium v3 geometry — same barcode structure as multiome GEX
/usr/bin/time -v ${CONDA}/simpleaf quant \
    --reads1 "${R1}" \
    --reads2 "${R2}" \
    --index "${SALMON_IDX}/index" \
    --chemistry 10xv3 \
    --resolution cr-like \
    --expected-ori fw \
    --threads 16 \
    --output "${SALMON_OUT}" \
    2>&1

echo "[$(date)] Salmon quant complete"
ls -la "${SALMON_OUT}/"

# ---- Step 4: Singlify pipeline ----
echo ""
echo "[$(date)] Step 4: Running singlify pipeline on .1fq..."

SINGLIFY_OUT=${VALDIR}/singlify_out/${SRR}
mkdir -p "${SINGLIFY_OUT}"

# Run singlify with profiling
/usr/bin/time -v ${SINGLIFY} \
    --threads 16 \
    --genome-dir "${REFDIR}/star_2.7.11b" \
    --input "${ONEFQ_FILE}" \
    --output-dir "${SINGLIFY_OUT}" \
    2>&1

echo "[$(date)] Singlify complete"
ls -la "${SINGLIFY_OUT}/"

# ---- Step 5: Singlify decode-only timing ----
echo ""
echo "[$(date)] Step 5: Timing 1fq decode..."

DECODE_DIR=${VALDIR}/corpus/${SRR}_decode_bench
mkdir -p "${DECODE_DIR}"

/usr/bin/time -v ${ONEFQ} decode \
    "${ONEFQ_FILE}" \
    "${DECODE_DIR}/${SRR}_bench_R1.fastq" \
    "${DECODE_DIR}/${SRR}_bench_R2.fastq" \
    2>&1

echo "[$(date)] Decode complete"
ls -la "${DECODE_DIR}/"

echo ""
echo "[$(date)] === All continuation steps complete ==="
