#!/bin/bash
#SBATCH --job-name=gt-continue2
#SBATCH --output=/mnt/projects/debruinz_project/singlify_validation/profiles/gt_continuation2.log
#SBATCH --error=/mnt/projects/debruinz_project/singlify_validation/profiles/gt_continuation2.log
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

export ALEVIN_FRY_HOME=${REFDIR}/af_home
export PATH="${CONDA}:${PATH}"

echo "[$(date)] === Continuation pipeline v2 for ${SRR} ==="
echo "Host: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}"

# ---- Step 3b: Salmon quant (index already built) ----
echo ""
echo "[$(date)] Step 3b: Running simpleaf quant..."
SALMON_OUT=${VALDIR}/salmon/${SRR}
mkdir -p "${SALMON_OUT}"

/usr/bin/time -v ${CONDA}/simpleaf quant \
    --reads1 "${R1}" \
    --reads2 "${R2}" \
    --index "${REFDIR}/salmon_idx/index" \
    --chemistry 10xv3 \
    --resolution cr-like \
    --expected-ori fw \
    --knee \
    --threads 16 \
    --output "${SALMON_OUT}" \
    2>&1

echo "[$(date)] Salmon quant complete"
ls -la "${SALMON_OUT}/" 2>/dev/null || true

# ---- Step 4: Singlify pipeline ----
echo ""
echo "[$(date)] Step 4: Running singlify pipeline on .1fq..."

SINGLIFY_OUT=${VALDIR}/singlify_out/${SRR}
mkdir -p "${SINGLIFY_OUT}"

# singlify uses --genome-dir and takes .1fq directly
/usr/bin/time -v ${SINGLIFY} \
    --threads 16 \
    --genome-dir "${REFDIR}/star_2.7.11b" \
    --input "${ONEFQ_FILE}" \
    --output-dir "${SINGLIFY_OUT}" \
    2>&1 || echo "Singlify failed (may need different flags — check help)"

echo "[$(date)] Singlify step done"
ls -la "${SINGLIFY_OUT}/" 2>/dev/null || true

# ---- Step 5: Check singlify help for correct CLI ----
echo ""
echo "[$(date)] Step 5: singlify help output for reference"
${SINGLIFY} --help 2>&1 || true
${SINGLIFY} pipeline --help 2>&1 || true
${SINGLIFY} align --help 2>&1 || true

echo ""
echo "[$(date)] === All continuation steps complete ==="
