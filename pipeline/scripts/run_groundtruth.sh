#!/bin/bash
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#
# Usage: sbatch --export=SRR=SRR27329891,PROTOCOL=10xv3,SPECIES=human scripts/run_groundtruth.sh
#
# Full pipeline: decode .1fq → FASTQ → STARsolo → simpleaf quant → singlify
# All steps profiled with /usr/bin/time -v
#
set -euo pipefail

: "${SRR:?SRR must be set via --export}"
: "${PROTOCOL:?PROTOCOL must be set (10xv3|10xv2|dropseq|10xv3_5prime|10xv4|10x_arc_gex)}"
: "${SPECIES:=human}"

VALDIR=/mnt/projects/debruinz_project/singlify_validation
REFBASE=/mnt/projects/debruinz_project/cellarium/reference
SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
ONEFQ=/mnt/home/debruinz/Singlet-AI/singlify/build/1fq
STAR_BIN=/mnt/home/debruinz/Singlet-AI/STAR/source/STAR_stock_baseline
CONDA=/mnt/home/debruinz/.conda/envs/cellarium/bin
WL_DIR=/mnt/home/debruinz/Singlet-AI/singlify/whitelists

# Species-dependent paths
if [[ "$SPECIES" == "mouse" ]]; then
    REFDIR=${REFBASE}/GRCm39-2024-A
else
    REFDIR=${REFBASE}/GRCh38-2024-A
fi
GENOME_DIR=${REFDIR}/star_2.7.11b
GTF=${REFDIR}/genes/genes.gtf
SNP_VCF=${REFBASE}/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.vcf.gz

# Protocol-dependent parameters
case "$PROTOCOL" in
    10xv3|10xv4)
        CB_LEN=16; UMI_LEN=12; CB_START=1; UMI_START=17
        WL=${WL_DIR}/3M-february-2018.txt
        SOLO_TYPE="CB_UMI_Simple"
        ;;
    10xv2)
        CB_LEN=16; UMI_LEN=10; CB_START=1; UMI_START=17
        WL=${WL_DIR}/737K-august-2016.txt
        SOLO_TYPE="CB_UMI_Simple"
        ;;
    10x_arc_gex)
        CB_LEN=16; UMI_LEN=12; CB_START=1; UMI_START=17
        WL=${WL_DIR}/gex_737K-arc-v1.txt
        SOLO_TYPE="CB_UMI_Simple"
        ;;
    10xv3_5prime)
        CB_LEN=16; UMI_LEN=12; CB_START=1; UMI_START=17
        WL=${WL_DIR}/3M-february-2018.txt
        SOLO_TYPE="CB_UMI_Simple"
        ;;
    dropseq)
        CB_LEN=12; UMI_LEN=8; CB_START=1; UMI_START=13
        WL="None"
        SOLO_TYPE="CB_UMI_Simple"
        ;;
    *)
        echo "ERROR: Unknown protocol $PROTOCOL"
        exit 1
        ;;
esac

LOG=${VALDIR}/profiles/${SRR}_groundtruth.log
echo "[$(date)] === Ground truth pipeline for ${SRR} (${PROTOCOL}, ${SPECIES}) ===" | tee "$LOG"
echo "Host: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}" | tee -a "$LOG"

# ---- Step 1: Decode .1fq to FASTQ ----
R1=${VALDIR}/corpus/${SRR}_decoded_R1.fastq
R2=${VALDIR}/corpus/${SRR}_decoded_R2.fastq
ONEFQ_FILE=${VALDIR}/corpus/${SRR}.1fq

if [[ ! -f "$R1" ]] || [[ ! -f "$R2" ]]; then
    echo "[$(date)] Step 1: Decoding .1fq to FASTQ..." | tee -a "$LOG"
    /usr/bin/time -v $ONEFQ decode "$ONEFQ_FILE" "$R1" "$R2" 2>&1 | tee -a "$LOG"
    echo "[$(date)] Decode complete" | tee -a "$LOG"
else
    echo "[$(date)] Step 1: FASTQs already decoded" | tee -a "$LOG"
fi

# Verify read lengths
R1_LEN=$(head -2 "$R1" | tail -1 | tr -d '\n' | wc -c)
R2_LEN=$(head -2 "$R2" | tail -1 | tr -d '\n' | wc -c)
echo "  R1 length: ${R1_LEN}bp, R2 length: ${R2_LEN}bp" | tee -a "$LOG"

# ---- Step 2: STARsolo ----
STAR_OUT=${VALDIR}/starsolo/${SRR}
if [[ ! -f "${STAR_OUT}/Log.final.out" ]]; then
    echo "[$(date)] Step 2: Running STARsolo (16 threads)..." | tee -a "$LOG"
    mkdir -p "${STAR_OUT}"
    
    STAR_CMD="${STAR_BIN} --runThreadN 16 --genomeDir ${GENOME_DIR}"
    
    # For 5' protocols, R1 has cDNA and R2 has barcode+UMI (reversed)
    if [[ "$PROTOCOL" == "10xv3_5prime" ]]; then
        STAR_CMD="${STAR_CMD} --readFilesIn ${R1} ${R2}"
        STAR_CMD="${STAR_CMD} --soloStrand Reverse"
    else
        # Standard: R2=cDNA, R1=barcode+UMI
        STAR_CMD="${STAR_CMD} --readFilesIn ${R2} ${R1}"
    fi
    
    STAR_CMD="${STAR_CMD} --soloType ${SOLO_TYPE}"
    if [[ "$WL" != "None" ]]; then
        STAR_CMD="${STAR_CMD} --soloCBwhitelist ${WL}"
    else
        STAR_CMD="${STAR_CMD} --soloCBwhitelist None"
    fi
    STAR_CMD="${STAR_CMD} --soloCBstart ${CB_START} --soloCBlen ${CB_LEN} --soloUMIstart ${UMI_START} --soloUMIlen ${UMI_LEN}"
    STAR_CMD="${STAR_CMD} --outSAMtype BAM SortedByCoordinate"
    STAR_CMD="${STAR_CMD} --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM"
    STAR_CMD="${STAR_CMD} --soloFeatures Gene GeneFull SJ"
    STAR_CMD="${STAR_CMD} --outFileNamePrefix ${STAR_OUT}/"
    
    echo "  CMD: ${STAR_CMD}" | tee -a "$LOG"
    /usr/bin/time -v ${STAR_CMD} 2>&1 | tee -a "$LOG"
    echo "[$(date)] STARsolo complete" | tee -a "$LOG"
    
    echo "  Mapping stats:" | tee -a "$LOG"
    grep -E "Number of input reads|Uniquely mapped|unmapped" "${STAR_OUT}/Log.final.out" | tee -a "$LOG"
    echo "  Solo Gene Summary:" | tee -a "$LOG"
    cat "${STAR_OUT}/Solo.out/Gene/Summary.csv" 2>/dev/null | tee -a "$LOG"
else
    echo "[$(date)] Step 2: STARsolo already complete" | tee -a "$LOG"
fi

# ---- Step 3: simpleaf/salmon quant ----
SALMON_IDX=${REFDIR}/salmon_idx
SALMON_OUT=${VALDIR}/salmon/${SRR}

if [[ ! -d "${SALMON_OUT}/af_quant" ]]; then
    echo "[$(date)] Step 3: Running simpleaf quant..." | tee -a "$LOG"
    export ALEVIN_FRY_HOME=${REFDIR}/af_home
    export PATH="${CONDA}:${PATH}"
    
    # Build salmon index if needed
    if [[ ! -d "${SALMON_IDX}/index" ]]; then
        echo "  Building salmon index..." | tee -a "$LOG"
        /usr/bin/time -v ${CONDA}/simpleaf index \
            --output "${SALMON_IDX}" \
            --fasta "${REFDIR}/fasta/genome.fa" \
            --gtf "${GTF}" \
            --threads 16 2>&1 | tee -a "$LOG"
    fi
    
    mkdir -p "${SALMON_OUT}"
    # Map protocol to simpleaf chemistry
    case "$PROTOCOL" in
        10xv3|10xv4|10x_arc_gex|10xv3_5prime) CHEM="10xv3" ;;
        10xv2) CHEM="10xv2" ;;
        dropseq) CHEM="dropseq" ;;
        *) CHEM="10xv3" ;;
    esac
    
    /usr/bin/time -v ${CONDA}/simpleaf quant \
        --reads1 "${R1}" --reads2 "${R2}" \
        --index "${SALMON_IDX}/index" \
        --chemistry "${CHEM}" \
        --resolution cr-like --expected-ori fw --knee \
        --threads 16 --output "${SALMON_OUT}" 2>&1 | tee -a "$LOG"
    echo "[$(date)] Salmon quant complete" | tee -a "$LOG"
else
    echo "[$(date)] Step 3: Salmon quant already complete" | tee -a "$LOG"
fi

# ---- Step 4: singlify ----
SINGLIFY_OUT=${VALDIR}/singlify_out/${SRR}
BARCODES=${VALDIR}/starsolo/${SRR}/Solo.out/Gene/filtered/barcodes.tsv

if [[ ! -d "${SINGLIFY_OUT}" ]] || [[ -z "$(ls -A ${SINGLIFY_OUT} 2>/dev/null)" ]]; then
    echo "[$(date)] Step 4: Running singlify..." | tee -a "$LOG"
    mkdir -p "${SINGLIFY_OUT}"
    
    SINGLIFY_CMD="${SINGLIFY} ${ONEFQ_FILE} --genome-dir ${GENOME_DIR} --whitelist ${WL} --exons ${GTF} --out-prefix ${SINGLIFY_OUT}/ --threads 16 --output-format mtx"
    
    # Add barcodes if available
    if [[ -f "$BARCODES" ]]; then
        SINGLIFY_CMD="${SINGLIFY_CMD} --barcodes ${BARCODES}"
    fi
    
    # Add SNPs for human
    if [[ "$SPECIES" == "human" ]] && [[ -f "$SNP_VCF" ]]; then
        SINGLIFY_CMD="${SINGLIFY_CMD} --snps ${SNP_VCF}"
    fi
    
    echo "  CMD: ${SINGLIFY_CMD}" | tee -a "$LOG"
    /usr/bin/time -v ${SINGLIFY_CMD} 2>&1 | tee -a "$LOG"
    echo "[$(date)] Singlify complete" | tee -a "$LOG"
    ls -la "${SINGLIFY_OUT}/" | tee -a "$LOG"
else
    echo "[$(date)] Step 4: Singlify already complete" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "[$(date)] === All ground truth steps complete for ${SRR} ===" | tee -a "$LOG"
