#!/bin/bash
#SBATCH --job-name=val2-process
#SBATCH --cpus-per-task=20
#SBATCH --mem=384G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu
#SBATCH --output=val2_logs/proc_%a.log
#SBATCH --error=val2_logs/proc_%a.log

# SLURM array processing script for VAL2
# Submit: sbatch --array=1-N%50 --dependency=afterok:<download_jobid> scripts/val2_process.sh

set -euo pipefail
source /opt/rh/gcc-toolset-13/enable
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/dev/shm

SINGLIFY="/mnt/home/debruinz/Singlet-AI/singlify/build/singlify"
VAL2_BASE="/mnt/projects/debruinz_project/singlify_validation/val2"
REF_BASE="/mnt/projects/debruinz_project/cellarium/reference"
SAMPLES_CSV="/mnt/home/debruinz/Singlet-AI/singlify/scripts/val2_samples.csv"

# Get SRR for this array task
LINE=$((SLURM_ARRAY_TASK_ID + 1))
ROW=$(sed -n "${LINE}p" "$SAMPLES_CSV")
SRR_FIELD=$(echo "$ROW" | cut -d',' -f1)

if [[ -z "$SRR_FIELD" ]]; then
    echo "ERROR: No SRR at line $LINE"
    exit 1
fi

# Handle multi-SRR samples (semicolon-separated) — use first SRR as primary ID
PRIMARY_SRR=$(echo "$SRR_FIELD" | cut -d';' -f1)
INPUT="$VAL2_BASE/$PRIMARY_SRR/${PRIMARY_SRR}.1fq"
OUTPUT="$VAL2_BASE/$PRIMARY_SRR/output"

# Skip if already processed
if [[ -f "$OUTPUT/pileup_stats.json" ]]; then
    echo "SKIP: $PRIMARY_SRR already processed"
    exit 0
fi

# Skip if no .1fq
if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: $PRIMARY_SRR has no .1fq file"
    exit 1
fi

mkdir -p "$OUTPUT"

echo "[$(date)] Processing $PRIMARY_SRR"

# ── Species → genome-dir resolution from sidecar metadata.json ──
GENOME_DIR=""
METADATA_JSON="$VAL2_BASE/$PRIMARY_SRR/metadata.json"
if [[ -f "$METADATA_JSON" ]]; then
    SPECIES_EXPECTED=$(python3 -c "import json; print(json.load(open('$METADATA_JSON')).get('species_expected',''))" 2>/dev/null || true)
    echo "[val2] species_expected=$SPECIES_EXPECTED"
    
    # Map species name → reference directory name
    declare -A SPECIES_MAP=(
        [human]="GRCh38-2024-A" [homo_sapiens]="GRCh38-2024-A"
        [mouse]="GRCm39-2024-A" [mus_musculus]="GRCm39-2024-A"
        [chicken]="GRCg7b" [gallus_gallus]="GRCg7b"
        [zebrafish]="GRCz11" [danio_rerio]="GRCz11"
        [rat]="mRatBN7.2" [rattus_norvegicus]="mRatBN7.2"
        [pig]="Sscrofa11.1" [sus_scrofa]="Sscrofa11.1"
        [cow]="ARS-UCD1.3" [bos_taurus]="ARS-UCD1.3"
        [dog]="ROS_Cfam_1.0" [canis_lupus_familiaris]="ROS_Cfam_1.0"
        [macaque]="Mmul_10" [macaca_mulatta]="Mmul_10"
        [drosophila]="BDGP6.46" [drosophila_melanogaster]="BDGP6.46"
        [c_elegans]="WBcel235" [caenorhabditis_elegans]="WBcel235"
        [cat]="Felis_catus_9.0" [felis_catus]="Felis_catus_9.0"
        [rabbit]="OryCun2.0" [oryctolagus_cuniculus]="OryCun2.0"
        [horse]="EquCab3.0" [equus_caballus]="EquCab3.0"
        [frog]="UCB_Xtro_10.0" [xenopus_tropicalis]="UCB_Xtro_10.0"
        [sheep]="ARS-UI_Ramb_v2.0" [ovis_aries]="ARS-UI_Ramb_v2.0"
        [yeast]="R64-1-1" [saccharomyces_cerevisiae]="R64-1-1"
    )
    
    REF_NAME="${SPECIES_MAP[$SPECIES_EXPECTED]:-}"
    if [[ -n "$REF_NAME" ]]; then
        CANDIDATE="$REF_BASE/$REF_NAME/star_2.7.11b"
        if [[ -d "$CANDIDATE" ]]; then
            GENOME_DIR="$CANDIDATE"
            echo "[val2] Resolved genome-dir=$GENOME_DIR"
        else
            # Try without star_2.7.11b — maybe just star/
            CANDIDATE="$REF_BASE/$REF_NAME/star"
            if [[ -d "$CANDIDATE" ]]; then
                GENOME_DIR="$CANDIDATE"
                echo "[val2] Resolved genome-dir=$GENOME_DIR"
            else
                echo "[val2] WARNING: Reference $REF_NAME not found at $REF_BASE/$REF_NAME/star*"
            fi
        fi
    else
        echo "[val2] WARNING: Unknown species '$SPECIES_EXPECTED' — trying auto-detection"
    fi
fi

# Build singlify command (default output is mtx + 1pz)
SINGLIFY_ARGS=("$INPUT" --out-prefix "$OUTPUT" --threads "${SLURM_CPUS_PER_TASK:-20}")

if [[ -n "$GENOME_DIR" ]]; then
    SINGLIFY_ARGS+=(--genome-dir "$GENOME_DIR")
else
    SINGLIFY_ARGS+=(--ref-base "$REF_BASE")
fi

# Pass metadata-json if available
if [[ -f "$METADATA_JSON" ]]; then
    SINGLIFY_ARGS+=(--metadata-json "$METADATA_JSON")
fi

# Find GTF for exon counting
if [[ -n "$GENOME_DIR" ]]; then
    # Go up from star dir to find genes.gtf
    PARENT_DIR=$(dirname "$GENOME_DIR")
    GTF=$(find "$PARENT_DIR" -name "genes.gtf" -o -name "*.gtf" 2>/dev/null | head -1)
    if [[ -n "$GTF" ]]; then
        SINGLIFY_ARGS+=(--exons "$GTF")
    fi
fi

echo "[$(date)] Running: $SINGLIFY ${SINGLIFY_ARGS[*]}"
$SINGLIFY "${SINGLIFY_ARGS[@]}" 2>&1

EXIT_CODE=$?
echo "[$(date)] $PRIMARY_SRR exit=$EXIT_CODE"

# Extract key metrics for summary
if [[ -f "$OUTPUT/star_Log.final.out" ]]; then
    MAPPING=$(grep 'Uniquely mapped reads %' "$OUTPUT/star_Log.final.out" | awk -F'|' '{gsub(/%/,"",$2); print $2}' || true)
else
    MAPPING="NA"
fi

if [[ -f "$OUTPUT/pileup_stats.json" ]]; then
    CELLS=$(python3 -c "import json; d=json.load(open('$OUTPUT/pileup_stats.json')); print(d.get('cells_called', d.get('n_barcodes', 'NA')))" 2>/dev/null || echo "NA")
else
    CELLS="NA"
fi

# Write per-sample result (append with lock)
echo "$PRIMARY_SRR,$EXIT_CODE,${MAPPING:-NA},${CELLS:-NA}" >> "$VAL2_BASE/val2_results.csv"

exit $EXIT_CODE
