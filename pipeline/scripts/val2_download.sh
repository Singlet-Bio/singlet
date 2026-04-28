#!/bin/bash
#SBATCH --job-name=val2-download
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --partition=cpu
#SBATCH --output=val2_logs/dl_%a.log
#SBATCH --error=val2_logs/dl_%a.log

# SLURM array download script for VAL2 (500+ samples)
# Submit: sbatch --array=1-N%100 scripts/val2_download.sh
#
# Input: scripts/val2_samples.csv (header: srr,species,protocol_expected,modality,reads_expected,source)
# Output: val2/<SRR>/<SRR>.1fq + metadata.json

set -euo pipefail
source /opt/rh/gcc-toolset-13/enable
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/dev/shm

SINGLIFY="/mnt/home/debruinz/Singlet-AI/singlify/build/singlify"
VAL2_BASE="/mnt/projects/debruinz_project/singlify_validation/val2"
SAMPLES_CSV="/mnt/home/debruinz/Singlet-AI/singlify/scripts/val2_samples.csv"

# Get SRR for this array task (skip header line)
LINE=$((SLURM_ARRAY_TASK_ID + 1))
ROW=$(sed -n "${LINE}p" "$SAMPLES_CSV")
SRR_FIELD=$(echo "$ROW" | cut -d',' -f1)
SPECIES=$(echo "$ROW" | cut -d',' -f2)
PROTOCOL=$(echo "$ROW" | cut -d',' -f3)

if [[ -z "$SRR_FIELD" ]]; then
    echo "ERROR: No SRR at line $LINE"
    exit 1
fi

# Handle multi-SRR samples (semicolon-separated) — use first SRR as primary ID
PRIMARY_SRR=$(echo "$SRR_FIELD" | cut -d';' -f1)
OUT_DIR="$VAL2_BASE/$PRIMARY_SRR"
mkdir -p "$OUT_DIR"

# Skip if already downloaded AND file is valid (footer magic check)
if [[ -f "$OUT_DIR/${PRIMARY_SRR}.1fq" ]]; then
    SIZE=$(stat -c%s "$OUT_DIR/${PRIMARY_SRR}.1fq")
    if [[ $SIZE -gt 1000 ]]; then
        # Validate footer magic: singlify decode fails fast on corrupt footer
        if $SINGLIFY decode "$OUT_DIR/${PRIMARY_SRR}.1fq" 2>&1 | head -1 | grep -q "footer magic invalid"; then
            echo "REDOWNLOAD: $PRIMARY_SRR .1fq is corrupt (footer magic invalid), removing"
            rm -f "$OUT_DIR/${PRIMARY_SRR}.1fq"
        else
            echo "SKIP: $PRIMARY_SRR already has valid .1fq ($SIZE bytes)"
            exit 0
        fi
    fi
fi

echo "[$(date)] Downloading $SRR_FIELD (species=$SPECIES, protocol=$PROTOCOL)"

# Map catalog protocol names to singlify --protocol tags
# normalize_tag() in singlify is case-insensitive with alias resolution
map_protocol() {
    local p="$1"
    case "$p" in
        CITE-seq)      echo "cite-seq-gex" ;;
        DNBelab)        echo "dnbelab-c4" ;;
        Parse-SPLIT-seq) echo "splitseq" ;;
        Seq-Well)       echo "seqwell" ;;
        *)              echo "$p" ;;  # singlify normalize_tag handles casing
    esac
}
SINGLIFY_PROTOCOL=$(map_protocol "$PROTOCOL")

# Download each SRR in the sample (may be multi-run)
# For multi-SRR, download first SRR only — it's the primary run
SRR="$PRIMARY_SRR"

# Download and encode to .1fq with retry
MAX_RETRIES=3
for attempt in $(seq 1 $MAX_RETRIES); do
    if $SINGLIFY download "$SRR" \
        -o "$OUT_DIR/${SRR}.1fq" \
        --protocol "$SINGLIFY_PROTOCOL" \
        --codec zstd \
        --codec-level 4 \
        --vdb-threads 4 \
        2>&1; then
        echo "[$(date)] SUCCESS: $SRR downloaded (attempt $attempt)"
        
        # Write metadata JSON
        cat > "$OUT_DIR/metadata.json" <<EOF
{
    "srr": "$SRR",
    "all_srrs": "$SRR_FIELD",
    "species_expected": "$SPECIES", 
    "protocol_expected": "$PROTOCOL",
    "download_date": "$(date -Iseconds)",
    "download_node": "$(hostname)",
    "onefq_size_bytes": $(stat -c%s "$OUT_DIR/${SRR}.1fq"),
    "slurm_job_id": "$SLURM_JOB_ID"
}
EOF
        exit 0
    fi
    
    echo "[$(date)] RETRY $attempt/$MAX_RETRIES for $SRR"
    rm -f "$OUT_DIR/${SRR}.1fq"
    sleep $((attempt * 30))
done

echo "[$(date)] FAILED: $SRR after $MAX_RETRIES attempts"
exit 1
