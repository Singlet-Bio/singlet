#!/bin/bash
#SBATCH --job-name=singlify-ref-build
#SBATCH --array=1-15%5
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output=ref_build_%a.log
#SBATCH --error=ref_build_%a.log

# Top-15 model organisms by GEO sample frequency (excluding human+mouse)
# Each task $SLURM_ARRAY_TASK_ID corresponds to one species

set -euo pipefail
source /opt/rh/gcc-toolset-13/enable

REF_BASE="/mnt/projects/debruinz_project/cellarium/reference"
STAR="/mnt/home/debruinz/.conda/envs/cellarium/bin/STAR"
THREADS=${SLURM_CPUS_PER_TASK:-20}
TMPDIR=/dev/shm

# Species lookup table: task_id → (common_name, ensembl_name, assembly, division, release)
# Division: "ensembl" for main Ensembl, "plants" for EnsemblGenomes plants, etc.
declare -A SPECIES
SPECIES[1]="zebrafish|danio_rerio|GRCz11|ensembl|111"
SPECIES[2]="rat|rattus_norvegicus|mRatBN7.2|ensembl|111"
SPECIES[3]="fruit_fly|drosophila_melanogaster|BDGP6.46|ensembl|111"
SPECIES[4]="nematode|caenorhabditis_elegans|WBcel235|ensembl|111"
SPECIES[5]="chicken|gallus_gallus|GRCg7b|ensembl|111"
SPECIES[6]="pig|sus_scrofa|Sscrofa11.1|ensembl|111"
SPECIES[7]="cow|bos_taurus|ARS-UCD1.3|ensembl|111"
SPECIES[8]="dog|canis_lupus_familiaris|ROS_Cfam_1.0|ensembl|111"
SPECIES[9]="rhesus_macaque|macaca_mulatta|Mmul_10|ensembl|111"
SPECIES[10]="horse|equus_caballus|EquCab3.0|ensembl|111"
SPECIES[11]="cat|felis_catus|Felis_catus_9.0|ensembl|111"
SPECIES[12]="rabbit|oryctolagus_cuniculus|OryCun2.0|ensembl|111"
SPECIES[13]="yeast|saccharomyces_cerevisiae|R64-1-1|ensembl|111"
SPECIES[14]="frog|xenopus_tropicalis|UCB_Xtro_10.0|ensembl|111"
SPECIES[15]="sheep|ovis_aries|ARS-UI_Ramb_v2.0|ensembl|111"

# Parse this task's species
IFS='|' read -r COMMON ENSEMBL_NAME ASSEMBLY DIVISION RELEASE <<< "${SPECIES[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID: $COMMON ($ENSEMBL_NAME, $ASSEMBLY)"

# Output directory structure: REF_BASE/<Assembly>/
OUT_DIR="$REF_BASE/$ASSEMBLY"
STAR_DIR="$OUT_DIR/star_2.7.11b"

if [[ -d "$STAR_DIR" && -f "$STAR_DIR/SA" ]]; then
    echo "[$(date)] STAR index already exists at $STAR_DIR — skipping"
    exit 0
fi

mkdir -p "$OUT_DIR/fasta" "$OUT_DIR/genes" "$STAR_DIR"

# Construct Ensembl FTP URLs
if [[ "$DIVISION" == "ensembl" ]]; then
    FASTA_BASE="https://ftp.ensembl.org/pub/release-${RELEASE}/fasta/${ENSEMBL_NAME}/dna"
    GTF_BASE="https://ftp.ensembl.org/pub/release-${RELEASE}/gtf/${ENSEMBL_NAME}"
elif [[ "$DIVISION" == "plants" ]]; then
    FASTA_BASE="https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-${RELEASE}/fasta/${ENSEMBL_NAME}/dna"
    GTF_BASE="https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-${RELEASE}/gtf/${ENSEMBL_NAME}"
elif [[ "$DIVISION" == "metazoa" ]]; then
    FASTA_BASE="https://ftp.ensemblgenomes.ebi.ac.uk/pub/metazoa/release-${RELEASE}/fasta/${ENSEMBL_NAME}/dna"
    GTF_BASE="https://ftp.ensemblgenomes.ebi.ac.uk/pub/metazoa/release-${RELEASE}/gtf/${ENSEMBL_NAME}"
elif [[ "$DIVISION" == "protists" ]]; then
    FASTA_BASE="https://ftp.ensemblgenomes.ebi.ac.uk/pub/protists/release-${RELEASE}/fasta/${ENSEMBL_NAME}/dna"
    GTF_BASE="https://ftp.ensemblgenomes.ebi.ac.uk/pub/protists/release-${RELEASE}/gtf/${ENSEMBL_NAME}"
fi

# Capitalize first letter for Ensembl file naming convention
ENSEMBL_CAP="$(echo "${ENSEMBL_NAME}" | sed 's/\b\(.\)/\u\1/')"

# Download genome FASTA (primary assembly or toplevel)
FASTA_FILE="$OUT_DIR/fasta/genome.fa"
if [[ ! -f "$FASTA_FILE" ]]; then
    echo "[$(date)] Downloading genome FASTA..."
    # Ensembl file naming is inconsistent — some species have complex intermediate names.
    # List the FTP directory and dynamically discover the correct filename.
    # Prefer single-file: primary_assembly.fa.gz > toplevel.fa.gz
    # Some species only have per-chromosome splits, so fall back to toplevel.
    FASTA_PAGE=$(curl -sL --connect-timeout 30 --max-time 120 --retry 3 "${FASTA_BASE}/")
    # Look for single-file primary_assembly (not per-chromosome splits like .1.fa.gz)
    # Use || true to prevent pipefail from killing the script when grep has no match
    FASTA_GZ=$(echo "$FASTA_PAGE" | grep -oP 'href="\K[^"]*\.dna\.primary_assembly\.fa\.gz' | grep -v '\.[0-9]\.fa\.gz' | sed 's/"$//' | head -1 || true)
    if [[ -z "$FASTA_GZ" ]]; then
        # Fall back to toplevel (always a single file, but includes haplotypes)
        FASTA_GZ=$(echo "$FASTA_PAGE" | grep -oP 'href="\K[^"]*\.dna\.toplevel\.fa\.gz' | grep -v 'dna_[sr]m\.' | sed 's/"$//' | head -1 || true)
    fi
    if [[ -z "$FASTA_GZ" ]]; then
        echo "FAILED to find FASTA filename for $COMMON at ${FASTA_BASE}/"
        exit 1
    fi
    wget -q --timeout=300 --tries=3 -O "${FASTA_FILE}.gz" "${FASTA_BASE}/${FASTA_GZ}" || \
    { echo "FAILED to download FASTA for $COMMON"; exit 1; }
    gunzip "${FASTA_FILE}.gz"
fi

# Download GTF annotation
GTF_FILE="$OUT_DIR/genes/genes.gtf"
if [[ ! -f "$GTF_FILE" ]]; then
    echo "[$(date)] Downloading GTF..."
    GTF_PAGE=$(curl -sL --connect-timeout 30 --max-time 120 --retry 3 "${GTF_BASE}/")
    GTF_GZ=$(echo "$GTF_PAGE" | grep -oP 'href="\K[^"]*\.'"${RELEASE}"'\.gtf\.gz' | grep -v 'abinitio\|chr\.' | sed 's/"$//' | head -1 || true)
    if [[ -z "$GTF_GZ" ]]; then
        GTF_GZ=$(echo "$GTF_PAGE" | grep -oP 'href="\K[^"]*\.gtf\.gz' | grep -v 'abinitio\|chr\.' | sed 's/"$//' | head -1 || true)
    fi
    if [[ -z "$GTF_GZ" ]]; then
        echo "FAILED to find GTF filename for $COMMON at ${GTF_BASE}/"
        exit 1
    fi
    wget -q --timeout=300 --tries=3 -O "${GTF_FILE}.gz" "${GTF_BASE}/${GTF_GZ}" || \
    { echo "FAILED to download GTF for $COMMON"; exit 1; }
    gunzip "${GTF_FILE}.gz"
fi

# Build STAR index
echo "[$(date)] Building STAR index at $STAR_DIR (${THREADS} threads)..."
GENOME_LENGTH=$(awk '/^>/{next}{n+=length}END{print n}' "$FASTA_FILE")
# Compute genomeSAindexNbases: min(14, floor(log2(genome_length)/2 - 1))
SA_NBASES=$(python3 -c "import math; print(min(14, int(math.log2($GENOME_LENGTH)/2 - 1)))")

$STAR \
    --runMode genomeGenerate \
    --genomeDir "$STAR_DIR" \
    --genomeFastaFiles "$FASTA_FILE" \
    --sjdbGTFfile "$GTF_FILE" \
    --runThreadN "$THREADS" \
    --genomeSAindexNbases "$SA_NBASES" \
    2>&1 | tee "$OUT_DIR/star_build.log"

echo "[$(date)] DONE: $COMMON STAR index at $STAR_DIR"
ls -lh "$STAR_DIR/SA"
