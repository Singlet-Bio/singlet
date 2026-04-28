#!/bin/bash
#SBATCH --job-name=val_mc4
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --output=/mnt/projects/debruinz_project/singlify_pipeline/logs/validate_mc4_%j.log

# Validate AUTOFIX-EMPTYDROPS-DEPTH-MC (Poisson MC p-value)
# Target: SRR32855204, 10x-3p-v3, 40.4M reads
# Acceptance criteria: cells between 1,500 and 3,780 (STARsolo=2,520 * 1.5x tolerance)
# Binary must be current HEAD (Poisson MC at 078ea33 or later)

set -euo pipefail

SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
REF_BASE=/mnt/projects/debruinz_project/cellarium/reference
GENOME_DIR=${REF_BASE}/GRCh38-2024-A/star_2.7.11b
GTF=${REF_BASE}/GRCh38-2024-A/genes/genes.gtf
SNPS=${REF_BASE}/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
SRR=SRR32855204
WORKDIR=/dev/shm/mc4_val_${SLURM_JOB_ID}

export SINGLIFY_REF_BASE=${REF_BASE}

echo "[val] singlify binary: $SINGLIFY"
echo "[val] Binary timestamp: $(stat -c '%y' ${SINGLIFY})"
echo "[val] HEAD: $(cd /mnt/home/debruinz/Singlet-AI/singlify && git log -1 --format='%h %s')"
echo "[val] GTF: ${GTF}"
echo "[val] GENOME: ${GENOME_DIR}"

# Pre-load genome
echo "[val] Pre-loading genome..."
${SINGLIFY} genome load ${GENOME_DIR} || true
${SINGLIFY} genome status ${GENOME_DIR} && echo "[val] Genome loaded." || echo "[val] Genome not in shared memory — will load fresh."

mkdir -p ${WORKDIR}
trap "rm -rf ${WORKDIR}; echo '[val] Cleanup done.'" EXIT

# Write metadata JSON
python3 -c "
import json, os
meta = {
    'gsm_id': 'GSM_VAL_MC4',
    'gse_id': 'GSE_VAL',
    'organism': 'Homo sapiens',
    'taxon_id': 9606,
    'protocol': '10x-3p-v3',
    'modality': 'scrna',
    'srr_ids': ['${SRR}'],
    'read_count': 40358185,
    'geo_title': 'MC EmptyDrops validation',
    'geo_source_name': 'validation',
    'singlify_version': '0.3.0',
    'pipeline_date': '2026-04-16',
}
json.dump(meta, open('${WORKDIR}/meta.json', 'w'), indent=2)
print('[val] Metadata written.')
"

# Download
echo "[val] Downloading ${SRR}..."
${SINGLIFY} download ${SRR} \
    -o ${WORKDIR}/${SRR}.1fq \
    --metadata-json ${WORKDIR}/meta.json \
    --quality none

echo "[val] Download complete. Size: $(du -sh ${WORKDIR}/${SRR}.1fq | cut -f1)"

# Process
echo "[val] Running pipeline..."
${SINGLIFY} ${WORKDIR}/${SRR}.1fq \
    --genome-dir ${GENOME_DIR} \
    --genome-shared \
    --exons ${GTF} \
    --snps ${SNPS} \
    --pipeline \
    --n-donors -1 \
    --metadata-json ${WORKDIR}/meta.json \
    --out-prefix ${WORKDIR}/output/ \
    --threads ${SLURM_CPUS_PER_TASK}

EXIT_CODE=$?
echo "[val] singlify exit code: ${EXIT_CODE}"

# Check result
if [ -f "${WORKDIR}/output/summary.json" ]; then
    echo "[val] summary.json:"
    cat ${WORKDIR}/output/summary.json
    CELLS=$(python3 -c "import json; d=json.load(open('${WORKDIR}/output/summary.json')); print(d.get('estimated_cells', d.get('cells_called', d.get('n_cells', 'NOT_FOUND'))))" 2>/dev/null || echo "PARSE_ERROR")
    echo "[val] CELLS_CALLED: ${CELLS}"
    if python3 -c "
import json, sys
d = json.load(open('${WORKDIR}/output/summary.json'))
n = d.get('estimated_cells', d.get('cells_called', d.get('n_cells', 0)))
if 1500 <= int(n) <= 3780:
    print(f'[val] VALIDATION PASS: {n} cells (target: 1500-3780)')
    sys.exit(0)
else:
    print(f'[val] VALIDATION FAIL: {n} cells (target: 1500-3780)')
    sys.exit(1)
" 2>/dev/null; then
        echo "[val] RESULT: PASS"
    else
        echo "[val] RESULT: FAIL"
    fi
else
    echo "[val] ERROR: summary.json not found"
    ls ${WORKDIR}/output/ 2>/dev/null || echo "(output dir empty/missing)"
    echo "[val] RESULT: FAIL (no output)"
fi

# Check mapping rate
if [ -f "${WORKDIR}/output/star_Log.final.out" ]; then
    MAPRATE=$(grep "Uniquely mapped reads %" ${WORKDIR}/output/star_Log.final.out | awk '{print $NF}')
    echo "[val] Mapping rate: ${MAPRATE}"
fi
