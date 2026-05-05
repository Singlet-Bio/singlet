#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/projects/debruinz_project/cellarium/pipeline/logs/pipeline_%a_%j.out
#SBATCH --error=/mnt/projects/debruinz_project/cellarium/pipeline/logs/pipeline_%a_%j.err
#
# Unified SLURM worker for the Singlet Bio reprocessing pipeline.
#
# This script is submitted by launch.sh with partition-specific overrides:
#   sbatch --partition=cpu    --mem=128G --time=2-00:00:00 --array=0-999%58  ...
#   sbatch --partition=bigmem --mem=128G --time=2-00:00:00 --array=0-199%18  ...
#   sbatch --partition=short  --mem=128G --time=1-00:00:00 --array=0-999%33  ...
#
# 8 CPUs per task, 128G per task → fills all 22 nodes (109 concurrent).
# simpleaf at 8 threads for faster mapping.
#
# Required environment variable (set via --export at sbatch time):
#   PHASE  — processing phase (1, 2a, 2b, 2c, 2d, 3, 4a, 4b)
#
# Optional overrides:
#   BATCH_SIZE — samples per task (default: 5)

set -euo pipefail

# ── Validate phase ──
if [[ -z "${PHASE:-}" ]]; then
    echo "ERROR: PHASE environment variable not set. Use launch.sh to submit."
    exit 1
fi

module load miniconda3/25.5.1
source /opt/gvsu/clipper/2025.12/spack/apps/linux-cascadelake/miniconda3-25.5.1-xe7kyofwhfxilia75rj5t63zf6wpzzcr/etc/profile.d/conda.sh
conda activate cellarium

# IMPORTANT: cd away from Singlet-AI/ — the singlepress/ repo dir there
# shadows the editable pip install as a namespace package (no __init__.py at repo root)
cd /tmp

export SCGEO_BASE="/mnt/projects/debruinz_project/cellarium"
export SCGEO_WORKSPACE="/mnt/home/debruinz/Singlet-AI/geo-reprocess"
export ALEVIN_FRY_HOME="/mnt/projects/debruinz_project/cellarium/af_home"
export PYTHONUNBUFFERED=1
ulimit -n 2048

SCRIPTS_DIR="/mnt/home/debruinz/Singlet-AI/scripts"
CLAIM_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
BATCH_SIZE="${BATCH_SIZE:-5}"
RESULT_DIR="/mnt/projects/debruinz_project/cellarium/pipeline/results"
mkdir -p "$RESULT_DIR"

# ── Signal handler: mark claim as failed on preemption/cancellation ──
CLAIM_MADE=0
cleanup_on_signal() {
    echo "Signal received — marking claim $CLAIM_ID as abandoned"
    if [[ "$CLAIM_MADE" -eq 1 ]]; then
        python3 "$SCRIPTS_DIR/grab_batch.py" --phase "$PHASE" --mark-abandoned "$CLAIM_ID" 2>/dev/null || true
    fi
    exit 1
}
trap cleanup_on_signal SIGTERM SIGINT

echo "════════════════════════════════════════════════════"
echo "phase $PHASE | task $SLURM_ARRAY_TASK_ID | $(hostname) | ${SLURM_CPUS_PER_TASK} CPUs, ${SLURM_MEM_PER_NODE}M | ${SLURM_JOB_PARTITION}"
echo "════════════════════════════════════════════════════"

# ── Grab a batch of unclaimed eligible samples ──
BATCH_FILE="${RESULT_DIR}/batch_${CLAIM_ID}.csv"
python3 "$SCRIPTS_DIR/grab_batch.py" --phase "$PHASE" --batch-size "$BATCH_SIZE" --output "$BATCH_FILE"
GRAB_EXIT=$?

if [[ $GRAB_EXIT -ne 0 ]] || [[ ! -f "$BATCH_FILE" ]]; then
    echo "No unclaimed eligible samples remaining — exiting cleanly"
    exit 0
fi
CLAIM_MADE=1

N_SAMPLES=$(tail -n +2 "$BATCH_FILE" | wc -l)
echo "Claimed $N_SAMPLES samples (claim: $CLAIM_ID, phase: $PHASE)"
echo "Disk: $(df -h /mnt/projects/debruinz_project/ | tail -1 | awk '{print $4 " free (" $5 " used)"}')"

# ── Process the batch ──
export BATCH_FILE CLAIM_ID RESULT_DIR SCRIPTS_DIR PHASE

python3 -B << 'PYEOF'
import sys, os, logging, time
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
from scgeo.pipeline.api import process_samples

batch_file = os.environ["BATCH_FILE"]
claim_id = os.environ["CLAIM_ID"]
result_dir = os.environ["RESULT_DIR"]
phase = os.environ["PHASE"]

df = pd.read_csv(batch_file)
print(f"\n▸ Processing {len(df)} samples (claim: {claim_id}, phase: {phase})\n")

samples = []
for _, row in df.iterrows():
    sample = {
        'gsm_id': row['gsm_id'],
        'gse_id': row['gse_id'],
        'organism': row.get('organism', ''),
    }
    if pd.notna(row.get('ena_fastq_r1')):
        sample['ena_r1_url'] = row['ena_fastq_r1']
    if pd.notna(row.get('ena_fastq_r2')):
        sample['ena_r2_url'] = row['ena_fastq_r2']
    if pd.notna(row.get('srr_accessions')):
        sample['srr_accession'] = str(row['srr_accessions'])
    if pd.notna(row.get('protocol_inferred')):
        sample['protocol_hint'] = row['protocol_inferred']
    if pd.notna(row.get('protocol_confidence')):
        sample['catalog_confidence'] = row['protocol_confidence']
    if pd.notna(row.get('read_count')):
        try:
            sample['read_count'] = int(row['read_count'])
        except (ValueError, TypeError):
            pass
    samples.append(sample)

t0 = time.time()
results = process_samples(samples=samples)
elapsed = time.time() - t0

results_df = pd.DataFrame([r.to_dict() for r in results])
result_file = os.path.join(result_dir, f"results_{claim_id}.csv")
results_df.to_csv(result_file, index=False)

n_ok = sum(1 for r in results if r.status in ('success', 'qc_warn'))
n_skip = sum(1 for r in results if r.status == 'skipped')
n_fail = sum(1 for r in results if r.status == 'failed')
total_cells = sum(getattr(r, 'n_cells', 0) or 0 for r in results)

print(f"\n{'═'*50}")
print(f"  ✓ {n_ok} success ({total_cells:,} cells)")
print(f"  ○ {n_skip} skipped")
print(f"  ✗ {n_fail} failed")
print(f"  ⏱ {elapsed:.0f}s elapsed")
print(f"{'═'*50}\n")

for r in results:
    if r.status == 'failed':
        print(f"  FAIL {r.gsm_id}: {r.error}")

# Write processed GSM list for shell to pick up
all_gsms = [r.gsm_id for r in results]
with open(os.environ.get("BATCH_FILE", "") + ".done_gsms", "w") as f:
    f.write(",".join(all_gsms))
PYEOF

# ── Report processed GSMs to the completed sidecar ──
if [[ -f "${BATCH_FILE}.done_gsms" ]]; then
    DONE_GSMS=$(cat "${BATCH_FILE}.done_gsms")
    if ! python3 "$SCRIPTS_DIR/grab_batch.py" --phase "$PHASE" --report-gsms "$CLAIM_ID" "$DONE_GSMS"; then
        echo "WARNING: --report-gsms failed, falling back to --mark-done"
        python3 "$SCRIPTS_DIR/grab_batch.py" --phase "$PHASE" --mark-done "$CLAIM_ID" || \
            echo "ERROR: --mark-done also failed for claim $CLAIM_ID"
    fi
    rm -f "${BATCH_FILE}.done_gsms"
else
    python3 "$SCRIPTS_DIR/grab_batch.py" --phase "$PHASE" --mark-done "$CLAIM_ID" || \
        echo "ERROR: --mark-done failed for claim $CLAIM_ID"
fi

echo "Job complete | $(date)"
