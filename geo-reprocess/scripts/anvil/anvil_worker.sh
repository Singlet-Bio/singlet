#!/bin/bash
#SBATCH --job-name=p4a_mouse
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -p shared
#SBATCH --array=0-499
#
# Anvil worker for Phase 4a (Mouse) processing.
#
# Usage:
#   sbatch -A YOUR_ALLOCATION anvil_worker.sh
#
# Memory: 64GB matches Clipper's proven production config (cpu/short).
# CPUs: 4 — piscem peak RSS scales with thread count. At 4 CPUs, the
#   peak stays under 64GB for nearly all samples. At 8 CPUs, peaks hit
#   90GB+ on large samples (OOM). Mouse index (2GB) is smaller than
#   human (3.3GB), giving further headroom.
#
# SU cost on shared partition: max(4/128, 64/257) × 128 ≈ 32 SU/hr.
# (Memory-dominated — 4 vs 8 CPUs costs the same SUs!)
# At ~1 hr per 10-sample batch → 32 SU per batch → ~3.2 SU per sample.
# Total: 21,304 samples × 3.2 SU ≈ 68,000 SU (within 100k budget).
#
# With 500 tasks × 10 samples each → 5000 samples per submission.
# Re-submit until grab_batch reports no more samples.
#
# Output logs go to $SCGEO_BASE/pipeline/logs/
# Results go to $SCGEO_BASE/pipeline/results/

set -euo pipefail

module load anaconda
conda activate "$PROJECT/envs/scgeo"

export SCGEO_BASE="$PROJECT/scgeo"
export SCGEO_WORKSPACE="$PROJECT/geo-reprocess"
export SCGEO_INDEX_DIR="$SCGEO_BASE/index"
export ALEVIN_FRY_HOME="$SCGEO_BASE/af_home"
export PYTHONUNBUFFERED=1

# Ensure grab_batch uses our Anvil paths
export SCGEO_PIPELINE_DIR="$SCGEO_BASE/pipeline"
export SCGEO_CATALOG_DIR="$SCGEO_BASE/catalog"

cd /tmp

echo "Memory: $(free -g | awk '/Mem:/{print $2}')GB total"

PHASE="${PHASE:-4a}"
BATCH_SIZE="${BATCH_SIZE:-10}"
CLAIM_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RESULT_DIR="$SCGEO_BASE/pipeline/results"
SCRIPTS_DIR="$SCGEO_WORKSPACE/scripts"
BATCH_FILE="$RESULT_DIR/batch_${CLAIM_ID}.csv"

mkdir -p "$RESULT_DIR" "$SCGEO_BASE/pipeline/logs"

# Signal handler
CLAIM_MADE=0
cleanup_on_signal() {
    echo "Signal received — marking claim $CLAIM_ID as abandoned"
    if [[ "$CLAIM_MADE" -eq 1 ]]; then
        python3 "$SCRIPTS_DIR/../scripts/grab_batch.py" --phase "$PHASE" --mark-abandoned "$CLAIM_ID" 2>/dev/null || true
    fi
    exit 1
}
trap cleanup_on_signal SIGTERM SIGINT

echo "════════════════════════════════════════════════════"
echo "Phase $PHASE | task $SLURM_ARRAY_TASK_ID | $(hostname) | $SLURM_CPUS_PER_TASK CPUs"
echo "════════════════════════════════════════════════════"

# NOTE: grab_batch.py is in the parent scripts/ dir, not in anvil/
GRAB_SCRIPT="$(dirname "$(dirname "$0")")/grab_batch.py"
if [[ ! -f "$GRAB_SCRIPT" ]]; then
    # Fallback: look relative to workspace
    GRAB_SCRIPT="$SCGEO_WORKSPACE/../scripts/grab_batch.py"
fi

python3 "$GRAB_SCRIPT" --phase "$PHASE" --batch-size "$BATCH_SIZE" --output "$BATCH_FILE"
GRAB_EXIT=$?

if [[ $GRAB_EXIT -ne 0 ]] || [[ ! -f "$BATCH_FILE" ]]; then
    echo "No unclaimed eligible samples remaining — exiting cleanly"
    exit 0
fi
CLAIM_MADE=1

N_SAMPLES=$(tail -n +2 "$BATCH_FILE" | wc -l)
echo "Claimed $N_SAMPLES samples (claim: $CLAIM_ID, phase: $PHASE)"

# Process the batch
python3 -B << 'PYEOF'
import sys, os, logging, time
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
from scgeo.pipeline.api import process_samples

batch_file = os.environ["BATCH_FILE"]
claim_id = os.environ["CLAIM_ID"]
result_dir = os.environ["RESULT_DIR"]
phase = os.environ.get("PHASE", "4a")

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

all_gsms = [r.gsm_id for r in results]
with open(os.environ.get("BATCH_FILE", "") + ".done_gsms", "w") as f:
    f.write(",".join(all_gsms))
PYEOF

# Report completed GSMs
export BATCH_FILE CLAIM_ID RESULT_DIR PHASE
if [[ -f "${BATCH_FILE}.done_gsms" ]]; then
    DONE_GSMS=$(cat "${BATCH_FILE}.done_gsms")
    python3 "$GRAB_SCRIPT" --phase "$PHASE" --report-gsms "$CLAIM_ID" "$DONE_GSMS" || \
        python3 "$GRAB_SCRIPT" --phase "$PHASE" --mark-done "$CLAIM_ID" || \
        echo "WARNING: could not mark claim $CLAIM_ID as done"
    rm -f "${BATCH_FILE}.done_gsms"
else
    python3 "$GRAB_SCRIPT" --phase "$PHASE" --mark-done "$CLAIM_ID" || true
fi

echo "Job complete | $(date)"
