#!/usr/bin/env bash
#SBATCH --job-name=scgeo-dl
#SBATCH --account=bio260157
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=/anvil/projects/x-bio260157/scgeo/pipeline/logs/dl_%j.out
#
# ── Download-only phase ──────────────────────────────────────────────
# SU cost: ceil(6/1.97) = 4 SU/hr  (down from 5 at 8GB)
#
# Can also run on login node for zero SU cost:
#   export PHASE=4a BATCH_SIZE=20
#   bash anvil_download.sh   # (without sbatch)
#
# Downloads FASTQs to $SCRATCH/scgeo_downloads/{gse_id}/{gsm_id}/
# Writes a download_manifest.json alongside the FASTQs for the quant phase.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT="${PROJECT:-/anvil/projects/x-bio260157}"
SCRATCH="${SCRATCH:-/anvil/scratch/x-zdebruine}"

# Load environment (no-op if already active, e.g. login node usage)
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    module load anaconda 2>/dev/null || true
    conda activate "$PROJECT/envs/scgeo"
fi

export SCGEO_BASE="$PROJECT/scgeo"
export SCGEO_WORKSPACE="$PROJECT/geo-reprocess"
export SCGEO_PIPELINE_DIR="$SCGEO_BASE/pipeline"
export SCGEO_CATALOG_DIR="$SCGEO_BASE/catalog"
export PYTHONUNBUFFERED=1

PHASE="${PHASE:-4a}"
BATCH_SIZE="${BATCH_SIZE:-20}"
DL_DIR="$SCRATCH/scgeo_downloads"
RESULT_DIR="$SCGEO_BASE/pipeline/results"
SCRIPTS_DIR="$SCGEO_WORKSPACE/scripts"

# Claim ID: use SLURM vars if available, else PID
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    CLAIM_ID="dl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
else
    CLAIM_ID="dl_login_$$"
fi
BATCH_FILE="$RESULT_DIR/batch_${CLAIM_ID}.csv"

export BATCH_FILE DL_DIR CLAIM_ID RESULT_DIR PHASE

mkdir -p "$RESULT_DIR" "$DL_DIR" "$SCGEO_BASE/pipeline/logs"

echo "════════════════════════════════════════════════════"
echo "Download phase | ${PHASE} | task ${SLURM_ARRAY_TASK_ID:-login} | $(hostname)"
echo "════════════════════════════════════════════════════"

# Locate grab_batch.py (lives in scripts/, one level above scripts/anvil/)
GRAB_SCRIPT="$SCGEO_WORKSPACE/scripts/grab_batch.py"
if [[ ! -f "$GRAB_SCRIPT" ]]; then
    GRAB_SCRIPT="$(dirname "$(dirname "$0")")/grab_batch.py"
fi

python3 "$GRAB_SCRIPT" --phase "$PHASE" --batch-size "$BATCH_SIZE" --output "$BATCH_FILE"
GRAB_EXIT=$?

if [[ $GRAB_EXIT -ne 0 ]] || [[ ! -f "$BATCH_FILE" ]]; then
    echo "No unclaimed eligible samples remaining — exiting cleanly"
    exit 0
fi

N_SAMPLES=$(tail -n +2 "$BATCH_FILE" | wc -l)
echo "Claimed $N_SAMPLES samples → downloading to $DL_DIR"

# ── Download loop (Python) ───────────────────────────────────────────
python3 -B << 'PYEOF'
import json, os, sys, time, logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from scgeo.pipeline.download import download_sample, compute_download_timeout
from scgeo.config import get_config

batch_file = os.environ["BATCH_FILE"]
dl_dir = Path(os.environ["DL_DIR"])
claim_id = os.environ["CLAIM_ID"]

config = get_config()

df = pd.read_csv(batch_file)
print(f"\n▸ Downloading {len(df)} samples (claim: {claim_id})\n")

ok, fail = 0, 0
for _, row in df.iterrows():
    gsm_id = row["gsm_id"]
    gse_id = row["gse_id"]
    sample_dir = dl_dir / gse_id / gsm_id
    manifest_path = sample_dir / "download_manifest.json"

    # Skip if already downloaded
    if manifest_path.exists():
        try:
            meta = json.loads(manifest_path.read_text())
            if meta.get("success"):
                print(f"  ✓ {gsm_id} already downloaded — skipping")
                ok += 1
                continue
        except json.JSONDecodeError:
            pass

    sample_dir.mkdir(parents=True, exist_ok=True)

    ena_r1 = str(row["ena_fastq_r1"]) if pd.notna(row.get("ena_fastq_r1")) else None
    ena_r2 = str(row["ena_fastq_r2"]) if pd.notna(row.get("ena_fastq_r2")) else None
    srr = str(row["srr_accessions"]) if pd.notna(row.get("srr_accessions")) else None

    # Compute per-sample timeout from read count
    read_count = 0
    if pd.notna(row.get("read_count")):
        try:
            read_count = int(row["read_count"])
        except (ValueError, TypeError):
            pass
    sample_max_time = compute_download_timeout(read_count)

    t0 = time.time()
    result = download_sample(
        gsm_id=gsm_id,
        ena_r1_url=ena_r1,
        ena_r2_url=ena_r2,
        srr_accession=srr,
        output_dir=sample_dir,
        config=config,
        sample_max_time=sample_max_time,
    )
    elapsed = time.time() - t0

    # Persist download result as JSON manifest for the quant phase
    manifest = {
        "gsm_id": gsm_id,
        "gse_id": gse_id,
        "organism": row.get("organism", ""),
        "success": result.success,
        "r1_paths": [str(p) for p in result.r1_paths],
        "r2_paths": [str(p) for p in result.r2_paths],
        "sra_path": str(result.sra_path) if result.sra_path else "",  # fix: include sra_path for s3_sra_pending routing
        "method": result.method,
        "time_s": result.time_s,
        "error": result.error,
        "fail_category": result.fail_category,
        # Carry forward catalog metadata for quant phase
        "protocol_hint": str(row.get("protocol_inferred", "")) if pd.notna(row.get("protocol_inferred")) else "",
        "catalog_confidence": str(row.get("protocol_confidence", "")) if pd.notna(row.get("protocol_confidence")) else "",
        "claim_id": claim_id,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if result.success:
        ok += 1
        r1_size = sum(Path(p).stat().st_size for p in result.r1_paths if Path(p).exists()) / 1e9
        r2_size = sum(Path(p).stat().st_size for p in result.r2_paths if Path(p).exists()) / 1e9
        print(f"  ✓ {gsm_id} ({result.method}, {r1_size+r2_size:.1f}GB, {elapsed:.0f}s)")
    else:
        fail += 1
        print(f"  ✗ {gsm_id}: {result.error}")

print(f"\n{'═'*50}")
print(f"  Downloaded: {ok}  |  Failed: {fail}  |  Total: {ok+fail}")
print(f"{'═'*50}\n")

# Write done GSMs for grab_batch bookkeeping
all_gsms = list(df["gsm_id"])
with open(batch_file + ".done_gsms", "w") as f:
    f.write(",".join(all_gsms))
PYEOF

# Report completed GSMs
if [[ -f "${BATCH_FILE}.done_gsms" ]]; then
    DONE_GSMS=$(cat "${BATCH_FILE}.done_gsms")
    python3 "$GRAB_SCRIPT" --phase "$PHASE" --report-gsms "$CLAIM_ID" "$DONE_GSMS" || \
        python3 "$GRAB_SCRIPT" --phase "$PHASE" --mark-done "$CLAIM_ID" || \
        echo "WARNING: could not mark claim $CLAIM_ID as done"
    rm -f "${BATCH_FILE}.done_gsms"
fi

echo "Download phase complete."
