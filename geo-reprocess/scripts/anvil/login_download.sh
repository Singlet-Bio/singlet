#!/usr/bin/env bash
# ── Login-node single-threaded download ──────────────────────────────
# Zero SU cost. Single curl segment (no parallel downloads per file).
# Usage:
#   bash login_download.sh           # download 5 samples
#   BATCH_SIZE=10 bash login_download.sh  # download 10 samples
#
# To launch many concurrently:
#   for i in $(seq 1 20); do bash login_download.sh &>>dl_login_$i.log & sleep 10; done
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT="${PROJECT:-/anvil/projects/x-bio260157}"
export SCRATCH="${SCRATCH:-/anvil/scratch/x-zdebruine}"

# Avoid "Disk quota exceeded" for bash here-documents on login nodes
# where /tmp may have per-user quotas
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "$TMPDIR"

# Load environment
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    module load anaconda 2>/dev/null || true
    conda activate "$PROJECT/envs/scgeo"
fi
umask 0022  # Reset after conda activate which may set 0117

export SCGEO_BASE="$PROJECT/scgeo"
export SCGEO_WORKSPACE="$PROJECT/geo-reprocess"
export SCGEO_PIPELINE_DIR="$SCGEO_BASE/pipeline"
export SCGEO_CATALOG_DIR="$SCGEO_BASE/catalog"
export PYTHONUNBUFFERED=1

PHASE="${PHASE:-4a}"

# Use pre-filtered catalog for phase 4a (avoids OOM from 222 MB full catalog)
if [[ "$PHASE" == "4a" && -f "$SCGEO_BASE/catalog/phase4a/processing_catalog.parquet" ]]; then
    export SCGEO_CATALOG_DIR="$SCGEO_BASE/catalog/phase4a"
fi
BATCH_SIZE="${BATCH_SIZE:-5}"
DL_DIR="$SCRATCH/scgeo_downloads"
RESULT_DIR="$SCGEO_BASE/pipeline/results"

CLAIM_ID="dl_login_$$"
BATCH_FILE="$RESULT_DIR/batch_${CLAIM_ID}.csv"

export BATCH_FILE DL_DIR CLAIM_ID RESULT_DIR PHASE

mkdir -p "$RESULT_DIR" "$DL_DIR" "$SCGEO_BASE/pipeline/logs"

echo "════════════════════════════════════════════════════"
echo "Login DL | segments=1 | batch=$BATCH_SIZE | $(hostname) | PID=$$"
echo "════════════════════════════════════════════════════"

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

# ── Download loop (segments=1 override) ──────────────────────────────
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
config.download.segments = 1  # Single-threaded curl — no parallel segments
config.sra.timeout = 14400    # 4 hr — large SRA samples need time on login nodes

df = pd.read_csv(batch_file)
print(f"\n▸ Downloading {len(df)} samples (claim: {claim_id}, segments=1)\n")

ok, fail = 0, 0
for _, row in df.iterrows():
    gsm_id = row["gsm_id"]
    gse_id = row["gse_id"]
    sample_dir = dl_dir / gse_id / gsm_id
    manifest_path = sample_dir / "download_manifest.json"

    if manifest_path.exists():
        try:
            meta = json.loads(manifest_path.read_text())
            if meta.get("success"):
                print(f"  ✓ {gsm_id} already downloaded — skipping")
                ok += 1
                continue
            # Retry previously-failed SRA downloads (fail_category starts with "sra_")
            fail_cat = meta.get("fail_category", "")
            if fail_cat.startswith("sra_"):
                print(f"  ↻ {gsm_id} retrying (prev: {fail_cat})")
                manifest_path.unlink()
            else:
                print(f"  ✗ {gsm_id} permanent failure ({fail_cat}) — skipping")
                fail += 1
                continue
        except json.JSONDecodeError:
            pass

    sample_dir.mkdir(parents=True, exist_ok=True)

    ena_r1 = str(row["ena_fastq_r1"]) if pd.notna(row.get("ena_fastq_r1")) else None
    ena_r2 = str(row["ena_fastq_r2"]) if pd.notna(row.get("ena_fastq_r2")) else None
    srr = str(row["srr_accessions"]) if pd.notna(row.get("srr_accessions")) else None

    read_count = 0
    if pd.notna(row.get("read_count")):
        try:
            read_count = int(row["read_count"])
        except (ValueError, TypeError):
            pass
    sample_max_time = compute_download_timeout(read_count, max_timeout=36000)  # 10 hr (login = 0 SU)

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

    manifest = {
        "gsm_id": gsm_id,
        "gse_id": gse_id,
        "organism": row.get("organism", ""),
        "success": result.success,
        "r1_paths": [str(p) for p in result.r1_paths],
        "r2_paths": [str(p) for p in result.r2_paths],
        "sra_path": str(result.sra_path) if result.sra_path else "",
        "method": result.method,
        "time_s": result.time_s,
        "error": result.error,
        "fail_category": result.fail_category,
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

        # P1: Validate R1 presence immediately after successful download.
        # ENA FASTQ downloads may succeed (HTTP 200) but produce no _1.fastq.gz
        # (e.g. single-ended or misnamed). Flag these as permanent failures here
        # so quant jobs never waste a slot on them.
        if result.method not in ("s3_sra_pending", "sra_prefetch"):
            r1_exists = any(Path(p).exists() and Path(p).stat().st_size > 0
                            for p in result.r1_paths)
            if not r1_exists:
                manifest["success"] = False
                manifest["fail_category"] = "no_r1_fastq"
                manifest["error"] = "No R1 FASTQ produced after download — marked permanent failure"
                manifest_path.write_text(json.dumps(manifest, indent=2))
                ok -= 1
                fail += 1
                print(f"  ✗ {gsm_id}: no R1 FASTQ found post-download (permanent)")
    else:
        fail += 1
        print(f"  ✗ {gsm_id}: {result.error}")

print(f"\n{'═'*50}")
print(f"  Downloaded: {ok}  |  Failed: {fail}  |  Total: {ok+fail}")
print(f"{'═'*50}\n")

with open(batch_file + ".done_gsms", "w") as f:
    f.write(",".join(list(df["gsm_id"])))
PYEOF

# Report completed GSMs
if [[ -f "${BATCH_FILE}.done_gsms" ]]; then
    DONE_GSMS=$(cat "${BATCH_FILE}.done_gsms")
    python3 "$GRAB_SCRIPT" --phase "$PHASE" --report-gsms "$CLAIM_ID" "$DONE_GSMS" || \
        python3 "$GRAB_SCRIPT" --phase "$PHASE" --mark-done "$CLAIM_ID" || \
        echo "WARNING: could not mark claim $CLAIM_ID as done"
    rm -f "${BATCH_FILE}.done_gsms"
fi

echo "Login download complete (PID $$)."
