#!/usr/bin/env bash
#SBATCH --job-name=scgeo-qt-xl
#SBATCH --account=bio260157
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/anvil/projects/x-bio260157/scgeo/pipeline/logs/qt_xl_%j.out
#
# ── XL Quantification phase (OOM retry) ──────────────────────────────────
# SU cost: max(8, ceil(32/1.97)) = max(8, 17) = 17 SU/hr (memory dominates)
#
# Processes samples marked .needs_xl by the normal quant job after:
#   - OUT_OF_MEMORY state in SLURM
#   - 2+ failed quant attempts from stale lock detection
#
# Only processes 1 sample per job (these are large samples). Clears the
# .needs_xl and .quant_locked markers so the sample can be re-claimed.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT="${PROJECT:-/anvil/projects/x-bio260157}"
SCRATCH="${SCRATCH:-/anvil/scratch/x-zdebruine}"

module load anaconda
conda activate "$PROJECT/envs/scgeo"
umask 0022  # Reset after conda activate (may set 0117)

export SCGEO_BASE="$PROJECT/scgeo"
export SCGEO_WORKSPACE="$PROJECT/geo-reprocess"
export SCGEO_INDEX_DIR="$SCGEO_BASE/index"
export ALEVIN_FRY_HOME="$SCGEO_BASE/af_home"
export SCGEO_PIPELINE_DIR="$SCGEO_BASE/pipeline"
export SCGEO_CATALOG_DIR="$SCGEO_BASE/catalog"
export PYTHONUNBUFFERED=1

export DL_DIR="$SCRATCH/scgeo_downloads"
export RESULT_DIR="$SCGEO_BASE/pipeline/results"
export QUANT_BATCH="${QUANT_BATCH:-1}"  # process 1 large sample per XL job

cd /tmp   # avoid singlepress namespace shadow bug

mkdir -p "$RESULT_DIR"

echo "════════════════════════════════════════════════════"
echo "Quant XL phase | job ${SLURM_JOB_ID:-local} | $(hostname) | $SLURM_CPUS_PER_TASK CPUs"
echo "Memory: $(free -g | awk '/Mem:/{print $2}')GB total"
echo "════════════════════════════════════════════════════"

# ── Find and process XL-marked samples (Python) ──────────────────────
python3 -B << 'PYEOF'
import json, os, sys, time, logging, glob, shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("quant_xl_worker")

from scgeo.pipeline.download import DownloadResult
from scgeo.pipeline.api import _process_after_download, SampleResult
from scgeo.config import get_config

dl_dir = Path(os.environ["DL_DIR"])
result_dir = Path(os.environ["RESULT_DIR"])
output_base = Path(os.environ["SCGEO_BASE"]) / "pipeline"
quant_batch = int(os.environ.get("QUANT_BATCH", "1"))
job_id = os.environ.get("SLURM_JOB_ID", "local")

config = get_config()
config.kraken2.enabled = False
config.cleanup.after_qc = True

# ── Discover .needs_xl samples ────────────────────────────────────────
# These are samples marked by normal quant jobs after OOM or repeat failure.
xl_markers = sorted(dl_dir.rglob(".needs_xl"))
ready = []
for marker in xl_markers:
    sample_dir = marker.parent
    mpath = sample_dir / "download_manifest.json"
    if not mpath.exists():
        continue
    done_marker = sample_dir / ".quant_done"
    if done_marker.exists():
        marker.unlink(missing_ok=True)
        (sample_dir / ".quant_locked").unlink(missing_ok=True)
        continue
    try:
        meta = json.loads(mpath.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if not meta.get("success"):
        continue
    ready.append((mpath, meta))

if not ready:
    print("No .needs_xl samples found — exiting cleanly")
    sys.exit(0)

print(f"\n▸ Found {len(ready)} XL samples, will process up to {quant_batch}\n")

ok, fail, skip = 0, 0, 0
results = []

for mpath, meta in ready:
    if ok + fail >= quant_batch:
        break

    gsm_id = meta["gsm_id"]
    gse_id = meta["gse_id"]
    sample_dir = mpath.parent
    lock_path = sample_dir / ".quant_locked"
    done_path = sample_dir / ".quant_done"
    xl_path = sample_dir / ".needs_xl"

    # Release the existing lock (from the dead job) and re-claim atomically
    lock_path.unlink(missing_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{job_id}\n".encode())
        os.close(fd)
    except FileExistsError:
        skip += 1
        continue

    # Reconstruct DownloadResult from manifest
    dl_result = DownloadResult(
        success=meta["success"],
        r1_paths=[Path(p) for p in meta["r1_paths"]],
        r2_paths=[Path(p) for p in meta["r2_paths"]],
        method=meta.get("method", ""),
        time_s=meta.get("time_s", 0.0),
        error=meta.get("error", ""),
        fail_category=meta.get("fail_category", ""),
    )

    # Verify FASTQ files still exist
    missing = [p for p in dl_result.r1_paths + dl_result.r2_paths if not p.exists()]
    if missing:
        logger.error(f"[{gsm_id}] Missing FASTQ files: {missing}")
        fail += 1
        result = SampleResult(gsm_id=gsm_id, gse_id=gse_id, organism=meta.get("organism", ""))
        result.status = "failed"
        result.fail_stage = "missing_fastq"
        result.error = f"FASTQ files purged from scratch: {missing}"
        result.total_time_s = 0.0
        results.append(result)
        done_path.write_text(json.dumps({
            "status": "failed", "error": "missing_fastq",
            "quant_job": job_id,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        lock_path.unlink(missing_ok=True)
        xl_path.unlink(missing_ok=True)
        continue

    t0 = time.time()
    try:
        result = _process_after_download(
            gsm_id=gsm_id,
            gse_id=gse_id,
            organism=meta.get("organism", ""),
            download_result=dl_result,
            output_base=output_base,
            config=config,
            t0=t0,
            protocol_hint=meta.get("protocol_hint") or None,
            catalog_confidence=meta.get("catalog_confidence") or None,
        )
        results.append(result)
        elapsed = time.time() - t0

        if result.status in ("success", "qc_warn"):
            ok += 1
            cells = getattr(result.qc, "n_cells", 0) or 0
            print(f"  ✓ {gsm_id}: {result.status} ({cells:,} cells, {elapsed:.0f}s)")
        else:
            fail += 1
            print(f"  ✗ {gsm_id}: {result.status} — {result.error}")

    except Exception as e:
        logger.exception(f"[{gsm_id}] Unhandled exception")
        fail += 1
        result = SampleResult(gsm_id=gsm_id, gse_id=gse_id, organism=meta.get("organism", ""))
        result.status = "failed"
        result.fail_stage = "quant_exception"
        result.error = str(e)
        result.total_time_s = time.time() - t0
        results.append(result)

    # Write done marker and clean up XL marker
    done_path.write_text(json.dumps({
        "status": result.status,
        "quant_job": job_id,
        "quant_xl": True,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    xl_path.unlink(missing_ok=True)
    (sample_dir / ".quant_attempts").unlink(missing_ok=True)

    # Clean up FASTQs + intermediates
    for pattern in ("**/*.fastq.gz", "**/*.fq.gz", "**/*.fastq", "**/*.fq",
                    "**/*.sra", "**/*.sra.cache", "**/.dl_segments_*"):
        for p in sample_dir.glob(pattern):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    for d in sorted(sample_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass

# ── Write results CSV ─────────────────────────────────────────────────
if results:
    import pandas as pd
    rows = []
    for r in results:
        d = r.to_dict() if hasattr(r, "to_dict") else {"gsm_id": r.gsm_id, "status": r.status, "error": r.error}
        rows.append(d)
    results_df = pd.DataFrame(rows)
    result_file = result_dir / f"results_quant_xl_{job_id}.csv"
    results_df.to_csv(result_file, index=False)
    print(f"\nResults written to {result_file}")

total_cells = sum(
    getattr(r.qc, "n_cells", 0) or 0
    for r in results
    if hasattr(r, "qc") and r.qc is not None
)
print(f"\n{'═'*50}")
print(f"  ✓ {ok} success | ✗ {fail} failed | ○ {skip} locked/skipped")
print(f"  Total cells: {total_cells:,}")
print(f"{'═'*50}\n")
PYEOF

echo "Quant XL phase complete."
