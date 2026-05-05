#!/usr/bin/env bash
#SBATCH --job-name=scgeo-kr
#SBATCH --account=bio260157
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/anvil/projects/x-bio260157/scgeo/pipeline/logs/kr_%j.out
#
# ── Dedicated Kraken2 classification ────────────────────────────────
# SU cost: max(8, ceil(128/1.97)) = 65 SU/hr
# Loads the 104 GB kraken2 DB into RAM (no --memory-mapping), so the
# DB load (~2 min) is amortised across KRAKEN_BATCH samples.
#
# Scans $SCRATCH/scgeo_downloads/ for samples with:
#   .quant_done (status = success|qc_warn) AND mode = droplet AND NO .kraken2_done
# Uses O_EXCL lock files to prevent duplicate processing.
# After each sample, cleans up FASTQs to free scratch space.
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
export KRAKEN_BATCH="${KRAKEN_BATCH:-20}"

cd /tmp

mkdir -p "$RESULT_DIR"

echo "════════════════════════════════════════════════════"
echo "Kraken2 phase | job ${SLURM_JOB_ID:-local} | $(hostname) | $SLURM_CPUS_PER_TASK CPUs"
echo "Memory: $(free -g | awk '/Mem:/{print $2}')GB total"
echo "════════════════════════════════════════════════════"

python3 -B << 'PYEOF'
import json, os, sys, time, logging, glob, shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("kraken2_worker")

from scgeo.pipeline.kraken2 import classify_nonhost
from scgeo.config import get_config

dl_dir = Path(os.environ["DL_DIR"])
result_dir = Path(os.environ["RESULT_DIR"])
output_base = Path(os.environ["SCGEO_BASE"]) / "pipeline"
kraken_batch = int(os.environ.get("KRAKEN_BATCH", "20"))
job_id = os.environ.get("SLURM_JOB_ID", "local")

config = get_config()
config.kraken2.memory_mapping = False  # Load DB into RAM (~104 GB, ~2 min)

# ── Discover quant-done samples needing kraken2 ──────────────────────
quant_dones = sorted(glob.glob(str(dl_dir / "*" / "*" / ".quant_done")))
ready = []
for qd_path in quant_dones:
    qd_path = Path(qd_path)
    sample_dir = qd_path.parent
    kr_done = sample_dir / ".kraken2_done"
    if kr_done.exists():
        continue

    # Read quant status
    try:
        qd_meta = json.loads(qd_path.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if qd_meta.get("status") not in ("success", "qc_warn"):
        continue

    # Need sample_manifest.json for chemistry + organism
    # It lives in the pipeline output dir: $SCGEO_BASE/pipeline/$GSE/$GSM/
    gsm_id = sample_dir.name
    gse_id = sample_dir.parent.name
    quant_dir = output_base / gse_id / gsm_id
    manifest_path = quant_dir / "sample_manifest.json"
    if not manifest_path.exists():
        continue

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        continue

    # Only droplet samples
    if manifest.get("mode") != "droplet":
        continue

    # Need download_manifest for R1/R2 paths
    dl_manifest_path = sample_dir / "download_manifest.json"
    if not dl_manifest_path.exists():
        continue

    try:
        dl_meta = json.loads(dl_manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        continue

    ready.append({
        "sample_dir": sample_dir,
        "quant_dir": quant_dir,
        "gsm_id": gsm_id,
        "gse_id": gse_id,
        "manifest": manifest,
        "dl_meta": dl_meta,
    })

if not ready:
    print("No samples need kraken2 — exiting cleanly")
    sys.exit(0)

print(f"\n▸ Found {len(ready)} samples needing kraken2, will process up to {kraken_batch}\n")

ok, fail, skip = 0, 0, 0

for info in ready:
    if ok + fail >= kraken_batch:
        break

    sample_dir = info["sample_dir"]
    quant_dir = info["quant_dir"]
    gsm_id = info["gsm_id"]
    manifest = info["manifest"]
    dl_meta = info["dl_meta"]

    lock_path = sample_dir / ".kraken2_locked"
    done_path = sample_dir / ".kraken2_done"

    # Atomically lock
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{job_id}\n".encode())
        os.close(fd)
    except FileExistsError:
        skip += 1
        continue

    # Resolve R1/R2 paths (swap if detection said so)
    r1_paths = [Path(p) for p in dl_meta.get("r1_paths", [])]
    r2_paths = [Path(p) for p in dl_meta.get("r2_paths", [])]
    if manifest.get("reads_swapped", False):
        r1_paths, r2_paths = r2_paths, r1_paths

    # Verify files exist
    missing = [p for p in r1_paths + r2_paths if not p.exists()]
    if missing:
        logger.error(f"[{gsm_id}] FASTQs missing: {missing}")
        fail += 1
        done_path.write_text(json.dumps({
            "status": "failed", "error": "fastq_missing",
            "kraken2_job": job_id,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        lock_path.unlink(missing_ok=True)
        continue

    chemistry = manifest.get("chemistry_used", "")
    organism = manifest.get("organism", "")
    taxon_id = manifest.get("taxon_id")
    if not chemistry or taxon_id is None:
        logger.error(f"[{gsm_id}] Missing chemistry or taxon_id in manifest")
        fail += 1
        done_path.write_text(json.dumps({
            "status": "failed", "error": "missing_metadata",
            "kraken2_job": job_id,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        lock_path.unlink(missing_ok=True)
        continue

    t0 = time.time()
    try:
        kr_result = classify_nonhost(
            r1_paths=r1_paths,
            r2_paths=r2_paths,
            chemistry=chemistry,
            host_taxon_id=taxon_id,
            quant_dir=quant_dir,
            output_dir=quant_dir,
            config=config,
        )
        elapsed = time.time() - t0

        if kr_result.success:
            ok += 1
            print(f"  ✓ {gsm_id}: kraken2 ok ({elapsed:.0f}s, "
                  f"nonhost={getattr(kr_result, 'frac_nonhost', 0):.3f})")
        else:
            fail += 1
            print(f"  ✗ {gsm_id}: {kr_result.error} ({elapsed:.0f}s)")

        done_path.write_text(json.dumps({
            "status": "success" if kr_result.success else "failed",
            "error": kr_result.error or "",
            "frac_nonhost": getattr(kr_result, "frac_nonhost", None),
            "cells_with_nonhost": getattr(kr_result, "cells_with_nonhost", None),
            "total_nonhost_umis": getattr(kr_result, "total_nonhost_umis", None),
            "time_s": kr_result.time_s,
            "kraken2_job": job_id,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))

        # Also update sample_manifest.json with kraken2 stats
        if kr_result.success:
            try:
                manifest_path = quant_dir / "sample_manifest.json"
                sm = json.loads(manifest_path.read_text())
                sm["kraken2_frac_nonhost"] = getattr(kr_result, "frac_nonhost", None)
                sm["kraken2_cells_with_nonhost"] = getattr(kr_result, "cells_with_nonhost", None)
                sm["kraken2_total_nonhost_umis"] = getattr(kr_result, "total_nonhost_umis", None)
                manifest_path.write_text(json.dumps(sm, indent=2))
            except Exception as e:
                logger.warning(f"[{gsm_id}] Could not update sample_manifest: {e}")

    except Exception as e:
        elapsed = time.time() - t0
        logger.exception(f"[{gsm_id}] Kraken2 exception")
        fail += 1
        done_path.write_text(json.dumps({
            "status": "failed", "error": str(e),
            "kraken2_job": job_id,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))

    lock_path.unlink(missing_ok=True)

    # ── Clean up FASTQs now that kraken2 is done ──────────────────────
    for p in r1_paths + r2_paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    # Also clean any remaining FASTQ/SRA files
    for pattern in ("**/*.fastq.gz", "**/*.fq.gz", "**/*.fastq", "**/*.fq"):
        for p in sample_dir.glob(pattern):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    # Remove empty subdirectories
    for d in sorted(sample_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass

print(f"\n{'═'*50}")
print(f"  ✓ {ok} success | ✗ {fail} failed | ○ {skip} locked/skipped")
print(f"{'═'*50}\n")
PYEOF

echo "Kraken2 phase complete."
