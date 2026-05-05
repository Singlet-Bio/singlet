#!/usr/bin/env bash
#SBATCH --job-name=scgeo-qt
#SBATCH --account=bio260157
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/anvil/projects/x-bio260157/scgeo/pipeline/logs/qt_%j.out
#
# ── Quantification phase ──────────────────────────────────────────────────
# SU cost: max(8, ceil(8/1.97)) = max(8, 5) = 8 SU/hr  (CPU dominates)
#
# Scans $SCRATCH/scgeo_downloads/ for samples with download_manifest.json
# that haven't been quantified yet. Uses O_EXCL lock files to claim
# samples, preventing duplicate processing across parallel jobs.
#
# After processing (success OR failure), FASTQs are removed to free
# scratch space. Results are written to results_*.csv for catalog
# reconciliation.
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
export QUANT_BATCH="${QUANT_BATCH:-10}"
# P4: FASTQ_ONLY=true → skip s3_sra_pending samples (they go to dedicated SRA jobs)
export FASTQ_ONLY="${FASTQ_ONLY:-true}"

cd /tmp   # avoid singlepress namespace shadow bug

mkdir -p "$RESULT_DIR"

echo "════════════════════════════════════════════════════"
echo "Quant phase | job ${SLURM_JOB_ID:-local} | $(hostname) | $SLURM_CPUS_PER_TASK CPUs"
echo "Memory: $(free -g | awk '/Mem:/{print $2}')GB total"
echo "════════════════════════════════════════════════════"

# ── Find and process downloaded samples (Python) ─────────────────────
python3 -B << 'PYEOF'
import json, os, sys, time, logging, glob, shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("quant_worker")

from scgeo.pipeline.download import DownloadResult
from scgeo.pipeline.api import _process_after_download, SampleResult
from scgeo.config import get_config

dl_dir = Path(os.environ["DL_DIR"])
result_dir = Path(os.environ["RESULT_DIR"])
output_base = Path(os.environ["SCGEO_BASE"]) / "pipeline"
quant_batch = int(os.environ.get("QUANT_BATCH", "10"))
job_id = os.environ.get("SLURM_JOB_ID", "local")
# P4: In FASTQ_ONLY mode, skip samples still needing fasterq-dump
fastq_only = os.environ.get("FASTQ_ONLY", "true").lower() == "true"

config = get_config()
config.kraken2.enabled = False  # Skip kraken2 — not needed for this pipeline run
config.cleanup.after_qc = True  # Clean FASTQs after quant

# P7: Clean stale fasterq_tmp dirs (>6h old) left by interrupted jobs
_fq_tmp = Path(os.environ.get("SCRATCH", "/tmp")) / "scgeo_fasterq_tmp"
if _fq_tmp.exists():
    _now = time.time()
    _cleaned = 0
    for _entry in _fq_tmp.iterdir():
        try:
            if _entry.is_dir() and (_now - _entry.stat().st_mtime) > 21600:
                shutil.rmtree(_entry, ignore_errors=True)
                _cleaned += 1
        except OSError:
            pass
    if _cleaned:
        logger.info(f"P7: Cleaned {_cleaned} stale fasterq_tmp dirs (>6h old)")

# ── Release stale locks from dead/finished SLURM jobs ────────────────
# Handles samples from cancelled/OOM/timed-out jobs that never wrote .quant_done
import subprocess
_released = 0
for _lock in dl_dir.rglob(".quant_locked"):
    try:
        _done = _lock.parent / ".quant_done"
        if _done.exists():
            _lock.unlink(missing_ok=True)
            continue
        _owner = _lock.read_text().strip().split()[0]  # first token = job ID
        if not _owner.isdigit():
            continue
        # Check if owning job is still in the SLURM queue
        _sq = subprocess.run(["squeue", "-j", _owner, "-h"], capture_output=True, text=True)
        if _sq.stdout.strip():
            continue  # job still running — leave lock alone
        # Job is gone. Check its exit state for OOM.
        _sa = subprocess.run(
            ["sacct", "-j", _owner, "-n", "--format=State"],
            capture_output=True, text=True
        )
        _state = _sa.stdout.upper()
        _needs_xl = _lock.parent / ".needs_xl"
        _attempts = _lock.parent / ".quant_attempts"
        _try = int(_attempts.read_text().strip()) if _attempts.exists() else 0
        _try += 1
        _attempts.write_text(str(_try))
        if "OUT_OF_MEMORY" in _state or _try >= 2:
            # Needs more memory — mark for XL job and leave lock so normal jobs skip it
            _needs_xl.touch()
            logger.info(f"Stale lock {_lock.parent.name}: OOM/repeated failure → marked .needs_xl")
        else:
            # Normal dead job (cancelled/timed-out before finishing) — release for retry
            _lock.unlink(missing_ok=True)
            logger.info(f"Released stale lock: {_lock.parent.name} (job {_owner} ended, attempt {_try})")
            _released += 1
    except Exception as _e:
        logger.warning(f"Lock cleanup error for {_lock}: {_e}")
if _released:
    logger.info(f"Released {_released} stale locks — those samples are now available for retry")

# ── Discover ready manifests ──────────────────────────────────────────
# A manifest is "ready" if download succeeded and no .quant_done marker exists
manifests = sorted(glob.glob(str(dl_dir / "*" / "*" / "download_manifest.json")))
ready = []
for mpath in manifests:
    mpath = Path(mpath)
    done_marker = mpath.parent / ".quant_done"
    if done_marker.exists():
        continue
    if (mpath.parent / ".needs_xl").exists():
        continue  # reserved for XL memory job
    try:
        meta = json.loads(mpath.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if not meta.get("success"):
        continue
    # P4: In FASTQ_ONLY mode, skip .sra files — they run in dedicated SRA quant jobs
    if fastq_only and meta.get("method") == "s3_sra_pending":
        continue
    ready.append((mpath, meta))

if not ready:
    print("No ready downloads found — exiting cleanly")
    sys.exit(0)

# P3-fix: pre-check for unlocked samples before entering the batch loop.
# Avoids burning 30 min in a job that would only encounter locked samples.
_unlocked_count = sum(
    1 for mpath, _ in ready
    if not (mpath.parent / ".quant_locked").exists()
)
if _unlocked_count == 0:
    print(f"All {len(ready)} ready samples are locked by other jobs — exiting cleanly (P3)")
    sys.exit(0)

print(f"\n▸ Found {len(ready)} ready samples ({_unlocked_count} unlocked), will process up to {quant_batch}\n")

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

    # Atomically lock this sample to prevent parallel processing
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
        sra_path=Path(meta["sra_path"]) if meta.get("sra_path") else None,
    )

    # ── fasterq-dump + pigz (deferred from login node) ───────────────
    # S3 downloads only store the .sra file; FASTQ conversion happens here
    # on the allocated compute node where CPU use is unrestricted.
    if dl_result.method == "s3_sra_pending" and not dl_result.sra_path:
        # No sra_path in manifest — .sra was never staged or is missing.
        # Skip without writing .quant_done so this sample stays available
        # for re-download (avoids permanently blocking it as failed).
        logger.warning(f"[{gsm_id}] s3_sra_pending but no sra_path in manifest — skipping (needs re-download)")
        skip += 1
        lock_path.unlink(missing_ok=True)
        continue

    if dl_result.method == "s3_sra_pending" and dl_result.sra_path:
        sra_file = dl_result.sra_path
        srr_id = sra_file.stem  # e.g. "SRR12345678"

        if not sra_file.exists():
            logger.error(f"[{gsm_id}] .sra file missing on quant node: {sra_file}")
            fail += 1
            result = SampleResult(gsm_id=gsm_id, gse_id=gse_id, organism=meta.get("organism", ""))
            result.status = "failed"
            result.fail_stage = "missing_sra"
            result.error = f".sra file purged from scratch: {sra_file}"
            result.total_time_s = 0.0
            results.append(result)
            done_path.write_text(json.dumps({
                "status": "failed", "error": "missing_sra",
                "quant_job": job_id,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }))
            lock_path.unlink(missing_ok=True)
            continue

        _fq_t0 = time.time()
        logger.info(f"[{gsm_id}] fasterq-dump: {sra_file} → {sample_dir}")
        _cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        _tmp_dir = Path(os.environ.get("SCRATCH", "/tmp")) / "scgeo_fasterq_tmp"
        _tmp_dir.mkdir(parents=True, exist_ok=True)

        _fq_cmd = [
            "fasterq-dump", str(sra_file),
            "--outdir", str(sample_dir),
            "--temp", str(_tmp_dir),
            "--split-3",
            "--threads", str(_cpus),
        ]
        try:
            _fq_result = subprocess.run(_fq_cmd, capture_output=True, text=True, timeout=7200)
        except subprocess.TimeoutExpired:
            _fq_result = None

        # Clean up .sra regardless of fasterq-dump outcome
        try:
            sra_file.unlink(missing_ok=True)
            sra_file.parent.rmdir()
        except OSError:
            pass

        if _fq_result is None or _fq_result.returncode != 0:
            _err = (_fq_result.stderr[:500] if _fq_result else "timed out")
            logger.error(f"[{gsm_id}] fasterq-dump failed: {_err}")
            fail += 1
            result = SampleResult(gsm_id=gsm_id, gse_id=gse_id, organism=meta.get("organism", ""))
            result.status = "failed"
            result.fail_stage = "fasterq_dump"
            result.error = f"fasterq-dump failed: {_err}"
            result.total_time_s = time.time() - _fq_t0
            results.append(result)
            done_path.write_text(json.dumps({
                "status": "failed", "error": "fasterq_dump",
                "quant_job": job_id,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }))
            lock_path.unlink(missing_ok=True)
            continue

        # Locate output FASTQs
        _r1 = sample_dir / f"{srr_id}_1.fastq"
        _r2 = sample_dir / f"{srr_id}_2.fastq"
        _se = sample_dir / f"{srr_id}.fastq"
        _raw_r1 = [_r1] if _r1.exists() else ([_se] if _se.exists() else [])
        _raw_r2 = [_r2] if _r2.exists() else []

        if not _raw_r1:
            logger.error(f"[{gsm_id}] fasterq-dump produced no output files")
            fail += 1
            result = SampleResult(gsm_id=gsm_id, gse_id=gse_id, organism=meta.get("organism", ""))
            result.status = "failed"
            result.fail_stage = "fasterq_dump"
            result.error = "fasterq-dump produced no output files"
            result.total_time_s = time.time() - _fq_t0
            results.append(result)
            done_path.write_text(json.dumps({
                "status": "failed", "error": "fasterq_no_output",
                "quant_job": job_id,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }))
            lock_path.unlink(missing_ok=True)
            continue

        # P5: Rewrite manifest so retried quant jobs see method=fastq (not s3_sra_pending).
        # This prevents missing_sra failures when the .sra has already been consumed.
        try:
            _manifest_path = sample_dir / "download_manifest.json"
            _updated_meta = dict(meta)
            _updated_meta["method"] = "s3_sra"          # signal: FASTQs are ready
            _updated_meta["sra_path"] = ""               # .sra is gone — clear it
            _updated_meta["r1_paths"] = [str(_r1.with_suffix('.fastq.gz')) if _r1.exists() else str(_r1) for _r1 in _raw_r1]
            _updated_meta["r2_paths"] = [str(_r2.with_suffix('.fastq.gz')) if _r2.exists() else str(_r2) for _r2 in _raw_r2]
            _manifest_path.write_text(json.dumps(_updated_meta, indent=2))
        except Exception as _me:
            logger.warning(f"[{gsm_id}] Could not update manifest after fasterq-dump: {_me}")

        # pigz compress in-place (replaces .fastq with .fastq.gz)
        _pigz_r1 = []
        _pigz_r2 = []
        for _raw_path in _raw_r1 + _raw_r2:
            try:
                subprocess.run(
                    ["pigz", "-p", str(_cpus), str(_raw_path)],
                    check=True, timeout=21600,   # P2: 6h — large 669GB FASTQs need ~2h to compress
                )
                _gz = _raw_path.parent / f"{_raw_path.name}.gz"
                _final = _gz if _gz.exists() else _raw_path
            except subprocess.TimeoutExpired:
                logger.warning(f"[{gsm_id}] pigz timed out for {_raw_path}, keeping uncompressed")
                _final = _raw_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning(f"[{gsm_id}] pigz failed for {_raw_path}, keeping uncompressed")
                _final = _raw_path
            if _raw_path in _raw_r1:
                _pigz_r1.append(_final)
            else:
                _pigz_r2.append(_final)

        dl_result.r1_paths = _pigz_r1
        dl_result.r2_paths = _pigz_r2
        dl_result.method = "s3_sra"
        logger.info(
            f"[{gsm_id}] fasterq-dump+pigz done in {time.time()-_fq_t0:.0f}s: "
            f"{len(_pigz_r1)} R1, {len(_pigz_r2)} R2"
        )

    # Verify FASTQ files still exist (scratch auto-purge protection)
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

    # Write done marker (prevents re-processing)
    # P2-fix: include success boolean + gsm so external scripts can validate
    done_path.write_text(json.dumps({
        "success": result.status in ("success", "qc_warn"),
        "status": result.status,
        "gsm": gsm_id,
        "quant_job": job_id,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    lock_path.unlink(missing_ok=True)  # P2-fix: release lock after done marker written

    # Clean up FASTQs + intermediates to free scratch space
    for pattern in ("**/*.fastq.gz", "**/*.fq.gz", "**/*.fastq", "**/*.fq",
                    "**/*.sra", "**/*.sra.cache", "**/.dl_segments_*"):
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

# ── Write results CSV ─────────────────────────────────────────────────
if results:
    import pandas as pd
    rows = []
    for r in results:
        d = r.to_dict() if hasattr(r, "to_dict") else {"gsm_id": r.gsm_id, "status": r.status, "error": r.error}
        rows.append(d)
    results_df = pd.DataFrame(rows)
    claim_id = f"quant_{job_id}"
    result_file = result_dir / f"results_{claim_id}.csv"
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

echo "Quant phase complete."
