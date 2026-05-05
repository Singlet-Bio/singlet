#!/usr/bin/env python3
"""Batch convert .spz files to .1pz format.

Scans a directory tree for counts.spz files, converts each to counts.1pz
using sparsepress (legacy reader) + singlepress (1pz writer).

Handles two pipeline generations:
  v3/v4 (early 2026): Matrix stored as barcodes × genes (transposed, unfiltered).
      Detected by: header m (rows) is NOT a known gene count.
      Action: transpose to genes × cells before writing .1pz.
  v5+  (Feb 17 2026+): Matrix stored as genes × cells (standard orientation).
      Detected by: header m IS a known gene count (115818, 171540, etc.).
      Action: write directly to .1pz.

All on-disk .spz files use legacy sparsepress_v2 format (128-byte header,
magic SPRZ, SPEN footer). The current singlepress.read() cannot read them
(CRC32 mismatch due to different footer layout). We use sparsepress.sp_read().

Usage:
    python convert_spz_to_1pz.py /path/to/data --delete-originals
    python convert_spz_to_1pz.py /path/to/data --dry-run
    python convert_spz_to_1pz.py /path/to/data --jobs 4

Requires:
    - sparsepress package (legacy reader): add /mnt/home/debruinz/SingletAI to PYTHONPATH
    - singlepress package (1pz writer): installed in cellarium env
"""

import argparse
import logging
import os
import struct
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Known standard gene counts (reference genome sizes)
STANDARD_GENE_COUNTS = {115818, 171540}  # human USA, mouse USA

# v3/v4 pipeline versions — these store barcodes × genes (transposed)
TRANSPOSED_PIPELINES = {"2026-02-12_v3_pipelined_kraken2", "2026-02-14_v4_per_gsm_cleanup"}


def _detect_transposed(spz_path: str) -> bool:
    """Detect if an SPZ file has transposed orientation (v3/v4 legacy).

    v3/v4 files store barcodes × genes:  m = n_cells_raw, n = gene_count
    v5+   files store genes × cells:     m = gene_count,   n = n_cells

    Detection strategy:
    1. Check manifest pipeline_version if available (authoritative).
    2. Fall back to heuristic: if n ∈ {115818, 171540} → transposed.
       (All 664 v3/v4 files have n equal to a standard gene count;
        v5+ files store genes as rows, so n is always a cell count.)
    """
    import json

    # Try manifest first (authoritative)
    manifest = Path(spz_path).parent / "sample_manifest.json"
    if manifest.exists():
        try:
            with open(manifest) as f:
                d = json.load(f)
            pv = d.get("pipeline_version", "")
            if pv in TRANSPOSED_PIPELINES:
                return True
            if pv:  # have a known version and it's not v3/v4
                return False
        except Exception:
            pass

    # Heuristic fallback: check if n is a standard gene count
    with open(spz_path, "rb") as f:
        f.seek(8)  # skip magic(4) + version(2) + header_size(2)
        m = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<I", f.read(4))[0]

    return n in STANDARD_GENE_COUNTS


def convert_one(spz_path: str, delete_original: bool = False) -> dict:
    """Convert a single .spz file to .1pz.

    Reads with sparsepress (legacy format), optionally transposes v3/v4
    files to standard genes×cells orientation, then writes .1pz with
    singlepress.

    Returns dict with conversion stats or error info.
    """
    import time
    import numpy as np
    import scipy.sparse as sp

    # Ensure sparsepress is importable
    if "/mnt/home/debruinz/SingletAI" not in sys.path:
        sys.path.insert(0, "/mnt/home/debruinz/SingletAI")
    # Ensure the real singlepress package (not namespace dir) is found
    _sp_pkg = "/mnt/home/debruinz/Singlet-AI/singlepress"
    if _sp_pkg not in sys.path:
        sys.path.insert(0, _sp_pkg)
    import sparsepress
    import singlepress

    spz_path = Path(spz_path)
    pz_path = spz_path.with_name("counts.1pz")
    result = {
        "spz_path": str(spz_path),
        "pz_path": str(pz_path),
        "success": False,
        "error": None,
        "transposed": False,
    }

    try:
        t0 = time.time()

        # Detect orientation before reading
        is_transposed = _detect_transposed(str(spz_path))
        result["transposed"] = is_transposed

        # Read with sparsepress (legacy v2 format), fallback to singlepress
        try:
            raw = sparsepress.sp_read(str(spz_path))
            shape = tuple(raw["shape"])
            data = np.asarray(raw["data"])
            indices = np.asarray(raw["indices"])
            indptr = np.asarray(raw["indptr"])

            # Fix non-monotonic indptr (legacy reader bug at chunk boundaries)
            indptr = np.maximum.accumulate(indptr)

            mat = sp.csc_matrix((data, indices, indptr), shape=shape)
        except Exception:
            # Fallback: try singlepress reader (newer .spz format)
            mat = singlepress.read(str(spz_path))
        read_ms = (time.time() - t0) * 1000

        # Transpose v3/v4 files from barcodes×genes → genes×cells
        if is_transposed:
            mat = mat.T.tocsc()

        # Write .1pz (genes × cells orientation)
        t1 = time.time()
        stats = singlepress.write_1pz(str(pz_path), mat)
        write_ms = (time.time() - t1) * 1000

        # Verify roundtrip: read back .1pz and check dimensions + nnz
        t2 = time.time()
        mat2 = singlepress.read_1pz(str(pz_path))
        verify_ms = (time.time() - t2) * 1000

        if mat.shape != mat2.shape:
            raise ValueError(
                f"Shape mismatch: {mat.shape} vs {mat2.shape}"
            )
        if mat.nnz != mat2.nnz:
            raise ValueError(
                f"NNZ mismatch: {mat.nnz} vs {mat2.nnz}"
            )

        result["success"] = True
        result["spz_bytes"] = spz_path.stat().st_size
        result["pz_bytes"] = pz_path.stat().st_size
        result["ratio"] = stats.get("ratio", 0)
        result["m"] = mat.shape[0]
        result["n"] = mat.shape[1]
        result["nnz"] = mat.nnz
        result["read_ms"] = read_ms
        result["write_ms"] = write_ms
        result["verify_ms"] = verify_ms

        if delete_original:
            spz_path.unlink()
            result["deleted"] = True

    except Exception as e:
        result["error"] = str(e)
        # Clean up failed .1pz if it exists
        if pz_path.exists():
            pz_path.unlink()

    return result


def find_spz_files(root: Path) -> list[Path]:
    """Find all counts.spz files under root."""
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn == "counts.spz":
                results.append(Path(dirpath) / fn)
    return sorted(results)


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert .spz files to .1pz format"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root directory to scan for counts.spz files",
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete .spz files after successful conversion + verification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files that would be converted",
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=1,
        help="Number of parallel conversion jobs (default: 1)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files where counts.1pz already exists",
    )
    parser.add_argument(
        "--file-list",
        help="Read file paths from this text file (one per line) instead of scanning",
    )
    parser.add_argument(
        "--chunk", type=int, default=None,
        help="0-based chunk index (for array jobs). Requires --total-chunks.",
    )
    parser.add_argument(
        "--total-chunks", type=int, default=None,
        help="Total number of chunks to split work across.",
    )
    args = parser.parse_args()

    # Load file list
    if args.file_list:
        with open(args.file_list) as f:
            spz_files = [Path(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded {len(spz_files)} files from {args.file_list}")
    else:
        root = Path(args.root)
        if not root.is_dir():
            logger.error(f"Not a directory: {root}")
            sys.exit(1)
        logger.info(f"Scanning {root} for counts.spz files...")
        spz_files = find_spz_files(root)
        logger.info(f"Found {len(spz_files)} .spz files")

    if args.skip_existing:
        before = len(spz_files)
        spz_files = [
            p for p in spz_files
            if not p.with_name("counts.1pz").exists()
        ]
        skipped = before - len(spz_files)
        if skipped:
            logger.info(f"Skipping {skipped} files (counts.1pz already exists)")

    # Chunk selection for array jobs
    if args.chunk is not None and args.total_chunks is not None:
        chunk_size = (len(spz_files) + args.total_chunks - 1) // args.total_chunks
        start = args.chunk * chunk_size
        end = min(start + chunk_size, len(spz_files))
        spz_files = spz_files[start:end]
        logger.info(f"Chunk {args.chunk}/{args.total_chunks}: files {start}-{end-1} ({len(spz_files)} files)")

    if args.dry_run:
        for p in spz_files:
            size_mb = p.stat().st_size / 1024**2
            print(f"  {p}  ({size_mb:.1f} MB)")
        logger.info(f"Dry run: {len(spz_files)} files would be converted")
        return

    if not spz_files:
        logger.info("Nothing to convert")
        return

    # Convert
    success = 0
    failed = 0
    n_transposed = 0
    total_spz_bytes = 0
    total_pz_bytes = 0

    if args.jobs <= 1:
        for i, spz_path in enumerate(spz_files):
            logger.info(f"[{i+1}/{len(spz_files)}] {spz_path}")
            r = convert_one(str(spz_path), args.delete_originals)
            if r["success"]:
                success += 1
                total_spz_bytes += r["spz_bytes"]
                total_pz_bytes += r["pz_bytes"]
                if r.get("transposed"):
                    n_transposed += 1
                logger.info(
                    f"  OK: {r['m']}×{r['n']} nnz={r['nnz']:,} "
                    f"ratio={r['ratio']:.1f}x "
                    f"read={r['read_ms']:.0f}ms write={r['write_ms']:.0f}ms"
                    f"{' [TRANSPOSED]' if r.get('transposed') else ''}"
                )
            else:
                failed += 1
                logger.error(f"  FAILED: {r['error']}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(convert_one, str(p), args.delete_originals): p
                for p in spz_files
            }
            for i, future in enumerate(as_completed(futures)):
                p = futures[future]
                r = future.result()
                if r["success"]:
                    success += 1
                    total_spz_bytes += r["spz_bytes"]
                    total_pz_bytes += r["pz_bytes"]
                    if r.get("transposed"):
                        n_transposed += 1
                    logger.info(
                        f"[{i+1}/{len(spz_files)}] OK: {p.name} "
                        f"ratio={r['ratio']:.1f}x"
                        f"{' [T]' if r.get('transposed') else ''}"
                    )
                else:
                    failed += 1
                    logger.error(f"[{i+1}/{len(spz_files)}] FAILED: {p} — {r['error']}")

    logger.info(
        f"\nDone: {success} converted, {failed} failed, "
        f"{n_transposed} transposed (v3/v4)\n"
        f"  Total .spz: {total_spz_bytes/1e9:.2f} GB\n"
        f"  Total .1pz: {total_pz_bytes/1e9:.2f} GB\n"
        f"  Overall reduction: {(1 - total_pz_bytes/max(total_spz_bytes,1))*100:.1f}%"
    )


if __name__ == "__main__":
    main()
