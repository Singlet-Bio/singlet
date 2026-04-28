#!/usr/bin/env python3
"""Convert .1pz v3 files to v4 (bit-plane + bitmap pre-filter).

Reads each file, rewrites as v4 with specified mode(s).
Preserves all metadata (obs, var, uns, rownames, colnames, transpose).

Usage:
  # Convert counts.1pz to zstd-1 (in-place) + lz4 copy, kraken2.1pz to zstd-1 only:
  python convert_v3_to_v4.py --dir /path/to/quant --counts --kraken2

  # Dry run:
  python convert_v3_to_v4.py --dir /path/to/quant --counts --kraken2 --dry-run
"""
import singlepress
import os
import sys
import glob
import tempfile
import time


def _read_metadata(mat, info):
    """Extract metadata kwargs from a matrix returned by read_1pz."""
    kwargs = {}
    if hasattr(mat, "rownames") and mat.rownames:
        kwargs["rownames"] = mat.rownames
    if hasattr(mat, "colnames") and mat.colnames:
        kwargs["colnames"] = mat.colnames
    if hasattr(mat, "obs") and mat.obs is not None:
        kwargs["obs"] = mat.obs
    if hasattr(mat, "var") and mat.var is not None:
        kwargs["var"] = mat.var
    if hasattr(mat, "uns") and mat.uns:
        kwargs["uns"] = mat.uns
    kwargs["store_transpose"] = info.get("has_transpose", False)
    return kwargs


def _write_verified(path, mat, info, mode="default", level=1):
    """Write to temp file, verify, then atomically replace."""
    dirn = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(suffix=".1pz.tmp", dir=dirn)
    os.close(fd)
    try:
        kwargs = _read_metadata(mat, info)
        singlepress.write_1pz(tmp_path, mat, mode=mode, level=level, **kwargs)
        new_info = singlepress.info_1pz(tmp_path)
        assert new_info["version"] == 4, f"Expected v4, got v{new_info['version']}"
        assert new_info["m"] == info["m"], f"m mismatch"
        assert new_info["n"] == info["n"], f"n mismatch"
        assert new_info["nnz"] == info["nnz"], f"nnz mismatch"
        os.replace(tmp_path, path)
        return os.path.getsize(path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def convert_counts(path, dry_run=False):
    """Convert counts.1pz → v4 zstd-1 (in-place) + counts_lz4.1pz (lz4)."""
    if not os.path.exists(path):
        return "skip (missing)"
    info = singlepress.info_1pz(path)
    if info["nnz"] > 2_000_000_000:
        return f"skip (nnz={info['nnz']} > 2B, too large)"
    old_sz = os.path.getsize(path)
    lz4_path = os.path.join(os.path.dirname(path), "counts_lz4.1pz")
    lz4_exists = os.path.exists(lz4_path)

    need_zstd = info["version"] != 4
    need_lz4 = not lz4_exists

    if not need_zstd and not need_lz4:
        return "skip (both exist)"

    if dry_run:
        parts = []
        if need_zstd:
            parts.append(f"zstd-1 ({old_sz/1e6:.1f}MB)")
        if need_lz4:
            parts.append("lz4")
        return f"would_convert: {' + '.join(parts)}, {info['m']}x{info['n']}, nnz={info['nnz']}"

    # Read once, write twice
    mat = singlepress.read_1pz(path, num_threads=4)

    results = []
    if need_zstd:
        new_sz = _write_verified(path, mat, info, mode="default", level=1)
        results.append(f"zstd-1:{old_sz/1e6:.1f}→{new_sz/1e6:.1f}MB")

    if need_lz4:
        # For lz4, write directly (new file, not replacing)
        dirn = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(suffix=".1pz.tmp", dir=dirn)
        os.close(fd)
        try:
            kwargs = _read_metadata(mat, info)
            singlepress.write_1pz(tmp_path, mat, mode="fast", **kwargs)
            new_info = singlepress.info_1pz(tmp_path)
            assert new_info["version"] == 4
            assert new_info["nnz"] == info["nnz"]
            os.replace(tmp_path, lz4_path)
            lz4_sz = os.path.getsize(lz4_path)
            results.append(f"lz4:{lz4_sz/1e6:.1f}MB")
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            results.append(f"lz4:error({e})")

    return "converted " + ", ".join(results)


def convert_kraken2(path, dry_run=False):
    """Convert kraken2.1pz → v4 zstd-1 (in-place)."""
    if not os.path.exists(path):
        return "skip (missing)"
    info = singlepress.info_1pz(path)
    if info["version"] == 4:
        return "skip_v4"

    old_sz = os.path.getsize(path)
    if dry_run:
        return f"would_convert ({old_sz/1e6:.1f}MB, {info['m']}x{info['n']}, nnz={info['nnz']})"

    mat = singlepress.read_1pz(path, num_threads=4)
    new_sz = _write_verified(path, mat, info, mode="default", level=1)
    return f"converted ({old_sz/1e6:.1f}→{new_sz/1e6:.1f}MB, {100*new_sz/old_sz:.1f}%)"


def run_batch(label, files, convert_fn, dry_run):
    """Process a batch of files with progress reporting."""
    print(f"\n{'='*60}")
    print(f"  {label}: {len(files)} files")
    print(f"{'='*60}")

    results = {"converted": 0, "skip": 0, "error": 0, "would": 0}
    t0 = time.time()

    for i, path in enumerate(files):
        try:
            status = convert_fn(path, dry_run=dry_run)
        except Exception as e:
            status = f"error: {e}"

        if status.startswith("converted"):
            results["converted"] += 1
        elif status.startswith("skip"):
            results["skip"] += 1
        elif status.startswith("would"):
            results["would"] += 1
        else:
            results["error"] += 1

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        # Show dirname as sample ID
        sample = os.path.basename(os.path.dirname(path))
        print(f"  [{i+1}/{len(files)}] {elapsed:.0f}s ({rate:.1f}/s) {sample}: {status}")

    elapsed = time.time() - t0
    print(f"\n  {label} done in {elapsed:.0f}s: {results}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert .1pz v3 → v4 with multi-mode support")
    parser.add_argument("--dir", help="Root quant directory to scan (uses glob)")
    parser.add_argument("--counts", action="store_true", help="Convert counts.1pz (zstd-1 + lz4)")
    parser.add_argument("--kraken2", action="store_true", help="Convert kraken2.1pz (zstd-1)")
    parser.add_argument("--file-list", help="File containing paths to convert (one per line), skips glob")
    parser.add_argument("--file-type", choices=["counts", "kraken2"], help="Type when using --file-list")
    parser.add_argument("--dry-run", action="store_true", help="Just report, don't convert")
    parser.add_argument("--reverse", action="store_true", help="Process files in reverse order (for parallel runs)")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    if not args.file_list and not args.counts and not args.kraken2:
        parser.error("Specify --file-list or at least one of --counts/--kraken2")
    if args.file_list and not args.file_type:
        parser.error("--file-type required when using --file-list")

    t_start = time.time()

    # --file-list mode: read paths from file, no glob needed
    if args.file_list:
        with open(args.file_list) as f:
            files = [line.strip() for line in f if line.strip()]
        if args.reverse:
            files = files[::-1]
        if args.num_shards > 1:
            files = [p for i, p in enumerate(files) if i % args.num_shards == args.shard]
        fn = convert_counts if args.file_type == "counts" else convert_kraken2
        shard_label = f" (shard {args.shard}/{args.num_shards})" if args.num_shards > 1 else ""
        run_batch(f"{args.file_type}{shard_label}", files, fn, args.dry_run)
        print(f"\nTotal wall time: {time.time()-t_start:.0f}s")
        return

    t_start = time.time()

    def shard_files(files):
        if args.reverse:
            files = files[::-1]
        if args.num_shards > 1:
            files = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
        return files

    shard_label = f" (shard {args.shard}/{args.num_shards})" if args.num_shards > 1 else ""

    if args.counts:
        files = shard_files(sorted(glob.glob(os.path.join(args.dir, "*/counts.1pz"))))
        run_batch(f"counts.1pz → zstd-1 + lz4{shard_label}", files, convert_counts, args.dry_run)

    if args.kraken2:
        files = shard_files(sorted(glob.glob(os.path.join(args.dir, "*/kraken2.1pz"))))
        run_batch(f"kraken2.1pz → zstd-1{shard_label}", files, convert_kraken2, args.dry_run)

    print(f"\nTotal wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
