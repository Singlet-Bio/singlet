#!/usr/bin/env python3
"""
Comprehensive dataset survey for the SinglePress manuscript.

Scans ALL production .1pz files, re-encodes each with the CURRENT codec
to measure accurate compression ratios, and benchmarks I/O speed on a
stratified subsample.

Key design: production .1pz files may have been written with an older codec.
This script re-encodes each dataset to /dev/shm (tmpfs) to measure the file
size produced by the CURRENT encoder, then immediately deletes the temp file.
This ensures all compression statistics reflect the codec as described in the
manuscript, without modifying production data.

Outputs (to code/data/):
    all_datasets_survey.csv    — Shape, nnz, species, protocol, compression
                                 ratio with current codec
    io_benchmarks.csv          — Stratified subsample: .1pz / H5AD / npz / H5
                                 file sizes and read timings
    read_throughput.csv        — .1pz decode throughput per dataset
    value_distributions.csv    — Histogram of nonzero count values per dataset

Usage:
    srun --time=360 --mem=128G --cpus-per-task=8 bash -c \\
        'source /mnt/home/debruinz/venv/bin/activate && cd /tmp && \\
         python3 -u <this_script>'
"""

import csv
import os
import sys
import time
import resource
import gc
import tempfile

# ── Path fixes ──────────────────────────────────────────────────
_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if "" in sys.path:
    sys.path.remove("")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import numpy as np
import singlepress as sp

# ── Constants ───────────────────────────────────────────────────
QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TMPFS = "/dev/shm"

# Maximum nnz for in-memory re-encoding (skip very large files).
# 2 billion nnz ~ 24 GB as CSC int32 + overhead.
MAX_NNZ_REENCODE = 2_000_000_000

# Gene rows -> species mapping (USA-resolved: nrows = 3 * n_genes).
SPECIES_MAP = {
    115818: ("Homo sapiens", 38606),
    171540: ("Mus musculus", 57180),
    97560: ("Danio rerio", 32520),
    91686: ("Rattus norvegicus", 30562),
    72834: ("Drosophila melanogaster", 24278),
    106296: ("Macaca mulatta", 35432),
    90324: ("Gallus gallus", 30108),
    73842: ("Xenopus laevis", 24614),
    78924: ("Macaca fascicularis", 26308),
    69072: ("Sus scrofa", 23024),
    139200: ("Ovis aries", 46400),
    60930: ("Canis lupus familiaris", 20310),
    89097: ("Pan troglodytes", 29699),
}


def load_catalog():
    """Load catalog and build GSE->(species, protocol) mapping."""
    try:
        import pandas as pd
        cat = pd.read_parquet(
            "/mnt/projects/debruinz_project/cellarium/catalog/"
            "geo_single_cell_catalog.parquet",
            columns=["gse_id", "organism", "protocol_inferred"],
        )
        gse_map = {}
        for _, row in cat.drop_duplicates("gse_id").iterrows():
            gse_map[row["gse_id"]] = {
                "catalog_species": str(row["organism"]),
                "catalog_protocol": str(row["protocol_inferred"]),
            }
        return gse_map
    except Exception as e:
        print(f"WARNING: Could not load catalog: {e}")
        return {}


def identify_species(nrows):
    """Determine species from the nrows dimension using USA-resolution mapping."""
    sp_info = SPECIES_MAP.get(nrows)
    if sp_info:
        return sp_info[0], sp_info[1], True
    # Try matching n/3 (non-USA)
    for nr, (sname, ng) in SPECIES_MAP.items():
        if nrows == ng:
            return sname, ng, False
    return "Unknown", nrows if nrows < 100000 else nrows // 3, (nrows % 3 == 0)


def reencode_measure(pz_path, nnz):
    """Re-encode a .1pz file with the current codec and return (new_bytes, read_s).

    Returns (file_bytes, read_seconds) or (None, None) if skipped/failed.
    """
    if nnz > MAX_NNZ_REENCODE:
        return None, None

    tmp_path = os.path.join(TMPFS, f"survey_reencode_{os.getpid()}.1pz")
    try:
        # Read with current decoder
        t0 = time.perf_counter()
        mat = sp.read_1pz(pz_path)
        read_s = time.perf_counter() - t0

        # Get metadata from file
        pz = sp.open_1pz(pz_path)
        rownames = list(pz.rownames) if pz.rownames else []
        colnames = list(pz.colnames) if pz.colnames else []
        del pz

        # Re-encode with current codec
        sp.write_1pz(tmp_path, mat.tocsc(),
                      rownames=rownames, colnames=colnames, num_threads=8)
        new_bytes = os.path.getsize(tmp_path)

        del mat
        gc.collect()
        return new_bytes, read_s
    except Exception:
        return None, None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ===================================================================
# PHASE 1: Full survey with re-encoding
# ===================================================================

def survey_all():
    """Survey all merged counts.1pz files, re-encoding each to measure
    the file size produced by the current codec."""
    catalog = load_catalog()
    print(f"Catalog loaded: {len(catalog)} GSEs")

    gse_dirs = sorted(os.listdir(QUANT_DIR))
    print(f"Total GSE directories: {len(gse_dirs)}")

    results = []
    errors = 0
    skipped_large = 0
    t0 = time.time()

    for i, gse_id in enumerate(gse_dirs):
        pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
        if not os.path.isfile(pz_path):
            continue

        try:
            info = sp.info_1pz(pz_path)
            old_file_size = os.path.getsize(pz_path)

            nrows = info["m"]
            ncols = info["n"]
            nnz = info["nnz"]
            species, n_genes, is_usa = identify_species(nrows)

            # Re-encode to measure current-codec file size
            new_bytes, read_s = reencode_measure(pz_path, nnz)

            # Use re-encoded size if available, else fall back to existing
            pz_bytes = new_bytes if new_bytes is not None else old_file_size
            was_reencoded = new_bytes is not None
            if not was_reencoded:
                skipped_large += 1

            # int32 CSC baseline: indptr + indices + data
            density = nnz / (nrows * ncols) if nrows * ncols > 0 else 0
            raw_int32_bytes = (ncols + 1) * 4 + nnz * 4 + nnz * 4
            ratio = raw_int32_bytes / pz_bytes if pz_bytes > 0 else 0

            cat_info = catalog.get(gse_id, {})

            row = {
                "gse_id": gse_id,
                "species": species,
                "catalog_species": cat_info.get("catalog_species", ""),
                "protocol": cat_info.get("catalog_protocol", ""),
                "nrows": nrows,
                "ncols": ncols,
                "n_genes": n_genes,
                "nnz": nnz,
                "density": round(density, 6),
                "is_usa": is_usa,
                "pz_bytes": pz_bytes,
                "old_pz_bytes": old_file_size,
                "raw_int32_bytes": raw_int32_bytes,
                "ratio": round(ratio, 3),
                "bits_per_nz": round(pz_bytes * 8 / nnz, 3) if nnz > 0 else 0,
                "reencoded": was_reencoded,
                "read_s": round(read_s, 4) if read_s is not None else None,
                "codec": info.get("codec", ""),
                "has_transpose": info.get("has_transpose", False),
                "has_obs_var": info.get("has_obs_var", False),
                "num_chunks": info.get("num_chunks", 0),
            }
            results.append(row)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(gse_dirs)}] {rate:.1f} GSEs/s, "
                      f"{len(results)} with .1pz, {errors} errors, "
                      f"{skipped_large} skipped (too large)")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {gse_id}: {e}")

    elapsed = time.time() - t0
    print(f"\nSurvey: {len(results)} datasets in {elapsed:.0f}s "
          f"({errors} errors, {skipped_large} skipped)")
    return results


# ===================================================================
# PHASE 2: I/O benchmarks on stratified subsample
# ===================================================================

def reencode_to_tmpfs(pz_path, gse_id):
    """Re-encode a .1pz file to tmpfs for benchmarking.

    Returns (tmpfs_path, mat_csc, rownames, colnames).
    """
    mat = sp.read_1pz(pz_path)
    pz = sp.open_1pz(pz_path)
    rownames = list(pz.rownames) if pz.rownames else [f"g{i}" for i in range(mat.shape[0])]
    colnames = list(pz.colnames) if pz.colnames else [f"c{i}" for i in range(mat.shape[1])]
    del pz

    csc = mat.tocsc()
    csc.data = csc.data.astype(np.int32)

    tmp_path = os.path.join(TMPFS, f"bench_{gse_id}.1pz")
    sp.write_1pz(tmp_path, csc, rownames=rownames, colnames=colnames)
    return tmp_path, csc, rownames, colnames


def benchmark_io_subsample(results, n_sample=150, n_trials=3):
    """Benchmark read speed on a stratified subsample across formats.

    .1pz reads use RE-ENCODED files (current codec).
    """
    import anndata as ad
    import scipy.sparse as ss

    sorted_results = sorted(results, key=lambda x: x["nnz"])
    step = max(1, len(sorted_results) // n_sample)
    sample = [sorted_results[i] for i in range(0, len(sorted_results), step)][:n_sample]
    print(f"\nI/O benchmark on {len(sample)} datasets (stratified by nnz)")

    io_results = []
    read_tp_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, rec in enumerate(sample):
            gse_id = rec["gse_id"]
            nnz = rec["nnz"]
            raw_bytes = rec["raw_int32_bytes"]
            pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")

            if idx % 20 == 0:
                print(f"  [{idx}/{len(sample)}] {gse_id} nnz={nnz:,}")

            # Skip very large datasets
            if nnz > 200_000_000:
                continue

            try:
                # Re-encode to tmpfs
                tmp_pz, csc, rownames, colnames = reencode_to_tmpfs(pz_path, gse_id)

                pz_bytes = os.path.getsize(tmp_pz)

                # Time .1pz reads from re-encoded file
                times_1pz = []
                for trial in range(n_trials + 1):
                    gc.collect()
                    t0 = time.perf_counter()
                    _ = sp.read_1pz(tmp_pz)
                    t1 = time.perf_counter()
                    if trial > 0:  # skip warmup
                        times_1pz.append(t1 - t0)
                t_1pz = float(np.median(times_1pz))

                io_rec = {
                    "gse_id": gse_id,
                    "nnz": nnz,
                    "raw_int32_bytes": raw_bytes,
                    "pz_bytes": pz_bytes,
                    "read_1pz_s": round(t_1pz, 4),
                    "read_1pz_mbps": round(raw_bytes / t_1pz / 1e6, 1),
                }

                # Decode throughput record
                read_tp_results.append({
                    "gse_id": gse_id,
                    "species": rec["species"],
                    "protocol": rec["protocol"],
                    "nnz": nnz,
                    "read_s": round(t_1pz, 4),
                    "read_gbps": round(raw_bytes / t_1pz / 1e9, 3),
                    "read_mbps": round(raw_bytes / t_1pz / 1e6, 1),
                })

                # Clean up re-encoded .1pz
                os.remove(tmp_pz)

                # Write and bench H5AD
                h5ad_path = os.path.join(tmpdir, f"{gse_id}.h5ad")
                try:
                    import pandas as pd
                    adata = ad.AnnData(
                        X=csc.T.tocsr(),
                        obs=pd.DataFrame(index=colnames),
                        var=pd.DataFrame(index=rownames),
                    )
                    adata.write_h5ad(h5ad_path, compression="gzip")
                    h5ad_size = os.path.getsize(h5ad_path)
                    io_rec["h5ad_bytes"] = h5ad_size
                    io_rec["h5ad_ratio"] = round(raw_bytes / h5ad_size, 3)

                    times_h5ad = []
                    for trial in range(n_trials + 1):
                        gc.collect()
                        t0 = time.perf_counter()
                        ad.read_h5ad(h5ad_path)
                        t1 = time.perf_counter()
                        if trial > 0:
                            times_h5ad.append(t1 - t0)
                    t_h5ad = float(np.median(times_h5ad))
                    io_rec["read_h5ad_s"] = round(t_h5ad, 4)
                    io_rec["read_h5ad_mbps"] = round(raw_bytes / t_h5ad / 1e6, 1)
                    os.remove(h5ad_path)
                except Exception as e:
                    io_rec["h5ad_error"] = str(e)[:60]

                # Write and bench scipy npz
                npz_path = os.path.join(tmpdir, f"{gse_id}.npz")
                try:
                    ss.save_npz(npz_path, csc)
                    npz_size = os.path.getsize(npz_path)
                    io_rec["npz_bytes"] = npz_size
                    io_rec["npz_ratio"] = round(raw_bytes / npz_size, 3)

                    times_npz = []
                    for trial in range(n_trials + 1):
                        gc.collect()
                        t0 = time.perf_counter()
                        ss.load_npz(npz_path)
                        t1 = time.perf_counter()
                        if trial > 0:
                            times_npz.append(t1 - t0)
                    t_npz = float(np.median(times_npz))
                    io_rec["read_npz_s"] = round(t_npz, 4)
                    io_rec["read_npz_mbps"] = round(raw_bytes / t_npz / 1e6, 1)
                    os.remove(npz_path)
                except Exception as e:
                    io_rec["npz_error"] = str(e)[:60]

                # Write and bench 10x HDF5
                h5_path = os.path.join(tmpdir, f"{gse_id}.h5")
                try:
                    import h5py
                    with h5py.File(h5_path, "w") as f:
                        grp = f.create_group("matrix")
                        grp.create_dataset("data", data=csc.data,
                                           compression="gzip")
                        grp.create_dataset("indices", data=csc.indices,
                                           compression="gzip")
                        grp.create_dataset("indptr", data=csc.indptr,
                                           compression="gzip")
                        grp.attrs["shape"] = np.array(csc.shape, dtype=np.int32)
                    h5_size = os.path.getsize(h5_path)
                    io_rec["h5_bytes"] = h5_size
                    io_rec["h5_ratio"] = round(raw_bytes / h5_size, 3)

                    def read_10x_h5():
                        with h5py.File(h5_path, "r") as f:
                            g = f["matrix"]
                            return ss.csc_matrix(
                                (g["data"][:], g["indices"][:], g["indptr"][:]),
                                shape=g.attrs["shape"],
                            )

                    times_h5 = []
                    for trial in range(n_trials + 1):
                        gc.collect()
                        t0 = time.perf_counter()
                        read_10x_h5()
                        t1 = time.perf_counter()
                        if trial > 0:
                            times_h5.append(t1 - t0)
                    t_h5 = float(np.median(times_h5))
                    io_rec["read_h5_s"] = round(t_h5, 4)
                    io_rec["read_h5_mbps"] = round(raw_bytes / t_h5 / 1e6, 1)
                    os.remove(h5_path)
                except Exception as e:
                    io_rec["h5_error"] = str(e)[:60]

                io_results.append(io_rec)
                del csc

            except Exception as e:
                if idx < 5:
                    print(f"  ERROR {gse_id}: {e}")

            gc.collect()

    print(f"I/O benchmarks complete: {len(io_results)} datasets")
    return io_results, read_tp_results


# ===================================================================
# PHASE 3: Value distributions (data-intrinsic, not codec-dependent)
# ===================================================================

def extract_value_dists(results, n_sample=200):
    """Extract nonzero value histograms from a stratified sample."""
    sorted_results = sorted(results, key=lambda x: x["nnz"])
    step = max(1, len(sorted_results) // n_sample)
    sample = [sorted_results[i] for i in range(0, len(sorted_results), step)][:n_sample]
    print(f"\nValue distributions from {len(sample)} datasets")

    rows = []
    for idx, rec in enumerate(sample):
        gse_id = rec["gse_id"]
        pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
        nnz = rec["nnz"]

        if nnz > 500_000_000:
            continue

        if idx % 50 == 0:
            print(f"  [{idx}/{len(sample)}] {gse_id}")

        try:
            mat = sp.read_1pz(pz_path)
            data = np.asarray(mat.data).ravel()

            # Histogram: count occurrences of each nonzero value 1..100
            for val in range(1, 101):
                cnt = int(np.sum(data == val))
                if cnt > 0:
                    rows.append({
                        "gse_id": gse_id,
                        "species": rec["species"],
                        "nnz": nnz,
                        "value": val,
                        "count": cnt,
                    })
            del mat, data
            gc.collect()
        except Exception:
            pass

    print(f"Value distributions: {len(rows)} rows from {n_sample} datasets")
    return rows


# ===================================================================
# Main
# ===================================================================

def write_csv(rows, path, fieldnames=None):
    """Write list of dicts to CSV."""
    if not rows:
        print(f"  No data for {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = fieldnames or rows[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {path} ({len(rows)} rows)")


def main():
    print("=" * 70)
    print("SinglePress Comprehensive Dataset Survey")
    print(f"  singlepress version: {getattr(sp, '__version__', 'dev')}")
    print(f"  Codec: VOCSC + bit-plane + zstd-3")
    print(f"  Max nnz for re-encode: {MAX_NNZ_REENCODE:,}")
    print("=" * 70)

    # Phase 1: Full survey with re-encoding
    results = survey_all()
    write_csv(results, os.path.join(CODE_DATA_DIR, "all_datasets_survey.csv"))

    # Summary
    if results:
        reencoded = [r for r in results if r["reencoded"]]
        ratios = [r["ratio"] for r in results if r["ratio"] > 0 and r["reencoded"]]
        species_counts = {}
        total_cells = 0
        total_nnz = 0
        for r in results:
            species_counts[r["species"]] = species_counts.get(r["species"], 0) + 1
            total_cells += r["ncols"]
            total_nnz += r["nnz"]

        print(f"\n{'='*50}")
        print(f"SURVEY SUMMARY")
        print(f"  Datasets: {len(results)} ({len(reencoded)} re-encoded)")
        print(f"  Total cells: {total_cells:,}")
        print(f"  Total NNZ: {total_nnz:,}")
        if ratios:
            print(f"\n  Compression ratio (current codec, vs int32 CSC):")
            print(f"    min:    {min(ratios):.1f}x")
            print(f"    median: {np.median(ratios):.1f}x")
            print(f"    max:    {max(ratios):.1f}x")

        print(f"\n  Species:")
        for s, c in sorted(species_counts.items(), key=lambda x: -x[1]):
            print(f"    {s}: {c}")

    # Phase 2: I/O benchmarks on stratified subsample
    io_results, read_tp = benchmark_io_subsample(results, n_sample=150, n_trials=3)
    write_csv(io_results, os.path.join(CODE_DATA_DIR, "io_benchmarks.csv"))
    write_csv(read_tp, os.path.join(CODE_DATA_DIR, "read_throughput.csv"))

    # Phase 3: Value distributions
    vdist = extract_value_dists(results, n_sample=200)
    write_csv(vdist, os.path.join(CODE_DATA_DIR, "value_distributions.csv"))

    print(f"\nPeak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    print("Survey complete.")


if __name__ == "__main__":
    main()
