#!/usr/bin/env python3
"""
Fast survey of ALL merged .1pz files in the production pipeline.
Reads only headers (info_1pz) + file sizes — no matrix decompression.
Cross-references with catalog for species/protocol.
Also benchmarks read speed on a stratified subsample for format comparisons.

Usage:
    ssh <node> "cd /tmp && python3 -u /path/to/survey_all_datasets.py"
"""

import csv
import json
import os
import sys
import time
import traceback
import tempfile
import subprocess
import resource

# Fix namespace-package shadowing
_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if '' in sys.path:
    sys.path.remove('')

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import numpy as np
import singlepress as sp

QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Gene rows → species mapping (USA-resolved: nrows = 3 × n_genes)
SPECIES_MAP = {
    115818: ("Homo sapiens", 38606),
    171540: ("Mus musculus", 57180),
    97560: ("Danio rerio", 32520),
    91686: ("Rattus norvegicus", 30562),
    72834: ("Drosophila melanogaster", 24278),
    106296: ("Macaca mulatta", 35432),
    90324: ("Gallus gallus", 30108),
    73842: ("Xenopus laevis", 24614),
    # Additional species
    78924: ("Macaca fascicularis", 26308),
    69072: ("Sus scrofa", 23024),
    139200: ("Ovis aries", 46400),
    60930: ("Canis lupus familiaris", 20310),
    89097: ("Pan troglodytes", 29699),
}


def load_catalog():
    """Load catalog and build GSE→(species, protocol) mapping."""
    try:
        import pandas as pd
        cat = pd.read_parquet(
            "/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet",
            columns=["gse_id", "organism", "protocol_inferred"]
        )
        # Take first per GSE
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


def survey_all():
    """Survey all merged counts.1pz files."""
    catalog = load_catalog()
    print(f"Catalog loaded: {len(catalog)} GSEs")

    # Find all merged counts.1pz
    gse_dirs = sorted(os.listdir(QUANT_DIR))
    print(f"Total GSE directories: {len(gse_dirs)}")

    results = []
    errors = 0
    t0 = time.time()

    for i, gse_id in enumerate(gse_dirs):
        pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
        if not os.path.isfile(pz_path):
            continue

        try:
            info = sp.info_1pz(pz_path)
            file_size = os.path.getsize(pz_path)

            nrows = info["m"]
            ncols = info["n"]
            nnz = info["nnz"]

            # Determine species from row count
            sp_info = SPECIES_MAP.get(nrows)
            if sp_info:
                species, n_genes = sp_info
                is_usa = True
            else:
                # Try matching n/3
                for nr, (sname, ng) in SPECIES_MAP.items():
                    if nrows == ng:
                        species, n_genes = sname, ng
                        is_usa = False
                        break
                else:
                    species = "Unknown"
                    n_genes = nrows if nrows < 100000 else nrows // 3
                    is_usa = (nrows % 3 == 0)

            # int32 CSC baseline
            density = nnz / (nrows * ncols) if nrows * ncols > 0 else 0
            raw_int32_bytes = (ncols + 1) * 4 + nnz * 4 + nnz * 4  # indptr + indices + data

            ratio = raw_int32_bytes / file_size if file_size > 0 else 0

            # Catalog info
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
                "pz_bytes": file_size,
                "raw_int32_bytes": raw_int32_bytes,
                "ratio": round(ratio, 3),
                "bits_per_nz": round(file_size * 8 / nnz, 3) if nnz > 0 else 0,
                "codec": info.get("codec", ""),
                "has_transpose": info.get("has_transpose", False),
                "has_obs_var": info.get("has_obs_var", False),
                "num_chunks": info.get("num_chunks", 0),
            }
            results.append(row)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(gse_dirs)}] {rate:.0f} GSEs/s, {len(results)} with .1pz, {errors} errors")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {gse_id}: {e}")

    elapsed = time.time() - t0
    print(f"\nSurvey complete: {len(results)} datasets in {elapsed:.1f}s ({errors} errors)")
    return results


def benchmark_io_subsample(results, n_sample=200, n_trials=3):
    """
    Benchmark read speed on a stratified subsample across formats.
    Only does read timing (.1pz, H5AD, npz) — no heavy writes.
    """
    import random
    import anndata as ad
    import scipy.sparse as ss

    # Stratified sample: pick across size range
    sorted_results = sorted(results, key=lambda x: x["nnz"])
    # Pick every N-th to get a spread
    step = max(1, len(sorted_results) // n_sample)
    sample_indices = list(range(0, len(sorted_results), step))[:n_sample]
    sample = [sorted_results[i] for i in sample_indices]
    print(f"\nBenchmarking I/O on {len(sample)} datasets (stratified by NNZ)")

    io_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, rec in enumerate(sample):
            gse_id = rec["gse_id"]
            pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
            nnz = rec["nnz"]
            raw_bytes = rec["raw_int32_bytes"]

            if idx % 20 == 0:
                print(f"  [{idx}/{len(sample)}] {gse_id} nnz={nnz:,}")

            try:
                # Skip very large (>200M nnz) for I/O bench to keep wall-clock reasonable
                if nnz > 200_000_000:
                    continue

                # Read .1pz — timed
                times_1pz = []
                mat = None
                for trial in range(n_trials + 1):
                    t0 = time.perf_counter()
                    mat = sp.read_1pz(pz_path)
                    t1 = time.perf_counter()
                    if trial > 0:  # skip warmup
                        times_1pz.append(t1 - t0)
                t_1pz = float(np.median(times_1pz))

                csc = mat.tocsc()
                csc.data = csc.data.astype(np.int32)
                rownames = getattr(mat, 'rownames', None) or [f"g{i}" for i in range(csc.shape[0])]
                colnames = getattr(mat, 'colnames', None) or [f"c{i}" for i in range(csc.shape[1])]

                io_rec = {
                    "gse_id": gse_id,
                    "nnz": nnz,
                    "raw_int32_bytes": raw_bytes,
                    "pz_bytes": rec["pz_bytes"],
                    "read_1pz_s": round(t_1pz, 4),
                    "read_1pz_mbps": round(raw_bytes / t_1pz / 1e6, 1),
                }

                # Write and read H5AD (gzip)
                h5ad_path = os.path.join(tmpdir, f"{gse_id}.h5ad")
                try:
                    import pandas as pd
                    adata = ad.AnnData(
                        X=csc.T.tocsr(),
                        obs=pd.DataFrame(index=list(colnames)),
                        var=pd.DataFrame(index=list(rownames)),
                    )
                    adata.write_h5ad(h5ad_path, compression="gzip")
                    h5ad_size = os.path.getsize(h5ad_path)
                    io_rec["h5ad_bytes"] = h5ad_size
                    io_rec["h5ad_ratio"] = round(raw_bytes / h5ad_size, 3)

                    times_h5ad = []
                    for trial in range(n_trials + 1):
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

                # Write and read scipy npz
                npz_path = os.path.join(tmpdir, f"{gse_id}.npz")
                try:
                    ss.save_npz(npz_path, csc)
                    npz_size = os.path.getsize(npz_path)
                    io_rec["npz_bytes"] = npz_size
                    io_rec["npz_ratio"] = round(raw_bytes / npz_size, 3)

                    times_npz = []
                    for trial in range(n_trials + 1):
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

                # 10x HDF5
                h5_path = os.path.join(tmpdir, f"{gse_id}.h5")
                try:
                    import h5py
                    with h5py.File(h5_path, "w") as f:
                        grp = f.create_group("matrix")
                        grp.create_dataset("data", data=csc.data, compression="gzip")
                        grp.create_dataset("indices", data=csc.indices, compression="gzip")
                        grp.create_dataset("indptr", data=csc.indptr, compression="gzip")
                        grp.attrs["shape"] = np.array(csc.shape, dtype=np.int32)
                    h5_size = os.path.getsize(h5_path)
                    io_rec["h5_bytes"] = h5_size
                    io_rec["h5_ratio"] = round(raw_bytes / h5_size, 3)

                    def read_10x_h5():
                        with h5py.File(h5_path, "r") as f:
                            g = f["matrix"]
                            return ss.csc_matrix(
                                (g["data"][:], g["indices"][:], g["indptr"][:]),
                                shape=g.attrs["shape"]
                            )

                    times_h5 = []
                    for trial in range(n_trials + 1):
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
                del mat, csc

            except Exception as e:
                if idx < 5:
                    print(f"  ERROR {gse_id}: {e}")

    print(f"I/O benchmarks complete: {len(io_results)} datasets")
    return io_results


def benchmark_threading(n_threads_list=[1, 2, 4, 8]):
    """Benchmark effect of OMP_NUM_THREADS on .1pz read speed."""
    # Pick a few representative datasets
    test_gses = []
    gse_dirs = sorted(os.listdir(QUANT_DIR))
    # Find a medium dataset (~5-20M nnz)
    for gse_id in gse_dirs:
        pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
        if os.path.isfile(pz_path):
            try:
                info = sp.info_1pz(pz_path)
                if 5_000_000 < info["nnz"] < 50_000_000:
                    test_gses.append((gse_id, info["nnz"]))
                    if len(test_gses) >= 10:
                        break
            except:
                pass

    print(f"\nThreading benchmark on {len(test_gses)} datasets")
    thread_results = []

    for n_threads in n_threads_list:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        print(f"  Threads={n_threads}")
        for gse_id, nnz in test_gses:
            pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
            info = sp.info_1pz(pz_path)
            raw_bytes = (info["n"] + 1) * 4 + nnz * 4 + nnz * 4

            times = []
            for trial in range(4):  # 1 warmup + 3 timed
                t0 = time.perf_counter()
                sp.read_1pz(pz_path)
                t1 = time.perf_counter()
                if trial > 0:
                    times.append(t1 - t0)

            t = float(np.median(times))
            thread_results.append({
                "gse_id": gse_id,
                "nnz": nnz,
                "n_threads": n_threads,
                "read_s": round(t, 4),
                "read_mbps": round(raw_bytes / t / 1e6, 1),
            })

    # Reset
    os.environ["OMP_NUM_THREADS"] = "8"
    return thread_results


def main():
    print("=" * 70)
    print("SinglePress Comprehensive Dataset Survey")
    print("=" * 70)

    # Phase 1: Fast header survey of ALL datasets
    results = survey_all()

    # Save survey CSV
    out_csv = os.path.join(SCRIPT_DIR, "all_datasets_survey.csv")
    if results:
        keys = results[0].keys()
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Survey saved: {out_csv} ({len(results)} rows)")

    # Print summary stats
    if results:
        species_counts = {}
        protocol_counts = {}
        total_cells = 0
        total_nnz = 0
        for r in results:
            species_counts[r["species"]] = species_counts.get(r["species"], 0) + 1
            protocol_counts[r["protocol"]] = protocol_counts.get(r["protocol"], 0) + 1
            total_cells += r["ncols"]
            total_nnz += r["nnz"]

        print(f"\n{'='*50}")
        print(f"SURVEY SUMMARY")
        print(f"  Datasets: {len(results)}")
        print(f"  Total cells: {total_cells:,}")
        print(f"  Total NNZ: {total_nnz:,}")
        print(f"\n  Species:")
        for s, c in sorted(species_counts.items(), key=lambda x: -x[1]):
            print(f"    {s}: {c}")
        print(f"\n  Protocols (top 15):")
        for p, c in sorted(protocol_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {p}: {c}")

        # Compression ratio stats
        ratios = [r["ratio"] for r in results if r["ratio"] > 0]
        print(f"\n  Compression ratio (vs int32 CSC):")
        print(f"    min: {min(ratios):.1f}x")
        print(f"    median: {np.median(ratios):.1f}x")
        print(f"    max: {max(ratios):.1f}x")

    # Phase 2: I/O benchmarks on subsample
    io_results = benchmark_io_subsample(results, n_sample=150, n_trials=3)
    io_csv = os.path.join(SCRIPT_DIR, "io_benchmarks.csv")
    if io_results:
        keys = io_results[0].keys()
        with open(io_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(io_results)
        print(f"I/O benchmarks saved: {io_csv} ({len(io_results)} rows)")

    # Phase 3: Threading benchmark
    thread_results = benchmark_threading([1, 2, 4, 8])
    thread_csv = os.path.join(SCRIPT_DIR, "threading_benchmarks.csv")
    if thread_results:
        keys = thread_results[0].keys()
        with open(thread_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(thread_results)
        print(f"Threading benchmarks saved: {thread_csv} ({len(thread_results)} rows)")

    print(f"\nPeak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
