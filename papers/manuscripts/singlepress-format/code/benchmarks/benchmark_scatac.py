#!/usr/bin/env python3
"""
Benchmark D: scATAC-seq peak matrix compression.

Downloads public scATAC peak×cell count matrices and benchmarks .1pz
compression alongside scRNA data for comparison.

Sources:
  - 10x Genomics public scATAC datasets (filtered_peak_bc_matrix.h5)
  - Existing scRNA .1pz files for comparison

Outputs: ../data/scatac_bench.csv

Usage:
    python3 benchmark_scatac.py
"""

import sys, os, time, csv, gc, tempfile
import numpy as np
import scipy.sparse as ss

sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/singlepress")
import singlepress as sp

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
WORK_DIR = "/dev/shm/sp_atac_bench"

N_WARMUP = 1
N_REPS = 5

# ── 10x Genomics public scATAC datasets ──────────────────────
# These are filtered peak-barcode matrices in HDF5 format
TENX_ATAC_URLS = {
    "pbmc_3k_atac": "https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_pbmc_3k/atac_pbmc_3k_filtered_peak_bc_matrix.h5",
    "pbmc_10k_atac": "https://cf.10xgenomics.com/samples/cell-atac/2.1.0/atac_pbmc_10k/atac_pbmc_10k_filtered_peak_bc_matrix.h5",
    "pbmc_10k_multiome_atac": "https://cf.10xgenomics.com/samples/cell-arc/2.0.2/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_atac_filtered_peak_bc_matrix.h5",
}

# ── Existing scRNA datasets for comparison ───────────────────
SCRNA_DATASETS = [
    "GSE210261",   # ~4.6M nnz
    "GSE189042",   # ~20M nnz
    "GSE290932",   # ~33M nnz
    "GSE142483",   # ~51M nnz
    "GSE207157",   # ~74M nnz
]


def download_10x_h5(name, url):
    """Download a 10x HDF5 file."""
    import urllib.request
    dest = os.path.join(WORK_DIR, f"{name}.h5")
    if os.path.exists(dest):
        print(f"  Using cached: {dest}")
        return dest
    print(f"  Downloading {name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Downloaded: {os.path.getsize(dest) / 1e6:.1f} MB")
    return dest


def read_10x_peak_matrix(h5_path):
    """Read a 10x ATAC filtered_peak_bc_matrix.h5 as scipy sparse CSC (peaks × cells)."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        # 10x ATAC H5 format: matrix/data, matrix/indices, matrix/indptr, matrix/shape
        grp = f["matrix"]
        data = grp["data"][:]
        indices = grp["indices"][:]
        indptr = grp["indptr"][:]
        shape = tuple(grp["shape"][:])
        barcodes = [b.decode() for b in grp["barcodes"][:]]
        # Features (peaks as chr:start-end)
        feature_grp = grp["features"]
        features = [b.decode() for b in feature_grp["id"][:]]

    # shape is (n_features, n_barcodes) but stored as CSC
    mat = ss.csc_matrix((data.astype(np.int32), indices, indptr), shape=shape)
    return mat, features, barcodes


def benchmark_compression(mat, features, barcodes, name, data_type):
    """Benchmark .1pz compression on a sparse matrix."""
    pz_path = os.path.join(WORK_DIR, f"{name}.1pz")

    # Write .1pz
    t0 = time.perf_counter()
    sp.write_1pz(pz_path, mat, rownames=features, colnames=barcodes)
    t_write = time.perf_counter() - t0
    pz_bytes = os.path.getsize(pz_path)

    raw_int32_bytes = mat.nnz * 8  # 4 bytes value + 4 bytes index per nnz
    ratio = raw_int32_bytes / pz_bytes
    bytes_per_nnz = pz_bytes / mat.nnz

    # Read .1pz
    times = []
    for i in range(N_WARMUP + N_REPS):
        gc.collect()
        t0 = time.perf_counter()
        m = sp.read_1pz(pz_path)
        t = time.perf_counter() - t0
        if i >= N_WARMUP:
            times.append(t)
        del m

    med_read = np.median(times)
    read_mbps = raw_int32_bytes / med_read / 1e6

    # Write H5AD for comparison
    h5ad_path = os.path.join(WORK_DIR, f"{name}.h5ad")
    import anndata
    adata = anndata.AnnData(X=mat.T.tocsr())  # cells × features
    t0 = time.perf_counter()
    adata.write_h5ad(h5ad_path)
    t_write_h5ad = time.perf_counter() - t0
    h5ad_bytes = os.path.getsize(h5ad_path)
    h5ad_ratio = raw_int32_bytes / h5ad_bytes

    # Value distribution analysis
    vals = mat.data
    val_max = int(vals.max()) if len(vals) > 0 else 0
    frac_one = float(np.sum(vals == 1)) / len(vals) if len(vals) > 0 else 0
    frac_two = float(np.sum(vals == 2)) / len(vals) if len(vals) > 0 else 0
    density = mat.nnz / (mat.shape[0] * mat.shape[1])

    result = {
        "name": name,
        "data_type": data_type,
        "nrows": mat.shape[0],
        "ncols": mat.shape[1],
        "nnz": mat.nnz,
        "density": density,
        "val_max": val_max,
        "frac_val_1": frac_one,
        "frac_val_2": frac_two,
        "raw_int32_bytes": raw_int32_bytes,
        "pz_bytes": pz_bytes,
        "pz_ratio": ratio,
        "pz_bytes_per_nnz": bytes_per_nnz,
        "pz_write_s": t_write,
        "pz_read_s": med_read,
        "pz_read_mbps": read_mbps,
        "h5ad_bytes": h5ad_bytes,
        "h5ad_ratio": h5ad_ratio,
    }

    print(f"  {name}: {mat.shape[0]:,}×{mat.shape[1]:,}, {mat.nnz:,} nnz ({density:.4%})")
    print(f"    Values: max={val_max}, frac=1: {frac_one:.2%}, frac≤2: {frac_one+frac_two:.2%}")
    print(f"    .1pz: {pz_bytes/1e6:.1f} MB ({ratio:.1f}×, {bytes_per_nnz:.3f} B/nnz)")
    print(f"    H5AD: {h5ad_bytes/1e6:.1f} MB ({h5ad_ratio:.1f}×)")
    print(f"    Read: {med_read:.3f}s ({read_mbps:.0f} MB/s)")

    return result


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    results = []

    # ── scATAC peak matrices from 10x Genomics ──────────────
    print("=" * 60)
    print("scATAC Peak Matrix Benchmarks")
    print("=" * 60)

    for name, url in TENX_ATAC_URLS.items():
        try:
            h5_path = download_10x_h5(name, url)
            mat, features, barcodes = read_10x_peak_matrix(h5_path)
            result = benchmark_compression(mat, features, barcodes, name, "scATAC_peaks")
            results.append(result)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()

    # ── Existing scRNA datasets for comparison ───────────────
    print("\n" + "=" * 60)
    print("scRNA-seq Comparison Datasets")
    print("=" * 60)

    for gse in SCRNA_DATASETS:
        pz_path = os.path.join(QUANT_DIR, gse, "counts.1pz")
        if not os.path.exists(pz_path):
            print(f"  {gse}: NOT FOUND, skipping")
            continue

        try:
            pz = sp.open_1pz(pz_path)
            info = sp.info_1pz(pz_path)
            mat = sp.read_1pz(pz_path)

            pz_bytes = os.path.getsize(pz_path)
            raw_int32_bytes = mat.nnz * 8
            ratio = raw_int32_bytes / pz_bytes
            bytes_per_nnz = pz_bytes / mat.nnz

            vals = mat.data
            val_max = int(vals.max()) if len(vals) > 0 else 0
            frac_one = float(np.sum(vals == 1)) / len(vals) if len(vals) > 0 else 0
            frac_two = float(np.sum(vals == 2)) / len(vals) if len(vals) > 0 else 0
            density = mat.nnz / (mat.shape[0] * mat.shape[1])

            # Read speed
            times = []
            for i in range(N_WARMUP + N_REPS):
                gc.collect()
                t0 = time.perf_counter()
                m = sp.read_1pz(pz_path)
                t = time.perf_counter() - t0
                if i >= N_WARMUP:
                    times.append(t)
                del m

            med_read = np.median(times)
            read_mbps = raw_int32_bytes / med_read / 1e6

            result = {
                "name": gse,
                "data_type": "scRNA",
                "nrows": mat.shape[0],
                "ncols": mat.shape[1],
                "nnz": mat.nnz,
                "density": density,
                "val_max": val_max,
                "frac_val_1": frac_one,
                "frac_val_2": frac_two,
                "raw_int32_bytes": raw_int32_bytes,
                "pz_bytes": pz_bytes,
                "pz_ratio": ratio,
                "pz_bytes_per_nnz": bytes_per_nnz,
                "pz_write_s": 0,  # not measured here
                "pz_read_s": med_read,
                "pz_read_mbps": read_mbps,
                "h5ad_bytes": 0,
                "h5ad_ratio": 0,
            }
            results.append(result)

            print(f"  {gse}: {mat.shape[0]:,}×{mat.shape[1]:,}, {mat.nnz:,} nnz ({density:.4%})")
            print(f"    Values: max={val_max}, frac=1: {frac_one:.2%}")
            print(f"    .1pz: {pz_bytes/1e6:.1f} MB ({ratio:.1f}×, {bytes_per_nnz:.3f} B/nnz)")
            del mat

        except Exception as e:
            print(f"  FAILED {gse}: {e}")
            import traceback
            traceback.print_exc()

    # ── Save results ─────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, "scatac_bench.csv")
    if results:
        keys = list(results[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"\nSaved {out_path}")

    # ── Summary ──────────────────────────────────────────────
    if results:
        atac = [r for r in results if r["data_type"] == "scATAC_peaks"]
        rna = [r for r in results if r["data_type"] == "scRNA"]
        if atac:
            ratios = [r["pz_ratio"] for r in atac]
            bpn = [r["pz_bytes_per_nnz"] for r in atac]
            print(f"\nscATAC summary: {len(atac)} datasets")
            print(f"  Compression: {np.median(ratios):.1f}× median (range {min(ratios):.1f}–{max(ratios):.1f}×)")
            print(f"  Bytes/nnz: {np.median(bpn):.3f} median")
        if rna:
            ratios = [r["pz_ratio"] for r in rna]
            bpn = [r["pz_bytes_per_nnz"] for r in rna]
            print(f"\nscRNA summary: {len(rna)} datasets")
            print(f"  Compression: {np.median(ratios):.1f}× median (range {min(ratios):.1f}–{max(ratios):.1f}×)")
            print(f"  Bytes/nnz: {np.median(bpn):.3f} median")


if __name__ == "__main__":
    main()
