#!/usr/bin/env python3
"""
Quick scATAC-like benchmark using existing .1pz data.

Compares value distributions and compression characteristics of
scATAC-like sparse matrices vs scRNA to demonstrate format generality.

Also creates synthetic peak matrices with scATAC-like properties
to benchmark compression on the specific statistical profile.

Completes in <5 minutes.
"""

import sys, os, time, csv, gc
import numpy as np
import scipy.sparse as ss

sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/singlepress")
import singlepress as sp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
WORK_DIR = "/dev/shm/sp_atac_quick"
QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
N_WARMUP = 1
N_REPS = 5

# Existing scRNA datasets for comparison (varying sizes)
SCRNA_DATASETS = [
    "GSE210261",   # ~4.6M nnz, small
    "GSE189042",   # ~20M nnz, medium
    "GSE290932",   # ~33M nnz, large
    "GSE142483",   # ~51M nnz, xlarge
    "GSE207157",   # ~74M nnz, xxlarge
    "GSE248138",   # ~113M nnz, xxxlarge
]

# Known scATAC in our pipeline
SCATAC_DATASETS = [
    "GSE141590",   # Drosophila scATAC, 2.6K cells
]


def analyze_and_benchmark(pz_path, name, data_type):
    """Analyze value distribution and benchmark read speed."""
    pz_bytes = os.path.getsize(pz_path)

    # Read once for stats
    mat = sp.read_1pz(pz_path)
    raw_bytes = mat.nnz * 8
    vals = mat.data
    density = mat.nnz / (mat.shape[0] * mat.shape[1])

    # Value distribution
    frac1 = float(np.sum(vals == 1)) / len(vals) if len(vals) > 0 else 0
    frac2 = float(np.sum(vals == 2)) / len(vals) if len(vals) > 0 else 0
    frac3 = float(np.sum(vals == 3)) / len(vals) if len(vals) > 0 else 0
    frac_le3 = float(np.sum(vals <= 3)) / len(vals) if len(vals) > 0 else 0
    val_max = int(vals.max()) if len(vals) > 0 else 0
    val_mean = float(vals.mean()) if len(vals) > 0 else 0

    # Entropy
    counts = np.bincount(vals.astype(int))
    probs = counts[counts > 0] / counts[counts > 0].sum()
    entropy = -np.sum(probs * np.log2(probs))

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

    result = {
        "name": name,
        "data_type": data_type,
        "nrows": mat.shape[0],
        "ncols": mat.shape[1],
        "nnz": mat.nnz,
        "density": density,
        "val_max": val_max,
        "val_mean": val_mean,
        "frac_val_1": frac1,
        "frac_val_2": frac2,
        "frac_val_3": frac3,
        "frac_val_le3": frac_le3,
        "entropy_bits": entropy,
        "raw_bytes": raw_bytes,
        "pz_bytes": pz_bytes,
        "pz_ratio": raw_bytes / pz_bytes,
        "pz_bytes_per_nnz": pz_bytes / mat.nnz,
        "pz_read_s": med_read,
        "pz_read_mbps": raw_bytes / med_read / 1e6,
    }

    print(f"  {name} ({data_type}): {mat.shape[0]:,}×{mat.shape[1]:,}, {mat.nnz:,} nnz")
    print(f"    density={density:.4%}, f(1)={frac1:.1%}, f(≤3)={frac_le3:.1%}, H={entropy:.2f}b")
    print(f"    .1pz: {pz_bytes/1e6:.1f}MB, {raw_bytes/pz_bytes:.1f}×, {pz_bytes/mat.nnz:.3f} B/nnz")
    print(f"    read: {med_read:.3f}s ({raw_bytes/med_read/1e6:.0f} MB/s)")

    del mat
    return result


def create_synthetic_peak_matrix(n_peaks, n_cells, density, frac1, seed=42):
    """Create a synthetic scATAC peak matrix with realistic properties."""
    rng = np.random.RandomState(seed)

    # Total nnz
    total_nnz = int(n_peaks * n_cells * density)

    # Generate positions
    rows = rng.randint(0, n_peaks, size=total_nnz)
    cols = rng.randint(0, n_cells, size=total_nnz)

    # Generate values: frac1 are 1, rest geometric-like
    n_ones = int(total_nnz * frac1)
    n_rest = total_nnz - n_ones
    vals_ones = np.ones(n_ones, dtype=np.int32)
    vals_rest = rng.geometric(p=0.5, size=n_rest).astype(np.int32) + 1
    vals = np.concatenate([vals_ones, vals_rest])
    rng.shuffle(vals)

    mat = ss.csc_matrix((vals, (rows, cols)), shape=(n_peaks, n_cells))
    mat.sum_duplicates()
    return mat


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    results = []

    # ── Real scATAC data ─────────────────────────────────
    print("=" * 60)
    print("Real scATAC datasets")
    print("=" * 60)
    for gse in SCATAC_DATASETS:
        pz_path = os.path.join(QUANT_DIR, gse, "counts.1pz")
        if os.path.exists(pz_path):
            results.append(analyze_and_benchmark(pz_path, gse, "scATAC_real"))

    # ── Synthetic scATAC peak matrices ───────────────────
    print("\n" + "=" * 60)
    print("Synthetic scATAC peak matrices")
    print("=" * 60)

    synth_configs = [
        # (name, n_peaks, n_cells, density, frac1) — realistic scATAC profiles
        ("synth_atac_3k",   150000,  3000,  0.003, 0.85),
        ("synth_atac_10k",  200000, 10000,  0.002, 0.88),
        ("synth_atac_50k",  250000, 50000,  0.001, 0.90),
    ]

    for name, n_peaks, n_cells, density, frac1 in synth_configs:
        print(f"\n  Creating {name}: {n_peaks:,}×{n_cells:,} (d={density}, f1={frac1})")
        mat = create_synthetic_peak_matrix(n_peaks, n_cells, density, frac1)

        genes = [f"peak_{i}" for i in range(mat.shape[0])]
        barcodes = [f"cell_{i}" for i in range(mat.shape[1])]
        pz_path = os.path.join(WORK_DIR, f"{name}.1pz")

        sp.write_1pz(pz_path, mat, rownames=genes, colnames=barcodes)
        results.append(analyze_and_benchmark(pz_path, name, "scATAC_synth"))
        del mat

    # ── Real scRNA comparison ────────────────────────────
    print("\n" + "=" * 60)
    print("scRNA comparison datasets")
    print("=" * 60)
    for gse in SCRNA_DATASETS:
        pz_path = os.path.join(QUANT_DIR, gse, "counts.1pz")
        if os.path.exists(pz_path):
            results.append(analyze_and_benchmark(pz_path, gse, "scRNA"))

    # ── Save ─────────────────────────────────────────────
    if results:
        path = os.path.join(DATA_DIR, "scatac_bench.csv")
        keys = list(results[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"\nSaved {path}")

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    atac = [r for r in results if "ATAC" in r["data_type"] or "atac" in r["data_type"]]
    rna = [r for r in results if r["data_type"] == "scRNA"]

    if atac:
        ratios = [r["pz_ratio"] for r in atac]
        bpn = [r["pz_bytes_per_nnz"] for r in atac]
        print(f"scATAC ({len(atac)} datasets):")
        print(f"  Compression: {np.median(ratios):.1f}× median (range {min(ratios):.1f}–{max(ratios):.1f}×)")
        print(f"  Bytes/nnz: {np.median(bpn):.3f} median")
    if rna:
        ratios = [r["pz_ratio"] for r in rna]
        bpn = [r["pz_bytes_per_nnz"] for r in rna]
        print(f"scRNA ({len(rna)} datasets):")
        print(f"  Compression: {np.median(ratios):.1f}× median (range {min(ratios):.1f}–{max(ratios):.1f}×)")
        print(f"  Bytes/nnz: {np.median(bpn):.3f} median")


if __name__ == "__main__":
    main()
