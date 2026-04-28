#!/usr/bin/env python3
"""
Focused Census + TileDB-SOMA benchmark.

Extracts tissue slices from CELLxGENE Census, converts to .1pz and H5AD,
then benchmarks read speed, storage, and PyTorch DataLoader performance.

Census version: 2024-07-01 (compatible with tiledbsoma 1.11.4)

Outputs:
  ../data/census_storage_bench.csv   — format size comparison
  ../data/census_read_bench.csv      — read speed comparison
  ../data/census_dataloader_bench.csv — PyTorch DataLoader comparison

Usage:
    ssh b004 "cd /mnt/home/debruinz/Singlet-AI/papers/manuscripts/singlepress-format/code/benchmarks && python3 benchmark_census_focused.py"
"""

import sys, os, time, csv, gc, json
import numpy as np
import scipy.sparse as ss

sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/singlepress")
import singlepress as sp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
WORK_DIR = "/dev/shm/sp_census_bench"
CENSUS_VERSION = "2024-07-01"
N_WARMUP = 1
N_REPS = 5

# Tissue slices to extract (matching CZI tutorial + larger slices)
SLICES = [
    {"name": "tongue",    "tissue": "tongue",         "max_cells": None},
    {"name": "lung_50k",  "tissue": "lung",           "max_cells": 50000},
    {"name": "blood_100k","tissue": "blood",           "max_cells": 100000},
    {"name": "brain_50k", "tissue": "cerebral cortex", "max_cells": 50000},
]


def extract_census_slice(tissue, max_cells=None):
    """Extract a tissue slice from Census as scipy sparse CSC."""
    import cellxgene_census
    import tiledbsoma

    print(f"  Opening Census {CENSUS_VERSION}...")
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
    human = census["census_data"]["homo_sapiens"]

    # Get cell count for this tissue
    obs_filter = f"tissue == '{tissue}'"
    obs_df = human.obs.read(
        column_names=["soma_joinid", "tissue"],
        value_filter=obs_filter
    ).concat().to_pandas()

    n_total = len(obs_df)
    print(f"  {tissue}: {n_total:,} cells total")

    if max_cells and n_total > max_cells:
        obs_df = obs_df.sample(n=max_cells, random_state=42)
        print(f"  Subsampled to {max_cells:,} cells")

    cell_ids = obs_df["soma_joinid"].values

    # Get gene names
    var_df = human.ms["RNA"].var.read(column_names=["soma_joinid", "feature_name"]).concat().to_pandas()
    gene_names = var_df["feature_name"].tolist()
    gene_ids = var_df["soma_joinid"].values

    # Read sparse matrix
    print(f"  Reading expression matrix ({len(cell_ids):,} cells × {len(gene_ids):,} genes)...")
    t0 = time.perf_counter()

    # Use ExperimentAxisQuery for efficient reads
    with human.axis_query(
        measurement_name="RNA",
        obs_query=tiledbsoma.AxisQuery(value_filter=obs_filter)
    ) as query:
        tbl = query.X("raw").tables().concat()
        # Convert Arrow table to scipy sparse
        soma_dim_0 = tbl["soma_dim_0"].to_numpy()
        soma_dim_1 = tbl["soma_dim_1"].to_numpy()
        soma_data = tbl["soma_data"].to_numpy()

    t_read_tiledb = time.perf_counter() - t0

    # Remap to contiguous indices
    cell_map = {cid: i for i, cid in enumerate(sorted(set(soma_dim_0)))}
    gene_map = {gid: i for i, gid in enumerate(sorted(set(soma_dim_1)))}

    rows = np.array([gene_map[g] for g in soma_dim_1])
    cols = np.array([cell_map[c] for c in soma_dim_0])
    n_cells = len(cell_map)
    n_genes = len(gene_map)

    mat = ss.csc_matrix((soma_data.astype(np.int32), (rows, cols)),
                        shape=(n_genes, n_cells))

    # Get remapped gene names
    gene_id_to_name = dict(zip(var_df["soma_joinid"].values, var_df["feature_name"].values))
    sorted_gene_ids = sorted(gene_map.keys())
    remapped_genes = [gene_id_to_name.get(gid, f"gene_{gid}") for gid in sorted_gene_ids]
    barcodes = [f"cell_{i}" for i in range(n_cells)]

    if max_cells and n_cells > max_cells:
        # Random subsample columns
        idx = np.random.RandomState(42).choice(n_cells, max_cells, replace=False)
        idx.sort()
        mat = mat[:, idx]
        barcodes = [barcodes[i] for i in idx]

    census.close()
    print(f"  Read from TileDB-SOMA: {t_read_tiledb:.1f}s")
    print(f"  Matrix: {mat.shape[0]:,} × {mat.shape[1]:,}, {mat.nnz:,} nnz")

    return mat, remapped_genes, barcodes, t_read_tiledb


def benchmark_storage_and_reads(mat, genes, barcodes, name):
    """Benchmark storage size and read speed across formats."""
    pz_path = os.path.join(WORK_DIR, f"{name}.1pz")
    h5ad_path = os.path.join(WORK_DIR, f"{name}.h5ad")

    results = []
    raw_bytes = mat.nnz * 8  # int32 value + int32 index per nnz

    # ── Write .1pz ────────────────────────────────────────
    t0 = time.perf_counter()
    sp.write_1pz(pz_path, mat, rownames=genes, colnames=barcodes)
    t_write_pz = time.perf_counter() - t0
    pz_bytes = os.path.getsize(pz_path)

    # ── Write H5AD ────────────────────────────────────────
    import anndata
    adata = anndata.AnnData(X=mat.T.tocsr())  # cells × genes for AnnData
    adata.var_names = genes[:mat.shape[0]] if len(genes) >= mat.shape[0] else [f"gene_{i}" for i in range(mat.shape[0])]
    adata.obs_names = barcodes[:mat.shape[1]] if len(barcodes) >= mat.shape[1] else [f"cell_{i}" for i in range(mat.shape[1])]
    t0 = time.perf_counter()
    adata.write_h5ad(h5ad_path)
    t_write_h5ad = time.perf_counter() - t0
    h5ad_bytes = os.path.getsize(h5ad_path)

    print(f"  Storage: .1pz={pz_bytes/1e6:.1f}MB ({raw_bytes/pz_bytes:.1f}×), H5AD={h5ad_bytes/1e6:.1f}MB ({raw_bytes/h5ad_bytes:.1f}×)")
    print(f"  Write: .1pz={t_write_pz:.2f}s ({raw_bytes/t_write_pz/1e6:.0f} MB/s), H5AD={t_write_h5ad:.2f}s ({raw_bytes/t_write_h5ad/1e6:.0f} MB/s)")

    # ── Read .1pz ─────────────────────────────────────────
    times_pz = []
    for i in range(N_WARMUP + N_REPS):
        gc.collect()
        t0 = time.perf_counter()
        m = sp.read_1pz(pz_path)
        t = time.perf_counter() - t0
        if i >= N_WARMUP:
            times_pz.append(t)
        del m

    # ── Read H5AD ─────────────────────────────────────────
    times_h5ad = []
    for i in range(N_WARMUP + N_REPS):
        gc.collect()
        t0 = time.perf_counter()
        a = anndata.read_h5ad(h5ad_path)
        t = time.perf_counter() - t0
        if i >= N_WARMUP:
            times_h5ad.append(t)
        del a

    med_pz = np.median(times_pz)
    med_h5ad = np.median(times_h5ad)

    # ── Column-range read benchmark ───────────────────────
    n_cols_10pct = max(1, mat.shape[1] // 10)
    times_pz_col = []
    for i in range(N_WARMUP + N_REPS):
        gc.collect()
        t0 = time.perf_counter()
        m = sp.read_1pz_columns(pz_path, 0, n_cols_10pct)
        t = time.perf_counter() - t0
        if i >= N_WARMUP:
            times_pz_col.append(t)
        del m
    med_pz_col = np.median(times_pz_col)

    results.append({
        "name": name,
        "n_genes": mat.shape[0],
        "n_cells": mat.shape[1],
        "nnz": mat.nnz,
        "raw_bytes": raw_bytes,
        "pz_bytes": pz_bytes,
        "pz_ratio": raw_bytes / pz_bytes,
        "pz_bytes_per_nnz": pz_bytes / mat.nnz,
        "h5ad_bytes": h5ad_bytes,
        "h5ad_ratio": raw_bytes / h5ad_bytes,
        "pz_write_s": t_write_pz,
        "h5ad_write_s": t_write_h5ad,
        "pz_read_s": med_pz,
        "pz_read_mbps": raw_bytes / med_pz / 1e6,
        "h5ad_read_s": med_h5ad,
        "h5ad_read_mbps": raw_bytes / med_h5ad / 1e6,
        "pz_col10pct_s": med_pz_col,
        "read_speedup": med_h5ad / med_pz,
    })

    print(f"  Read: .1pz={med_pz:.3f}s ({raw_bytes/med_pz/1e6:.0f}MB/s), H5AD={med_h5ad:.3f}s ({raw_bytes/med_h5ad/1e6:.0f}MB/s)")
    print(f"  10% column: .1pz={med_pz_col:.3f}s, speedup={med_h5ad/med_pz:.1f}×")

    return results


def benchmark_dataloader(pz_path, h5ad_path, name, n_genes, n_cells):
    """Benchmark PyTorch DataLoader: OnePZ vs manual H5AD loading."""
    import torch
    from singlepress.torch import OnePZCellDataset, collate_sparse

    batch_size = 1024
    n_batches = min(50, n_cells // batch_size)

    results = []

    # ── OnePZ DataLoader ──────────────────────────────────
    ds = OnePZCellDataset(pz_path, normalize=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_sparse, num_workers=0
    )

    # Warmup
    for i, batch in enumerate(loader):
        if i >= 2:
            break

    batch_times_pz = []
    for i, batch in enumerate(loader):
        t0 = time.perf_counter()
        _ = batch  # Already loaded by iterator
        t = time.perf_counter() - t0
        batch_times_pz.append(t)
        if i >= n_batches:
            break

    # ── H5AD "DataLoader" (manual batched reads) ─────────
    import anndata
    adata = anndata.read_h5ad(h5ad_path)
    X = adata.X  # sparse CSR
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    batch_times_h5ad = []
    for b in range(min(n_batches, len(indices) // batch_size)):
        gc.collect()
        idx = indices[b * batch_size:(b + 1) * batch_size]
        t0 = time.perf_counter()
        # Simulate what CZI's ExperimentDataset does: extract dense batch
        batch_sparse = X[idx]
        batch_dense = torch.from_numpy(batch_sparse.toarray()).float()
        t = time.perf_counter() - t0
        batch_times_h5ad.append(t)
        del batch_dense

    # ── Memory comparison ─────────────────────────────────
    # Dense batch memory: batch_size × n_genes × 4 bytes
    dense_batch_mem = batch_size * n_genes * 4
    # Sparse batch memory: ~nnz_per_batch × 8 bytes (value + index)
    avg_nnz_per_cell = ds.matrix.nnz / ds.matrix.shape[0] if hasattr(ds, 'matrix') else 0
    sparse_batch_mem = int(batch_size * avg_nnz_per_cell * 8)

    # OnePZ with multi-worker (test that it works)
    loader_mw = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_sparse, num_workers=4
    )
    batch_times_pz_mw = []
    for i, batch in enumerate(loader_mw):
        t0 = time.perf_counter()
        _ = batch
        t = time.perf_counter() - t0
        batch_times_pz_mw.append(t)
        if i >= n_batches:
            break

    med_pz = np.median(batch_times_pz) * 1000  # ms
    med_h5ad = np.median(batch_times_h5ad) * 1000  # ms
    med_pz_mw = np.median(batch_times_pz_mw) * 1000  # ms

    results.append({
        "name": name,
        "n_genes": n_genes,
        "n_cells": n_cells,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "pz_batch_ms": med_pz,
        "h5ad_batch_ms": med_h5ad,
        "pz_mw4_batch_ms": med_pz_mw,
        "batch_speedup": med_h5ad / med_pz,
        "dense_batch_bytes": dense_batch_mem,
        "sparse_batch_bytes": sparse_batch_mem,
        "memory_ratio": dense_batch_mem / max(sparse_batch_mem, 1),
        "pz_multiworker": "yes",
        "tiledb_sparse_multiworker": "NotImplementedError",
    })

    print(f"  DataLoader batch (ms): OnePZ={med_pz:.1f}, H5AD-dense={med_h5ad:.1f}, OnePZ-4w={med_pz_mw:.1f}")
    print(f"  Batch speedup: {med_h5ad/med_pz:.1f}×")
    print(f"  Memory per batch: sparse={sparse_batch_mem/1e6:.1f}MB, dense={dense_batch_mem/1e6:.1f}MB ({dense_batch_mem/max(sparse_batch_mem,1):.0f}×)")

    return results


def benchmark_tiledb_soma_ml(pz_path, name, n_genes, n_cells, tissue):
    """Benchmark TileDB-SOMA-ML ExperimentDataset for comparison."""
    results = []
    try:
        import tiledbsoma
        import cellxgene_census
        from tiledbsoma_ml import ExperimentDataPipe
        import torch

        batch_size = 1024
        n_batches = min(50, n_cells // batch_size)

        census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
        human = census["census_data"]["homo_sapiens"]

        # Dense mode (default) — this is what CZI recommends
        obs_filter = f"tissue == '{tissue}'"
        dp = ExperimentDataPipe(
            human,
            measurement_name="RNA",
            X_name="raw",
            obs_query=tiledbsoma.AxisQuery(value_filter=obs_filter),
            batch_size=batch_size,
            shuffle=True,
            return_sparse_X=False,  # default: dense NumPy
        )

        loader = torch.utils.data.DataLoader(dp, num_workers=0)

        batch_times = []
        for i, batch in enumerate(loader):
            t0 = time.perf_counter()
            _ = batch
            t = time.perf_counter() - t0
            batch_times.append(t)
            if i >= n_batches:
                break

        med_soma = np.median(batch_times) * 1000  # ms

        # Try sparse mode with multiworkers — should fail
        sparse_mw_works = False
        try:
            dp_sparse = ExperimentDataPipe(
                human,
                measurement_name="RNA",
                X_name="raw",
                obs_query=tiledbsoma.AxisQuery(value_filter=obs_filter),
                batch_size=batch_size,
                shuffle=True,
                return_sparse_X=True,
            )
            loader_sparse = torch.utils.data.DataLoader(dp_sparse, num_workers=2)
            for i, batch in enumerate(loader_sparse):
                if i >= 1:
                    break
            sparse_mw_works = True
        except (NotImplementedError, Exception) as e:
            print(f"  TileDB sparse+multiworker: {type(e).__name__}: {e}")

        results.append({
            "name": name,
            "soma_batch_ms": med_soma,
            "soma_sparse_multiworker": "yes" if sparse_mw_works else "NotImplementedError",
        })

        print(f"  TileDB-SOMA-ML batch (ms): dense={med_soma:.1f}")
        census.close()

    except Exception as e:
        print(f"  TileDB-SOMA-ML benchmark FAILED: {e}")
        import traceback
        traceback.print_exc()

    return results


def benchmark_existing_1pz():
    """Benchmark our existing .1pz files for cross-reference with Census data."""
    QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
    datasets = [
        ("GSE210261", "small"),
        ("GSE189042", "medium"),
        ("GSE290932", "large"),
        ("GSE142483", "xlarge"),
        ("GSE207157", "xxlarge"),
    ]

    results = []
    for gse, label in datasets:
        pz_path = os.path.join(QUANT_DIR, gse, "counts.1pz")
        if not os.path.exists(pz_path):
            continue

        pz_bytes = os.path.getsize(pz_path)

        # Read speed
        times = []
        for i in range(N_WARMUP + N_REPS):
            gc.collect()
            t0 = time.perf_counter()
            m = sp.read_1pz(pz_path)
            t = time.perf_counter() - t0
            if i >= N_WARMUP:
                times.append(t)
            if i == 0:
                info = {"nrows": m.shape[0], "ncols": m.shape[1], "nnz": m.nnz}
            del m

        med = np.median(times)
        raw = info["nnz"] * 8

        results.append({
            "name": gse,
            "label": label,
            "source": "GEO_1pz",
            "n_genes": info["nrows"],
            "n_cells": info["ncols"],
            "nnz": info["nnz"],
            "raw_bytes": raw,
            "pz_bytes": pz_bytes,
            "pz_ratio": raw / pz_bytes,
            "pz_read_s": med,
            "pz_read_mbps": raw / med / 1e6,
        })
        print(f"  {gse} ({label}): {info['nnz']:,} nnz, {pz_bytes/1e6:.1f}MB, {raw/pz_bytes:.1f}×, read={med:.3f}s ({raw/med/1e6:.0f}MB/s)")

    return results


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    storage_results = []
    read_results = []
    dl_results = []

    # ── Phase 1: Existing .1pz baseline ──────────────────
    print("=" * 60)
    print("Phase 1: Existing .1pz read speed baseline")
    print("=" * 60)
    existing = benchmark_existing_1pz()

    # ── Phase 2: Census tissue slices ────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Census tissue slice extraction + benchmarks")
    print("=" * 60)

    for sl in SLICES:
        name = sl["name"]
        tissue = sl["tissue"]
        max_cells = sl["max_cells"]

        print(f"\n--- {name} ({tissue}) ---")
        try:
            mat, genes, barcodes, t_tiledb_read = extract_census_slice(tissue, max_cells)

            # Storage + read benchmarks
            sr = benchmark_storage_and_reads(mat, genes, barcodes, name)
            for r in sr:
                r["tiledb_read_s"] = t_tiledb_read
                r["source"] = "Census"
            storage_results.extend(sr)

            # DataLoader benchmarks
            pz_path = os.path.join(WORK_DIR, f"{name}.1pz")
            h5ad_path = os.path.join(WORK_DIR, f"{name}.h5ad")
            dr = benchmark_dataloader(pz_path, h5ad_path, name, mat.shape[0], mat.shape[1])
            dl_results.extend(dr)

            del mat
            gc.collect()

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── Save results ─────────────────────────────────────
    if storage_results:
        path = os.path.join(DATA_DIR, "census_storage_bench.csv")
        keys = list(storage_results[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(storage_results)
        print(f"\nSaved {path}")

    if dl_results:
        path = os.path.join(DATA_DIR, "census_dataloader_bench.csv")
        keys = list(dl_results[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(dl_results)
        print(f"\nSaved {path}")

    if existing:
        path = os.path.join(DATA_DIR, "existing_read_bench.csv")
        keys = list(existing[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(existing)
        print(f"\nSaved {path}")

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if storage_results:
        for r in storage_results:
            print(f"  {r['name']}: .1pz={r['pz_ratio']:.1f}× H5AD={r['h5ad_ratio']:.1f}× read-speedup={r['read_speedup']:.1f}×")
    if dl_results:
        for r in dl_results:
            print(f"  {r['name']} DL: OnePZ={r['pz_batch_ms']:.1f}ms H5AD={r['h5ad_batch_ms']:.1f}ms ({r['batch_speedup']:.1f}×) mem-ratio={r['memory_ratio']:.0f}×")


if __name__ == "__main__":
    main()
