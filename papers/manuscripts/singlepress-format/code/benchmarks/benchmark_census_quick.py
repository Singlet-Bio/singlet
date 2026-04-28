#!/usr/bin/env python3
"""
Quick Census benchmark — tongue tissue slice only.
Extracts ~10K cells, writes .1pz and H5AD, compares storage + read + DataLoader.
Designed to complete in <10 minutes.

Usage:
    ssh b004 "cd /mnt/home/debruinz/Singlet-AI/papers/manuscripts/singlepress-format/code/benchmarks && python3 benchmark_census_quick.py"
"""

import sys, os, time, csv, gc, json
import numpy as np
import scipy.sparse as ss

sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/singlepress")
import singlepress as sp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
WORK_DIR = "/dev/shm/sp_census_quick"
CENSUS_VERSION = "2024-07-01"
N_WARMUP = 1
N_REPS = 5

SLICES = [
    {"name": "tongue",     "tissue": "tongue",          "max_cells": None},
    {"name": "lung_50k",   "tissue": "lung",            "max_cells": 50000},
    {"name": "blood_100k", "tissue": "blood",           "max_cells": 100000},
]


def extract_census_slice(tissue, max_cells=None):
    """Extract a tissue slice from Census as scipy sparse CSC."""
    import cellxgene_census
    import tiledbsoma

    print(f"  Opening Census {CENSUS_VERSION}...")
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
    human = census["census_data"]["homo_sapiens"]

    obs_filter = f"tissue == '{tissue}'"

    # Read expression data via ExperimentAxisQuery
    print(f"  Reading {tissue} expression data...")
    t0 = time.perf_counter()

    with human.axis_query(
        measurement_name="RNA",
        obs_query=tiledbsoma.AxisQuery(value_filter=obs_filter)
    ) as query:
        # Get obs/var metadata
        obs_df = query.obs(column_names=["soma_joinid"]).concat().to_pandas()
        var_df = query.var(column_names=["soma_joinid", "feature_name"]).concat().to_pandas()

        if max_cells and len(obs_df) > max_cells:
            obs_df = obs_df.sample(n=max_cells, random_state=42)

        # Read X data
        tbl = query.X("raw").tables().concat()
        soma_dim_0 = tbl["soma_dim_0"].to_numpy()
        soma_dim_1 = tbl["soma_dim_1"].to_numpy()
        soma_data = tbl["soma_data"].to_numpy()

    t_tiledb = time.perf_counter() - t0

    # Filter to selected cells if subsampled
    if max_cells and len(obs_df) <= max_cells:
        pass  # use all
    elif max_cells:
        keep = set(obs_df["soma_joinid"].values)
        mask = np.isin(soma_dim_0, list(keep))
        soma_dim_0 = soma_dim_0[mask]
        soma_dim_1 = soma_dim_1[mask]
        soma_data = soma_data[mask]

    # Remap to contiguous indices
    unique_cells = np.unique(soma_dim_0)
    unique_genes = np.unique(soma_dim_1)
    cell_map = {c: i for i, c in enumerate(unique_cells)}
    gene_map = {g: i for i, g in enumerate(unique_genes)}

    rows = np.array([gene_map[g] for g in soma_dim_1], dtype=np.int32)
    cols = np.array([cell_map[c] for c in soma_dim_0], dtype=np.int32)

    mat = ss.csc_matrix(
        (soma_data.astype(np.int32), (rows, cols)),
        shape=(len(unique_genes), len(unique_cells))
    )

    gene_id_to_name = dict(zip(var_df["soma_joinid"].values, var_df["feature_name"].values))
    genes = [gene_id_to_name.get(gid, f"gene_{gid}") for gid in unique_genes]
    barcodes = [f"cell_{i}" for i in range(len(unique_cells))]

    census.close()
    print(f"  TileDB-SOMA read: {t_tiledb:.1f}s → {mat.shape[0]:,}×{mat.shape[1]:,}, {mat.nnz:,} nnz")
    return mat, genes, barcodes, t_tiledb


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    import anndata
    all_results = []

    for sl in SLICES:
        name = sl["name"]
        print(f"\n{'='*60}\n{name} ({sl['tissue']})\n{'='*60}")

        try:
            mat, genes, barcodes, t_tiledb = extract_census_slice(sl["tissue"], sl["max_cells"])
        except Exception as e:
            print(f"  EXTRACTION FAILED: {e}")
            import traceback; traceback.print_exc()
            continue

        raw_bytes = mat.nnz * 8
        pz_path = os.path.join(WORK_DIR, f"{name}.1pz")
        h5ad_path = os.path.join(WORK_DIR, f"{name}.h5ad")

        # ── Write .1pz ──────────────────────────────────
        t0 = time.perf_counter()
        sp.write_1pz(pz_path, mat, rownames=genes, colnames=barcodes)
        t_write_pz = time.perf_counter() - t0
        pz_bytes = os.path.getsize(pz_path)

        # ── Write H5AD ──────────────────────────────────
        adata = anndata.AnnData(X=mat.T.tocsr())
        adata.var_names = genes[:mat.shape[0]]
        adata.obs_names = barcodes[:mat.shape[1]]
        t0 = time.perf_counter()
        adata.write_h5ad(h5ad_path)
        t_write_h5ad = time.perf_counter() - t0
        h5ad_bytes = os.path.getsize(h5ad_path)

        # ── Read .1pz ───────────────────────────────────
        times_pz = []
        for i in range(N_WARMUP + N_REPS):
            gc.collect()
            t0 = time.perf_counter()
            m = sp.read_1pz(pz_path)
            t = time.perf_counter() - t0
            if i >= N_WARMUP:
                times_pz.append(t)
            del m
        med_pz = np.median(times_pz)

        # ── Read H5AD ───────────────────────────────────
        times_h5ad = []
        for i in range(N_WARMUP + N_REPS):
            gc.collect()
            t0 = time.perf_counter()
            a = anndata.read_h5ad(h5ad_path)
            t = time.perf_counter() - t0
            if i >= N_WARMUP:
                times_h5ad.append(t)
            del a
        med_h5ad = np.median(times_h5ad)

        # ── 10% column range read ───────────────────────
        n10 = max(1, mat.shape[1] // 10)
        times_col = []
        for i in range(N_WARMUP + N_REPS):
            gc.collect()
            t0 = time.perf_counter()
            m = sp.read_1pz_columns(pz_path, 0, n10)
            t = time.perf_counter() - t0
            if i >= N_WARMUP:
                times_col.append(t)
            del m
        med_col = np.median(times_col)

        # ── Value stats ─────────────────────────────────
        vals = mat.data
        frac1 = float(np.sum(vals == 1)) / len(vals)
        density = mat.nnz / (mat.shape[0] * mat.shape[1])

        # ── DataLoader benchmark ────────────────────────
        import torch
        from singlepress.torch import OnePZCellDataset

        batch_size = min(1024, mat.shape[1] // 2)
        if batch_size < 8:
            batch_size = mat.shape[1]
        n_batches_target = min(50, mat.shape[1] // batch_size)
        avg_nnz_per_cell = mat.nnz / mat.shape[1]

        # OnePZ sparse DataLoader (0 workers) — uses dataset's own collate_fn
        ds = OnePZCellDataset(pz_path, normalize=True)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            collate_fn=ds.collate_fn, num_workers=0
        )
        batch_times = []
        for i, batch in enumerate(loader):
            if i < 2:
                continue  # warmup
            t0 = time.perf_counter()
            if hasattr(batch, 'to_dense'):
                _ = batch.to_dense()
            t = time.perf_counter() - t0
            batch_times.append(t)
            if i >= n_batches_target + 2:
                break
        med_dl_pz = np.median(batch_times) * 1000 if batch_times else 0

        # H5AD dense batch (simulating CZI ExperimentDataset default)
        X_csr = mat.T.tocsr()
        indices = np.arange(X_csr.shape[0])
        np.random.shuffle(indices)
        batch_times_h5 = []
        for b in range(n_batches_target):
            idx = indices[b * batch_size:(b + 1) * batch_size]
            gc.collect()
            t0 = time.perf_counter()
            batch_sparse = X_csr[idx]
            batch_dense = torch.from_numpy(batch_sparse.toarray()).float()
            t = time.perf_counter() - t0
            batch_times_h5.append(t)
            del batch_dense
        med_dl_h5 = np.median(batch_times_h5) * 1000 if batch_times_h5 else 0

        dense_batch_mem = batch_size * mat.shape[0] * 4
        sparse_batch_mem = int(batch_size * avg_nnz_per_cell * 8)

        result = {
            "name": name,
            "tissue": sl["tissue"],
            "n_genes": mat.shape[0],
            "n_cells": mat.shape[1],
            "nnz": mat.nnz,
            "density": density,
            "frac_val_1": frac1,
            "raw_bytes": raw_bytes,
            "pz_bytes": pz_bytes,
            "pz_ratio": raw_bytes / pz_bytes,
            "pz_bytes_per_nnz": pz_bytes / mat.nnz,
            "h5ad_bytes": h5ad_bytes,
            "h5ad_ratio": raw_bytes / h5ad_bytes,
            "tiledb_read_s": t_tiledb,
            "pz_write_s": t_write_pz,
            "pz_write_mbps": raw_bytes / t_write_pz / 1e6,
            "h5ad_write_s": t_write_h5ad,
            "h5ad_write_mbps": raw_bytes / t_write_h5ad / 1e6,
            "pz_read_s": med_pz,
            "pz_read_mbps": raw_bytes / med_pz / 1e6,
            "h5ad_read_s": med_h5ad,
            "h5ad_read_mbps": raw_bytes / med_h5ad / 1e6,
            "pz_col10pct_s": med_col,
            "read_speedup": med_h5ad / med_pz,
            "write_speedup": t_write_h5ad / t_write_pz,
            "pz_dl_batch_ms": med_dl_pz,
            "h5ad_dl_batch_ms": med_dl_h5,
            "dl_speedup": med_dl_h5 / med_dl_pz if med_dl_pz > 0 else 0,
            "dense_batch_bytes": dense_batch_mem,
            "sparse_batch_bytes": sparse_batch_mem,
            "memory_ratio": dense_batch_mem / max(sparse_batch_mem, 1),
        }
        all_results.append(result)

        print(f"\n  Summary for {name}:")
        print(f"    Storage: .1pz={pz_bytes/1e6:.1f}MB ({raw_bytes/pz_bytes:.1f}×) H5AD={h5ad_bytes/1e6:.1f}MB ({raw_bytes/h5ad_bytes:.1f}×)")
        print(f"    Write:  .1pz={t_write_pz:.2f}s H5AD={t_write_h5ad:.2f}s ({t_write_h5ad/t_write_pz:.1f}× slower)")
        print(f"    Read:   .1pz={med_pz:.3f}s H5AD={med_h5ad:.3f}s ({med_h5ad/med_pz:.1f}× slower)")
        print(f"    Col10%: .1pz={med_col:.3f}s")
        print(f"    DL batch: OnePZ={med_dl_pz:.1f}ms H5AD-dense={med_dl_h5:.1f}ms ({med_dl_h5/med_dl_pz:.1f}× slower)" if med_dl_pz > 0 else "")
        print(f"    Memory: sparse={sparse_batch_mem/1e6:.1f}MB dense={dense_batch_mem/1e6:.1f}MB ({dense_batch_mem/max(sparse_batch_mem,1):.0f}× ratio)")
        print(f"    TileDB-SOMA read: {t_tiledb:.1f}s vs .1pz read: {med_pz:.3f}s ({t_tiledb/med_pz:.0f}× slower)")

        del mat, adata
        gc.collect()

    # ── Save ────────────────────────────────────────────
    if all_results:
        path = os.path.join(DATA_DIR, "census_bench.csv")
        keys = list(all_results[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
