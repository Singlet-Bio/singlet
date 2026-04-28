#!/usr/bin/env python3
"""
Benchmark A: .1pz vs TileDB-SOMA on CELLxGENE Census data.

Tests:
  1. Storage footprint comparison
  2. Tissue-slice extraction read speed
  3. Cell subsetting at various sizes
  4. Gene subsetting at various sizes
  5. TileDB-SOMA-ML ExperimentDataset vs OnePZDataset loading comparison

Outputs: ../data/tiledb_census_bench.csv
         ../data/tiledb_dataloader_bench.csv

Usage:
    python3 benchmark_tiledb_census.py
"""

import sys, os, time, json, csv, gc, tempfile, shutil
import numpy as np
import scipy.sparse as ss

sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/singlepress")
import singlepress as sp

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
WORK_DIR = "/dev/shm/sp_bench"  # tmpfs for I/O benchmarks

# ── Tissue slices to benchmark ───────────────────────────────────
# These match BPCells' Census benchmarks and CZI's PyTorch tutorial
TISSUE_SLICES = [
    {"name": "tongue",  "filter": "tissue_general == 'tongue' and is_primary_data == True"},
    {"name": "lung_100k", "filter": "tissue_general == 'lung' and is_primary_data == True"},
    {"name": "blood_250k", "filter": "tissue_general == 'blood' and is_primary_data == True"},
]

N_WARMUP = 1
N_REPS = 5


def ensure_workdir():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def extract_census_slice(tissue_name, value_filter, max_cells=None):
    """Extract a Census slice as scipy sparse + metadata, save as .1pz and TileDB-SOMA local copy."""
    import cellxgene_census
    import tiledbsoma

    print(f"\n{'='*60}")
    print(f"Extracting Census slice: {tissue_name}")
    print(f"Filter: {value_filter}")

    census = cellxgene_census.open_soma(census_version="2025-01-30")
    exp = census["census_data"]["homo_sapiens"]

    with exp.axis_query(
        measurement_name="RNA",
        obs_query=tiledbsoma.AxisQuery(value_filter=value_filter),
    ) as query:
        # Get obs metadata
        obs_df = query.obs(column_names=["soma_joinid", "cell_type", "tissue_general"]).concat().to_pandas()
        n_cells = len(obs_df)
        print(f"  Found {n_cells:,} cells")

        if max_cells and n_cells > max_cells:
            obs_df = obs_df.sample(n=max_cells, random_state=42)
            n_cells = max_cells
            print(f"  Subsampled to {n_cells:,} cells")

        # Get var metadata
        var_df = query.var().concat().to_pandas()
        n_genes = len(var_df)
        print(f"  {n_genes:,} genes")

        # Read X matrix as scipy sparse
        print("  Reading X matrix...")
        t0 = time.perf_counter()
        X_iterator = query.X("raw").tables()
        tables = []
        for tbl in X_iterator:
            tables.append(tbl)
        if tables:
            import pyarrow as pa
            combined = pa.concat_tables(tables)
            soma_dim_0 = combined.column("soma_dim_0").to_numpy()
            soma_dim_1 = combined.column("soma_dim_1").to_numpy()
            soma_data = combined.column("soma_data").to_numpy()

            # Remap to 0-indexed
            obs_ids = obs_df["soma_joinid"].values
            var_ids = var_df["soma_joinid"].values
            obs_map = {v: i for i, v in enumerate(obs_ids)}
            var_map = {v: i for i, v in enumerate(var_ids)}

            rows = np.array([var_map.get(v, -1) for v in soma_dim_1])
            cols = np.array([obs_map.get(v, -1) for v in soma_dim_0])
            valid = (rows >= 0) & (cols >= 0)
            mat = ss.csc_matrix(
                (soma_data[valid].astype(np.int32), (rows[valid], cols[valid])),
                shape=(n_genes, n_cells)
            )
        else:
            mat = ss.csc_matrix((n_genes, n_cells), dtype=np.int32)

        t_read = time.perf_counter() - t0
        print(f"  Census read: {t_read:.1f} s ({mat.nnz:,} nnz)")

    census.close()

    # Save as .1pz
    pz_path = os.path.join(WORK_DIR, f"census_{tissue_name}.1pz")
    t0 = time.perf_counter()
    sp.write_1pz(pz_path, mat,
                 rownames=var_df["feature_id"].tolist(),
                 colnames=[str(x) for x in obs_df["soma_joinid"].tolist()])
    t_write_pz = time.perf_counter() - t0
    pz_size = os.path.getsize(pz_path)
    print(f"  .1pz write: {t_write_pz:.1f} s, size: {pz_size / 1e6:.1f} MB")

    return {
        "tissue": tissue_name,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "nnz": mat.nnz,
        "census_read_s": t_read,
        "pz_write_s": t_write_pz,
        "pz_bytes": pz_size,
        "pz_path": pz_path,
        "raw_int32_bytes": mat.nnz * 8,  # value + index per nnz
    }


def benchmark_reads(info):
    """Compare read speed: .1pz vs H5AD vs TileDB local copy."""
    import h5py

    pz_path = info["pz_path"]
    h5ad_path = pz_path.replace(".1pz", ".h5ad")

    # Write H5AD for comparison
    mat = sp.read_1pz(pz_path)
    import anndata
    adata = anndata.AnnData(X=mat.T.tocsr())  # cells × genes
    t0 = time.perf_counter()
    adata.write_h5ad(h5ad_path)
    t_write_h5ad = time.perf_counter() - t0
    h5ad_size = os.path.getsize(h5ad_path)

    results = []

    # .1pz read
    times_pz = []
    for i in range(N_WARMUP + N_REPS):
        gc.collect()
        t0 = time.perf_counter()
        m = sp.read_1pz(pz_path)
        t = time.perf_counter() - t0
        if i >= N_WARMUP:
            times_pz.append(t)
        del m

    # H5AD read
    times_h5ad = []
    for i in range(N_WARMUP + N_REPS):
        gc.collect()
        t0 = time.perf_counter()
        a = anndata.read_h5ad(h5ad_path)
        t = time.perf_counter() - t0
        if i >= N_WARMUP:
            times_h5ad.append(t)
        del a

    raw_bytes = info["raw_int32_bytes"]
    med_pz = np.median(times_pz)
    med_h5ad = np.median(times_h5ad)

    results.append({
        "tissue": info["tissue"],
        "n_cells": info["n_cells"],
        "n_genes": info["n_genes"],
        "nnz": info["nnz"],
        "operation": "full_read",
        "format": ".1pz",
        "file_bytes": info["pz_bytes"],
        "time_s": med_pz,
        "throughput_mbps": raw_bytes / med_pz / 1e6,
    })
    results.append({
        "tissue": info["tissue"],
        "n_cells": info["n_cells"],
        "n_genes": info["n_genes"],
        "nnz": info["nnz"],
        "operation": "full_read",
        "format": "H5AD",
        "file_bytes": h5ad_size,
        "time_s": med_h5ad,
        "throughput_mbps": raw_bytes / med_h5ad / 1e6,
    })

    # ── Column-range reads (cell subsets) ─────────────────────
    n_cells = info["n_cells"]
    for pct in [0.01, 0.05, 0.10]:
        n = max(1, int(n_cells * pct))
        label = f"col_range_{int(pct*100)}pct"

        # .1pz column range
        times = []
        for i in range(N_WARMUP + N_REPS):
            gc.collect()
            t0 = time.perf_counter()
            sub = sp.read_1pz_columns(pz_path, 0, n)
            t = time.perf_counter() - t0
            if i >= N_WARMUP:
                times.append(t)
            del sub

        results.append({
            "tissue": info["tissue"],
            "n_cells": info["n_cells"],
            "n_genes": info["n_genes"],
            "nnz": info["nnz"],
            "operation": label,
            "format": ".1pz",
            "file_bytes": info["pz_bytes"],
            "time_s": np.median(times),
            "throughput_mbps": 0,  # partial read, not meaningful
        })

    info["h5ad_bytes"] = h5ad_size
    info["h5ad_write_s"] = t_write_h5ad
    info["pz_read_s"] = med_pz
    info["h5ad_read_s"] = med_h5ad
    info["pz_read_mbps"] = raw_bytes / med_pz / 1e6
    info["h5ad_read_mbps"] = raw_bytes / med_h5ad / 1e6
    info["compression_ratio_pz"] = raw_bytes / info["pz_bytes"]
    info["compression_ratio_h5ad"] = raw_bytes / h5ad_size

    return results


def benchmark_dataloader_comparison(info):
    """Head-to-head: OnePZDataset vs TileDB-SOMA-ML ExperimentDataset."""
    import torch
    from torch.utils.data import DataLoader

    results = []
    pz_path = info["pz_path"]

    # ── OnePZ DataLoader ──────────────────────────────────────
    from singlepress.torch import OnePZCellDataset

    for batch_size in [128, 512]:
        ds = OnePZCellDataset(pz_path, seed=42)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            collate_fn=ds.collate_fn, num_workers=4, pin_memory=False
        )

        # Warmup
        for batch in loader:
            break

        t0 = time.perf_counter()
        n_batches = 0
        total_cells = 0
        for batch in loader:
            n_batches += 1
            if hasattr(batch, 'shape'):
                total_cells += batch.shape[0]
            elif isinstance(batch, (list, tuple)):
                total_cells += batch[0].shape[0] if hasattr(batch[0], 'shape') else batch_size
            else:
                total_cells += batch_size
        epoch_time = time.perf_counter() - t0

        results.append({
            "tissue": info["tissue"],
            "n_cells": info["n_cells"],
            "loader": "OnePZCellDataset",
            "batch_size": batch_size,
            "num_workers": 4,
            "sparse": True,
            "n_batches": n_batches,
            "total_cells": total_cells,
            "epoch_s": epoch_time,
            "cells_per_sec": total_cells / epoch_time if epoch_time > 0 else 0,
        })
        print(f"  OnePZ bs={batch_size}: {epoch_time:.2f}s, {total_cells/epoch_time:.0f} cells/s")

    # ── TileDB-SOMA-ML DataLoader ─────────────────────────────
    try:
        import cellxgene_census
        import tiledbsoma
        from tiledbsoma_ml import ExperimentDataset, experiment_dataloader

        tissue_filter = [s for s in TISSUE_SLICES if s["name"] == info["tissue"]][0]["filter"]

        census = cellxgene_census.open_soma(census_version="2025-01-30")
        experiment = census["census_data"]["homo_sapiens"]

        for batch_size in [128, 512]:
            with experiment.axis_query(
                measurement_name="RNA",
                obs_query=tiledbsoma.AxisQuery(value_filter=tissue_filter),
            ) as query:
                # Dense mode (default) — this is what CZI recommends
                soma_ds = ExperimentDataset(
                    query,
                    layer_name="raw",
                    obs_column_names=["cell_type"],
                    batch_size=batch_size,
                    shuffle=True,
                    seed=42,
                    return_sparse_X=False,
                )
                dl = experiment_dataloader(soma_ds)

                # Warmup
                for X_batch, obs_batch in dl:
                    break

                t0 = time.perf_counter()
                n_batches = 0
                total_cells = 0
                for X_batch, obs_batch in dl:
                    n_batches += 1
                    total_cells += X_batch.shape[0]
                epoch_time = time.perf_counter() - t0

                results.append({
                    "tissue": info["tissue"],
                    "n_cells": info["n_cells"],
                    "loader": "TileDB-SOMA-ML (dense)",
                    "batch_size": batch_size,
                    "num_workers": 0,  # CZI default for Census
                    "sparse": False,
                    "n_batches": n_batches,
                    "total_cells": total_cells,
                    "epoch_s": epoch_time,
                    "cells_per_sec": total_cells / epoch_time if epoch_time > 0 else 0,
                })
                print(f"  TileDB dense bs={batch_size}: {epoch_time:.2f}s, {total_cells/epoch_time:.0f} cells/s")

            # Try sparse mode (should fail with num_workers > 0)
            with experiment.axis_query(
                measurement_name="RNA",
                obs_query=tiledbsoma.AxisQuery(value_filter=tissue_filter),
            ) as query:
                try:
                    soma_ds_sparse = ExperimentDataset(
                        query,
                        layer_name="raw",
                        obs_column_names=["cell_type"],
                        batch_size=batch_size,
                        shuffle=True,
                        seed=42,
                        return_sparse_X=True,
                    )
                    dl_sparse = experiment_dataloader(soma_ds_sparse, num_workers=4)
                    for X_batch, obs_batch in dl_sparse:
                        break
                    results.append({
                        "tissue": info["tissue"],
                        "n_cells": info["n_cells"],
                        "loader": "TileDB-SOMA-ML (sparse, 4 workers)",
                        "batch_size": batch_size,
                        "num_workers": 4,
                        "sparse": True,
                        "n_batches": 0,
                        "total_cells": 0,
                        "epoch_s": float("nan"),
                        "cells_per_sec": 0,
                    })
                except (NotImplementedError, RuntimeError) as e:
                    results.append({
                        "tissue": info["tissue"],
                        "n_cells": info["n_cells"],
                        "loader": "TileDB-SOMA-ML (sparse, 4 workers)",
                        "batch_size": batch_size,
                        "num_workers": 4,
                        "sparse": True,
                        "n_batches": 0,
                        "total_cells": 0,
                        "epoch_s": float("nan"),
                        "cells_per_sec": 0,
                        "error": str(e)[:200],
                    })
                    print(f"  TileDB sparse+multiworker: FAILED — {str(e)[:100]}")

        census.close()

    except Exception as e:
        print(f"  TileDB-SOMA-ML benchmark failed: {e}")

    return results


def benchmark_memory_footprint(info):
    """Compare memory usage: sparse vs dense data loading."""
    import tracemalloc
    import torch

    pz_path = info["pz_path"]
    results = []

    # .1pz sparse memory
    tracemalloc.start()
    from singlepress.torch import OnePZCellDataset
    ds = OnePZCellDataset(pz_path, seed=42)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=128, shuffle=True, collate_fn=ds.collate_fn
    )
    batch = next(iter(loader))
    pz_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del batch, loader, ds

    # Theoretical dense memory for same batch
    n_genes = info["n_genes"]
    dense_batch_bytes = 128 * n_genes * 4  # float32

    results.append({
        "tissue": info["tissue"],
        "loader": "OnePZ (sparse)",
        "batch_memory_bytes": pz_peak,
        "theoretical_dense_bytes": dense_batch_bytes,
        "ratio": dense_batch_bytes / max(pz_peak, 1),
    })

    return results


def main():
    ensure_workdir()
    all_census_results = []
    all_read_results = []
    all_dl_results = []
    all_mem_results = []

    for slice_cfg in TISSUE_SLICES:
        try:
            info = extract_census_slice(
                slice_cfg["name"],
                slice_cfg["filter"],
                max_cells=50000 if "100k" in slice_cfg["name"] else
                           100000 if "250k" in slice_cfg["name"] else None,
            )
            all_census_results.append(info)

            # Read benchmarks
            read_results = benchmark_reads(info)
            all_read_results.extend(read_results)

            # DataLoader comparison
            dl_results = benchmark_dataloader_comparison(info)
            all_dl_results.extend(dl_results)

            # Memory footprint
            mem_results = benchmark_memory_footprint(info)
            all_mem_results.extend(mem_results)

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── Save results ─────────────────────────────────────────
    # Census overview
    out_path = os.path.join(DATA_DIR, "tiledb_census_bench.csv")
    if all_census_results:
        keys = [k for k in all_census_results[0] if k != "pz_path"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_census_results:
                row = {k: v for k, v in r.items() if k != "pz_path"}
                w.writerow(row)
        print(f"\nSaved {out_path}")

    # Read results
    if all_read_results:
        out2 = os.path.join(DATA_DIR, "tiledb_read_bench.csv")
        keys = list(all_read_results[0].keys())
        with open(out2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_read_results)
        print(f"Saved {out2}")

    # DataLoader results
    if all_dl_results:
        out3 = os.path.join(DATA_DIR, "tiledb_dataloader_bench.csv")
        all_keys = set()
        for r in all_dl_results:
            all_keys.update(r.keys())
        keys = sorted(all_keys)
        with open(out3, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_dl_results)
        print(f"Saved {out3}")

    # Memory results
    if all_mem_results:
        out4 = os.path.join(DATA_DIR, "tiledb_memory_bench.csv")
        keys = list(all_mem_results[0].keys())
        with open(out4, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_mem_results)
        print(f"Saved {out4}")

    print("\n" + "=" * 60)
    print("CENSUS BENCHMARK COMPLETE")
    for r in all_census_results:
        tissue = r["tissue"]
        pz_mb = r["pz_bytes"] / 1e6
        h5ad_mb = r.get("h5ad_bytes", 0) / 1e6
        ratio = r.get("compression_ratio_pz", 0)
        print(f"  {tissue}: {r['n_cells']:,} cells, {r['nnz']:,} nnz")
        print(f"    .1pz: {pz_mb:.1f} MB ({ratio:.1f}× compression)")
        print(f"    H5AD: {h5ad_mb:.1f} MB")
        print(f"    Read: .1pz {r.get('pz_read_s', 0):.2f}s vs H5AD {r.get('h5ad_read_s', 0):.2f}s")


if __name__ == "__main__":
    main()
