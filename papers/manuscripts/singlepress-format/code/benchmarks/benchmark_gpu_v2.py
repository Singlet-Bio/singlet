#!/usr/bin/env python3
"""Expanded GPU benchmark: .1pz vs H5AD autoencoder training.

Selects ~50 datasets spanning 1K-1M cells.
For each: runs 3-epoch autoencoder, times load vs compute for both formats.
Output: gpu_benchmark_v2.csv

Usage (H100):
    sbatch --partition=gpu --gres=gpu:1 --constraint=h100 \
           --time=360 --mem=128G --cpus-per-task=8 \
           --wrap='source /mnt/home/debruinz/venv/bin/activate && \
                   cd /tmp && python3 -u /path/to/benchmark_gpu_v2.py'
"""
import os, sys, time, csv, gc, tempfile
import numpy as np

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if '' in sys.path:
    sys.path.remove('')

import torch
import torch.nn as nn
import pandas as pd
import scipy.sparse as ss
import anndata as ad
import singlepress as sp
from singlepress.torch import OnePZDataset, collate_sparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
OUT = os.path.join(SCRIPT_DIR, "gpu_benchmark_v2.csv")
N_EPOCHS = 3
BATCH_CELLS = 1024  # cells per mini-batch
NUM_WORKERS = 4
TARGET_N = 100  # aim for ~100 datasets

# ── Dataset selection ────────────────────────────────────────────
survey = pd.read_csv(os.path.join(SCRIPT_DIR, "all_datasets_survey.csv"))
# Filter: must have counts.1pz, 1K-1M cells, >1M nnz
cands = survey[
    (survey["ncols"] >= 1_000) &
    (survey["ncols"] <= 1_000_000) &
    (survey["nnz"] > 1_000_000)
].sort_values("ncols")

# Evenly sample across the cell-count range
step = max(1, len(cands) // TARGET_N)
picked = cands.iloc[::step].head(TARGET_N)
datasets = []
for _, row in picked.iterrows():
    pz_path = os.path.join(QUANT_DIR, row["gse_id"], "counts.1pz")
    if os.path.exists(pz_path):
        datasets.append((row["gse_id"], int(row["nnz"]),
                         int(row["ncols"]), int(row["nrows"])))

print(f"Selected {len(datasets)} datasets ({picked.shape[0]} candidates)")
print(f"Cell range: {min(d[2] for d in datasets):,} – {max(d[2] for d in datasets):,}")

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Device: {device_name}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def benchmark_1pz(gse_id, n_genes, n_cells):
    """Benchmark .1pz loading + autoencoder training."""
    pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
    chunk_size = min(1024, max(64, n_cells // 20))

    ds = OnePZDataset(pz_path, chunk_size=chunk_size, normalize=True,
                      dtype="float32", sparse_format="csr")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=max(1, BATCH_CELLS // chunk_size),
        shuffle=True, collate_fn=collate_sparse,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model = SparseAutoencoder(n_genes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Warmup
    for batch in loader:
        x = batch.to_dense().T.cuda(non_blocking=True)
        torch.cuda.synchronize()
        break

    load_times, compute_times = [], []
    total_cells = 0
    t_load_start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        for batch_sparse in loader:
            t_load_end = time.perf_counter()
            load_times.append(t_load_end - t_load_start)

            t_comp_start = time.perf_counter()
            x = batch_sparse.to_dense().T.cuda(non_blocking=True)
            torch.cuda.synchronize()
            output = model(x)
            loss = criterion(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            compute_times.append(time.perf_counter() - t_comp_start)
            total_cells += x.shape[0]

            t_load_start = time.perf_counter()

    total_load = sum(load_times)
    total_compute = sum(compute_times)
    total = total_load + total_compute

    del model, optimizer, loader, ds
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "format": "1pz", "gse_id": gse_id, "n_genes": n_genes,
        "n_cells": n_cells, "n_epochs": N_EPOCHS,
        "total_cells": total_cells,
        "total_s": round(total, 3),
        "load_s": round(total_load, 3),
        "compute_s": round(total_compute, 3),
        "load_pct": round(100 * total_load / max(total, 1e-9), 1),
        "compute_pct": round(100 * total_compute / max(total, 1e-9), 1),
        "cells_per_sec": round(total_cells / max(total, 1e-9), 0),
        "device": device_name,
    }


def benchmark_h5ad(gse_id, n_genes, n_cells, h5ad_path):
    """Benchmark H5AD loading + autoencoder training via anndata backed mode."""
    model = SparseAutoencoder(n_genes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Pre-compute batch indices
    batch_size = BATCH_CELLS
    indices = list(range(0, n_cells, batch_size))

    # Warmup
    adata = ad.read_h5ad(h5ad_path, backed='r')
    chunk = adata[0:min(batch_size, n_cells)].X
    if ss.issparse(chunk):
        chunk = chunk.toarray()
    x = torch.tensor(chunk, dtype=torch.float32).cuda()
    torch.cuda.synchronize()
    del x, chunk

    load_times, compute_times = [], []
    total_cells = 0

    for epoch in range(N_EPOCHS):
        for start in indices:
            end = min(start + batch_size, n_cells)

            t_load_start = time.perf_counter()
            chunk = adata[start:end].X
            if ss.issparse(chunk):
                chunk = chunk.toarray()
            x = torch.tensor(np.ascontiguousarray(chunk), dtype=torch.float32)
            x = x.cuda(non_blocking=True)
            torch.cuda.synchronize()
            t_load_end = time.perf_counter()
            load_times.append(t_load_end - t_load_start)

            t_comp_start = time.perf_counter()
            output = model(x)
            loss = criterion(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            compute_times.append(time.perf_counter() - t_comp_start)
            total_cells += x.shape[0]

    adata.file.close()
    total_load = sum(load_times)
    total_compute = sum(compute_times)
    total = total_load + total_compute

    del model, optimizer, adata
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "format": "h5ad", "gse_id": gse_id, "n_genes": n_genes,
        "n_cells": n_cells, "n_epochs": N_EPOCHS,
        "total_cells": total_cells,
        "total_s": round(total, 3),
        "load_s": round(total_load, 3),
        "compute_s": round(total_compute, 3),
        "load_pct": round(100 * total_load / max(total, 1e-9), 1),
        "compute_pct": round(100 * total_compute / max(total, 1e-9), 1),
        "cells_per_sec": round(total_cells / max(total, 1e-9), 0),
        "device": device_name,
    }


# ── Main ─────────────────────────────────────────────────────────
results = []
tmpdir = tempfile.mkdtemp(prefix="gpu_bench_")

for i, (gse_id, nnz, ncols, nrows) in enumerate(datasets):
    print(f"\n[{i+1}/{len(datasets)}] {gse_id}: {ncols:,} cells, {nrows:,} genes, {nnz:,} nnz")

    try:
        # ── .1pz benchmark ──
        r_pz = benchmark_1pz(gse_id, nrows, ncols)
        results.append(r_pz)
        print(f"  .1pz:  total={r_pz['total_s']:.1f}s  "
              f"load={r_pz['load_pct']:.0f}%  compute={r_pz['compute_pct']:.0f}%  "
              f"{r_pz['cells_per_sec']:.0f} cells/s")

        # ── Convert to H5AD ──
        h5ad_path = os.path.join(tmpdir, f"{gse_id}.h5ad")
        if not os.path.exists(h5ad_path):
            pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
            mat = sp.read_1pz(pz_path)
            adata = ad.AnnData(X=mat.T)  # cells × genes
            adata.write_h5ad(h5ad_path)
            del mat, adata
            gc.collect()
            print(f"  Converted to H5AD: {os.path.getsize(h5ad_path)/1e6:.1f} MB")

        # ── H5AD benchmark ──
        r_h5 = benchmark_h5ad(gse_id, nrows, ncols, h5ad_path)
        results.append(r_h5)
        print(f"  H5AD:  total={r_h5['total_s']:.1f}s  "
              f"load={r_h5['load_pct']:.0f}%  compute={r_h5['compute_pct']:.0f}%  "
              f"{r_h5['cells_per_sec']:.0f} cells/s")

        speedup = r_h5['total_s'] / max(r_pz['total_s'], 0.01)
        print(f"  → .1pz {speedup:.1f}× faster")

        # Clean up H5AD to save disk
        os.remove(h5ad_path)

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Periodic save
    if len(results) >= 2:
        df = pd.DataFrame(results)
        df.to_csv(OUT, index=False)

# Final save
if results:
    df = pd.DataFrame(results)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(results)} rows to {OUT}")

    # Summary
    pz_rows = [r for r in results if r["format"] == "1pz"]
    h5_rows = [r for r in results if r["format"] == "h5ad"]
    if pz_rows:
        print(f"\n.1pz: median compute%={np.median([r['compute_pct'] for r in pz_rows]):.0f}%")
    if h5_rows:
        print(f"H5AD: median compute%={np.median([r['compute_pct'] for r in h5_rows]):.0f}%")

# Cleanup temp dir
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
