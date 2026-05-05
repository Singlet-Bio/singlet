#!/usr/bin/env python3
"""GPU FLOP sweep: GPU utilization vs. model compute intensity.

Instead of reporting GPU utilization for one arbitrary autoencoder, this
benchmark sweeps model FLOPs per training step and shows how GPU utilization
scales with compute intensity for each file format.

Design:
  - .1pz: uses DataLoader with multi-worker prefetch (PZDenseDataset returns
    dense tensors from workers, avoiding sparse-tensor IPC issues)
  - H5AD: uses manual backed-mode slicing (no multi-worker prefetch available)
  - This mirrors real-world usage: .1pz users get DataLoader parallelism;
    H5AD users are stuck with sequential reads
  - FLOPs per training step = 12 × batch_size × n_features × hidden_dim
    (two dense linear layers, forward + backward)

Usage (H100):
    sbatch --partition=gpu --gres=gpu:1 --constraint=nvidia_h100_nvl \\
           --time=120 --mem=64G --cpus-per-task=8 \\
           --wrap='source /mnt/home/debruinz/venv/bin/activate && \\
                   cd /tmp && python3 -u \\
                   /mnt/home/debruinz/Singlet-AI/papers/manuscripts/singlepress-format/benchmark_gpu_flops.py'
"""

import os, sys, time, csv, gc, tempfile, shutil
import numpy as np

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if "" in sys.path:
    sys.path.remove("")

import torch
# Use default 'fork' start method (compatible with batch scripts).
# PZDenseDataset returns dense tensors, so no sparse-IPC issues.

import torch.nn as nn
import pandas as pd
import scipy.sparse as ss
import anndata as ad
import singlepress as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
OUT_CSV = os.path.join(SCRIPT_DIR, "gpu_flop_sweep.csv")

# ── Parameters ─────────────────────────────────────────────────────
HIDDEN_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
N_STEPS = 100
N_WARMUP = 10
BATCH_SIZE = 1024
NUM_WORKERS = 4
N_FEATURES = 20_000  # gene subsetting for realistic input dim


# ── Dataset selection ──────────────────────────────────────────────
survey = pd.read_csv(os.path.join(SCRIPT_DIR, "all_datasets_survey.csv"))
survey_m = survey[(survey["nrows"] == 171540) &
                  (survey["ncols"] >= BATCH_SIZE * 5)].copy()


def pick_dataset(df, target, used, tol=0.5):
    lo, hi = int(target * (1 - tol)), int(target * (1 + tol))
    c = df[(df["ncols"] >= lo) & (df["ncols"] <= hi) &
           ~df["gse_id"].isin(used)].copy()
    c["d"] = abs(c["ncols"] - target)
    for _, row in c.sort_values("d").iterrows():
        pz = os.path.join(QUANT_DIR, row["gse_id"], "counts.1pz")
        if os.path.exists(pz):
            return row
    return None


used = set()
datasets = []
for label, target in [("small", 10_000), ("medium", 50_000), ("large", 150_000)]:
    row = pick_dataset(survey_m, target, used)
    if row is not None:
        used.add(row["gse_id"])
        datasets.append(dict(
            label=label, gse_id=row["gse_id"],
            n_genes=int(row["nrows"]), n_cells=int(row["ncols"]),
            nnz=int(row["nnz"]),
        ))
        print(f"  {label}: {row['gse_id']}  "
              f"{int(row['ncols']):,} cells x {int(row['nrows']):,} genes")
    else:
        print(f"  {label}: no dataset found near {target:,}")

if not datasets:
    sys.exit("ERROR: no datasets found")

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"\nDevice: {device_name}")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"Feature dim: {N_FEATURES}  Hidden dims: {HIDDEN_DIMS}")
print(f"Batch: {BATCH_SIZE}  Steps/config: {N_STEPS}")
print(f"Configurations: {len(datasets) * len(HIDDEN_DIMS) * 2}")


# ── Model ──────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, n_hid), nn.ReLU())
        self.dec = nn.Linear(n_hid, n_in)

    def forward(self, x):
        return self.dec(self.enc(x))


def calc_flops(n_feat, h, B):
    return 12 * B * n_feat * h


# ── .1pz Dataset (returns dense tensors — worker-safe) ────────────
class PZDenseDataset(torch.utils.data.Dataset):
    """Read .1pz chunks in workers, return dense [B, n_features] tensors."""

    def __init__(self, pz_path, batch_size, gene_idx):
        info = sp.info_1pz(pz_path)
        self.pz_path = pz_path
        self.batch_size = batch_size
        self.gene_idx = gene_idx
        self.n_chunks = info["n"] // batch_size

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.batch_size
        mat = sp.read_1pz_columns(self.pz_path, start,
                                  start + self.batch_size, num_threads=1)
        sub = mat[self.gene_idx, :]
        arr = sub.toarray().T
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


# ── Bench: .1pz with DataLoader ───────────────────────────────────
def bench_1pz(pz_path, gene_idx, hidden_dim):
    n_feat = len(gene_idx)
    ds = PZDenseDataset(pz_path, BATCH_SIZE, gene_idx)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    model = Autoencoder(n_feat, hidden_dim).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    step = 0
    prep_times, compute_times = [], []

    n_epochs = max(2, (N_WARMUP + N_STEPS) // max(len(ds), 1) + 2)
    for _epoch in range(n_epochs):
        t_wait = time.perf_counter()
        for x_cpu in loader:
            t_got = time.perf_counter()

            x = x_cpu.cuda(non_blocking=True)
            torch.cuda.synchronize()
            t_gpu = time.perf_counter()

            loss = crit(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize()
            t_done = time.perf_counter()

            if step >= N_WARMUP:
                prep_times.append(t_gpu - t_wait)
                compute_times.append(t_done - t_gpu)

            step += 1
            if step >= N_WARMUP + N_STEPS:
                break
            t_wait = time.perf_counter()

        if step >= N_WARMUP + N_STEPS:
            break

    total_cells = N_STEPS * BATCH_SIZE
    del model, opt, crit, loader, ds
    gc.collect(); torch.cuda.empty_cache()
    return sum(prep_times), sum(compute_times), total_cells


# ── Bench: H5AD backed mode ───────────────────────────────────────
def bench_h5ad(adata, n_cells, gene_idx, hidden_dim):
    n_feat = len(gene_idx)
    model = Autoencoder(n_feat, hidden_dim).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    starts = list(range(0, n_cells - BATCH_SIZE + 1, BATCH_SIZE))
    si = 0
    step = 0
    prep_times, compute_times = [], []

    while step < N_WARMUP + N_STEPS:
        s = starts[si % len(starts)]
        si += 1

        t0 = time.perf_counter()
        chunk = adata[s : s + BATCH_SIZE].X
        if ss.issparse(chunk):
            arr = chunk[:, gene_idx].toarray()
        else:
            arr = np.asarray(chunk[:, gene_idx])
        x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        x = x.cuda(non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss = crit(model(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        if step >= N_WARMUP:
            prep_times.append(t1 - t0)
            compute_times.append(t2 - t1)
        step += 1

    del model, opt, crit
    gc.collect(); torch.cuda.empty_cache()
    return sum(prep_times), sum(compute_times), N_STEPS * BATCH_SIZE


# ── Main ───────────────────────────────────────────────────────────
FIELDS = [
    "gse_id", "label", "n_features", "n_cells", "nnz",
    "format", "hidden_dim", "flops_per_step",
    "n_steps", "total_cells",
    "prep_s", "compute_s", "total_s",
    "gpu_util_pct", "cells_per_sec", "device",
]

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    f.flush()

    for ds in datasets:
        gse = ds["gse_id"]
        n_cells = ds["n_cells"]
        pz_path = os.path.join(QUANT_DIR, gse, "counts.1pz")

        print(f"\n{'='*65}")
        print(f"{ds['label'].upper()}: {gse}  ({n_cells:,} cells)")

        # Identify top-expressed genes
        print("  Identifying features... ", end="", flush=True)
        mat_full = sp.read_1pz(pz_path)
        gene_totals = np.array(mat_full.sum(axis=1)).ravel()
        gene_idx = np.sort(np.argsort(gene_totals)[-N_FEATURES:])
        n_feat = len(gene_idx)
        print(f"{n_feat} genes selected")

        # Copy .1pz to /tmp for consistent I/O
        local_pz = f"/tmp/{gse}_counts.1pz"
        if not os.path.exists(local_pz):
            print(f"  Copying .1pz to /tmp... ", end="", flush=True)
            shutil.copy2(pz_path, local_pz)
            print("done")

        # Create H5AD in /tmp
        h5ad_path = f"/tmp/{gse}.h5ad"
        if not os.path.exists(h5ad_path):
            print(f"  Writing H5AD to /tmp... ", end="", flush=True)
            t0 = time.time()
            adata_w = ad.AnnData(X=mat_full.T)
            adata_w.write_h5ad(h5ad_path)
            del adata_w
            print(f"done ({time.time()-t0:.1f}s)")

        del mat_full; gc.collect()
        adata = ad.read_h5ad(h5ad_path, backed="r")

        for hidden_dim in HIDDEN_DIMS:
            F = calc_flops(n_feat, hidden_dim, BATCH_SIZE)
            print(f"\n  h={hidden_dim:>5d}   FLOPs/step={F/1e9:>7.1f}G")

            # .1pz
            try:
                prep, comp, cells = bench_1pz(local_pz, gene_idx, hidden_dim)
                tot = prep + comp
                pct = 100 * comp / max(tot, 1e-9)
                row = dict(
                    gse_id=gse, label=ds["label"], n_features=n_feat,
                    n_cells=n_cells, nnz=ds["nnz"],
                    format="1pz", hidden_dim=hidden_dim,
                    flops_per_step=F, n_steps=N_STEPS, total_cells=cells,
                    prep_s=round(prep, 4), compute_s=round(comp, 4),
                    total_s=round(tot, 4),
                    gpu_util_pct=round(pct, 1),
                    cells_per_sec=round(cells / max(tot, 1e-9)),
                    device=device_name,
                )
                writer.writerow(row); f.flush()
                print(f"    .1pz   GPU {pct:5.1f}%  prep={prep:.2f}s  "
                      f"compute={comp:.2f}s  ({cells/max(tot,1e-9):,.0f} c/s)")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"    .1pz   FAILED: {e}")

            # H5AD
            try:
                prep, comp, cells = bench_h5ad(
                    adata, n_cells, gene_idx, hidden_dim)
                tot = prep + comp
                pct = 100 * comp / max(tot, 1e-9)
                row = dict(
                    gse_id=gse, label=ds["label"], n_features=n_feat,
                    n_cells=n_cells, nnz=ds["nnz"],
                    format="h5ad", hidden_dim=hidden_dim,
                    flops_per_step=F, n_steps=N_STEPS, total_cells=cells,
                    prep_s=round(prep, 4), compute_s=round(comp, 4),
                    total_s=round(tot, 4),
                    gpu_util_pct=round(pct, 1),
                    cells_per_sec=round(cells / max(tot, 1e-9)),
                    device=device_name,
                )
                writer.writerow(row); f.flush()
                print(f"    h5ad   GPU {pct:5.1f}%  prep={prep:.2f}s  "
                      f"compute={comp:.2f}s  ({cells/max(tot,1e-9):,.0f} c/s)")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"    h5ad   FAILED: {e}")

        adata.file.close(); del adata
        gc.collect(); torch.cuda.empty_cache()
        print(f"\n  Done with {gse}")

print(f"\n{'='*65}")
print(f"Results: {OUT_CSV}")
df = pd.read_csv(OUT_CSV)
for fmt in ["1pz", "h5ad"]:
    sub = df[df["format"] == fmt]
    if sub.empty: continue
    print(f"\n  {fmt}: GPU util {sub['gpu_util_pct'].min():.1f}%"
          f" - {sub['gpu_util_pct'].max():.1f}%")
    for _, r in sub.sort_values("flops_per_step").iterrows():
        if r["gpu_util_pct"] >= 50:
            print(f"    50% crossover at h={int(r['hidden_dim'])} "
                  f"({r['flops_per_step']/1e9:.0f} GFLOPs)")
            break
