#!/usr/bin/env python3
"""GPU FLOP sweep: GPU utilization vs. model compute intensity.
Produces data for Figure 2 panels (e) and (f).

Sweeps model FLOPs per training step and shows how GPU utilization scales
with compute intensity for .1pz (DataLoader with multi-worker prefetch)
vs H5AD (backed-mode sequential reads).

Re-encodes .1pz datasets with the CURRENT codec before benchmarking.

Output: code/data/gpu_flop_sweep.csv

Usage (H100):
    sbatch --partition=gpu --gres=gpu:1 --constraint=nvidia_h100_nvl \\
           --time=120 --mem=64G --cpus-per-task=8 \\
           --wrap='source /mnt/home/debruinz/venv/bin/activate && \\
                   cd /tmp && python3 -u <this_script>'
"""

import os, sys, time, csv, gc, shutil
import numpy as np

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if "" in sys.path:
    sys.path.remove("")

import torch
import torch.nn as nn
import pandas as pd
import scipy.sparse as ss
import anndata as ad
import singlepress as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
TMPFS = "/dev/shm"

# ── Parameters ─────────────────────────────────────────────────────
HIDDEN_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
N_STEPS = 100
N_WARMUP = 10
BATCH_SIZE = 1024
NUM_WORKERS = 4
N_FEATURES = 20_000


# ── Dataset selection ──────────────────────────────────────────────
def select_datasets():
    """Select small/medium/large mouse datasets from the survey."""
    survey_path = os.path.join(CODE_DATA_DIR, "all_datasets_survey.csv")
    survey = pd.read_csv(survey_path)
    survey_m = survey[(survey["nrows"] == 171540) &
                      (survey["ncols"] >= BATCH_SIZE * 5)].copy()

    def pick(df, target, used, tol=0.5):
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
    for label, target in [("small", 10_000), ("medium", 50_000),
                           ("large", 150_000)]:
        row = pick(survey_m, target, used)
        if row is not None:
            used.add(row["gse_id"])
            datasets.append(dict(
                label=label, gse_id=row["gse_id"],
                n_genes=int(row["nrows"]), n_cells=int(row["ncols"]),
                nnz=int(row["nnz"]),
            ))
            print(f"  {label}: {row['gse_id']}  "
                  f"{int(row['ncols']):,} cells x {int(row['nrows']):,} genes")
    return datasets


def reencode(gse_id):
    """Re-encode a production .1pz with the current codec to tmpfs."""
    src = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
    dst = os.path.join(TMPFS, f"gpu_bench_{gse_id}.1pz")
    if os.path.exists(dst):
        return dst
    print(f"  Re-encoding {gse_id}...", end="", flush=True)
    mat = sp.read_1pz(src)
    pz = sp.open_1pz(src)
    sp.write_1pz(dst, mat.tocsc(),
                  rownames=list(pz.rownames) if pz.rownames else [],
                  colnames=list(pz.colnames) if pz.colnames else [])
    del mat, pz; gc.collect()
    print(f" {os.path.getsize(dst)/1e6:.1f} MB")
    return dst


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


# ── .1pz Dataset (returns dense tensors -- worker-safe) ────────────
class PZDenseDataset(torch.utils.data.Dataset):
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


def main():
    print(f"GPU FLOP sweep benchmark")
    print(f"singlepress {getattr(sp, '__version__', 'dev')}")

    datasets = select_datasets()
    if not datasets:
        sys.exit("ERROR: no datasets found")

    device_name = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "CPU")
    print(f"\nDevice: {device_name}")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Feature dim: {N_FEATURES}  Hidden dims: {HIDDEN_DIMS}")
    print(f"Batch: {BATCH_SIZE}  Steps/config: {N_STEPS}")
    print(f"Configurations: {len(datasets) * len(HIDDEN_DIMS) * 2}")

    os.makedirs(CODE_DATA_DIR, exist_ok=True)
    out_csv = os.path.join(CODE_DATA_DIR, "gpu_flop_sweep.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        f.flush()

        for ds in datasets:
            gse = ds["gse_id"]
            n_cells = ds["n_cells"]

            print(f"\n{'='*65}")
            print(f"{ds['label'].upper()}: {gse}  ({n_cells:,} cells)")

            # Re-encode with current codec
            local_pz = reencode(gse)

            # Identify top-expressed genes
            print("  Identifying features... ", end="", flush=True)
            mat_full = sp.read_1pz(local_pz)
            gene_totals = np.array(mat_full.sum(axis=1)).ravel()
            gene_idx = np.sort(np.argsort(gene_totals)[-N_FEATURES:])
            n_feat = len(gene_idx)
            print(f"{n_feat} genes selected")

            # Create H5AD in tmpfs
            h5ad_path = os.path.join(TMPFS, f"gpu_bench_{gse}.h5ad")
            if not os.path.exists(h5ad_path):
                print(f"  Writing H5AD...", end="", flush=True)
                t0 = time.time()
                adata_w = ad.AnnData(X=mat_full.T)
                adata_w.write_h5ad(h5ad_path)
                del adata_w
                print(f" done ({time.time()-t0:.1f}s)")

            del mat_full; gc.collect()
            adata = ad.read_h5ad(h5ad_path, backed="r")

            for hidden_dim in HIDDEN_DIMS:
                F = calc_flops(n_feat, hidden_dim, BATCH_SIZE)
                print(f"\n  h={hidden_dim:>5d}   FLOPs/step={F/1e9:>7.1f}G")

                # .1pz
                try:
                    prep, comp, cells = bench_1pz(local_pz, gene_idx,
                                                  hidden_dim)
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
                          f"compute={comp:.2f}s  "
                          f"({cells/max(tot,1e-9):,.0f} c/s)")
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
                          f"compute={comp:.2f}s  "
                          f"({cells/max(tot,1e-9):,.0f} c/s)")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"    h5ad   FAILED: {e}")

            adata.file.close(); del adata

            # Cleanup tmpfs
            for p in [local_pz, h5ad_path]:
                if os.path.exists(p):
                    os.remove(p)
            gc.collect(); torch.cuda.empty_cache()
            print(f"\n  Done with {gse}")

    print(f"\n{'='*65}")
    print(f"Results: {out_csv}")
    df = pd.read_csv(out_csv)
    for fmt in ["1pz", "h5ad"]:
        sub = df[df["format"] == fmt]
        if sub.empty:
            continue
        print(f"\n  {fmt}: GPU util {sub['gpu_util_pct'].min():.1f}%"
              f" - {sub['gpu_util_pct'].max():.1f}%")
        for _, r in sub.sort_values("flops_per_step").iterrows():
            if r["gpu_util_pct"] >= 50:
                print(f"    50% crossover at h={int(r['hidden_dim'])} "
                      f"({r['flops_per_step']/1e9:.0f} GFLOPs)")
                break


if __name__ == "__main__":
    main()
