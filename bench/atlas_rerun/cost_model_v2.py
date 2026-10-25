"""Clipper-only quote, present vs optimized pipeline, from the perf matrix and download benchmarks.

    python cost_model_v2.py <universe.parquet>

Measured on PBMC 3' v3 (66.6M pairs, Xeon Gold 6230R, 4 threads; bench/atlas_rerun/perf_*):
  compute wall = F + s * U        F = 300 s, s = 12.3 s/M without the non-host screen,
                                  16.8 s/M with it (campaign default; +27%, 53 GB RSS)
  packing      x 1.15             4 concurrent samples per 16 cores ran 15% slower than one alone
  present download  15 s/M held inside the 4-core job (campaign logs: remote `singlet download`,
                    median 0.065M reads/s)
  optimized download separate 1-core I/O stage: samples under the cap are fetched whole with prefetch
                    (~20 MB/s per stream) and converted locally at 1M reads/s; capped samples stream
                    only their share of each run (remote `singlet download --max-reads`, 0.065M
                    reads/s per stream). Decoded reads are identical either way.
  aggregate bandwidth into Clipper: ~150 MB/s whatever the source (NCBI 125-160 MB/s, AWS S3 116 MB/s)
  SRA bytes: 0.44 B/base
U = read pairs used per sample (millions) = min(reads, cap); 20% retry margin on everything.
"""

import sys

import numpy as np
import pandas as pd

u = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "universe.parquet")
u = u[u.route == "gex"].copy()
u["reads"] = u.reads.fillna(u.reads.median())
u["bases"] = u.bases.fillna(u.bases.median())
bpr = u.bases / u.reads                     # bases per read pair, per sample

F, PACK, RETRY = 300.0, 1.15, 1.2
S = {"screen": 16.8, "noscreen": 12.3}
DL_PRESENT = 15.0                           # s/M, 4 cores held
BW_AGG = 150e6                              # B/s aggregate
SRA_BPB = 0.44
STREAM = 20e6                               # B/s per prefetch stream (under aggregate load)
CONVERT = 1.0                               # s/M, 1 core

rows = []
for cap in [None, 300e6, 200e6, 150e6, 100e6, 50e6]:
    U = (u.reads if cap is None else np.minimum(u.reads, cap)) / 1e6
    dl_bytes = U * 1e6 * bpr * SRA_BPB
    for mode in ("present", "optimized"):
        for screen in ("screen", "noscreen"):
            if mode == "present" and screen == "noscreen":
                continue
            wall = F + S[screen] * U + (DL_PRESENT * U if mode == "present" else 0.0)
            core_h = 4 * wall * PACK / 3600
            if mode == "optimized":
                capped = (u.reads / 1e6) > U
                io_s = np.where(capped, U * 1e6 / 65e3, dl_bytes / STREAM + CONVERT * U)
                core_h = core_h + io_s / 3600                                  # 1-core I/O stage
            core_h = core_h.sum() * RETRY
            dl_days = dl_bytes.sum() * RETRY / BW_AGG / 86400
            r = {"depth": "full" if cap is None else f"{int(cap / 1e6)}M",
                 "pipeline": mode + ("" if mode == "present" else (", screen" if screen == "screen" else ", no screen")),
                 "core_h_k": core_h / 1e3, "download_PB": dl_bytes.sum() * RETRY / 1e15, "download_days": dl_days}
            for cores in (200, 300, 400):
                r[f"days@{cores}"] = max(core_h / cores / 24, dl_days)
            rows.append(r)
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(t.round(1).to_string(index=False))
t.to_csv("cost_options_v2.csv", index=False)

# --- Anvil (ACCESS) quote, all reads ---------------------------------------------------------------
# Whole nodes (128 cores, 257 GB, charged all 128 cores): one genome in /dev/shm, 28 four-thread samples
# (112 cores busy; the rest run the 1-core I/O stages). The non-host screen fits only if its indices are
# memory-mapped once per node (26 GB); today each job reads its own 35 GB copy, which would leave ~6 jobs
# per node. Per-core speed of Anvil's EPYC 7763 is taken equal to Clipper's 6230R until a pilot measures it.
# MAP is the speedup of STAR mapping alone (10.8 of the 12.3 s/M slope at 4 threads is mapping).
NODE_FILL = 128 / 112
BAL = 112_226
print("\nAnvil, all reads (SU = core-h x 128/112, retries included)")
U = u.reads / 1e6
for screen in ("noscreen", "screen"):
    for MAP in (1.0, 1.5, 2.0, 3.0):
        slope = S[screen] - 10.8 * (1 - 1 / MAP)
        core_h = 4 * (F + slope * U) * PACK / 3600
        su = core_h.sum() * RETRY * NODE_FILL
        print(f"  {screen:8s} mapping x{MAP:.1f}: {su / 1e3:6.0f}k SU = {su / BAL:4.1f}x balance;"
              f" {su / 128 / 24:5.0f} node-days -> {su / 128 / 24 / 30:4.0f} nodes for 30 days")

# --- all reads, measured STAR speedup (PR #34: mapping 793 -> 515 s on PBMC, x1.54; deeper samples hit
# the sequence cache more, 66% simulated at 489M reads vs 48% at 67M, so x1.54 is conservative) ----------
print("\nClipper, all reads, optimized pipeline")
for screen in ("noscreen", "screen"):
    for MAP in (1.0, 1.54, 2.0):
        slope = S[screen] - 10.8 * (1 - 1 / MAP)
        core_h = (4 * (F + slope * U) * PACK / 3600).sum() * RETRY
        print(f"  {screen:8s} mapping x{MAP:.2f}: {core_h / 1e3:5.0f}k core-h;"
              f" compute days at 300 cores {core_h / 300 / 24:5.0f}; download days 145")
