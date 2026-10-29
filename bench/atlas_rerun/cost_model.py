"""Cost of the GEX re-run per depth option, from measured rates.

Per sample, with U = read pairs used (millions) = min(reads, cap):
  wall at 4 threads (s)  = F + s * U          F = 300 s fixed (genome attach, export, cell calls)
                                              s = 17.5 s/M (benchmark: 15-17 s/M Singlet at 4 workers
                                                  + 1.5 s/M .1fq encode)
  core-h                  = 4 * wall / 3600
  download bytes          = U/reads * bases * 0.44 B/base (SRA files, measured on 3 probe runs);
                            a cap is taken as the same share of every run, so only that share is fetched
  download stream time    = bytes / 30 MB/s (measured per run from Clipper)
Unknown-depth samples (207) get the median.
"""
import numpy as np
import pandas as pd

u = pd.read_parquet("universe.parquet")
u = u[u.route == "gex"].copy()
med_r, med_b = u.reads.median(), u.bases.median()
u["reads"] = u.reads.fillna(med_r)
u["bases"] = u.bases.fillna(med_b)
F, S, BPB, BW = 300.0, 17.5, 0.44, 30e6
rows = []
for cap in [None, 300e6, 200e6, 150e6, 100e6, 50e6]:
    used = u.reads if cap is None else np.minimum(u.reads, cap)
    wall = F + S * used / 1e6
    coreh = 4 * wall / 3600
    dl = used / u.reads * u.bases * BPB
    rows.append({
        "option": "full depth" if cap is None else f"cap {int(cap/1e6)}M pairs",
        "samples": len(u), "read pairs used (T)": used.sum() / 1e12,
        "samples capped": float((u.reads > cap).mean()) if cap else 0.0,
        "median pairs used (M)": float(np.median(used) / 1e6),
        "core-h (4-thread jobs)": coreh.sum(), "core-h +20% retries": coreh.sum() * 1.2,
        "download (PB)": dl.sum() / 1e15, "download stream-hours": (dl / BW).sum() / 3600,
        "p99 sample wall (h)": float(np.percentile(wall, 99) / 3600),
        "max sample wall (h)": float(wall.max() / 3600),
    })
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
t.to_csv("cost_options.csv", index=False)
old = 52756 * 1.56
print("old campaign core-h (52,756 samples)", round(old))
for wave, x in u.groupby("wave"):
    used = np.minimum(x.reads, 200e6)
    print(wave, len(x), "core-h@200M", round((4 * (F + S * used / 1e6) / 3600).sum()), "core-h@full", round((4 * (F + S * x.reads / 1e6) / 3600).sum()))
