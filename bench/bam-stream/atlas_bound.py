#!/usr/bin/env python3
"""Phase 2 upper bound: atlas-weighted saving from retiring the sorted BAM.  Runs inside a Slurm job (needs pandas).

Inputs : $W/bam-stream/runs/<GSM>/t<T>_r<REP>/stages.json   (phase 1)   and   p2/<GSM>/ streaming-pileup timing
         $W/data/atlas-rerun/samples.parquet                (atlas depth: `reads`, else `catalog_read_count`)
Model  : per measured dataset d (median over replicates at T threads):
           wall(U) = FIXED + (wall_d - FIXED) * U / U_d          U = reads in the sample (wall seconds, T threads)
           saved   = (sort + index + cb_refresh + pileup_parallel - pileup_streaming) * U / U_d
         pileup_streaming is bracketed: B = the current parallel pileup cost (an order-independent streaming pileup that
         scales across threads: the upper bound), A = the measured serial streaming run() cost (the lower bound).
         Core-hours = wall * T, so the saving share is the same as the wall share at fixed T.
The atlas weights are all samples in the Beta scope (single-cell, 10x chemistries), by depth bucket and by chemistry class.
"""
import json, glob, os, statistics as st, sys
import pandas as pd

W = "/mnt/projects/debruinz_project/debruinz/singlet"
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
FIXED = 130.0                       # s, audit §3: the fixed part of the wall is closer to 130 s than 300 s
STREAM_SERIAL_S_PER_M = float(sys.argv[2]) if len(sys.argv) > 2 else 1.47
SUF = sys.argv[3] if len(sys.argv) > 3 else ""   # runs directory suffix: "", "_fa", "_memsort"   # s per M total reads (measured, phase 2 job)

runs = {}
for f in glob.glob(f"{W}/bam-stream/runs{SUF}/*/t{T}_r*/stages.json"):
    d = json.load(open(f)); g = f.split(f"/runs{SUF}/")[1].split("/")[0]
    if d.get("sort") is None or not d.get("reads") or d.get("pileup_total") is None: continue
    runs.setdefault(g, []).append(d)
med = lambda xs: st.median(xs)
ds = {}
for g, L in runs.items():
    ds[g] = {k: med([x.get(k, 0) or 0 for x in L]) for k in ("wall", "sort", "index", "cb_refresh", "pileup_total", "reads", "cpu_s")}
    ds[g]["n_reps"] = len(L); ds[g]["wall_range"] = [min(x["wall"] for x in L), max(x["wall"] for x in L)]
    ds[g]["sort_range"] = [min(x["sort"] for x in L), max(x["sort"] for x in L)]

s = pd.read_parquet(f"{W}/data/atlas-rerun/samples.parquet")
s = s[s.target_assay.eq("scrna") & s.protocol_inferred.isin(["10xv2", "10xv3", "10xv3_5prime", "10x_suspect"])]
U = (s.reads.where(s.reads > 0, s.catalog_read_count)).dropna() / 1e6
U = U[U > 0.1]
print(f"atlas weights: {len(U)} samples, total {U.sum():.0f} M reads, median {U.median():.1f} M")

out = {"T": T, "n_samples": int(len(U)), "total_Mreads": float(U.sum()), "median_Mreads": float(U.median()), "datasets": ds, "scenarios": {}}
for g, d in ds.items():
    U0 = d["reads"] / 1e6
    sav_B = (d["sort"] + d["index"] + d["cb_refresh"]) / U0              # s per M reads, pileup unchanged
    pile_stream = STREAM_SERIAL_S_PER_M                                   # serial streaming
    sav_A = sav_B + d["pileup_total"] / U0 - pile_stream
    var = (d["wall"] - FIXED) / U0
    tot = (FIXED * len(U) + var * U.sum())                                # wall seconds, summed over the atlas
    for name, sav in (("B_parallel_stream", sav_B), ("A_serial_stream", sav_A)):
        saved = sav * U.sum()
        out["scenarios"][f"{g}:{name}"] = {"saving_share": float(saved / tot), "atlas_wall_core_h": float(tot * T / 3600),
                                           "saved_core_h": float(saved * T / 3600)}
shares_B = [v["saving_share"] for k, v in out["scenarios"].items() if k.endswith("B_parallel_stream")]
shares_A = [v["saving_share"] for k, v in out["scenarios"].items() if k.endswith("A_serial_stream")]
out["summary"] = {"upper_B": [min(shares_B), st.median(shares_B), max(shares_B)] if shares_B else None,
                  "lower_A": [min(shares_A), st.median(shares_A), max(shares_A)] if shares_A else None,
                  "stop_rule_threshold": 0.15}
print(json.dumps(out["summary"], indent=1))
os.makedirs(f"{W}/bam-stream/results", exist_ok=True)
json.dump(out, open(f"{W}/bam-stream/results/atlas_bound_T{T}{SUF}.json", "w"), indent=1)
