#!/usr/bin/env python3
"""E2 census aggregation (docs/gpu-engine/fable/E2-census.md).

Reads the per-read and per-window records that `m5c --census PREFIX` writes (layout: bench/gpu/m5/census.cuh) and writes one
JSON with the distributions, cost shares, correlations, a cost predictor from phase-1 state and the capacity quantiles, plus a
Markdown summary on stdout.

    python3 census_agg.py PREFIX OUT.json [--label NAME]

Only numpy is needed. Cycles are SM clock cycles of one thread (clock64), i.e. elapsed time of that read's thread including
the time its warp and SM spend on other work; they rank reads by cost, they are not exclusive instruction counts.
"""
import json
import sys

import numpy as np

LANES = ["A", "M", "B", "host"]
OV = ["none", "lread", "split", "pc", "win", "wa", "winbin", "exons", "wtr", "keep", "blob", "steps"]
STATUS = ["ok", "overflow", "blobfull", "unsupported"]

REC = np.dtype([
    ("read", "<u4"), ("count", "<u4"),
    ("Lread", "<u2"), ("lane", "u1"), ("status", "u1"),
    ("ov", "u1"), ("final", "u1"), ("ident", "u1"), ("unmapType", "i1"),
    ("stitched", "u1"), ("nSplit", "u1"), ("blobLen", "<u2"),
    ("mapMarker", "<u4"),
    ("nP", "<u4"), ("maxNrep", "<u4"), ("nRepGt1", "<u4"), ("nRepGtWin", "<u4"),
    ("sumNrep", "<u8"),
    ("nWall", "<u4"), ("nW", "<u4"),
    ("nWAsum", "<u4"), ("nWAmax", "<u4"), ("nWAne", "<u4"), ("nWApeak", "<u4"),
    ("nEntries", "<u8"), ("nHits", "<u8"),
    ("nSteps", "<u8"),
    ("maxWinSteps", "<u4"), ("nWinStitched", "<u4"), ("winTrMax", "<u4"), ("keepMax", "<u4"),
    ("trNtotal", "<u4"), ("nTr", "<u4"), ("maxEx", "<u4"), ("wbN", "<u4"),
    ("nStitch", "<u8"), ("nFinal", "<u8"), ("nRec", "<u8"),
    ("maxWinW", "<u4"), ("pad2", "<u4"),
    ("clk", "<i8", (8,)), ("cn", "<i8", (4,)),
])
WIN = np.dtype([("read", "<u4"), ("iW", "<u2"), ("lane", "u1"), ("nWA", "u1"), ("steps", "<u4"), ("cyc", "<u4"),
                ("nTr", "<u2"), ("pad", "<u2")])
assert REC.itemsize == 256 and WIN.itemsize == 20
PCTS = [50, 90, 99, 99.9, 100]


def load(path, magic, dt):
    raw = np.fromfile(path, dtype=np.uint8)
    assert bytes(raw[:8]) == magic, (path, bytes(raw[:8]))
    hdr = raw[8:24].view("<u4")
    assert hdr[0] == REC.itemsize and hdr[1] == WIN.itemsize, hdr
    body = raw[24:]
    return body[: (len(body) // dt.itemsize) * dt.itemsize].view(dt)


def pct(x, w=None):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {str(p): None for p in PCTS} | {"mean": None, "n": 0}
    if w is None:
        q = np.percentile(x, PCTS, method="higher")
        m = float(x.mean())
    else:
        o = np.argsort(x, kind="stable")
        cw = np.cumsum(np.asarray(w, dtype=np.float64)[o])
        q = [float(x[o][min(np.searchsorted(cw, p / 100 * cw[-1], side="left"), len(o) - 1)]) for p in PCTS]
        m = float(np.average(x, weights=w))
    return {str(p): float(v) for p, v in zip(PCTS, q)} | {"mean": m, "n": int(x.size)}


def wquant(x, w, q):
    """smallest value v with (weight of x <= v) >= q * total weight"""
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(np.asarray(w, dtype=np.float64)[o])
    return float(np.asarray(x)[o][min(np.searchsorted(cw, q * cw[-1], side="left"), len(o) - 1)])


def log2hist(x):
    """counts per bucket: 0, 1, 2-3, 4-7, ... (bucket k >= 1 holds [2^(k-1), 2^k))"""
    x = np.asarray(x, dtype=np.float64)
    b = np.zeros(x.shape, dtype=np.int64)
    pos = x >= 1
    b[pos] = np.floor(np.log2(x[pos])).astype(np.int64) + 1
    c = np.bincount(b) if b.size else np.zeros(1, dtype=np.int64)
    labels = ["0"] + [f"{1 << (k - 1)}-{(1 << k) - 1}" if k > 1 else "1" for k in range(1, len(c))]
    return {"buckets": labels, "counts": c.tolist()}


def buckets(x, edges, names):
    """index of the bucket: edges are inclusive upper bounds"""
    idx = np.searchsorted(np.asarray(edges), np.asarray(x), side="left")
    return idx, names


def share_table(key_idx, names, weights):
    tot = {k: float(np.sum(v)) for k, v in weights.items()}
    rows = []
    for i, nm in enumerate(names):
        m = key_idx == i
        row = {"bucket": nm}
        for k, v in weights.items():
            row[k] = float(np.sum(v[m])) / tot[k] if tot[k] else 0.0
        rows.append(row)
    return rows


def r2(y, yh):
    y = np.asarray(y, dtype=np.float64)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yh) ** 2) / ss) if ss > 0 else 0.0


def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


class Tree:
    """small regression tree on quantile-binned features (histogram split search), squared error"""

    def __init__(self, depth=6, nbins=64, min_leaf=200):
        self.depth, self.nbins, self.min_leaf = depth, nbins, min_leaf

    def fit(self, X, y):
        n, d = X.shape
        self.edges = []
        B = np.empty((n, d), dtype=np.int16)
        for j in range(d):
            e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.nbins + 1)[1:-1], method="higher"))
            self.edges.append(e)
            B[:, j] = np.searchsorted(e, X[:, j], side="right")
        self.nodes = []
        self._grow(B, y, np.arange(n), 0)
        return self

    def _grow(self, B, y, idx, dep):
        me = len(self.nodes)
        self.nodes.append(None)
        yy = y[idx]
        best = None
        if dep < self.depth and len(idx) >= 2 * self.min_leaf:
            tot, tot2, cnt = yy.sum(), (yy * yy).sum(), len(idx)
            base = tot2 - tot * tot / cnt
            for j in range(B.shape[1]):
                bj = B[idx, j].astype(np.int64)
                nb = len(self.edges[j]) + 1
                c = np.bincount(bj, minlength=nb).astype(np.float64)
                s = np.bincount(bj, weights=yy, minlength=nb)
                s2 = np.bincount(bj, weights=yy * yy, minlength=nb)
                cl, sl, s2l = np.cumsum(c)[:-1], np.cumsum(s)[:-1], np.cumsum(s2)[:-1]
                cr, sr, s2r = cnt - cl, tot - sl, tot2 - s2l
                ok = (cl >= self.min_leaf) & (cr >= self.min_leaf)
                if not ok.any():
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    sse = (s2l - sl * sl / cl) + (s2r - sr * sr / cr)
                sse = np.where(ok, sse, np.inf)
                k = int(np.argmin(sse))
                if sse[k] < base - 1e-9 and (best is None or sse[k] < best[0]):
                    best = (sse[k], j, k)
        if best is None:
            self.nodes[me] = ("leaf", float(yy.mean()), len(idx))
            return me
        _, j, k = best
        m = B[idx, j] <= k
        l = self._grow(B, y, idx[m], dep + 1)
        r = self._grow(B, y, idx[~m], dep + 1)
        self.nodes[me] = ("split", j, k, l, r)
        return me

    def predict(self, X):
        n, d = X.shape
        B = np.empty((n, d), dtype=np.int16)
        for j in range(d):
            B[:, j] = np.searchsorted(self.edges[j], X[:, j], side="right")
        out = np.empty(n)
        stack = [(0, np.arange(n))]
        while stack:
            node, idx = stack.pop()
            nd = self.nodes[node]
            if nd[0] == "leaf":
                out[idx] = nd[1]
                continue
            _, j, k, l, r = nd
            m = B[idx, j] <= k
            stack.append((l, idx[m]))
            stack.append((r, idx[~m]))
        return out

    def rules(self, names, node=0, ind=""):
        nd = self.nodes[node]
        if nd[0] == "leaf":
            return [f"{ind}-> {nd[1]:.0f} cycles (n={nd[2]})"]
        _, j, k, l, r = nd
        thr = self.edges[j][k] if k < len(self.edges[j]) else float("inf")
        return ([f"{ind}if {names[j]} < {thr:g}:"] + self.rules(names, l, ind + "  ")
                + [f"{ind}else:"] + self.rules(names, r, ind + "  "))


def router_eval(pred, y, heavy_today, fracs=(0.7, 0.8, 0.9)):
    """route the reads with predicted cost above the q-quantile to the wide class"""
    out = []
    tot = y.sum()
    order_true = np.sort(y)[::-1]
    for q in fracs:
        thr = np.quantile(pred, q)
        wide = pred > thr
        if wide.mean() > (1 - q) + 0.02:   # ties: take exactly the top share by rank
            r = np.argsort(np.argsort(-pred, kind="stable"), kind="stable")
            wide = r < int(round((1 - q) * len(pred)))
        k = int(wide.sum())
        out.append({
            "quantile": q, "wide_share_reads": float(wide.mean()),
            "cycles_captured": float(y[wide].sum() / tot),
            "oracle_cycles_captured": float(order_true[:k].sum() / tot),
            "today_overflow_reads_routed_wide": float(wide[heavy_today].mean()) if heavy_today.any() else None,
            "narrow_class_p99_cycles": float(np.percentile(y[~wide], 99)) if (~wide).any() else None,
            "wide_class_median_cycles": float(np.median(y[wide])) if wide.any() else None,
        })
    return out


def main():
    prefix, outp = sys.argv[1], sys.argv[2]
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else prefix
    meta = json.load(open(prefix + ".meta.json"))
    R = load(prefix + ".reads.bin", b"M5CEN001", REC)
    W = load(prefix + ".wins.bin", b"M5CENW01", WIN)
    nD = int(meta["distinct_reads"])
    res = {"label": label, "meta": meta, "records": int(len(R)), "window_records": int(len(W))}
    md = [f"# E2 census: {label}", ""]

    # ---- per-read views
    F = R[R["final"] == 1]
    F = F[np.argsort(F["read"], kind="stable")]
    assert len(F) == nD and np.all(F["read"] == np.arange(nD)), (len(F), nD)
    lane_f = F["lane"].astype(np.int64)
    cnt = F["count"].astype(np.float64)
    gpu = R["lane"] < 3
    # per record cost (cycles of the read's thread): phase 1 call + phase 3 call (assignment + stitching + select + encode)
    rec_cost = np.where(gpu, R["cn"][:, 0] + R["cn"][:, 3], 0).astype(np.float64)
    rec_post = np.where(gpu, R["clk"][:, 1] + R["cn"][:, 3], 0).astype(np.float64)   # windows + assignment + stitch + select
    cascade = np.bincount(R["read"], weights=rec_cost, minlength=nD)       # every GPU lane the read went through
    cascade_post = np.bincount(R["read"], weights=rec_post, minlength=nD)
    wasted = cascade - np.where(F["lane"] < 3, F["cn"][:, 0] + F["cn"][:, 3], 0)
    fcost = np.where(F["lane"] < 3, F["cn"][:, 0] + F["cn"][:, 3], 0).astype(np.float64)
    fstitch = np.where(F["lane"] < 3, F["clk"][:, 2], 0).astype(np.float64)
    task_cost = np.where(F["lane"] < 3, F["clk"][:, 1] + F["clk"][:, 2] + F["clk"][:, 3], 0).astype(np.float64)
    # stitching cycles over all lanes: clk[2] of the lane that finished; in a lane that overflowed during stitching, its
    # phase-3 time after the assignment
    rec_stitch = np.where(gpu, np.where(R["status"] == 0, R["clk"][:, 2], R["cn"][:, 3] - R["cn"][:, 2]), 0).astype(np.float64)
    stitch_all = np.bincount(R["read"], weights=rec_stitch, minlength=nD)
    scan_all = np.bincount(R["read"], weights=np.where(gpu, R["cn"][:, 1], 0).astype(np.float64), minlength=nD)
    host = lane_f == 3

    # ---- 0. identity and lane flow
    res["identity"] = {"identical_distinct": meta["identical_distinct"], "distinct": nD,
                       "identical_reads": meta["identical_reads"], "reads": meta["reads"],
                       "ident_flag_final": int(F["ident"].sum())}
    flow = []
    for li, ln in enumerate(LANES):
        m = R["lane"] == li
        st = R["status"][m]
        ovc = np.bincount(R["ov"][m][st != 0], minlength=len(OV))
        flow.append({"lane": ln, "reads_in": int(m.sum()), "ok": int((st == 0).sum()),
                     "final": int(((lane_f == li)).sum()), "share_of_distinct": float(m.sum() / nD),
                     "overflow_why": {OV[k]: int(c) for k, c in enumerate(ovc) if c},
                     "thread_cycles": float(rec_cost[m].sum()),
                     "scan_warp_cycles": float(R["cn"][m][:, 1].sum()) if li < 3 else 0.0,
                     "lane_ms": meta["lane_ms"][li] if li < 3 else meta["heavy_host_s"] * 1e3})
    res["lane_flow"] = flow
    md += ["## Lane flow", "", "| lane | reads in | share | finished here | overflow reasons | thread Gcycles | lane ms |", "|---|---|---|---|---|---|---|"]
    for f in flow:
        why = ", ".join(f"{k} {v}" for k, v in sorted(f["overflow_why"].items(), key=lambda kv: -kv[1]))
        md.append(f"| {f['lane']} | {f['reads_in']:,} | {100 * f['share_of_distinct']:.2f}% | {f['final']:,} | {why} | "
                  f"{f['thread_cycles'] / 1e9:.1f} | {f['lane_ms']:.0f} |")
    tot_casc = cascade.sum()
    md += ["", f"Thread cycles: cascade total {tot_casc / 1e9:.1f} G; wasted in lanes the read overflowed {wasted.sum() / tot_casc * 100:.1f}%;"
           f" stitching {stitch_all.sum() / tot_casc * 100:.1f}%; phase 1 {np.where(gpu, R['cn'][:, 0], 0).sum() / tot_casc * 100:.1f}%;"
           f" assignment {np.where(gpu, R['cn'][:, 2], 0).sum() / tot_casc * 100:.1f}%. Scan (k_p2, warp cycles): {scan_all.sum() / 1e9:.1f} G.", ""]

    # ---- 1. distributions per final lane
    qty = {
        "nP": F["nP"], "nSplit": F["nSplit"], "maxNrep": F["maxNrep"], "nRepGtWin": F["nRepGtWin"],
        "nWall": F["nWall"], "nW_final": F["nW"], "nWAsum": F["nWAsum"], "nWAmax": F["nWAmax"], "nWApeak": F["nWApeak"],
        "nWAne": F["nWAne"], "nEntries": F["nEntries"], "nHits": F["nHits"], "nSteps": F["nSteps"],
        "maxWinSteps": F["maxWinSteps"], "nWinStitched": F["nWinStitched"], "keepMax": F["keepMax"], "winTrMax": F["winTrMax"],
        "nTr": F["nTr"], "trNtotal": F["trNtotal"], "maxEx": F["maxEx"], "wbN": F["wbN"], "blobLen": F["blobLen"],
        "nStitch": F["nStitch"], "nFinal": F["nFinal"], "nRec": F["nRec"],
        "cycles_final_lane": fcost, "cycles_cascade": cascade, "cycles_task_def": task_cost, "cycles_stitch": fstitch,
    }
    dist = {}
    for name, x in qty.items():
        d = {"all": pct(x), "all_weighted": pct(x, cnt), "hist_all": log2hist(x)}
        for li, ln in enumerate(LANES):
            m = lane_f == li
            if name.startswith("cycles") and li == 3:
                continue
            d[ln] = pct(x[m])
            d["hist_" + ln] = log2hist(x[m])
        dist[name] = d
    res["distributions"] = dist
    md += ["## Percentiles per read (final lane record: the read's full state)", "",
           "| quantity | lane | p50 | p90 | p99 | p99.9 | max | mean |", "|---|---|---|---|---|---|---|---|"]
    for name in ["nP", "maxNrep", "nWall", "nW_final", "nWAsum", "nWAmax", "nEntries", "nHits", "nSteps", "maxWinSteps",
                 "keepMax", "winTrMax", "nTr", "maxEx", "cycles_final_lane", "cycles_cascade"]:
        for ln in ["all"] + LANES:
            if ln not in dist[name]:
                continue
            p = dist[name][ln]
            if not p["n"]:
                continue
            md.append(f"| {name} | {ln} (n={p['n']:,}) | {p['50']:g} | {p['90']:g} | {p['99']:g} | {p['99.9']:g} | {p['100']:g} | {p['mean']:.1f} |")
    md.append("")

    # ---- 2. shares by window-count and max-aligns buckets
    wts = {"reads": np.ones(nD), "reads_weighted": cnt, "cycles_cascade": cascade, "cycles_stitch": stitch_all,
           "cycles_wasted": wasted, "steps": F["nSteps"].astype(np.float64)}
    nwb, nwn = buckets(F["nWall"], [0, 1, 4, 16, 64, 128], ["0", "1", "2-4", "5-16", "17-64", "65-128", ">128"])
    res["share_by_nW"] = share_table(nwb, nwn, wts)
    wab, wan = buckets(F["nWAmax"], [0, 1, 2, 4, 8, 16, 32], ["0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-50"])
    res["share_by_maxNWA"] = share_table(wab, wan, wts)
    enb, enn = buckets(F["nEntries"], [0, 1, 16, 64, 256, 1024, 4096, 16384],
                       ["0", "1", "2-16", "17-64", "65-256", "257-1024", "1025-4096", "4097-16384", ">16384"])
    res["share_by_nEntries"] = share_table(enb, enn, wts)
    for key, title in [("share_by_nW", "windows per read (nWall)"), ("share_by_maxNWA", "largest window (aligns)"),
                       ("share_by_nEntries", "SA entries per read")]:
        md += [f"## Shares by {title}", "", "| bucket | reads | reads (weighted) | cascade cycles | stitching cycles | wasted cycles | steps |",
               "|---|---|---|---|---|---|---|"]
        for r in res[key]:
            md.append(f"| {r['bucket']} | {100 * r['reads']:.2f}% | {100 * r['reads_weighted']:.2f}% | {100 * r['cycles_cascade']:.2f}% | "
                      f"{100 * r['cycles_stitch']:.2f}% | {100 * r['cycles_wasted']:.2f}% | {100 * r['steps']:.2f}% |")
        md.append("")

    # ---- 3. joint distributions and correlations
    def joint(x, y):
        bx = np.where(x >= 1, np.floor(np.log2(np.maximum(x, 1))) + 1, 0).astype(np.int64)
        by = np.where(y >= 1, np.floor(np.log2(np.maximum(y, 1))) + 1, 0).astype(np.int64)
        H = np.zeros((bx.max() + 1, by.max() + 1), dtype=np.int64)
        np.add.at(H, (bx, by), 1)
        per = []
        for b in range(bx.max() + 1):
            m = bx == b
            if m.any():
                per.append({"x_bucket": b, "n": int(m.sum()), "median_cycles": float(np.median(y[m])),
                            "mean_cycles": float(y[m].mean()), "p99_cycles": float(np.percentile(y[m], 99)),
                            "cycle_share": float(y[m].sum() / y.sum())})
        return {"log2_hist2d": H.tolist(), "per_x_bucket": per}

    g = ~host
    yj = cascade[g]
    res["joint"] = {}
    md += ["## Cost vs read state (GPU-finished reads; cycles = cascade thread cycles)", "",
           "| x | Spearman(x, cycles) | Pearson(log x, log cycles) |", "|---|---|---|"]
    for nm in ["nWall", "nEntries", "nHits", "nSteps", "nWAsum", "nWAmax", "nP", "maxNrep", "sumNrep"]:
        x = F[nm][g].astype(np.float64)
        res["joint"][nm] = joint(x, yj) | {"spearman": spearman(x, yj),
                                            "pearson_log": float(np.corrcoef(np.log1p(x), np.log1p(yj))[0, 1])}
        md.append(f"| {nm} | {res['joint'][nm]['spearman']:.3f} | {res['joint'][nm]['pearson_log']:.3f} |")
    md += ["", "Median cascade cycles by window-count bucket (log2 bucket b holds 2^(b-1)..2^b-1):", "",
           "| nWall bucket | reads | median | mean | p99 | cycle share |", "|---|---|---|---|---|---|"]
    for r in res["joint"]["nWall"]["per_x_bucket"]:
        lo = 0 if r["x_bucket"] == 0 else 1 << (r["x_bucket"] - 1)
        md.append(f"| {lo}-{max(lo, (1 << r['x_bucket']) - 1)} | {r['n']:,} | {r['median_cycles']:.0f} | {r['mean_cycles']:.0f} | "
                  f"{r['p99_cycles']:.0f} | {100 * r['cycle_share']:.1f}% |")
    md.append("")

    # ---- 4. windows: steps per window, dominance
    Wf = W
    wsteps = Wf["steps"].astype(np.float64)
    res["windows"] = {"n": int(len(Wf)), "steps": pct(wsteps), "steps_hist": log2hist(wsteps), "nWA": pct(Wf["nWA"]),
                      "cycles": pct(Wf["cyc"][Wf["lane"] < 3])}
    per_nwa = []
    for a in range(1, 51):
        m = Wf["nWA"] == a
        if not m.any():
            continue
        s = wsteps[m]
        per_nwa.append({"nWA": a, "windows": int(m.sum()), "steps_p50": float(np.median(s)), "steps_p90": float(np.percentile(s, 90)),
                        "steps_p99": float(np.percentile(s, 99)), "steps_max": float(s.max()), "steps_share": float(s.sum() / wsteps.sum()),
                        "cyc_mean": float(Wf["cyc"][m & (Wf["lane"] < 3)].mean()) if (m & (Wf["lane"] < 3)).any() else None,
                        "nTr_mean": float(Wf["nTr"][m].mean())})
    res["windows"]["by_nWA"] = per_nwa
    st = F["nSteps"].astype(np.float64)
    multi = (F["nWinStitched"] >= 2) & (st > 0)
    frac = np.where(st > 0, F["maxWinSteps"] / np.maximum(st, 1), 0)
    dom = {"reads_with_2plus_windows": int(multi.sum()),
           "share_max_window_gt50": float((frac[multi] > 0.5).mean()) if multi.any() else None,
           "share_max_window_gt90": float((frac[multi] > 0.9).mean()) if multi.any() else None,
           "steps_weighted_gt50": float(st[multi & (frac > 0.5)].sum() / st[multi].sum()) if multi.any() else None,
           "steps_weighted_gt90": float(st[multi & (frac > 0.9)].sum() / st[multi].sum()) if multi.any() else None,
           "cycles_weighted_gt50": float(cascade[multi & (frac > 0.5)].sum() / cascade[multi].sum()) if multi.any() else None,
           "max_window_share": pct(frac[multi]) if multi.any() else None}
    for li, ln in enumerate(LANES):
        m = multi & (lane_f == li)
        if m.any():
            dom[ln] = {"reads": int(m.sum()), "gt50": float((frac[m] > 0.5).mean()), "gt90": float((frac[m] > 0.9).mean()),
                       "steps_in_max_window": float(F["maxWinSteps"][m].sum() / st[m].sum())}
    res["window_dominance"] = dom
    # steps outside the biggest window: what a per-window parallel lane saves at best (critical path = the largest window)
    res["window_parallel_bound"] = {
        "steps_total": float(st.sum()), "steps_in_largest_window": float(F["maxWinSteps"].sum()),
        "critical_path_share": float(F["maxWinSteps"].sum() / max(st.sum(), 1)),
        "by_lane": {ln: float(F["maxWinSteps"][lane_f == li].sum() / max(st[lane_f == li].sum(), 1)) for li, ln in enumerate(LANES)}}
    md += ["## Windows (per-window records of the finishing lane)", "",
           f"{len(Wf):,} stitched windows; steps per window p50 {res['windows']['steps']['50']:g}, p90 {res['windows']['steps']['90']:g}, "
           f"p99 {res['windows']['steps']['99']:g}, p99.9 {res['windows']['steps']['99.9']:g}, max {res['windows']['steps']['100']:g}.", "",
           f"Reads with >= 2 stitched windows: {dom['reads_with_2plus_windows']:,}; the largest window holds > 50% of the read's steps in "
           f"{100 * (dom['share_max_window_gt50'] or 0):.1f}% of them, > 90% in {100 * (dom['share_max_window_gt90'] or 0):.1f}%. "
           f"Steps in the largest window / all steps (critical path of a window-parallel lane): {100 * res['window_parallel_bound']['critical_path_share']:.1f}%.", "",
           "| aligns in window | windows | steps p50 | p90 | p99 | max | share of steps | mean cycles | transcripts |", "|---|---|---|---|---|---|---|---|---|"]
    for r in per_nwa:
        md.append(f"| {r['nWA']} | {r['windows']:,} | {r['steps_p50']:g} | {r['steps_p90']:g} | {r['steps_p99']:g} | {r['steps_max']:g} | "
                  f"{100 * r['steps_share']:.2f}% | {r['cyc_mean'] or 0:.0f} | {r['nTr_mean']:.2f} |")
    md.append("")

    # ---- 5. multimappers, multiplicity, unmapType
    def split(mask_list):
        rows = []
        for nm, m in mask_list:
            rows.append({"class": nm, "reads": float(m.mean()), "reads_weighted": float(cnt[m].sum() / cnt.sum()),
                         "cycles_cascade": float(cascade[m].sum() / cascade.sum()), "stitch_cycles": float(stitch_all[m].sum() / stitch_all.sum()),
                         "lane_mix": {ln: float((lane_f[m] == li).mean()) if m.any() else 0.0 for li, ln in enumerate(LANES)},
                         "median_nWall": float(np.median(F["nWall"][m])) if m.any() else None})
        return rows
    res["by_nTr"] = split([("nTr=0", F["nTr"] == 0), ("nTr=1", F["nTr"] == 1), ("nTr 2-10", (F["nTr"] >= 2) & (F["nTr"] <= 10)),
                           ("nTr>10", F["nTr"] > 10)])
    res["by_multiplicity"] = split([("singleton", F["count"] == 1), ("2-9", (F["count"] >= 2) & (F["count"] <= 9)), (">=10", F["count"] >= 10)])
    res["by_unmapType"] = split([("mapped", F["unmapType"] == -1), ("unmapped other(0)", F["unmapType"] == 0),
                                 ("too short(1)", F["unmapType"] == 1), ("mismatches(2)", F["unmapType"] == 2),
                                 ("multimapping(3)", F["unmapType"] == 3)])
    for key, title in [("by_nTr", "transcripts (nTr; > 1 = multimapper)"), ("by_multiplicity", "read multiplicity"),
                       ("by_unmapType", "STAR outcome (unmapType)")]:
        md += [f"## By {title}", "", "| class | reads | reads (weighted) | cascade cycles | stitching cycles | A / M / B / host finish | median nWall |",
               "|---|---|---|---|---|---|---|"]
        for r in res[key]:
            lm = " / ".join(f"{100 * r['lane_mix'][ln]:.1f}" for ln in LANES)
            md.append(f"| {r['class']} | {100 * r['reads']:.2f}% | {100 * r['reads_weighted']:.2f}% | {100 * r['cycles_cascade']:.2f}% | "
                      f"{100 * r['stitch_cycles']:.2f}% | {lm} | {r['median_nWall']} |")
        md.append("")

    # ---- 6. cost predictor from phase-1 state
    feats1 = ["nP", "nWall", "nEntries", "maxNrep", "sumNrep", "nRepGtWin", "nRepGt1", "Lread"]
    sets = {"phase1": feats1, "phase1+scan": feats1 + ["nHits"], "phase1+scan+assign": feats1 + ["nHits", "nWAsum", "nWAmax", "nWAne"]}
    heavy_today = (lane_f != 0)[g]
    pred_res = {}
    rng = np.random.default_rng(1)
    perm = rng.permutation(int(g.sum()))
    ntr = len(perm) // 2
    tr, te = perm[:ntr], perm[ntr:]
    for target_name, Y in [("cycles_cascade", cascade[g]), ("cycles_post_seed", cascade_post[g]), ("cycles_final_lane", fcost[g]),
                           ("cycles_task_def", task_cost[g]), ("steps", st[g])]:
        pr = {}
        for sname, fl in sets.items():
            X = np.column_stack([F[f][g].astype(np.float64) for f in fl])
            Xd = np.column_stack([np.ones(len(X)), X])
            beta, *_ = np.linalg.lstsq(Xd[tr], Y[tr], rcond=None)
            lin = r2(Y[te], Xd[te] @ beta)
            Xl = np.column_stack([np.ones(len(X)), np.log1p(X)])
            bl, *_ = np.linalg.lstsq(Xl[tr], np.log1p(Y[tr]), rcond=None)
            yl = Xl[te] @ bl
            logr2 = r2(np.log1p(Y[te]), yl)
            tree = Tree(depth=6).fit(X[tr], Y[tr])
            yt = tree.predict(X[te])
            pr[sname] = {"features": fl, "ols_r2": lin, "ols_coef": beta.tolist(), "loglog_r2_log": logr2,
                         "loglog_r2_raw": r2(Y[te], np.expm1(yl)), "tree6_r2": r2(Y[te], yt),
                         "spearman_tree": spearman(yt, Y[te]), "spearman_loglog": spearman(yl, Y[te])}
            if target_name == "cycles_cascade":
                pr[sname]["router_tree"] = router_eval(yt, Y[te], heavy_today[te])
                pr[sname]["router_loglog"] = router_eval(yl, Y[te], heavy_today[te])
                if sname == "phase1":
                    pr[sname]["tree_rules"] = tree.rules(fl)[:80]
        pred_res[target_name] = pr
    # simple rules after phase 1 (no fit): windows > 16 (today's lane-A cap, known after phase 1), entries
    Yc, te_g = cascade[g][te], None
    simple = {}
    for nm, sc in [("nWall", F["nWall"][g][te]), ("nEntries", F["nEntries"][g][te]), ("nHits", F["nHits"][g][te])]:
        simple[nm] = router_eval(sc.astype(np.float64) + 1e-6 * rng.random(len(sc)), Yc, heavy_today[te])
    m16 = F["nWall"][g][te] > 16
    simple["rule_nWall_gt16"] = {"wide_share_reads": float(m16.mean()), "cycles_captured": float(Yc[m16].sum() / Yc.sum())}
    pred_res["simple_rules"] = simple
    res["predictor"] = pred_res
    md += ["## Cost predictor (held-out half; GPU-finished reads)", "",
           "| target | features | OLS R2 | log-log R2 (log space) | tree depth 6 R2 | Spearman (tree) |", "|---|---|---|---|---|---|"]
    for tn, pr in pred_res.items():
        if tn == "simple_rules":
            continue
        for sn, v in pr.items():
            md.append(f"| {tn} | {sn} | {v['ols_r2']:.3f} | {v['loglog_r2_log']:.3f} | {v['tree6_r2']:.3f} | {v['spearman_tree']:.3f} |")
    md += ["", "Router (route predicted top share to the wide class; target = cascade cycles):", "",
           "| features | model | wide share | cycles captured | oracle | today's overflow reads routed wide | narrow p99 cycles |", "|---|---|---|---|---|---|---|"]
    for sn in sets:
        for mdl in ["router_tree", "router_loglog"]:
            for r in pred_res["cycles_cascade"][sn][mdl]:
                md.append(f"| {sn} | {mdl[7:]} | {100 * r['wide_share_reads']:.0f}% | {100 * r['cycles_captured']:.1f}% | {100 * r['oracle_cycles_captured']:.1f}% | "
                          f"{100 * (r['today_overflow_reads_routed_wide'] or 0):.1f}% | {r['narrow_class_p99_cycles']:.0f} |")
    for nm in ["nWall", "nEntries", "nHits"]:
        for r in simple[nm]:
            md.append(f"| rank by {nm} | - | {100 * r['wide_share_reads']:.0f}% | {100 * r['cycles_captured']:.1f}% | {100 * r['oracle_cycles_captured']:.1f}% | "
                      f"{100 * (r['today_overflow_reads_routed_wide'] or 0):.1f}% | {r['narrow_class_p99_cycles']:.0f} |")
    md += [f"| nWall > 16 | rule | {100 * simple['rule_nWall_gt16']['wide_share_reads']:.1f}% | {100 * simple['rule_nWall_gt16']['cycles_captured']:.1f}% | | | |", ""]

    # ---- 7. capacities
    req = {
        "NSPLIT": F["nSplit"].astype(np.int64),
        "NP": F["nP"].astype(np.int64),
        "NW": F["nWall"].astype(np.int64),
        "NWA": F["nWApeak"].astype(np.int64),
        "NEX": np.where(F["nStitch"] > 0, np.minimum(F["maxEx"].astype(np.int64) + 1, 20), 1),
        "NWTR": F["winTrMax"].astype(np.int64),
        "NKEEP": F["keepMax"].astype(np.int64),
        "NWB": np.ceil(F["wbN"].astype(np.float64) * 4 / 3 + 1).astype(np.int64),
        "NBLOB": F["blobLen"].astype(np.int64),
        "steps": F["nSteps"].astype(np.int64),
    }
    qs = [0.99, 0.999, 0.9999]
    caps = {}
    for k, v in req.items():
        caps[k] = {"q": {str(q): int(np.quantile(v, q, method="higher")) for q in qs},
                   "q_weighted": {str(q): wquant(v, cnt, q) for q in qs}, "max": int(v.max())}
    capsets = {}
    for q in qs:
        cs = {k: caps[k]["q"][str(q)] for k in req}
        over = np.zeros(nD, dtype=bool)
        per = {}
        for k, v in req.items():
            o = v > cs[k]
            per[k] = float(o.mean())
            over |= o
        capsets[str(q)] = {"caps": cs, "overflow_share": float(over.mean()), "overflow_share_weighted": float(cnt[over].sum() / cnt.sum()),
                           "overflow_cycle_share": float(cascade[over].sum() / cascade.sum()), "per_cap_overflow": per}
    # today's lane A caps, for comparison
    capA = dict(zip(meta["caps"]["fields"], meta["caps"]["A"]))
    capA_req = {"NSPLIT": capA["NSPLIT"], "NP": capA["NP"], "NW": capA["NW"], "NWA": capA["NWA"], "NEX": capA["NEX"],
                "NWTR": capA["NWTR"], "NKEEP": capA["NKEEP"], "NWB": capA["NWB"], "NBLOB": capA["NBLOB"], "steps": meta["budget"][0] or 10 ** 18}
    over = np.zeros(nD, dtype=bool)
    perA = {}
    for k, v in req.items():
        o = v > capA_req[k]
        perA[k] = float(o.mean())
        over |= o
    capsets["laneA_today"] = {"caps": capA_req, "overflow_share": float(over.mean()), "per_cap_overflow": perA,
                              "overflow_cycle_share": float(cascade[over].sum() / cascade.sum())}
    # a narrow lane with only 8 windows (Part I 5.2: 24 seeds, 8 windows)
    for nm, cs in [("narrow_24seeds_8win", {"NP": 24, "NW": 8}), ("narrow_32seeds_16win", {"NP": 32, "NW": 16}),
                   ("narrow_16win_8wa", {"NW": 16, "NWA": 8})]:
        o = np.zeros(nD, dtype=bool)
        for k, c in cs.items():
            o |= req[k] > c
        capsets[nm] = {"caps": cs, "overflow_share": float(o.mean()), "overflow_cycle_share": float(cascade[o].sum() / cascade.sum())}
    res["caps"] = caps
    res["capsets"] = capsets
    md += ["## Capacities (smallest cap covering the share of distinct reads)", "", "| cap | 99% | 99.9% | 99.99% | max | today A / M / B |", "|---|---|---|---|---|---|"]
    capM = dict(zip(meta["caps"]["fields"], meta["caps"]["M"]))
    capB = dict(zip(meta["caps"]["fields"], meta["caps"]["B"]))
    for k in req:
        today = f"{capA.get(k, meta['budget'][0])} / {capM.get(k, meta['budget'][1])} / {capB.get(k, meta['budget'][2])}"
        md.append(f"| {k} | {caps[k]['q']['0.99']} | {caps[k]['q']['0.999']} | {caps[k]['q']['0.9999']} | {caps[k]['max']} | {today} |")
    md += ["", "| cap set | overflow share (distinct) | weighted | cycle share of the overflow |", "|---|---|---|---|"]
    for k, v in capsets.items():
        md.append(f"| {k} | {100 * v['overflow_share']:.3f}% | {100 * v.get('overflow_share_weighted', float('nan')):.3f}% | {100 * v['overflow_cycle_share']:.1f}% |")
    md.append("")

    # ---- 8. wide-lane shape: windows x aligns for reads with > 16 windows
    wide = F["nWall"] > 16
    H = {}
    for a_lo, a_hi in [(0, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 50)]:
        for w_lo, w_hi in [(17, 32), (33, 64), (65, 128), (129, 10 ** 9)]:
            m = wide & (F["nWAmax"] >= a_lo) & (F["nWAmax"] <= a_hi) & (F["nWall"] >= w_lo) & (F["nWall"] <= w_hi)
            H[f"nWAmax {a_lo}-{a_hi} | nWall {w_lo}-{w_hi if w_hi < 10 ** 9 else 'inf'}"] = {
                "reads": int(m.sum()), "cycle_share_of_wide": float(cascade[m].sum() / max(cascade[wide].sum(), 1))}
    res["wide_shape"] = {"reads": int(wide.sum()), "share": float(wide.mean()), "grid": H,
                         "nWAsum": pct(F["nWAsum"][wide]), "nWAne": pct(F["nWAne"][wide]),
                         "smem_bytes_16B_per_align": pct(F["nWAsum"][wide].astype(np.float64) * 16)}
    md += ["## Wide-lane shape (reads with > 16 windows)", "",
           f"{wide.sum():,} reads ({100 * wide.mean():.1f}%); windows with >= 1 align p50 {res['wide_shape']['nWAne']['50']:g}, "
           f"p99 {res['wide_shape']['nWAne']['99']:g}; aligns over all windows p50 {res['wide_shape']['nWAsum']['50']:g}, p99 {res['wide_shape']['nWAsum']['99']:g}, "
           f"max {res['wide_shape']['nWAsum']['100']:g}.", "", "| largest window \\ windows | 17-32 | 33-64 | 65-128 | >128 |", "|---|---|---|---|---|"]
    for a_lo, a_hi in [(0, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 50)]:
        cells = []
        for w_lo, w_hi in [(17, 32), (33, 64), (65, 128), (129, 10 ** 9)]:
            c = H[f"nWAmax {a_lo}-{a_hi} | nWall {w_lo}-{w_hi if w_hi < 10 ** 9 else 'inf'}"]
            cells.append(f"{c['reads']:,} ({100 * c['cycle_share_of_wide']:.1f}%)")
        md.append(f"| {a_lo}-{a_hi} aligns | " + " | ".join(cells) + " |")
    md.append("")

    json.dump(res, open(outp, "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
