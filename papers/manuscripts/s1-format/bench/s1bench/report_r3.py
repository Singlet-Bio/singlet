"""Report sections for the later measurements: the primed cold-read protocol, random-access costs (access),
latency, requests in flight and DataLoader workers (netsweep), the real object store (netsweep --r2) and write cost
(writecost). Called from report.run; writes generated/fig*_r3 / tab*_r3 TSVs and \\nm{} numbers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .report import FORMAT_LABELS, LOADER_LABELS, Numbers, _f, _stage, _write
from .util import Config, read_tsv

# Prices per request and per GB, as the providers publish them (cited in the Methods). R2: Class B (GetObject,
# HeadObject) $0.36 per million, no egress charge (developers.cloudflare.com/r2/pricing, 2026-10-30). S3 Standard,
# us-east-1 (AWS price list API, AmazonS3 2026-10-15 and AWSDataTransfer 2026-09-16): GET $0.004 per 10,000 requests;
# transfer out to the internet $0.09 per GB for the first 10 TB a month beyond the free tier.
PRICE = {"r2": {"get": 0.36e-6, "gb": 0.0}, "s3": {"get": 0.4e-6, "gb": 0.09}}
# S3 Standard storage, us-east-1, first 50 TB a month: $0.023 per GB-month (aws.amazon.com/s3/pricing). AWS bills per
# GB of 2^30 bytes.
S3_STORE_GB_MONTH, GB = 0.023, 2 ** 30

DS_LABELS = {"hlca_core": "HLCA core", "pbmc10k_v3": "PBMC 10k", "tabula_muris_facs": "Tabula Muris FACS"}
R3_LOADERS = {"s1_stream": ".s1", "anndata_zarr_lazy": "AnnData Zarr", "soma_ml": "TileDB-SOMA ML",
              "zarr3_shard": "Zarr v3 sharded", "anndata_inmem": "h5ad gzip, downloaded", "s1_cells": ".s1 (S1Cells)",
              "scdataset": "scDataset", "scdataset_default": "scDataset, fetch 16"}


def _rows(cfg: Config, stage: str, suffix: str = "") -> list[dict]:
    d = cfg.results_dir / stage
    keep = {r.get("dataset") for r in _stage(cfg, "write")}
    out = []
    for p in sorted(d.glob(f"*{suffix}.tsv")) if d.exists() else []:
        if suffix == "" and p.name.count(".") > 2:   # hlca_core.rna.tsv, not hlca_core.rna.r2.tsv
            continue
        out += [r for r in read_tsv(p) if r.get("dataset") in keep]
    return out


def protocol_section(cfg: Config, N: Numbers, out: Path) -> None:
    """Cold reads with the libraries primed: how the first of the three repeats compares with their median."""
    ratios = defaultdict(list)
    for r in _stage(cfg, "decode"):
        if r.get("status") != "ok" or r.get("cache") != "cold" or r.get("what") != "file" or r.get("storage", "local") != "local":
            continue
        ts = [float(x) for x in str(r.get("times_s", "")).split(",") if x]
        if len(ts) >= 3:
            ratios[r["format"]].append(ts[0] / float(np.median(ts)))
    allr = [x for v in ratios.values() for x in v]
    if allr:
        N.put("prot-first-over-median", float(np.median(allr)), "{:.2f}")
        N.put("prot-first-over-median-p90", float(np.quantile(allr, 0.9)), "{:.2f}")
        N.put("prot-n", len(allr))


def access_section(cfg: Config, N: Numbers, out: Path) -> None:
    rows = [r for r in _rows(cfg, "access") if r.get("format") != "s1_chunk"]
    by = defaultdict(list)
    for r in rows:
        by[r["format"]].append(r)
    tab = []
    for f in ("s1", "zarr", "h5ad_gzip", "h5ad_lzf", "h5ad_none", "tenx_h5"):
        v = by.get(f)
        if not v:
            continue
        med = lambda k: float(np.median([_f(x[k]) for x in v if _f(x.get(k)) is not None]))  # noqa: E731
        tab.append({"format": FORMAT_LABELS.get(f, f), "fid": f, "kib": med("bytes_per_batch") / 1024, "req": med("requests_per_batch"),
                    "overread": med("overread"), "index_kib": med("index_bytes") / 1024, "n": len(v)})
        N.put(f"acc-{f}-overread", med("overread"), "{:.1f}")
        N.put(f"acc-{f}-req", med("requests_per_batch"), "{:.0f}")
        N.put(f"acc-{f}-kib", med("bytes_per_batch") / 1024, "{:,.0f}")
    _write(tab, out / "tab_access.tsv", ["format", "fid", "kib", "req", "overread", "index_kib", "n"])
    if tab:
        N.put("accN", max(t["n"] for t in tab))
    # chunk sweep: relative to the library default (49,152), per dataset
    sw = [r for r in _rows(cfg, "access") if r.get("format") == "s1_chunk"]
    pts = []
    for ds in sorted({r["dataset"] for r in sw}):
        v = sorted((r for r in sw if r["dataset"] == ds), key=lambda r: int(r["chunk_nnz_max"]))
        base = next((r for r in v if int(r["chunk_nnz_max"]) == 49152), None)
        if not base:
            continue
        for r in v:
            pts.append({"dataset": ds, "chunk": int(r["chunk_nnz_max"]), "size_rel": _f(r["bits_per_nnz"]) / _f(base["bits_per_nnz"]),
                        "kib": _f(r["bytes_per_batch"]) / 1024, "req": _f(r["requests_per_batch"]), "overread": _f(r["overread"]),
                        "dec_rel": _f(r["decode_mnnz_s"]) / _f(base["decode_mnnz_s"]), "index_kib": _f(r["index_bytes"]) / 1024})
    _write(pts, out / "figacc_chunk.tsv", ["dataset", "chunk", "size_rel", "kib", "req", "overread", "dec_rel", "index_kib"])
    for p in pts:
        if p["chunk"] in (4096, 49152, 393216):
            N.put(f"chunk-{p['dataset']}-{p['chunk']}-size", 100 * (p["size_rel"] - 1), "{:+.1f}")
            N.put(f"chunk-{p['dataset']}-{p['chunk']}-overread", p["overread"], "{:.1f}")


def _cps(r) -> float | None:
    return _f(r.get("sustained_cells_per_s"))


def netsweep_section(cfg: Config, N: Numbers, out: Path) -> None:
    rows = [r for r in _rows(cfg, "netsweep") if r.get("status") == "ok"]
    if not rows:
        return
    ds = cfg.get("report", {}).get("netsweep_dataset", "hlca_core") if isinstance(cfg.get("report"), dict) else "hlca_core"
    if ds not in {r["dataset"] for r in rows}:
        ds = sorted({r["dataset"] for r in rows})[-1]
    N.put("ns-dataset", DS_LABELS.get(ds, ds.replace("_", r"\_")))
    lat = [r for r in rows if r["dataset"] == ds and r["grid"] == "latency"]
    # one .s1: where the netsweep133 stage measured the singlet 1.3.3 defaults (32 requests in flight, four minibatches
    # ahead) at every round-trip time, those rows replace the .s1 rows of the latency grid (run with 1.3.2's 8 requests)
    v133 = [r for r in _rows(cfg, "netsweep133") if r.get("status") == "ok" and r["dataset"] == ds and r.get("grid") == "inflight"
            and int(float(r.get("io_threads") or 0)) == 32 and int(float(r.get("ahead") or 0)) == 4]
    if v133:
        lat = [r for r in lat if r["loader"] != "s1_stream"] + v133
        N.put("ns-s1-version", "1.3.3")
    rtts = sorted({int(float(r["rtt_ms"])) for r in lat})
    loaders = [ld for ld in cfg["netsweep"]["loaders"] if any(r["loader"] == ld for r in lat)]
    curve = []
    for t in rtts:
        row = {"rtt": max(t, 0.5)}   # 0 ms is drawn at 0.5 on the log axis
        for ld in loaders:
            v = [r for r in lat if r["loader"] == ld and int(float(r["rtt_ms"])) == t]
            if v:
                row[f"{ld}_first"] = _f(v[0].get("first_batch_s"))
                row[f"{ld}_cps"] = _cps(v[0])
                b = [float(x) for x in str(v[0].get("batch_s", "")).split(",") if x]
                if b:
                    row[f"{ld}_p50"], row[f"{ld}_p99"] = float(np.median(b)), float(np.quantile(b, 0.99))
                row[f"{ld}_req"] = _f(v[0].get("proxy_requests_per_batch"))
                row[f"{ld}_kib"] = (_f(v[0].get("proxy_bytes_per_batch")) or 0) / 1024 or None
        curve.append(row)
        for ld in loaders:
            if row.get(f"{ld}_cps"):
                N.put(f"ns-{ld}-cps-{t}", row[f"{ld}_cps"], "{:,.0f}")
            if row.get(f"{ld}_first"):
                N.put(f"ns-{ld}-first-{t}", row[f"{ld}_first"], "{:.2f}")
    cols = ["rtt"] + [f"{ld}_{k}" for ld in loaders for k in ("first", "cps", "p50", "p99", "req", "kib")]
    _write(curve, out / "fignet_rtt.tsv", cols)
    # per-minibatch latency CDF at 30 ms
    cdf = []
    for ld in loaders:
        v = [r for r in lat if r["loader"] == ld and int(float(r["rtt_ms"])) == 30]
        b = sorted(float(x) for x in str(v[0].get("batch_s", "")).split(",") if x) if v else []
        for i, x in enumerate(b):
            cdf.append({"loader": ld, "s": x, "q": (i + 1) / len(b)})
    _write(cdf, out / "fignet_cdf.tsv", ["loader", "s", "q"])
    # requests and bytes per minibatch, every loader, at the lowest added latency
    for ld in loaders:
        v = [r for r in lat if r["loader"] == ld and int(float(r["rtt_ms"])) == rtts[0]]
        if v and _f(v[0].get("proxy_requests_per_batch")):
            N.put(f"ns-{ld}-req", _f(v[0]["proxy_requests_per_batch"]), "{:.0f}")
            N.put(f"ns-{ld}-kib", _f(v[0]["proxy_bytes_per_batch"]) / 1024, "{:,.0f}")
    # in flight
    inf = [r for r in rows if r["dataset"] == ds and r["grid"] == "inflight"]
    gbit = float(cfg["netsweep"]["gbit"])
    pts = []
    for r in inf:
        kib = _f(r.get("proxy_bytes_per_batch"))
        cps = _cps(r)
        share = cps / 512 * kib * 8 / (gbit * 1e9) if cps and kib else None
        pts.append({"rtt": int(float(r["rtt_ms"])), "io": int(r["io_threads"]), "ahead": int(r["ahead"]), "cps": cps, "share": share})
    _write(sorted(pts, key=lambda p: (p["rtt"], p["io"])), out / "fignet_inflight.tsv", ["rtt", "io", "ahead", "cps", "share"])
    for t in sorted({p["rtt"] for p in pts}):
        v = [p for p in pts if p["rtt"] == t and p["share"]]
        if v:
            best = max(v, key=lambda p: p["cps"])
            N.put(f"inf-{t}-best-share", 100 * best["share"], "{:.0f}")
            N.put(f"inf-{t}-best-io", best["io"])
            N.put(f"inf-{t}-best-cps", best["cps"], "{:,.0f}")
            one = next((p for p in v if p["io"] == 1), None)
            if one:
                N.put(f"inf-{t}-gain", best["cps"] / one["cps"], "{:.0f}")
            eight = next((p for p in v if p["io"] == 8), None)
            if eight:
                N.put(f"inf-{t}-share-8", 100 * eight["share"], "{:.0f}")
    # DataLoader workers
    wk = [r for r in _rows(cfg, "netsweep") if r["dataset"] == ds and r.get("grid") == "workers"]
    tab = []
    for r in wk:
        tab.append({"loader": R3_LOADERS.get(r["loader"], r["loader"]), "lid": r["loader"], "where": r.get("where", ""),
                    "workers": int(r["workers"]), "cps": _cps(r) if r.get("status") == "ok" else None,
                    "first": _f(r.get("first_batch_s")), "status": r.get("status"), "note": (r.get("note") or "")[:60]})
        if r.get("status") == "ok":
            N.put(f"wk-{r['loader']}-{r.get('where')}-{r['workers']}", _cps(r), "{:,.0f}")
    _write(sorted(tab, key=lambda t: (t["lid"], t["where"], t["workers"])), out / "tab_workers.tsv",
           ["loader", "lid", "where", "workers", "cps", "first", "status", "note"])
    wide = {}
    for t in tab:   # one row per loader and source, one column per worker count ("--" where the run failed)
        w = wide.setdefault((t["lid"], t["where"]), {"loader": t["loader"], "where": "node disk" if t["where"] == "local" else
                                                     f"store, {cfg['netsweep']['workers_rtt_ms']} ms"})
        w[f"w{t['workers']}"] = f"{t['cps']:,.0f}" if t["cps"] else "--"
    _write([wide[k] for k in sorted(wide)], out / "tab_workers_wide.tsv", ["loader", "where"] + [f"w{w}" for w in cfg["netsweep"]["workers"]])


def r2_section(cfg: Config, N: Numbers, out: Path) -> None:
    rows = [r for r in _rows(cfg, "netsweep", ".r2") if r.get("grid") == "r2"]
    if not rows:
        return
    emu = {(r["dataset"], r["loader"]): r for r in _rows(cfg, "netsweep") if r.get("grid") == "latency" and r.get("status") == "ok"
           and int(float(r["rtt_ms"])) == 0}
    stats = {r["dataset"]: int(r["cells"]) for r in _stage(cfg, "stats")} if _stage(cfg, "stats") else {}
    tab = []
    for (ds, ld), v in sorted(_group(rows).items()):
        base = ld.split("@")[0]   # s1_stream@32x4: the .s1 reader with 32 requests and 4 minibatches in flight
        ok = [r for r in v if r.get("status") == "ok"]
        first = [_f(r.get("first_batch_s")) for r in ok if _f(r.get("first_batch_s"))]
        cps = [_cps(r) for r in ok if _cps(r)]
        label = R3_LOADERS.get(base, base) + (f", {ld.split('@')[1].replace('x', ' requests, ')} ahead" if "@" in ld else "")
        rec = {"dataset": DS_LABELS.get(ds, ds.replace("_", " ")), "loader": label, "lid": ld, "first": float(np.median(first)) if first else None,
               "cps": float(np.median(cps)) if cps else None, "rtt_ms": _f(v[0].get("rtt_median_ms")),
               "edge_ms": _f(v[0].get("edge_rtt_median_ms")),
               "gbit1": _f(v[0].get("gbit_1conn")), "gbit8": _f(v[0].get("gbit_8conn"))}
        e = emu.get((ds, base))   # requests and bytes per minibatch do not depend on how many are in flight
        cells = stats.get(ds)
        if e and cells and _f(e.get("proxy_requests_per_batch")) is not None:
            nb = cells / 512
            req = _f(e["proxy_requests_per_batch"]) * nb
            gb = _f(e["proxy_bytes_per_batch"]) * nb / 2**30
            rec.update({"req_epoch": req, "gb_epoch": gb, "usd_r2": req * PRICE["r2"]["get"] + gb * PRICE["r2"]["gb"],
                        "usd_s3": req * PRICE["s3"]["get"] + gb * PRICE["s3"]["gb"]})
        elif base == "anndata_inmem" and cells:
            gb = next((_f(r["bytes"]) for r in _stage(cfg, "write") if r["dataset"] == ds and r["format"] == "h5ad_gzip"), 0) / 2**30
            rec.update({"req_epoch": 1, "gb_epoch": gb, "usd_r2": PRICE["r2"]["get"], "usd_s3": PRICE["s3"]["get"] + gb * PRICE["s3"]["gb"]})
        tab.append(rec)
        for k in ("first", "cps", "req_epoch", "gb_epoch", "usd_s3"):
            if rec.get(k) is not None:
                N.put(f"r2-{ds}-{ld}-{k}", rec[k], {"first": "{:.2f}", "cps": "{:,.0f}", "req_epoch": "{:,.0f}", "gb_epoch": "{:.2f}",
                                                     "usd_s3": "{:.3f}"}[k])
        N.put(f"r2-{ds}-rtt", rec["rtt_ms"], "{:.0f}")
        if rec.get("edge_ms") is not None:
            N.put(f"r2-{ds}-edge", rec["edge_ms"], "{:.0f}")
        N.put(f"r2-{ds}-gbit1", rec["gbit1"], "{:.2f}")
        N.put(f"r2-{ds}-gbit8", rec["gbit8"], "{:.2f}")
    _write(tab, out / "tab_r2.tsv", ["dataset", "loader", "lid", "first", "cps", "req_epoch", "gb_epoch", "usd_r2", "usd_s3",
                                     "rtt_ms", "edge_ms", "gbit1", "gbit8"])
    # Fig. 3d: first minibatch against sustained rate on R2, one point per loader (HLCA core)
    # (one .s1: the singlet 1.3.3 defaults, 32 requests in flight and four minibatches ahead; fmt: the format's color)
    short = {"anndata_inmem": ("h5ad, downloaded first", "east", "other", "h5ad_gzip"),
             "anndata_zarr_lazy": ("AnnData Zarr", "south east", "other", "zarr"),
             "zarr3_shard": ("Zarr v3, sharded", "east", "other", "zarr3"),
             "s1_stream@32x4": (".s1", "west", "s1", "s1")}
    pts = [{"lid": r["lid"], "first": r["first"], "cps": r["cps"], "label": short[r["lid"]][0], "anchor": short[r["lid"]][1],
            "group": short[r["lid"]][2], "fmt": short[r["lid"]][3]} for r in tab if r["dataset"] == "HLCA core" and r["lid"] in short]
    _write(pts, out / "figr2.tsv", ["lid", "first", "cps", "label", "anchor", "group", "fmt"])
    # Fig. 3d guide lines: every (first, rate) on a line takes the same time to deliver one epoch of the matrix,
    # first + cells / rate, through the .s1 point and through the downloaded h5ad file
    cells = stats.get("hlca_core")
    by = {p["lid"]: p for p in pts}
    s, h, z = by.get("s1_stream@32x4"), by.get("anndata_inmem"), by.get("anndata_zarr_lazy")
    if cells and s and h and s["cps"] and h["cps"]:
        curves, labels = [], []
        for key, p, fmt in (("s1", s, "s1"), ("h5ad", h, "h5ad_gzip"), ("zarr", z, "zarr")):
            if not p or not p["cps"]:
                continue
            T = p["first"] + cells / p["cps"]
            N.put(f"r2-epoch-{key}", T, "{:.0f}")
            for x in np.geomspace(0.3, T * 0.995, 120):
                curves.append({"curve": key, "first": float(x), "cps": cells / (T - x)})
            # label: horizontal, left-aligned at 2 s, just below the flat part of its line (above it for AnnData Zarr,
            # whose line runs just over the Zarr v3 point)
            txt = "one epoch in " + f"{T:,.0f}".replace(",", "{,}") + "\\,s"
            side = "south west" if key == "zarr" else "north west"
            labels.append(f"\\node[font=\\sffamily\\tiny, text=c-{fmt}!80!black, anchor={side}, inner sep=1pt] "
                          f"at (axis cs:2,{cells / (T - 2):.1f}) {{{txt}}};")
        (out / "figr2_epoch_labels.tex").write_text("% Generated by `python -m s1bench report` (report_r3.r2_section).\n"
                                                    + "\n".join(labels) + "\n")
        N.put("r2-epoch-cells", cells, "{:,}")
        # epochs after which the downloaded file catches up: first_h + k c / cps_h = first_s + k c / cps_s
        k = (h["first"] - s["first"]) / (cells / s["cps"] - cells / h["cps"])
        N.put("r2-epoch-crossover", k, "{:.1f}")
        _write(curves, out / "figr2_epoch.tsv", ["curve", "first", "cps"])


def _group(rows):
    g = defaultdict(list)
    for r in rows:
        io = r.get("io_threads")
        g[(r["dataset"], r["loader"] + (f"@{io}x{r.get('ahead')}" if io not in (None, "", "NA") else ""))].append(r)
    return g


def writecost_section(cfg: Config, N: Numbers, out: Path) -> None:
    rows = [r for r in _rows(cfg, "writecost") if r.get("status") == "ok"]
    if not rows:
        return
    nnz = {(r["dataset"], r["format"]): _f(r["nnz"]) for r in _stage(cfg, "write") if r.get("status") == "ok"}
    tab = []
    for r in rows:
        n = nnz.get((r["dataset"], r["format"]))
        extra = max(_f(r["peak_rss_mb"]), _f(r.get("peak_child_rss_mb")) or 0) - _f(r["baseline_rss_mb"])
        tab.append({"dataset": r["dataset"], "format": FORMAT_LABELS.get(r["format"], r["format"]), "fid": r["format"],
                    "write_s": _f(r["write_s"]), "mnnz_s": n / _f(r["write_s"]) / 1e6 if n else None,
                    "extra_mb": max(0.0, extra), "peak_mb": max(_f(r["peak_rss_mb"]), _f(r.get("peak_child_rss_mb")) or 0)})
    _write(tab, out / "tab_writecost.tsv", ["dataset", "format", "fid", "write_s", "mnnz_s", "extra_mb", "peak_mb"])
    dss = [d for d in cfg["access"]["sweep_datasets"] if any(t["dataset"] == d for t in tab)]
    wide = {}
    for t in tab:   # one row per format: seconds and extra memory per dataset
        w = wide.setdefault(t["fid"], {"format": t["format"]})
        w[f"{t['dataset']}_s"], w[f"{t['dataset']}_gb"] = t["write_s"], t["extra_mb"] / 1024
    order = sorted(wide, key=lambda f: -(wide[f].get(f"{dss[0]}_s") or 0)) if dss else []
    _write([wide[f] for f in order], out / "tab_writecost_wide.tsv", ["format"] + [f"{d}_{k}" for d in dss for k in ("s", "gb")])
    for i, d in enumerate(dss):
        N.put(f"wcds-{i}", d.replace("_", r"\_"))
    for t in tab:
        if t["fid"] in ("s1", "h5ad_gzip", "zarr", "zarr3", "soma", "csc_i32"):
            N.put(f"wc-{t['dataset']}-{t['fid']}-s", t["write_s"], "{:.1f}")
            N.put(f"wc-{t['dataset']}-{t['fid']}-mnnzs", t["mnnz_s"], "{:.0f}")
            N.put(f"wc-{t['dataset']}-{t['fid']}-mb", t["extra_mb"], "{:,.0f}")


def cores_section(cfg: Config, N: Numbers, out: Path) -> None:
    """The thread sweep to a whole cluster node (stage sweep44: 1 to 44 cores, an exclusive 48-core bigmem node): .s1 read
    cold from disk, per matrix, and the link it keeps up with, b x D(p) for that matrix's stored bits b; medians."""
    rows = [r for r in _rows(cfg, "sweep44/decode", ".sweep") if r.get("status") == "ok" and r["format"] == "s1"
            and r["what"] == "file" and r["cache"] == "cold" and r["mode"] == "native"]
    if not rows:
        return
    bits = {r["dataset"] + "." + r["matrix"]: _f(r["bits_per_nnz"]) for r in _stage(cfg, "write")
            if r.get("status") == "ok" and r["format"] == "s1"}
    tab = []
    for p in sorted({int(r["threads"]) for r in rows}):
        d = [(r["dataset"] + "." + r["matrix"], _f(r["mnnz_per_s"])) for r in rows if int(r["threads"]) == p]
        gbit = [bits[k] * v * 1e6 / 1e9 for k, v in d if bits.get(k) and v]
        tab.append({"threads": p, "mnnzs": float(np.median([v for _, v in d])), "gbit": float(np.median(gbit)), "n": len(d)})
        N.put(f"core-{p}-mnnzs", tab[-1]["mnnzs"], "{:.0f}")
        N.put(f"core-{p}-gbit", tab[-1]["gbit"], "{:.1f}")
    N.put("core-max", tab[-1]["threads"])
    N.put("core-n", min(t["n"] for t in tab))
    _write(tab, out / "fig_cores.tsv", ["threads", "mnnzs", "gbit", "n"])


def scaling_section(cfg: Config, N: Numbers, out: Path) -> None:
    """Fig. 2g: read rate and speed-up from a cold file against cores, 1 to 44 on an exclusive bigmem node, for .s1 and the formats
    whose readers use more than one core (stage sweep44p), with uncompressed CSC and gzip-compressed AnnData (sweep44) as
    single-core references. Per format: median and quartiles over the matrices measured at every core count."""
    rows = [r for r in _rows(cfg, "sweep44/decode", ".sweep") + _rows(cfg, "sweep44p/decode", ".sweep")
            if r.get("status") == "ok" and r["what"] == "file" and r["cache"] == "cold" and r.get("mode") in ("", "native", None)]
    if not rows:
        return
    tab = []
    for fmt in ("s1", "soma", "zarr3", "parquet_coo", "mtx_gz", "csc_i32", "h5ad_gzip"):
        v = defaultdict(dict)
        for r in rows:
            if r["format"] == fmt and _f(r["mnnz_per_s"]):
                v[r["dataset"] + "." + r["matrix"]][int(r["threads"])] = _f(r["mnnz_per_s"])
        ps = sorted({p for d in v.values() for p in d})
        full = [d for d in v.values() if all(p in d for p in ps)]
        if not full:
            continue
        for p in ps:
            x = [d[p] for d in full]
            s = [d[p] / d[ps[0]] for d in full]
            tab.append({"format": fmt, "threads": p, "mnnzs": float(np.median(x)), "q1": float(np.percentile(x, 25)),
                        "q3": float(np.percentile(x, 75)), "speedup": float(np.median(s)),
                        "sp_q1": float(np.percentile(s, 25)), "sp_q3": float(np.percentile(s, 75)), "n": len(full)})
        last = tab[-1]
        N.put(f"scale-{fmt}-max", last["speedup"], "{:.1f}")
        N.put(f"scale-{fmt}-top", last["mnnzs"], "{:.0f}")
        N.put(f"scale-{fmt}-n", last["n"], "{:d}")
    # at the most cores: .s1's lead over the fastest other format, and the largest gain among the formats after .s1 and SOMA
    top = {r["format"]: r for r in tab if r["threads"] == max(q["threads"] for q in tab if q["format"] == r["format"])}
    if "s1" in top and len(top) > 1:
        N.put("scale-s1-lead", top["s1"]["mnnzs"] / max(r["mnnzs"] for f, r in top.items() if f != "s1"), "{:.1f}")
        rest = [r["speedup"] for f, r in top.items() if f not in ("s1", "soma")]
        if rest:
            N.put("scale-rest-max", max(rest), "{:.1f}")
    _write(tab, out / "fig1_scaling.tsv", ["format", "threads", "mnnzs", "q1", "q3", "speedup", "sp_q1", "sp_q3", "n"])


def minibatch_spread_section(cfg: Config, N: Numbers, out: Path) -> None:
    """Fig. 1e-f: the streaming loaders at 1 Gbit/s (stage mb3: three seeds, each with its sustained loop). First
    minibatch: median of the seeds, whiskers their range. Each later minibatch: median of every timed minibatch of the
    three seeds, whiskers the 5th and 95th percentiles."""
    rows = [r for r in _rows(cfg, "mb3/minibatch") if r.get("tier") == "s3_1g" and r.get("status") == "ok"]
    if not rows:
        return
    show = [("s1_stream", r"\textbf{.s1}", "s1"), ("zarr3_shard", "Zarr v3 sharded", "zarr3"),
            ("anndata_zarr_lazy", "AnnData Zarr", "zarr"), ("soma_ml", "TileDB-SOMA ML", "soma")]
    tab = []
    for i, (ld, label, fmt) in enumerate(show):
        rs = [r for r in rows if r["loader"] == ld]
        first = [_f(r["first_batch_s"]) for r in rs if _f(r.get("first_batch_s"))]
        per = [float(x) for r in rs if r.get("batch_s") not in (None, "", "NA") for x in r["batch_s"].split(",")]
        if not first or not per:
            continue
        rec = {"pos": i, "loader": ld, "label": label, "fmt": fmt,
               "first": float(np.median(first)), "first_lo": min(first), "first_hi": max(first),
               "batch_ms": 1000 * float(np.median(per)), "batch_lo": 1000 * float(np.percentile(per, 5)),
               "batch_hi": 1000 * float(np.percentile(per, 95)), "seeds": len(first), "batches": len(per)}
        tab.append(rec)
        N.put(f"mb3-{ld}-first", rec["first"], "{:.2f}")
        N.put(f"mb3-{ld}-batch", rec["batch_ms"], "{:.0f}")
    if tab:
        N.put("mb3-batches", min(r["batches"] for r in tab), "{:,}")
    _write(tab, out / "fig1_mb.tsv", ["pos", "loader", "label", "fmt", "first", "first_lo", "first_hi", "batch_ms",
                                      "batch_lo", "batch_hi", "seeds", "batches"])


def speed_boxes_section(cfg: Config, N: Numbers, out: Path) -> None:
    """Fig. 1c: the read rate of every matrix from a cold file on the headline cores, per format, in the row order of the
    size panel (generated/fig1a_size.tsv, written earlier in the same report) so the two panels share their rows."""
    import zlib
    order = out / "fig1a_size.tsv"
    if not order.exists():
        return
    pos = {r["format"]: int(r["pos"]) for r in read_tsv(order)}
    H = int(cfg["threads"]["headline"])
    rows = [r for r in _stage(cfg, "decode") if r.get("status") == "ok" and r["what"] == "file" and r["cache"] == "cold"
            and int(r["threads"]) == H and r["format"] in pos and (r["format"] != "s1" or r.get("mode") == "native")
            and r.get("storage", "local") == "local" and r.get("run", "main") == "main" and _f(r.get("mnnz_per_s"))]
    pts, boxes = [], []
    for f, i in sorted(pos.items(), key=lambda kv: kv[1]):
        v = [(r["dataset"] + "/" + r["matrix"], _f(r["mnnz_per_s"])) for r in rows if r["format"] == f]
        if not v:
            continue
        for k, x in v:
            jit = (zlib.crc32(k.encode()) % 1000 / 999 - 0.5) * 0.5
            pts.append({"pos": i, "y": i + jit, "format": f, "mnnzs": x})
        a_ = np.array([x for _, x in v])
        q1, med, q3 = np.quantile(a_, [0.25, 0.5, 0.75])
        inside = a_[(a_ >= q1 - 1.5 * (q3 - q1)) & (a_ <= q3 + 1.5 * (q3 - q1))]
        boxes.append(f"\\figbox{{{i}}}{{{q1:.4f}}}{{{med:.4f}}}{{{q3:.4f}}}{{{inside.min():.4f}}}{{{inside.max():.4f}}}{{c-{f}}}")
        N.put(f"speed-{f}", med, "{:.0f}")
    _write(pts, out / "fig1_speed_points.tsv", ["pos", "y", "format", "mnnzs"])
    (out / "fig1_speed_boxes.tex").write_text("% Generated by `python -m s1bench report` (report_r3.speed_boxes_section). "
                                             "Do not edit.\n" + "\n".join(boxes) + "\n")


# Fig. 1d geometry, as in figures/fig1b_panel.tex: the square axis (side in cm) and its data ranges
FIG1D_SIDE, FIG1D_X, FIG1D_Y = 4.9, (0.0, 70.0), (0.0, 260.0)


def place_labels(pts: dict, labels: dict, w_cm: float, h_cm: float, obstacles=()) -> dict:
    """In-place labels for points given in cm: for each point the anchor (the side of the label that faces the point) and
    the label's offset from the point, chosen to overlap other labels, markers and obstacles as little as possible
    (greedy, most crowded point first, then refined for a few passes). Labels are set in 5-point type."""
    cw, ch, rm = 0.08, 0.16, 0.075                           # cm per character, label height, marker radius
    marks = [(x - rm, y - rm, x + rm, y + rm) for x, y in pts.values()] + list(obstacles)
    width = {f: cw * len(labels[f].replace("\\textbf{", "").replace("}", "")) + 0.04 for f in pts}
    dirs = {"west": (1, 0), "east": (-1, 0), "south": (0, 1), "north": (0, -1),
            "south west": (1, 1), "north west": (1, -1), "south east": (-1, 1), "north east": (-1, -1)}
    gaps = (0.10, 0.22, 0.36)                                   # distance from the point's center to the label (cm)

    def box(f, a, g):
        (x, y), (dx, dy), w = pts[f], dirs[a], width[f]
        ax, ay = x + dx * g * (0.75 if dy else 1), y + dy * g * (0.75 if dx else 1)   # the anchor's position
        x0 = ax if dx > 0 else ax - w if dx < 0 else ax - w / 2
        y0 = ay if dy > 0 else ay - ch if dy < 0 else ay - ch / 2
        return (x0, y0, x0 + w, y0 + ch), (ax - x, ay - y)

    def ov(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    def cost(f, b, a, g, placed):
        c = sum(ov(b, q) for h, q in placed.items() if h != f) * 60 + sum(ov(b, m) for m in marks) * 40
        c += 8 * (max(0, -b[0]) + max(0, b[2] - w_cm) + max(0, -b[1]) + max(0, b[3] - h_cm))
        return c + 0.01 * g + 0.002 * (" " in a) + 0.001 * (dirs[a][0] == 0)   # prefer close, straight beside

    crowd = {f: sum(1 for g in pts if g != f and abs(pts[g][0] - pts[f][0]) < 0.9 and abs(pts[g][1] - pts[f][1]) < 0.5)
             for f in pts}
    placed, choice = {}, {}
    for _ in range(4):
        for f in sorted(pts, key=lambda f: -crowd[f]):
            choice[f] = min((cost(f, box(f, a, g)[0], a, g, placed), a, g) for a in dirs for g in gaps)[1:]
            placed[f] = box(f, *choice[f])[0]
    return {f: (a, box(f, a, g)[1]) for f, (a, g) in choice.items()}


def tradeoff_labels_section(cfg: Config, N: Numbers, out: Path) -> None:
    """Fig. 1d on linear axes: every median labeled in place, close to its point (generated/fig1d_labels.tex)."""
    src = out / "fig1c_tradeoff.tsv"
    if not src.exists():
        return
    rows = read_tsv(src)
    S, (x0, x1), (y0, y1) = FIG1D_SIDE, FIG1D_X, FIG1D_Y
    data = {r["format"]: (float(r["bits"]), float(r["mnnzs"])) for r in rows}
    pts = {f: ((x - x0) / (x1 - x0) * S, (y - y0) / (y1 - y0) * S) for f, (x, y) in data.items()}
    note = (S - 2.1, S - 0.95, S, S - 0.5)                        # the dashed-line note in the top right corner
    place = place_labels(pts, {r["format"]: r["label"] for r in rows}, S, S, [note])
    lines = ["% Generated by `python -m s1bench report` (report_r3.tradeoff_labels_section). Do not edit."]
    for r in rows:
        a, (sx, sy) = place[r["format"]]
        lines.append(f"\\node[fblabel, anchor={a}, xshift={sx:.3f}cm, yshift={sy:.3f}cm] "
                     f"at (axis cs:{data[r['format']][0]:.3f},{data[r['format']][1]:.3f}) {{{r['label']}}};")
    (out / "fig1d_labels.tex").write_text("\n".join(lines) + "\n")


def billion_cells_section(cfg: Config, N: Numbers, out: Path) -> None:
    """What a billion cells would cost in S3 Standard as .s1 and as gzip-compressed AnnData: bytes per cell is the median
    over the benchmark matrices of each file's size (counts, gene identifiers and barcodes) divided by its cells; a year of
    storage and one download to the internet at list prices (PRICE, S3_STORE_GB_MONTH). Both sizes stay within the
    first price tier (10 TB a month out, 50 TB stored)."""
    per = defaultdict(list)
    for r in _stage(cfg, "write"):
        if r.get("status") == "ok" and _f(r.get("bytes")) and _f(r.get("cells")):
            per[r["format"]].append(_f(r["bytes"]) / _f(r["cells"]))
    for fmt, key in (("s1", "s1"), ("h5ad_gzip", "gz")):
        if not per.get(fmt):
            return
        gb = float(np.median(per[fmt])) * 1e9 / GB
        N.put(f"bn-{key}-tb", gb * GB / 1e12, "{:.1f}")
        N.put(f"bn-{key}-store", gb * S3_STORE_GB_MONTH * 12, "{:,.0f}")
        N.put(f"bn-{key}-egress", gb * PRICE["s3"]["gb"], "{:,.0f}")
        N.put(f"bn-{key}-bpc", float(np.median(per[fmt])), "{:,.0f}")


# Training compute per cell for published single-cell models, to ask whether training is bound by the GPU or by data
# delivery. Architectures: Geneformer V1 (6 layers, width 256, feed-forward 512, 2,048 genes per cell), scGPT (12 layers,
# width 512, feed-forward 512, 1,200 genes), scVI/scANVI with scvi-tools defaults (one hidden layer of 128, latent 10,
# zero-inflated negative binomial: a scale and a dropout head), and a plain dense autoencoder (1,024-256-64, mirrored).
# A transformer pass costs 2 N n + 4 L n^2 d FLOPs for N non-embedding parameters over n tokens (matrix products and
# attention); a dense network 2 P per cell; training is three passes (forward and backward). GPU: NVIDIA H100 SXM, dense
# BF16 989 TFLOP/s (FP32 67 TFLOP/s for scVI, which trains in 32-bit by default), at 40% of peak.
H100 = {"bf16": 989.4e12, "fp32": 66.9e12}
GPU_UTIL = 0.40
COMPUTE_MODELS = [   # key, label, kind, shape, precision
    ("geneformer", "Geneformer V1", "tf", dict(L=6, d=256, ff=512, n=2048), "bf16"),
    ("scgpt", "scGPT", "tf", dict(L=12, d=512, ff=512, n=1200), "bf16"),
    ("ae", "Autoencoder, 20,000 genes", "dense", dict(G=20000, layers=(1024, 256, 64)), "bf16"),
    ("scvi", "scVI / scANVI, 20,000 genes", "scvi", dict(G=20000, H=128, Z=10), "fp32"),
    ("scvihvg", "scVI / scANVI, 2,000 genes", "scvi", dict(G=2000, H=128, Z=10), "fp32"),
]


def _train_flops(kind: str, s: dict) -> float:
    if kind == "tf":
        n_par = s["L"] * (4 * s["d"] ** 2 + 2 * s["d"] * s["ff"])
        return 3 * (2 * n_par * s["n"] + 4 * s["L"] * s["n"] ** 2 * s["d"])
    if kind == "dense":
        dims = (s["G"], *s["layers"])
        p = sum(a * b for a, b in zip(dims, dims[1:]))
        return 3 * 2 * (2 * p)                       # encoder and mirrored decoder
    p = 3 * s["G"] * s["H"] + 3 * s["H"] * s["Z"]   # scVI: encoder G-H-(2Z), decoder Z-H-(2G)
    return 3 * 2 * p


def compute_bound_section(cfg: Config, N: Numbers, out: Path) -> None:
    """For each model: training FLOPs per cell, cells per second one H100 can train on, and what feeding it takes: link
    bandwidth at the median bytes per cell of .s1 and of gzip-compressed AnnData, the number of GPUs one 1 Gbit/s link
    keeps busy, and the cores .s1 needs to decode that many cells (the one-core AVX2 codec rate from memory, at the median
    non-zeros per cell)."""
    per, nz = defaultdict(list), []
    for r in _stage(cfg, "write"):
        if r.get("status") == "ok" and _f(r.get("bytes")) and _f(r.get("cells")):
            per[r["format"]].append(_f(r["bytes"]) / _f(r["cells"]))
            if r["format"] == "s1" and _f(r.get("nnz")):
                nz.append(_f(r["nnz"]) / _f(r["cells"]))
    core = [_f(r["mnnz_per_s"]) for r in _stage(cfg, "decode") if r.get("status") == "ok" and r["format"] == "s1"
            and r["what"] == "codec_cpp" and r["mode"] == "native" and r.get("isa") == "avx2" and r["threads"] == "1"]
    if not per.get("s1") or not per.get("h5ad_gzip") or not nz or not core:
        return
    bpc = {"s1": float(np.median(per["s1"])), "gz": float(np.median(per["h5ad_gzip"]))}
    cells_core = float(np.median(core)) * 1e6 / float(np.median(nz))
    N.put("comp-nnz-cell", float(np.median(nz)), "{:,.0f}")
    N.put("comp-cells-core", cells_core / 1e3, "{:.0f}")
    N.put("comp-util", GPU_UTIL * 100, "{:.0f}")
    rows = []
    for key, label, kind, shape, prec in COMPUTE_MODELS:
        fl = _train_flops(kind, shape)
        cps = H100[prec] * GPU_UTIL / fl
        rec = {"model": label, "gflop": fl / 1e9, "cps": cps, "prec": prec.upper(),
               "gbit_s1": cps * bpc["s1"] * 8 / 1e9, "gbit_gz": cps * bpc["gz"] * 8 / 1e9,
               "gpus_s1": 1e9 / 8 / bpc["s1"] / cps, "gpus_gz": 1e9 / 8 / bpc["gz"] / cps, "cores": cps / cells_core}
        rows.append(rec)
        N.put(f"comp-{key}-gflop", rec["gflop"], "{:.3g}")
        N.put(f"comp-{key}-cps", cps, "{:,.0f}" if cps < 1e5 else "{:,.0f}")
        N.put(f"comp-{key}-kcps", cps / 1e3, "{:,.0f}")
        N.put(f"comp-{key}-mcps", cps / 1e6, "{:.1f}")
        N.put(f"comp-{key}-gbit-s1", rec["gbit_s1"], "{:.2g}")
        N.put(f"comp-{key}-gbit-gz", rec["gbit_gz"], "{:.2g}")
        N.put(f"comp-{key}-gpus-s1", rec["gpus_s1"], "{:.0f}" if rec["gpus_s1"] >= 1 else "{:.2f}")
        N.put(f"comp-{key}-gpus-gz", rec["gpus_gz"], "{:.0f}" if rec["gpus_gz"] >= 1 else "{:.2f}")
        N.put(f"comp-{key}-cores", rec["cores"], "{:.1f}" if rec["cores"] < 10 else "{:.0f}")
    N.put("bn-ratio", bpc["gz"] / bpc["s1"], "{:.1f}")
    # what .s1 stores besides the entropy-coded payload (tables, chunk index, headers), per non-zero: the ladder's member
    # minus its payload, median over matrices, and as a share of the member
    lad = [r for r in _stage(cfg, "ladder") if _f(r.get("L4_s1_member")) and _f(r.get("L4_s1_payload"))]
    if lad:
        ov = [_f(r["L4_s1_member"]) - _f(r["L4_s1_payload"]) for r in lad]
        N.put("s1-overhead-bpn", float(np.median(ov)), "{:.2f}")
        N.put("s1-overhead-pct", float(np.median([100 * o / _f(r["L4_s1_member"]) for o, r in zip(ov, lad)])), "{:.1f}")
    _write(rows, out / "tab_compute.tsv", ["model", "prec", "gflop", "cps", "gbit_s1", "gbit_gz", "gpus_s1", "gpus_gz", "cores"])


def _repel(targets: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """1-D label placement: positions as close as possible to ``targets`` (any order) with at least ``gap`` between
    neighbours, inside [lo, hi]. Overlapping labels merge into groups spaced by ``gap`` around their mean target."""
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    groups = [[i] for i in order]                     # each group: label indices in ascending target order

    def place(g):
        c = float(np.mean([targets[i] for i in g]))
        start = c - gap * (len(g) - 1) / 2
        start = min(max(start, lo), hi - gap * (len(g) - 1))
        return start

    merged = True
    while merged:
        merged = False
        for k in range(len(groups) - 1):
            a, b = groups[k], groups[k + 1]
            if place(a) + gap * len(a) > place(b) + 1e-12:
                groups[k:k + 2] = [a + b]
                merged = True
                break
    pos = [0.0] * len(targets)
    for g in groups:
        s = place(g)
        for j, i in enumerate(g):
            pos[i] = s + j * gap
    return pos


def _label_lines(rows: list[tuple[str, float, float]], gap: float, lo: float, hi: float, log: bool, path: Path) -> None:
    """Direct labels at the right end of a panel's lines: rows are (format, x end, y end) in axis units; each label is
    moved vertically by _repel (in log10 units on a log axis) and joined to its line's end by a short leader."""
    ys = [np.log10(y) if log else y for _f_, _x, y in rows]
    pos = _repel(ys, gap, lo, hi)
    out = ["% Direct labels at the right end of each line (report_r3._label_lines). Do not edit."]
    for (fmt, x, y), q in zip(rows, pos):
        yl = 10 ** q if log else q
        lab = "\\textbf{\\sone}" if fmt == "s1" else FORMAT_LABELS.get(fmt, fmt)
        col = "sone!85!black" if fmt == "s1" else "ink2!80"
        out.append("\\draw[gen!55, line width=0.3pt] ($(axis cs:%g,%g)+(0.07cm,0)$) -- ++(0.1cm,0) -- ($(axis cs:%g,%g)+(0.32cm,0)$) "
                   "node[anchor=west, font=\\sffamily\\tiny, text=%s, inner sep=1pt] {%s};" % (x, y, x, yl, col, lab))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def direct_labels_section(cfg: Config, N: Numbers, out: Path) -> None:
    """Labels in place of legends where every format but .s1 is gray: Fig. 2g (speed-up against cores, at the most
    cores) and Fig. 3b (what a link delivers, at the fastest link). Gaps are sized for the smaller (bioRxiv) panels."""
    sc = out / "fig1_scaling.tsv"
    if sc.exists():
        last = {}
        for r in read_tsv(sc):
            if _f(r.get("speedup")) is not None:
                last[r["format"]] = (int(r["threads"]), _f(r["speedup"]))
        rows = [(f, float(x), y) for f, (x, y) in last.items()]
        if rows:
            _label_lines(rows, gap=0.82, lo=0.1, hi=9.9, log=False, path=out / "fig1g_labels.tex")
    cv = out / "fignet_curves.tsv"
    if cv.exists():
        cur = [r for r in read_tsv(cv) if _f(r.get("gbit")) is not None]
        if cur:
            end = max(cur, key=lambda r: _f(r["gbit"]))
            fmts = [k for k in end if k not in ("gbit", "s1_net", "s1_dec") and _f(end[k])]
            rows = [(f, _f(end["gbit"]), _f(end[f])) for f in fmts]
            _label_lines(rows, gap=0.185, lo=np.log10(0.35), hi=np.log10(1400), log=True, path=out / "fignet_labels.tex")


def run(cfg: Config, N: Numbers, out: Path) -> None:
    billion_cells_section(cfg, N, out)
    compute_bound_section(cfg, N, out)
    cores_section(cfg, N, out)
    tradeoff_labels_section(cfg, N, out)
    speed_boxes_section(cfg, N, out)
    minibatch_spread_section(cfg, N, out)
    scaling_section(cfg, N, out)
    protocol_section(cfg, N, out)
    access_section(cfg, N, out)
    netsweep_section(cfg, N, out)
    r2_section(cfg, N, out)
    writecost_section(cfg, N, out)
    direct_labels_section(cfg, N, out)
