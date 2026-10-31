"""Report sections for the review-round-3 measurements: the primed cold-read protocol, random-access costs (access),
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


def run(cfg: Config, N: Numbers, out: Path) -> None:
    protocol_section(cfg, N, out)
    access_section(cfg, N, out)
    netsweep_section(cfg, N, out)
    r2_section(cfg, N, out)
    writecost_section(cfg, N, out)
