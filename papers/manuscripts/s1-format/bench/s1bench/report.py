"""Stage `report`: aggregate results/ into generated/*.tsv (read by the figures and tables) and generated/numbers.tex.

The manuscript never contains a typed result. Each value it prints is a macro `\\nm{key}` defined here; a key the
pipeline has not produced prints as a red [??key] in the PDF, and `report` lists every key the .tex files use but this
stage did not define. Every generated table is written even when empty (header only), so the manuscript always builds.

Column conventions: `pos` is the plotting position (0..n-1, top to bottom); `label` is the text on the axis.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .util import PAPER_DIR, Config, load_datasets, log, read_stage

# ----------------------------------------------------------------------------------------------- labels and orders
FORMAT_LABELS = {
    "h5ad_none": "h5ad (none)", "csc_i32": "CSC, int32", "mtx_gz": "MTX.gz", "loom": "loom", "h5ad_lzf": "h5ad lzf",
    "h5ad_gzip": "h5ad gzip", "tenx_h5": "10x HDF5", "zarr": "AnnData Zarr", "soma": "TileDB-SOMA",
    "zarr3": "Zarr v3 sharded",
    "parquet_coo": "Parquet COO", "rds_gz": "rds (gzip)", "rds_xz": "rds (xz)", "csc_zstd19": "CSC + zstd-19",
    "bpcells": "BPCells", "s1": "\\textbf{.s1}", "s1_pack": ".s1, shared gene pack",
}
LOADER_LABELS = {
    "anndata_inmem": "AnnData in memory", "anndata_backed": "AnnData backed", "scdataset": "scDataset",
    "anndata_zarr_lazy": "AnnData Zarr", "soma_ml": "TileDB-SOMA ML", "bpcells": "BPCells (R)",
    "s1_stream": "\\textbf{.s1}", "s1_stream_canonical": ".s1, source gene order",
    "s1_stream_seq": ".s1, one request at a time", "zarr3_shard": "Zarr v3 sharded",
    "scdataset_default": "scDataset, fetch 16",
}
TIER_ORDER = ["s3_0.1g", "s3_1g", "s3_10g", "nfs", "local"]
FIG1D_TIERS = ["local", "nfs", "s3_10g", "s3_1g", "s3_1g_wan", "s3_0.1g"]   # fastest storage first
TIER_LABELS = {"s3_0.1g": "0.1 Gbit/s", "s3_1g": "1 Gbit/s", "s3_10g": "10 Gbit/s", "nfs": "shared filesystem (NFS)",
               "local": "node-local disk", "s3_1g_wan": "1 Gbit/s, 30 ms"}
CLASS_LABELS = {
    "droplet_3p": "droplet 3'", "droplet_3p_nuclei": "droplet nuclei", "droplet_5p": "droplet 5'", "droplet_probe": "droplet probe",
    "droplet_other": "other droplet", "combinatorial": "combinatorial", "plate_umi": "plate, UMI", "plate_reads": "plate, reads",
    "spatial_seq": "spatial, sequencing", "spatial_imaging": "spatial, imaging", "multimodal": "multimodal",
}
LADDER = [("L0_csc_i32", "CSC as 32-bit integers"), ("L1_leb128", "variable-length integers"),
          ("L2_gaps_leb_zstd3", "gaps + LEB128 + zstd-3"), ("L3_gaps_leb_zstd19", "gaps + LEB128 + zstd-19"),
          ("L4_s1_payload", "\\textbf{.s1}"), ("floor", "best count model (cross-validated)")]
MODELS = [("poisson", "Poisson, per gene"), ("nb", "negative binomial, per gene"), ("gene_depth", "per gene $\\times$ depth"),
          ("cluster", "per gene $\\times$ depth $\\times$ cluster")]
CODERS = [("zstd22", "zstd -22"), ("xz9e", "xz -9e"), ("brotli11", "brotli -11"), ("bzip2_9", "bzip2 -9"), ("cm", "context mixing")]


# ----------------------------------------------------------------------------------------------- helpers
def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _headline_decode(cfg: Config) -> list[dict]:
    """Decode rows of the main run from node-local disk (and the in-memory codec rows): not the thread sweep, not the
    shared-filesystem tier, which have their own tables."""
    return [r for r in _stage(cfg, "decode")
            if r.get("run", "main") == "main" and r.get("storage", "local") in ("local", "memory")]


def _stats(vals: Iterable) -> dict:
    """Median, quartiles and range; `em`/`ep` (median - q1, q3 - median) and `rm`/`rp` (median - min, max - median) are the
    error-bar lengths pgfplots wants."""
    a = np.array([v for v in (_f(x) for x in vals) if v is not None])
    if not len(a):
        return {"median": None, "q1": None, "q3": None, "lo": None, "hi": None, "n": 0}
    med, q1, q3 = float(np.median(a)), float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))
    return {"median": med, "q1": q1, "q3": q3, "lo": float(a.min()), "hi": float(a.max()), "n": int(len(a)),
            "em": med - q1, "ep": q3 - med, "rm": med - float(a.min()), "rp": float(a.max()) - med}


class Numbers:
    def __init__(self):
        self.d: dict[str, str] = {}

    def put(self, key: str, v, fmt: str = "{:.2f}") -> None:
        if v is None:
            return
        if isinstance(v, str):
            self.d[key] = v
        elif isinstance(v, (int, np.integer)):
            self.d[key] = f"{int(v):,}".replace(",", "{,}")
        else:
            self.d[key] = fmt.format(float(v)).replace(",", "{,}")

    def write(self, path: Path) -> None:
        lines = ["% Generated by `python -m s1bench report`. Do not edit.", ""]
        for k in sorted(self.d):
            lines.append(f"\\defnum{{{k}}}{{{self.d[k]}}}")
        path.write_text("\n".join(lines) + "\n")


def _stage(cfg: Config, stage: str) -> list[dict]:
    """A stage's rows for the benchmark panel: excluded datasets (datasets.yaml `excluded`) and the out-of-domain
    extensions are left out; the extensions are reported on their own (extension_numbers)."""
    keep = {d["id"] for d in load_datasets(cfg) if not d.get("excluded")}
    return [r for r in read_stage(cfg, stage) if r.get("dataset") in keep]


def _tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


# ----------------------------------------------------------------------------------------------- sections
def datasets_table(cfg: Config, N: Numbers, out: Path) -> dict:
    st = _stage(cfg, "stats")
    cite = {d["id"]: d.get("cite") for d in load_datasets(cfg, True)}
    rows = []
    for r in sorted(st, key=lambda r: (list(CLASS_LABELS).index(r["protocol_class"]) if r["protocol_class"] in CLASS_LABELS else 99,
                                        int(r["cells"]))):
        rows.append({"dataset": _tex_escape(r["dataset"]), "matrix": _tex_escape(r["matrix"]), "class": CLASS_LABELS.get(r["protocol_class"], r["protocol_class"]),
                     "tier": r["tier"], "cells": int(r["cells"]), "genes": int(r["genes"]), "nnz": int(r["nnz"]),
                     "density": _f(r["density"]), "frac_ones": _f(r["frac_ones"]), "counts_per_cell": _f(r["counts_per_cell_median"]),
                     "ref": ("\\cite{%s}" % cite[r["dataset"]]) if cite.get(r["dataset"]) else ""})
    _write(rows, out / "tab_datasets.tsv", ["dataset", "matrix", "class", "tier", "cells", "genes", "nnz", "density", "frac_ones", "counts_per_cell", "ref"])
    N.put("nMatrices", len(st))
    N.put("nDatasets", len({r["dataset"] for r in st}))
    N.put("nProtocolClasses", len({r["protocol_class"] for r in st}))
    if st:
        N.put("nCellsTotal", sum(int(r["cells"]) for r in st if r["matrix"] in ("rna",)))
        N.put("nnzTotalBillions", sum(int(r["nnz"]) for r in st) / 1e9, "{:.1f}")
        fo = _stats(r["frac_ones"] for r in st)
        N.put("fracOnesMedianPct", 100 * fo["median"] if fo["median"] is not None else None, "{:.0f}")
        N.put("fracOnesMinPct", 100 * fo["lo"] if fo["lo"] is not None else None, "{:.0f}")
        N.put("fracOnesMaxPct", 100 * fo["hi"] if fo["hi"] is not None else None, "{:.0f}")
    return {(r["dataset"], r["matrix"]): r for r in st}


def size_section(cfg: Config, N: Numbers, out: Path) -> dict:
    w = [r for r in _stage(cfg, "write") if r.get("status") == "ok"]
    by_fmt = defaultdict(list)
    by_unit = defaultdict(dict)
    for r in w:
        by_fmt[r["format"]].append(_f(r["bits_per_nnz"]))
        by_unit[(r["dataset"], r["matrix"])][r["format"]] = _f(r["bytes"])
    fmts = [f for f in cfg["formats"] if f in by_fmt]
    stats = {f: _stats(by_fmt[f]) for f in fmts}
    order = sorted(fmts, key=lambda f: -(stats[f]["median"] or 0))
    rows = [{"pos": i, "format": f, "label": FORMAT_LABELS.get(f, f), "group": _group(f), **stats[f]} for i, f in enumerate(order)]
    _write(rows, out / "fig1a_size.tsv", ["pos", "format", "label", "group", "median", "q1", "q3", "lo", "hi", "n", "em", "ep", "rm", "rp"])
    for f in fmts:
        N.put(f"size-{f}", stats[f]["median"])
        ratios = [u[f] / u["s1"] for u in by_unit.values() if u.get(f) and u.get("s1")]
        rs = _stats(ratios)
        N.put(f"ratio-{f}", rs["median"], "{:.1f}")
        N.put(f"ratiomin-{f}", rs["lo"], "{:.1f}")
        N.put(f"ratiomax-{f}", rs["hi"], "{:.1f}")
    if "s1" in stats:
        N.put("s1BpnMedian", stats["s1"]["median"])
        N.put("s1BpnMin", stats["s1"]["lo"])
        N.put("s1BpnMax", stats["s1"]["hi"])
        smallest = sum(1 for u in by_unit.values() if u.get("s1") and all(u["s1"] <= v for k, v in u.items() if v and k != "s1_pack"))
        N.put("s1SmallestCount", smallest)
        others = [f for f in fmts if f not in ("s1", "s1_pack")]
        if others:
            # paired by matrix: the smallest other file on each matrix over .s1, and the format that is most often that file
            pair, who = [], defaultdict(int)
            for u in by_unit.values():
                cand = {f: u[f] for f in others if u.get(f)}
                if cand and u.get("s1"):
                    b = min(cand, key=cand.get)
                    pair.append(cand[b] / u["s1"])
                    who[b] += 1
            best = max(who, key=who.get) if who else min(others, key=lambda f: stats[f]["median"] or 1e9)
            N.put("bestOtherFormat", FORMAT_LABELS.get(best, best).replace("\\textbf{", "").replace("}", ""))
            N.put("bestOtherFormatCount", who.get(best, 0))
            N.put("ratioBestOther", _stats(pair)["median"], "{:.1f}")
    # every matrix's size relative to .s1, per format (Fig. 1a: one point per matrix)
    st_cls = {(r["dataset"], r["matrix"]): r.get("protocol_class", "") for r in _stage(cfg, "stats")}
    pts = []
    for row in rows:
        f = row["format"]
        for (ds, m), u in by_unit.items():
            if u.get(f) and u.get("s1"):
                pts.append({"pos": row["pos"], "format": f, "group": row["group"], "ratio": u[f] / u["s1"],
                            "class": st_cls.get((ds, m), "")})
    _write(pts, out / "fig1a_points.tsv", ["pos", "format", "group", "ratio", "class"])
    for row in rows:   # the median ratio per format, for the marker over the points
        rr = _stats(u[row["format"]] / u["s1"] for u in by_unit.values() if u.get(row["format"]) and u.get("s1"))
        row["ratio_median"] = rr["median"]
    _write(rows, out / "fig1a_size.tsv", ["pos", "format", "label", "group", "median", "q1", "q3", "lo", "hi", "n", "em", "ep",
                                          "rm", "rp", "ratio_median"])
    # per-dataset size table (bits per non-zero), formats as columns
    tab = []
    for (ds, m), u in sorted(by_unit.items()):
        rec = {"dataset": _tex_escape(ds) + ("" if m == "rna" else f" ({_tex_escape(m)})")}
        nnz = next((_f(r["nnz"]) for r in w if r["dataset"] == ds and r["matrix"] == m), None)
        for f in FORMAT_LABELS:
            rec[f] = 8 * u[f] / nnz if u.get(f) and nnz else None
        tab.append(rec)
    _write(tab, out / "tab_size.tsv", ["dataset", *FORMAT_LABELS])        # every format, so the paper's columns always exist
    return stats


def _best(f: dict) -> Optional[float]:
    v = [_f(f.get(f"{k}_bpn")) for k, _ in MODELS]
    v = [x for x in v if x is not None]
    return min(v) if v else None


def _best_name(f: dict) -> Optional[str]:
    v = {k: _f(f.get(f"{k}_bpn")) for k, _ in MODELS}
    v = {k: x for k, x in v.items() if x is not None}
    return min(v, key=v.get) if v else None


def _group(f: str) -> str:
    return "s1" if f.startswith("s1") else ("domain" if f in ("bpcells", "soma") else "general")


def decode_section(cfg: Config, N: Numbers, out: Path, size_stats: dict) -> None:
    d = [r for r in _headline_decode(cfg) if r.get("status") == "ok"]
    H = int(cfg["threads"]["headline"])

    def ran(r, isa):   # a codec row counts for an ISA only if that kernel ran (the CPU may not support it)
        return isa is None or (r["isa"] == isa and r.get("isa_used") in (None, "", isa))

    def pick(fmt, what, threads, cache=None, mode=None, isa=None):
        return [_f(r["mnnz_per_s"]) for r in d if r["format"] == fmt and r["what"] == what and int(r["threads"]) == threads
                and (cache is None or r["cache"] == cache) and (mode is None or r["mode"] == mode) and ran(r, isa)]

    # .s1 is compared in source gene order, the order every other format returns; stored order is an extra row
    rows, trade = [], []
    plan = [(f, "canonical" if f == "s1" else None, FORMAT_LABELS.get(f, f)) for f in cfg["formats"] if f != "s1_pack"]
    for f, mode, label in plan + [("s1", "native", ".s1, stored gene order")]:
        c1 = _stats(pick(f, "file", 1, "cold", mode))
        c4 = _stats(pick(f, "file", H, "cold", mode))
        w4 = _stats(pick(f, "file", H, "warm", mode))
        if c4["n"] == 0 and c1["n"] == 0:
            continue
        rows.append({"format": label, "cold_t1": c1["median"], "cold_th": c4["median"], "warm_th": w4["median"]})
        if mode == "native":
            N.put(f"dec-s1native-cold-t{H}", c4["median"], "{:.0f}")
            continue
        if c4["median"] and f in size_stats:
            trade.append({"format": f, "label": FORMAT_LABELS.get(f, f).replace("\\textbf{", "").replace("}", ""), "group": _group(f),
                          "bits": size_stats[f]["median"], "mnnzs": c4["median"]})
        N.put(f"dec-{f}-cold-t{H}", c4["median"], "{:.0f}")
    # the fastest other format to read cold at the headline thread count, and how much faster .s1 is: the median over
    # matrices of the per-matrix ratio (paired), not a ratio of medians
    cold = {f: _stats(pick(f, "file", H, "cold", "canonical" if f == "s1" else None))["median"] for f in cfg["formats"]}
    others = {f: v for f, v in cold.items() if v and not f.startswith("s1")}
    if others and cold.get("s1"):
        best = max(others, key=others.get)
        per = {}
        for r in d:
            if r["what"] == "file" and r["cache"] == "cold" and int(r["threads"]) == H and r["format"] in ("s1", best) \
                    and (r["format"] != "s1" or r["mode"] == "canonical"):
                per.setdefault((r["dataset"], r["matrix"]), {})[r["format"]] = _f(r["mnnz_per_s"])
        ratios = _stats(u["s1"] / u[best] for u in per.values() if u.get("s1") and u.get(best))
        N.put("dec-best-other-name", FORMAT_LABELS.get(best, best))
        N.put("dec-speedup-best-other", ratios["median"], "{:.1f}")
        N.put("dec-speedup-best-other-lo", ratios["lo"], "{:.1f}")
        N.put("dec-speedup-best-other-hi", ratios["hi"], "{:.1f}")
    # against the fastest other format on each matrix (which is not always the same format): the paired ratio, how many
    # matrices .s1 reads fastest, and the largest matrix on which another format is faster
    each, nnz = defaultdict(dict), {}
    for r in d:
        if r["what"] == "file" and r["cache"] == "cold" and int(r["threads"]) == H and r["format"] != "s1_pack" \
                and (r["format"] != "s1" or r["mode"] == "canonical"):
            each[(r["dataset"], r["matrix"])][r["format"]] = _f(r["mnnz_per_s"])
            nnz[(r["dataset"], r["matrix"])] = _f(r.get("nnz"))
    paired = {k: u["s1"] / max(v for f, v in u.items() if f != "s1" and v)
              for k, u in each.items() if u.get("s1") and any(v for f, v in u.items() if f != "s1")}
    if paired:
        s = _stats(paired.values())
        N.put("dec-speedup-each", s["median"], "{:.2f}")
        N.put("dec-speedup-each-lo", s["lo"], "{:.2f}")
        N.put("dec-speedup-each-hi", s["hi"], "{:.2f}")
        N.put("dec-s1-fastest-count", sum(v > 1 for v in paired.values()), "{:d}")
        N.put("dec-paired-n", len(paired), "{:d}")
        slower = [nnz[k] for k, v in paired.items() if v <= 1 and nnz.get(k)]
        N.put("dec-s1-slower-max-mnnz", max(slower) / 1e6 if slower else 0, "{:.0f}")
    used = [r.get("isa_used") for r in d if r["format"] == "s1" and r["what"] == "file" and r.get("isa_used")]
    if used:
        N.put("isaBest", max(set(used), key=used.count).replace("avx", "AVX").replace("scalar", "scalar"))
    _write(rows, out / "tab_decode.tsv", ["format", "cold_t1", "cold_th", "warm_th"])
    # label sides for Fig 1c, chosen where the formats sit (west: label to the right of the point); others alternate
    sides = {"h5ad_none": "west", "csc_i32": "east", "h5ad_lzf": "south west", "mtx_gz": "west", "loom": "west",
             "h5ad_gzip": "west", "rds_gz": "east", "parquet_coo": "south west", "bpcells": "west", "zarr": "south",
             "soma": "east", "tenx_h5": "east", "rds_xz": "north", "csc_zstd19": "north", "s1": "north west", "s1_pack": "north"}
    for i, r in enumerate(sorted(trade, key=lambda r: r["mnnzs"])):
        r["anchor"] = sides.get(r["format"], ("west", "east")[i % 2])
    _write(trade, out / "fig1c_tradeoff.tsv", ["format", "label", "group", "bits", "mnnzs", "anchor"])
    for what in ("codec", "codec_cpp"):
        for mode in cfg["s1"]["modes"]:
            for isa in cfg["s1"]["isas"]:
                for t in sorted({1, H}):
                    s = _stats(pick("s1", what, t, None, mode, isa))
                    N.put(f"{what}-{mode}-{isa}-t{t}", s["median"], "{:.0f}")
                    N.put(f"{what}-{mode}-{isa}-t{t}-lo", s["lo"], "{:.0f}")
                    N.put(f"{what}-{mode}-{isa}-t{t}-hi", s["hi"], "{:.0f}")


def summary_section(cfg: Config, N: Numbers, out: Path) -> None:
    """One row per format: median size, geometric-mean size relative to .s1, the matrices on which it is the smallest file,
    median cold read rate on the headline cores, and the matrices on which it reads fastest (all formats on the same
    matrices; .s1 in source gene order)."""
    w = [r for r in _stage(cfg, "write") if r.get("status") == "ok"]
    H = int(cfg["threads"]["headline"])
    size = defaultdict(dict)
    for r in w:
        size[(r["dataset"], r["matrix"])][r["format"]] = _f(r["bytes"]) * 8 / _f(r["nnz"])
    rate = defaultdict(dict)
    for r in _headline_decode(cfg):
        if r.get("status") == "ok" and r["what"] == "file" and r["cache"] == "cold" and int(r["threads"]) == H \
                and (r["format"] != "s1" or r["mode"] == "canonical"):
            rate[(r["dataset"], r["matrix"])][r["format"]] = _f(r["mnnz_per_s"])
    fmts = [f for f in cfg["formats"] if f != "s1_pack" and any(f in u for u in size.values())]
    win_size, win_rate = defaultdict(int), defaultdict(int)
    for u in size.values():
        c = {f: v for f, v in u.items() if f in fmts and v}
        if c:
            win_size[min(c, key=c.get)] += 1
    for u in rate.values():
        c = {f: v for f, v in u.items() if f in fmts and v}
        if c:
            win_rate[max(c, key=c.get)] += 1
    n_size = sum(1 for u in size.values() if any(f in u for f in fmts))
    n_rate = sum(1 for u in rate.values() if u)
    rows = []
    for f in fmts:
        rel = [u[f] / u["s1"] for u in size.values() if u.get(f) and u.get("s1")]
        gm = float(np.exp(np.mean(np.log(rel)))) if rel else None
        rows.append({"format": FORMAT_LABELS.get(f, f), "bits": _stats(u.get(f) for u in size.values())["median"],
                     "rel": gm, "smallest": f"{win_size.get(f, 0)}/{n_size}",
                     "read": _stats(u.get(f) for u in rate.values())["median"], "fastest": f"{win_rate.get(f, 0)}/{n_rate}",
                     "n": len(rel)})
    rows.sort(key=lambda r: r["bits"] or 1e9)
    _write(rows, out / "tab_summary.tsv", ["format", "bits", "rel", "smallest", "read", "fastest", "n"])
    N.put("sumNsize", n_size)
    N.put("sumNread", n_rate)


def scaling_section(cfg: Config, N: Numbers, out: Path) -> None:
    """Read rate against threads, median over the matrices of the thread sweep: .s1 cold from disk and the codec from memory,
    the uncompressed arrays and gzip AnnData cold from disk; and the rate a 1 Gbit/s link delivers as .s1."""
    d = [r for r in _stage(cfg, "decode") if r.get("run") == "sweep" and r.get("status") == "ok"]
    if not d:
        return
    series = {
        "s1_codec": lambda r: r["format"] == "s1" and r["what"] == "codec" and r["mode"] == "canonical" and r["isa"] == "avx2",
        "s1_file": lambda r: r["format"] == "s1" and r["what"] == "file" and r["cache"] == "cold" and r["mode"] == "canonical",
        "csc_i32": lambda r: r["format"] == "csc_i32" and r["what"] == "file" and r["cache"] == "cold",
        "h5ad_gzip": lambda r: r["format"] == "h5ad_gzip" and r["what"] == "file" and r["cache"] == "cold",
    }
    threads = sorted({int(r["threads"]) for r in d})
    rows = []
    for th in threads:
        rec = {"threads": th}
        for k, sel in series.items():
            rec[k] = _stats(r["mnnz_per_s"] for r in d if int(r["threads"]) == th and sel(r))["median"]
        rows.append(rec)
    s1 = [r for r in _stage(cfg, "write") if r.get("status") == "ok" and r["format"] == "s1"]
    b = _stats(_f(r["bits_per_nnz"]) for r in s1)["median"]
    for rec in rows:   # what a 1 Gbit/s link delivers as .s1 (non-zeros per second, millions): above it, the link is the limit
        rec["link1g"] = 1e9 / b / 1e6 if b else None
    _write(rows, out / "fig1d_scaling.tsv", ["threads", *series, "link1g"])
    if b:
        N.put("scale-link-1g", 1e9 / b / 1e6, "{:.0f}")
    for rec in rows:
        if rec.get("s1_codec"):
            N.put(f"scale-s1codec-t{rec['threads']}", rec["s1_codec"], "{:.0f}")
            if b:
                N.put(f"scale-s1codec-gbit-t{rec['threads']}", rec["s1_codec"] * 1e6 * b / 1e9, "{:.1f}")
    N.put("scaleNmatrices", len({(r["dataset"], r["matrix"]) for r in d}))


def wire_section(cfg: Config, N: Numbers) -> None:
    """Q5 from measured size and decode: R_eff(B) = (b/B + 1/D)^-1 for .s1 against uncompressed CSC, per matrix."""
    w = {(r["dataset"], r["matrix"], r["format"]): r for r in _stage(cfg, "write") if r.get("status") == "ok"}
    H = int(cfg["threads"]["headline"])
    # D is the rate with the file already in memory (warm): Eq. 1 adds the transfer itself, so a cold read from the node's
    # disk would count reading twice. .s1 in source gene order, like the others.
    d = {(r["dataset"], r["matrix"], r["format"]): r for r in _headline_decode(cfg)
         if r.get("status") == "ok" and r["what"] == "file" and r["cache"] == "warm" and int(r["threads"]) == H
         and r["mode"] in ("", "canonical")}
    cross = []
    for (ds, m, f) in d:   # the bandwidth above which uncompressed CSC delivers faster: b0/B + 1/D0 = b1/B + 1/D1
        if f != "s1" or (ds, m, "csc_i32") not in d:
            continue
        nnz = _f(w[(ds, m, "s1")]["nnz"])
        b1, b0 = 8 * _f(w[(ds, m, "s1")]["bytes"]) / nnz, 8 * _f(w[(ds, m, "csc_i32")]["bytes"]) / nnz      # bits per nnz
        D1, D0 = _f(d[(ds, m, "s1")]["mnnz_per_s"]) * 1e6, _f(d[(ds, m, "csc_i32")]["mnnz_per_s"]) * 1e6
        cross.append(float("inf") if 1 / D1 <= 1 / D0 else (b0 - b1) / (1 / D1 - 1 / D0) / 1e9)
    if cross:
        fin = [c for c in cross if math.isfinite(c)]
        N.put("wire-crossover-gbit", _stats(fin)["median"] if fin else None, "{:.0f}")
        N.put("wire-crossover-gbit-lo", _stats(fin)["lo"] if fin else None, "{:.0f}")
        N.put("wire-crossover-never", sum(1 for c in cross if not math.isfinite(c)))
        N.put("wire-crossover-n", len(cross))
    for gbit in (0.1, 1, 10, 25):
        wins, n, gains = 0, 0, []
        for (ds, m, f) in list(d):
            if f != "s1" or (ds, m, "csc_i32") not in d:
                continue
            nnz = _f(w[(ds, m, "s1")]["nnz"])
            B = gbit * 1e9 / 8                                           # bytes per second

            def reff(fmt):
                b = _f(w[(ds, m, fmt)]["bytes"]) / nnz                   # bytes per non-zero
                D = _f(d[(ds, m, fmt)]["mnnz_per_s"]) * 1e6              # non-zeros per second
                return 1.0 / (b / B + 1.0 / D)

            r1, r0 = reff("s1"), reff("csc_i32")
            n += 1
            wins += r1 > r0
            gains.append(r1 / r0)
        tag = str(gbit).replace(".", "p")
        N.put(f"wire-wins-{tag}", wins)
        N.put(f"wire-n-{tag}", n)
        N.put(f"wire-gain-{tag}", _stats(gains)["median"], "{:.1f}")


# formats a reader cannot read in part: the whole object is fetched (or decompressed) before any cell
WHOLE_FILE = {"mtx_gz", "rds_gz", "rds_xz", "csc_zstd19"}


def network_section(cfg: Config, N: Numbers, out: Path) -> None:
    """The network figure. A reader runs at the speed of the network on a link of bandwidth B when it consumes stored bits at
    least as fast as the link delivers them, D_p * b >= B; it then delivers B / b non-zeros per second. (a) R_eff(B) for
    every format, from median b and in-memory D on p cores; (b) D_p * b against b, the link speed each format keeps up with;
    (c) the measured loaders on one dataset against the network and decoder ceilings; (d) first minibatch against latency."""
    w = {(r["dataset"], r["matrix"], r["format"]): r for r in _stage(cfg, "write") if r.get("status") == "ok"}
    H = int(cfg["threads"]["headline"])
    d = {(r["dataset"], r["matrix"], r["format"]): _f(r["mnnz_per_s"]) * 1e6 for r in _headline_decode(cfg)
         if r.get("status") == "ok" and r["what"] == "file" and r["cache"] == "warm" and int(r["threads"]) == H
         and r["mode"] in ("", "canonical")}
    per = defaultdict(list)   # format -> [(bits per nnz, nnz/s)]
    for k, D in d.items():
        if k in w and D:
            per[k[2]].append((8 * _f(w[k]["bytes"]) / _f(w[k]["nnz"]), D))
    per.pop("s1_pack", None)   # the same codec as .s1
    med = {f: (float(np.median([b for b, _ in v])), float(np.median([x for _, x in v]))) for f, v in per.items() if len(v) >= 5}
    if "s1" not in med:
        return
    fmts = [f for f in cfg["formats"] if f in med]
    grid = np.logspace(-2, 2, 81)   # Gbit/s
    curves = [{"gbit": g, **{f: 1.0 / (med[f][0] / (g * 1e9) + 1.0 / med[f][1]) / 1e6 for f in fmts},
               "s1_net": g * 1e9 / med["s1"][0] / 1e6 if g * 1e9 / med["s1"][0] / 1e6 <= 1500 else None, "s1_dec": med["s1"][1] / 1e6} for g in grid]
    _write(curves, out / "fignet_curves.tsv", ["gbit", *fmts, "s1_net", "s1_dec"])
    # label sides for Fig. net b, chosen where the formats sit (west: label to the right of the point)
    sides = {"csc_i32": "east", "h5ad_none": "east", "h5ad_lzf": "west", "zarr": "west", "csc_zstd19": "north",
             "parquet_coo": "south east", "soma": "north", "h5ad_gzip": "west", "tenx_h5": "east", "rds_gz": "south west",
             "mtx_gz": "west", "rds_xz": "east", "bpcells": "west", "loom": "west", "s1": "south"}
    roof = [{"format": f, "label": FORMAT_LABELS.get(f, f), "group": _group(f), "whole": int(f in WHOLE_FILE),
             "bits": med[f][0], "mnnzs": med[f][1] / 1e6, "gbit": med[f][0] * med[f][1] / 1e9, "anchor": sides.get(f, "west")}
            for f in fmts]
    _write(roof, out / "fignet_roof.tsv", ["format", "label", "group", "whole", "bits", "mnnzs", "gbit", "anchor"])
    pts = [{"format": f, "group": _group(f), "bits": b, "gbit": b * x / 1e9} for f in fmts for b, x in per[f]]
    _write(pts, out / "fignet_roof_pts.tsv", ["format", "group", "bits", "gbit"])
    b1, D1 = med["s1"]
    N.put("net-s1-bits", b1, "{:.2f}")
    N.put("net-s1-mnnzs", D1 / 1e6, "{:.0f}")
    N.put("net-knee-s1", b1 * D1 / 1e9, "{:.1f}")
    N.put("net-s1-at-1g", 1e9 / b1 / 1e6, "{:.0f}")   # non-zeros per second a 1 Gbit/s link carries as .s1 (millions)
    if "csc_i32" in med:
        N.put("net-csc-at-1g", 1e9 / med["csc_i32"][0] / 1e6, "{:.0f}")
    # the compact formats (under 12 bits per non-zero) and the link speed their decoders keep up with
    small = sorted((f for f in fmts if med[f][0] < 12 and f != "s1"), key=lambda f: -med[f][0] * med[f][1])
    if small:
        f = small[0]
        N.put("net-knee-compact-best", med[f][0] * med[f][1] / 1e9, "{:.2f}")
        N.put("net-knee-compact-best-name", FORMAT_LABELS.get(f, f))
    # per core: the C++ codec in memory, source order, one thread
    c1 = [_f(r["mnnz_per_s"]) for r in _headline_decode(cfg) if r.get("status") == "ok" and r["what"] == "codec_cpp"
          and int(r["threads"]) == 1 and r["mode"] == "canonical" and r["isa"] == "avx2"]
    if c1:
        N.put("net-knee-s1-per-core", b1 * float(np.median(c1)) * 1e6 / 1e9, "{:.2f}")

    # (c) one dataset: measured sustained cells/s per loader and tier, and the two ceilings for .s1
    rc = cfg.get("report", {}) or {}
    ds = rc.get("fig1d_dataset", rc.get("fig1b_dataset", "hlca_core"))
    mb = [r for r in _stage(cfg, "minibatch") if r.get("status") == "ok"]
    st = {(r["dataset"], r["matrix"]): r for r in _stage(cfg, "stats")}
    s = st.get((ds, "rna"))
    gb = {"s3_0.1g": 0.1, "s3_1g": 1.0, "s3_10g": 10.0}
    rows = []
    for ld in ("s1_stream", "s1_stream_seq", "anndata_zarr_lazy", "soma_ml"):
        for t, g in gb.items():
            v = _stats(r.get("sustained_cells_per_s") for r in mb if r["dataset"] == ds and r["loader"] == ld and r["tier"] == t)["median"]
            if v:
                rows.append({"loader": ld, "gbit": g, "cells_per_s": v})
    _write(rows, out / "fignet_loaders.tsv", ["loader", "gbit", "cells_per_s"])
    if s and (ds, "rna", "s1") in w:
        cells, nnz = _f(s["cells"]), _f(s["nnz"])
        bpc = _f(w[(ds, "rna", "s1")]["bytes"]) / cells
        N.put("net-ds-bytes-per-cell", bpc, "{:,.0f}")
        N.put("net-ds-cells-at-1g", 1e9 / 8 / bpc, "{:,.0f}")
        dd = d.get((ds, "rna", "s1"))
        if dd:
            N.put("net-ds-decoder-cells", dd / (nnz / cells), "{:,.0f}")
            ceil = [{"gbit": g, "net": g * 1e9 / 8 / bpc, "dec": dd / (nnz / cells)} for g in np.logspace(-1.5, 1.5, 31)]
            _write(ceil, out / "fignet_ceilings.tsv", ["gbit", "net", "dec"])
        loc = _stats(r.get("sustained_cells_per_s") for r in mb if r["dataset"] == ds and r["loader"] == "s1_stream" and r["tier"] == "local")["median"]
        if loc:
            N.put("net-ds-local-cells", loc, "{:,.0f}")

    # (d) first minibatch at 1 and 30 ms (1 Gbit/s), median over the panel, and the round trips each pays in sequence
    lat = []
    for ld in ("s1_stream", "s1_stream_seq", "anndata_zarr_lazy"):
        pts_ = []
        for t, ms in (("s3_1g", 1.0), ("s3_1g_wan", 30.0)):
            per_ds = [_stats(r.get("first_batch_s") for r in mb if r["tier"] == t and r["loader"] == ld and r["dataset"] == x)["median"]
                      for x in sorted({r["dataset"] for r in mb})]
            v = _stats(per_ds)["median"]
            if v is not None:
                pts_.append((ms, v))
        if len(pts_) == 2:
            (x0, y0), (x1, y1) = pts_
            k = (y1 - y0) / ((x1 - x0) / 1000)   # seconds per second of round trip = round trips paid one after another
            N.put(f"net-rtts-{ld}", k, "{:.0f}")
            for ms in (0.0, 100.0):
                lat.append({"loader": ld, "rtt_ms": ms, "first_s": y0 + (ms - x0) / 1000 * k, "measured": 0})
            for ms, v in pts_:
                lat.append({"loader": ld, "rtt_ms": ms, "first_s": v, "measured": 1})
    _write(lat, out / "fignet_latency.tsv", ["loader", "rtt_ms", "first_s", "measured"])


def minibatch_section(cfg: Config, N: Numbers, out: Path) -> None:
    mb = [r for r in _stage(cfg, "minibatch") if r.get("status") == "ok"]
    rc = cfg.get("report", {}) or {}
    ds_b = rc.get("fig1b_dataset", "hlca_core")
    tier_b = rc.get("fig1b_tier", "s3_1g")
    loaders = list(cfg["minibatch"]["loaders"])
    rows = []
    for i, ld in enumerate(loaders):
        rs = [r for r in mb if r["dataset"] == ds_b and r["tier"] == tier_b and r["loader"] == ld]
        if not rs:
            continue
        med = lambda k: _stats(r.get(k) for r in rs)["median"]
        rows.append({"pos": len(rows), "loader": ld, "label": LOADER_LABELS.get(ld, ld), "open": med("open_s"), "read": med("first_read_s"),
                     "tensor": med("to_tensor_s"), "total": med("first_batch_s"), "group": "s1" if ld.startswith("s1") else "other"})
    _write(rows, out / "fig1b_ttfm.tsv", ["pos", "loader", "label", "open", "read", "tensor", "total", "group"])
    tot = {r["loader"]: r["total"] for r in rows}
    if tot.get("s1_stream"):
        N.put("ttfm-s1", tot["s1_stream"], "{:.2f}")
        others = {k: v for k, v in tot.items() if not k.startswith("s1") and v}
        if others:
            best = min(others, key=others.get)
            N.put("ttfm-best-other", others[best], "{:.2f}")
            N.put("ttfm-best-other-name", LOADER_LABELS[best])
            N.put("ttfm-speedup", others[best] / tot["s1_stream"], "{:.0f}")
    N.put("fig1b-dataset", _tex_escape(ds_b))
    N.put("fig1b-tier", TIER_LABELS.get(tier_b, tier_b))
    # sustained throughput by tier (Fig 1d)
    ds_d = rc.get("fig1d_dataset", ds_b)
    thr = []
    for ld in loaders:
        for j, t in enumerate(FIG1D_TIERS):
            rs = [r for r in mb if r["dataset"] == ds_d and r["tier"] == t and r["loader"] == ld]
            if rs:
                s = _stats(r.get("sustained_cells_per_s") for r in rs)
                capped = any(str(r.get("capped")) == "True" for r in rs)
                thr.append({"loader": ld, "label": LOADER_LABELS.get(ld, ld), "tierpos": j, "tier": t, "cells_per_s": s["median"], "capped": int(capped)})
                N.put(f"thr-{ld}-{t}", s["median"], "{:,.0f}")
    _write(thr, out / "fig1d_throughput.tsv", ["loader", "label", "tierpos", "tier", "cells_per_s", "capped"])
    # how much of each link the .s1 reader fills: cells per second x the file's bytes per cell (chunk over-read not counted)
    st = {(r["dataset"], r["matrix"]): r for r in _stage(cfg, "stats")}
    wr = {(r["dataset"], r["matrix"], r["format"]): r for r in _stage(cfg, "write") if r.get("status") == "ok"}
    for t, gbit in (("s3_0.1g", 0.1), ("s3_1g", 1), ("s3_10g", 10)):
        rs = [r for r in mb if r["dataset"] == ds_d and r["tier"] == t and r["loader"] == "s1_stream"]
        cps = _stats(r.get("sustained_cells_per_s") for r in rs)["median"]
        w, s = wr.get((ds_d, "rna", "s1")), st.get((ds_d, "rna"))
        if cps and w and s:
            N.put(f"linkuse-{t}", 100 * cps * _f(w["bytes"]) / _f(s["cells"]) * 8 / (gbit * 1e9), "{:.0f}")
    # the 30 ms tier: .s1 sending a batch's range requests together, one at a time, and Zarr (median over panel datasets)
    for ld in ("s1_stream", "s1_stream_seq", "anndata_zarr_lazy"):
        per_ds = [_stats(r.get("first_batch_s") for r in mb if r["tier"] == "s3_1g_wan" and r["loader"] == ld and r["dataset"] == d)["median"]
                  for d in sorted({r["dataset"] for r in mb})]
        N.put(f"ttfm-wan-{ld}", _stats(per_ds)["median"], "{:.2f}")
    # sustained rates only where the workload reads each cell at most once (on a smaller matrix it wraps around, and a reader
    # that caches what it fetched would be timed partly from memory)
    mbc = cfg["minibatch"]
    need = (int(mbc["n_batches"]) + int(mbc["warmup_batches"]) + 1) * int(mbc["batch_size"])
    ncells = {}
    for r in _stage(cfg, "stats"):
        ncells[r["dataset"]] = max(ncells.get(r["dataset"], 0), int(_f(r["cells"]) or 0))
    ratios = {"first": [], "sust": []}
    for d in sorted({r["dataset"] for r in mb}):
        def m(ld, k):
            return _stats(r.get(k) for r in mb if r["dataset"] == d and r["tier"] == "s3_1g_wan" and r["loader"] == ld)["median"]
        a, b = m("s1_stream_seq", "first_batch_s"), m("s1_stream", "first_batch_s")
        if a and b:
            ratios["first"].append(a / b)
        a, b = m("s1_stream", "sustained_cells_per_s"), m("s1_stream_seq", "sustained_cells_per_s")
        if a and b and ncells.get(d, 0) >= need:
            ratios["sust"].append(a / b)
    N.put("concurrency-gain-first", _stats(ratios["first"])["median"], "{:.1f}")
    N.put("concurrency-gain-sustained", _stats(ratios["sust"])["median"], "{:.1f}")
    N.put("concurrency-gain-sustained-n", len(ratios["sust"]))
    # first-batch table: per loader and tier, the median over panel datasets of each dataset's median over seeds. Where a
    # download-first loader was not run (above the M tier) the dataset contributes the computed download time instead, a
    # lower bound, and the cell is marked with a superscript e. Cells are strings so the table can carry the mark.
    skipped = [r for r in _stage(cfg, "minibatch") if r.get("status") == "skipped" and _f(r.get("est_download_s"))]
    tab = []
    for ld in loaders:
        rec = {"loader": LOADER_LABELS.get(ld, ld)}
        for t in TIER_ORDER + ["s3_1g_wan"]:
            per, est = [], False
            for ds in sorted({r["dataset"] for r in mb} | {r["dataset"] for r in skipped}):
                v = _stats(r.get("first_batch_s") for r in mb if r["tier"] == t and r["loader"] == ld and r["dataset"] == ds)["median"]
                if v is None:
                    v = next((_f(r["est_download_s"]) for r in skipped if r["tier"] == t and r["loader"] == ld and r["dataset"] == ds), None)
                    est = est or v is not None
                if v is not None:
                    per.append(v)
            m = _stats(per)["median"]
            rec[t] = "--" if m is None else (f"{m:.2f}" + ("\\textsuperscript{e}" if est else ""))
        tab.append(rec)
    _write(tab, out / "tab_minibatch.tsv", ["loader", *TIER_ORDER, "s3_1g_wan"])
    capped = sorted({r["tier"] for r in mb if str(r.get("capped")) == "True"})
    N.put("cappedTiers", "; ".join(TIER_LABELS.get(t, t) for t in capped) if capped else "none")
    # the rate the proxy delivered at each tier with no latency, over `threads` connections (median over datasets)
    for t in sorted({r["tier"] for r in mb if r.get("proxy_ceiling_par_gbit")}):
        g = _stats(r.get("proxy_ceiling_par_gbit") for r in mb if r["tier"] == t)["median"]
        if g is not None:
            N.put(f"delivered-{t}", f"{g:.2f}")


def ladder_floor_section(cfg: Config, N: Numbers, out: Path) -> None:
    lad = _stage(cfg, "ladder")
    fl = _stage(cfg, "floor")
    floor_by = {(r["dataset"], r["matrix"]): r for r in fl}
    vals = defaultdict(list)
    for r in lad:
        for k, _ in LADDER[:-1]:
            vals[k].append(r.get(k))
        f = floor_by.get((r["dataset"], r["matrix"]))
        if f:
            vals["floor"].append(_best(f))
    rows = [{"pos": i, "key": k, "label": lab, **_stats(vals[k])} for i, (k, lab) in enumerate(LADDER)]
    _write(rows, out / "fig3a_ladder.tsv", ["pos", "key", "label", "median", "q1", "q3", "lo", "hi", "n", "em", "ep", "rm", "rp"])
    for k, _ in LADDER:
        N.put(f"ladder-{k}", _stats(vals[k])["median"])
    above = [(_f(f.get("s1_payload_bpn")) / _best(f) - 1) * 100 for f in fl if _f(f.get("s1_payload_bpn")) and _best(f)]
    a = _stats(above)
    N.put("s1AboveFloorPct", a["median"], "{:.1f}")
    N.put("s1AboveFloorPctLo", a["lo"], "{:.1f}")
    N.put("s1AboveFloorPctHi", a["hi"], "{:.1f}")
    # model gains over .s1 (Fig 3b): positive = the model's ideal code is shorter than .s1's actual payload
    mrows = []
    for i, (k, lab) in enumerate(MODELS):
        g = [(_f(f.get("s1_payload_bpn")) - _f(f.get(f"{k}_bpn"))) for f in fl if _f(f.get("s1_payload_bpn")) and _f(f.get(f"{k}_bpn"))]
        s = _stats(g)
        mrows.append({"pos": i, "model": k, "label": lab, **s})
        N.put(f"gain-{k}", s["median"])
        N.put(f"gain-{k}-lo", s["lo"])
        N.put(f"gain-{k}-hi", s["hi"])
    _write(mrows, out / "fig3b_models.tsv", ["pos", "model", "label", "median", "q1", "q3", "lo", "hi", "n", "em", "ep", "rm", "rp"])
    # decomposition by protocol class (Fig 3c), medians over matrices in each class:
    #   sampling = H(Multinomial(N_c, pi_c)) per non-zero (sampled cells); state = best model - sampling (what the counts
    #   say about the cell's own profile, as far as the best model can tell); above = .s1 payload - best model (>= 0)
    by_class = defaultdict(list)
    for f in fl:
        by_class[f["protocol_class"]].append(f)
    crow = []
    for cls in CLASS_LABELS:
        fs = by_class.get(cls)
        if not fs:
            continue
        samp = _stats(f.get("sample_sampling_bpn") for f in fs)["median"]
        best = _stats(_best(f) for f in fs)["median"]
        over = _stats(max((_f(f.get("s1_payload_bpn")) or np.nan) - (_best(f) or np.nan), 0.0) for f in fs)["median"]
        if None in (samp, best):
            continue
        crow.append({"pos": len(crow), "class": cls, "label": CLASS_LABELS[cls], "sampling": samp, "state": max(best - samp, 0.0),
                     "above": over or 0.0})
    _write(crow, out / "fig3c_decomp.tsv", ["pos", "class", "label", "sampling", "state", "above"])
    bestm = defaultdict(int)
    for f in fl:
        b = _best_name(f)
        if b:
            bestm[b] += 1
    for k, _ in MODELS:
        N.put(f"bestModelCount-{k}", bestm.get(k, 0))
    # the multinomial sampling model describes UMI counts (molecules), not read counts, peaks, tags or imaging spots
    value = {(d["id"], m["name"]): m.get("value") for d in load_datasets(cfg) for m in d["matrices"]}
    fl_umi = [f for f in fl if value.get((f["dataset"], f["matrix"])) == "umi"]
    N.put("samplingN", len(fl_umi))
    frac = [(_f(f.get("sample_sampling_bpn")) / _f(f.get("s1_payload_bpn"))) * 100 for f in fl_umi
            if _f(f.get("sample_sampling_bpn")) and _f(f.get("s1_payload_bpn"))]
    s = _stats(frac)
    N.put("samplingPctMedian", s["median"], "{:.0f}")
    N.put("samplingPctLo", s["lo"], "{:.0f}")
    N.put("samplingPctHi", s["hi"], "{:.0f}")
    sh = _stats((_f(f.get("sample_gene_depth_bpn")) or np.nan) - (_f(f.get("sample_cluster_bpn")) or np.nan) for f in fl)
    N.put("sharedBpnMedian", sh["median"])
    N.put("sharedBpnLo", sh["lo"])
    N.put("sharedBpnHi", sh["hi"])
    nb_exact = [abs(_f(f["nb_exact_bpn"]) / _f(f["nb_bpn"]) - 1) * 100 for f in fl if _f(f.get("nb_exact_bpn")) and _f(f.get("nb_bpn"))]
    N.put("nbBinningErrPctMax", _stats(nb_exact)["hi"], "{:.2f}")
    tab = []
    for cls in CLASS_LABELS:
        fs = by_class.get(cls)
        if fs:
            tab.append({"class": CLASS_LABELS[cls], "n": len(fs), "s1": _stats(f.get("s1_payload_bpn") for f in fs)["median"],
                        **{k: _stats(f.get(f"{k}_bpn") for f in fs)["median"] for k, _ in MODELS}})
    _write(tab, out / "tab_floor.tsv", ["class", "n", "s1", *[k for k, _ in MODELS]])


def second_section(cfg: Config, N: Numbers, out: Path) -> None:
    sec = _stage(cfg, "second")
    rows = []
    for i, (k, lab) in enumerate(CODERS):
        s = _stats(r["ratio"] for r in sec if r["coder"] == k)
        if s["n"]:
            rows.append({"pos": len(rows), "coder": k, "label": lab, **s})
    _write(rows, out / "fig3d_second.tsv", ["pos", "coder", "label", "median", "q1", "q3", "lo", "hi", "n", "em", "ep", "rm", "rp"])
    allr = _stats(r["ratio"] for r in sec)
    N.put("secondMin", allr["lo"], "{:.3f}")
    N.put("secondMax", allr["hi"], "{:.3f}")


def eco_section(cfg: Config, N: Numbers, out: Path) -> None:
    eco = [r for r in _stage(cfg, "eco") if _f(r.get("median_s")) is not None]
    rows = [{"dataset": _tex_escape(r["dataset"]), "lang": r["lang"], "call": "\\texttt{" + _tex_escape(r["call"]) + "}",
             "seconds": _f(r["median_s"]), "equal": r.get("equal", "")} for r in eco]
    _write(rows, out / "tab_eco.tsv", ["dataset", "lang", "call", "seconds", "equal"])
    # seconds per call on the first eco dataset, for the text: eco-<short key>
    keys = {"singlet.load": "py-singlet", "anndata.read_h5ad": "py-read-h5ad", "scanpy.read_10x_h5": "py-read-10x",
            "singlet::load(as='sce')": "r-singlet-sce", "singlet::load(as='seurat')": "r-singlet-seurat",
            "Seurat::Read10X_h5": "r-read10x", "zellkonverter::readH5AD": "r-readh5ad", "CreateSeuratObject(Read10X_h5)": "r-read10x-seurat",
            "pz2::Matrix::open + decode (member in memory)": "cpp-s1"}
    first = (cfg["eco"].get("datasets") or [None])[0]
    for r in eco:
        if r["dataset"] == first and r["call"] in keys:
            N.put(f"eco-{keys[r['call']]}", _f(r["median_s"]), "{:.2f}")


def env_section(cfg: Config, N: Numbers) -> None:
    N.put("threadsHeadline", int(cfg["threads"]["headline"]))
    p = cfg.results_dir / "env.json"
    if not p.exists():
        return
    e = json.loads(p.read_text())
    N.put("cpu", _tex_escape(e.get("cpu") or ""))
    N.put("cores", e.get("cores"))
    N.put("memGB", e.get("mem_gb"), "{:.0f}")
    if e.get("gpu"):
        N.put("gpu", _tex_escape(", ".join(e["gpu"].splitlines())))
    disk = dict(re.findall(r'(\w+)="([^"]*)"', e.get("data_disk") or ""))
    if disk:  # lsblk -P pairs: MODEL, SIZE, ROTA (1 = spinning)
        kind = "spinning disk" if disk.get("ROTA") == "1" else "solid-state disk"
        N.put("localDisk", _tex_escape(f"{disk.get('SIZE', '')} {kind} ({disk.get('MODEL', '').strip()})"))
    N.put("threadsHeadline", int(cfg["threads"]["headline"]))
    for pkg, ver in (e.get("packages") or {}).items():
        if ver:
            N.put(f"ver-{pkg}", _tex_escape(str(ver)))
    # the released singlet versions the paper cites (config `release`): their package code is the code that was measured
    for k, ver in (cfg.get("release") or {}).items():
        N.put(f"rel-{k}", _tex_escape(str(ver)))
    if e.get("R"):  # e.g. "R version 4.4.3 (2025-02-28)  BPCells 0.3.1"
        rv, bp = re.search(r"R version (\S+)", e["R"]), re.search(r"BPCells (\S+)", e["R"])
        N.put("ver-R", rv.group(1) if rv else _tex_escape(e["R"]))
        if bp:
            N.put("ver-BPCells", bp.group(1))


def _write(rows: list[dict], path: Path, cols: list[str]) -> None:
    """Tab-separated, header always present, missing values as empty cells (pgfplots skips them; tables print --)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell(v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return ""
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.6g}"
        return str(v)

    lines = ["\t".join(cols)] + ["\t".join(cell(r.get(c)) for c in cols) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def used_keys() -> set[str]:
    keys = set()
    for p in list(PAPER_DIR.glob("*.tex")) + list((PAPER_DIR / "sections").glob("*.tex")) + list((PAPER_DIR / "figures").glob("*.tex")):
        keys |= set(re.findall(r"\\nm\{([^}]+)\}", p.read_text()))
    return keys


def extension_numbers(cfg: Config, N: Numbers) -> None:
    """The out-of-domain count matrices (datasets.yaml `extensions`): .s1 size, and how much smaller it is than gzip AnnData
    and than the best other format, per matrix (ext-<id>-...)."""
    ext = {d["id"] for d in load_datasets(cfg, True)} - {d["id"] for d in load_datasets(cfg)}
    w = [r for r in read_stage(cfg, "write") if r.get("dataset") in ext and r.get("status") == "ok"]
    for ds in sorted(ext):
        bpn = {r["format"]: _f(r.get("bits_per_nnz")) for r in w if r["dataset"] == ds}
        s1 = bpn.get("s1")
        if not s1:
            continue
        N.put(f"ext-{ds}-s1", s1, "{:.2f}")
        if bpn.get("h5ad_gzip"):
            N.put(f"ext-{ds}-ratio-h5ad_gzip", bpn["h5ad_gzip"] / s1, "{:.1f}")
        others = [v for f, v in bpn.items() if v and not f.startswith("s1")]
        if others:
            N.put(f"ext-{ds}-ratio-best", min(others) / s1, "{:.1f}")


def run(cfg: Config) -> int:
    out = cfg.generated_dir
    out.mkdir(parents=True, exist_ok=True)
    N = Numbers()
    env_section(cfg, N)
    datasets_table(cfg, N, out)
    size_stats = size_section(cfg, N, out)
    decode_section(cfg, N, out, size_stats)
    summary_section(cfg, N, out)
    scaling_section(cfg, N, out)
    wire_section(cfg, N)
    minibatch_section(cfg, N, out)
    network_section(cfg, N, out)
    ladder_floor_section(cfg, N, out)
    second_section(cfg, N, out)
    eco_section(cfg, N, out)
    from . import report_r3

    report_r3.run(cfg, N, out)
    extension_numbers(cfg, N)
    N.write(out / "numbers.tex")
    missing = sorted(used_keys() - set(N.d))
    log(f"report: {len(N.d)} numbers defined; {len(missing)} used in the manuscript but not produced")
    for k in missing:
        print(f"missing\t{k}")
    return 1 if missing else 0
