"""Report section for the limits figure: how fast textbook integer codes decode on one core against .s1 (kernels stage),
what a link of a given bandwidth per decoding core delivers with each of them relative to an ideal code, and what each
format costs to write (writecost stage). Called from report.run; writes generated/figlim_*.tsv and \\nm{} numbers.

The delivered rate is Eq. 1 with the decoder on p cores, R = (b/B + 1/(p D1))^-1, so per core it depends on the link
only through x = B/p, the bandwidth per decoding core. The ideal code stores the best cross-validated count model's code
length (the floor stage) and decodes as fast as the output can be written (`store_bound`); neither is attainable, so the
ratio to it is a bound on what any lossless format could add, not a forecast.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .report import Numbers, _best, _f, _stage, _stats, _write
from .util import Config

KERNEL_LABELS = {
    "raw_u32": "raw arrays", "fixed_bytes": "fixed-width bytes", "leb128_gaps": "LEB128", "streamvbyte_gaps": "Stream VByte",
    "bitpack256_gaps": "SIMD bit-packing", "rans0_gaps": "order-0 rANS", "zstd3_leb_gaps": "zstd-3", "zstd19_leb_gaps": "zstd-19",
    "pz2_avx2": ".s1", "pz2_avx2_native": ".s1, stored order",
}
NATIVE = "pz2_avx2_native"   # the .s1 member in its stored gene order (s1order stage): row ids plus one permutation, no sort
SOURCE = "pz2_avx2_source"   # the .s1 member in source gene order (a sort per cell): reported as a number, not plotted
KERNEL_ANCHORS = {"raw_u32": "north", "fixed_bytes": "north west", "leb128_gaps": "north", "streamvbyte_gaps": "south west",
                  "bitpack256_gaps": "east", "rans0_gaps": "north west", "zstd3_leb_gaps": "west", "zstd19_leb_gaps": "east",
                  "pz2_avx2": "north west", "pz2_avx2_native": "south east"}
ENTROPY = ("rans0_gaps", "zstd3_leb_gaps", "zstd19_leb_gaps")   # entropy-coded streams: the fastest sets the practical optimum


def _r(b: float, d: float, x: float) -> float:
    """Non-zeros per second per core (millions) at x Gbit/s per core, b bits per non-zero, d million non-zeros/s per core."""
    return 1.0 / (b / (x * 1e3) + 1.0 / d)


def _trend_points(per: dict, floor: dict, mats: list, leave_out: str = "") -> list[tuple]:
    """(matrix, log b/floor, log t/t_out) for every code on a matrix's frontier other than .s1: t is the decoding time per
    non-zero beyond writing the output (t_out, from store_bound); a code is on the frontier if it decodes faster than every
    other code that stores fewer bits. Codes at the output bound (t < 5% of t_out) carry no information about the slope."""
    pts = []
    for k in mats:
        ts = 1e3 / per[k]["store_bound"][1]
        cand = sorted((b, 1e3 / d - ts) for c, (b, d) in per[k].items() if c not in ("store_bound", "pz2_avx2", NATIVE, SOURCE, leave_out))
        best = np.inf
        for b, t in cand:
            if t < best:
                best = t
                if t > 0.05 * ts:
                    pts.append((k, np.log(b / floor[k]), np.log(t / ts)))
    return pts


def _trend_fit(pts: list[tuple]) -> tuple[float, float]:
    """log(t/t_out) = a + s log(b/floor), least squares over the pooled points: slope s, intercept a."""
    s, a = np.polyfit([p[1] for p in pts], [p[2] for p in pts], 1)
    return float(s), float(a)


def _trend_rate(b, fl: float, d_store: float, s: float, a: float):
    """M non-zeros/s per core that the trend of the measured codes predicts for a code storing b bits per non-zero."""
    return d_store / (1.0 + np.exp(a) * (np.asarray(b) / fl) ** s)


def kernels_section(cfg: Config, N: Numbers, out: Path) -> None:
    rows = [r for r in _stage(cfg, "kernels") if _f(r.get("ok")) == 1]
    if not rows:
        return
    floor = {(r["dataset"], r["matrix"]): _best(r) for r in _stage(cfg, "floor")}
    per = defaultdict(dict)   # (dataset, matrix) -> codec -> (bits, M nnz/s on one core)
    for r in rows:
        per[(r["dataset"], r["matrix"])][r["codec"]] = (_f(r["bits"]), _f(r["mnnz_per_s"]))
    # stored order, timed next to a second source-order run in the s1order stage: its rate relative to that run, applied to
    # the source-order rate of the kernels run, so both .s1 points share the other codes' timing conditions
    rerun, native_x, native_ok = [], [], []
    so = defaultdict(dict)
    for r in _stage(cfg, "s1order"):
        so[(r["dataset"], r["matrix"])][r["codec"]] = r
    for k, v in so.items():
        if k in per and "pz2_avx2" in per[k] and {"pz2_avx2", NATIVE} <= v.keys():
            x = _f(v[NATIVE]["mnnz_per_s"]) / _f(v["pz2_avx2"]["mnnz_per_s"])
            # .s1 as the paper reports it (stored order); source order kept apart, for the text only
            per[k][SOURCE] = per[k]["pz2_avx2"]
            per[k]["pz2_avx2"] = (_f(v[NATIVE]["bits"]), per[k][SOURCE][1] * x)
            native_x.append(x)
            rerun.append(_f(v["pz2_avx2"]["mnnz_per_s"]) / per[k]["pz2_avx2"][1])
            native_ok.append(_f(v[NATIVE]["ok"]) == 1)
    if native_x:
        N.put("s1order-n", len(native_x), "{:d}")
        N.put("s1order-x", float(np.median(native_x)), "{:.1f}")
        N.put("s1order-x-lo", float(min(native_x)), "{:.1f}")
        N.put("s1order-x-hi", float(max(native_x)), "{:.1f}")
        N.put("s1order-rerun-pct", float(np.median(rerun)) * 100, "{:.0f}")
        N.put("s1order-ok", "all" if all(native_ok) else f"{sum(native_ok)} of {len(native_ok)}")
    mats = [k for k, v in per.items() if "pz2_avx2" in v and "store_bound" in v and floor.get(k)]
    codecs = [c for c in KERNEL_LABELS if all(c in per[k] for k in mats)]
    N.put("kern-n", len(mats), "{:d}")

    # (b) the plane: medians and every matrix, the floor and the output bound
    med = {}
    for c in codecs:
        b = _stats(per[k][c][0] for k in mats)
        d = _stats(per[k][c][1] for k in mats)
        med[c] = (b["median"], d["median"])
        N.put(f"kern-{c}-bits", b["median"], "{:.1f}")
        N.put(f"kern-{c}-mnnzs", d["median"], "{:.0f}")
        N.put(f"kern-{c}-gbit", b["median"] * d["median"] / 1e3, "{:.2f}")
    # the stored-order point is named in the caption: it sits next to the trend's diamond
    plane = [{"codec": c, "label": "{}" if c == NATIVE else KERNEL_LABELS[c], "group": {"pz2_avx2": "s1", NATIVE: "s1n"}.get(c) or ("entropy" if c in ENTROPY else "fast"),
              "bits": med[c][0], "mnnzs": med[c][1], "anchor": KERNEL_ANCHORS.get(c, "west")} for c in codecs]
    _write(plane, out / "figlim_plane.tsv", ["codec", "label", "group", "bits", "mnnzs", "anchor"])
    pts = [{"codec": c, "group": {"pz2_avx2": "s1", NATIVE: "s1n"}.get(c, "other"), "bits": per[k][c][0], "mnnzs": per[k][c][1]}
           for k in mats for c in codecs]
    _write(pts, out / "figlim_plane_pts.tsv", ["codec", "group", "bits", "mnnzs"])
    fl = _stats(floor[k] for k in mats)["median"]
    sb = _stats(per[k]["store_bound"][1] for k in mats)["median"]
    N.put("kern-floor-bits", fl, "{:.2f}")
    N.put("kern-store-mnnzs", sb, "{:,.0f}")
    # the frontier of the measured codes: each code that decodes faster than every code that stores fewer bits
    front, best = [], 0.0
    for c in sorted((c for c in codecs if c != NATIVE), key=lambda c: med[c][0]):   # codes that return source order
        if med[c][1] > best:
            front.append({"codec": c, "bits": med[c][0], "mnnzs": med[c][1]})
            best = med[c][1]
    _write(front, out / "figlim_front.tsv", ["codec", "bits", "mnnzs"])
    _write([{"bits": fl, "mnnzs": sb}], out / "figlim_ideal.tsv", ["bits", "mnnzs"])
    _write([{"bits": fl, "mnnzs": 20}, {"bits": fl, "mnnzs": 2500}], out / "figlim_floorline.tsv", ["bits", "mnnzs"])
    _write([{"bits": 3, "mnnzs": sb}, {"bits": 80, "mnnzs": sb}], out / "figlim_storeline.tsv", ["bits", "mnnzs"])
    N.put("kern-front-n", len(front), "{:d}")
    N.put("kern-s1-on-front", "yes" if any(f["codec"] == "pz2_avx2" for f in front) else "no")
    s1b, s1d = med["pz2_avx2"]
    N.put("kern-s1-knee-gbit", s1b * s1d / 1e3, "{:.2f}")   # the link one core of .s1 keeps up with
    N.put("kern-s1-above-floor-pct", (s1b / fl - 1) * 100, "{:.0f}")
    fastest_entropy = max((c for c in codecs if c in ENTROPY), key=lambda c: med[c][1])
    N.put("kern-entropy-fastest", KERNEL_LABELS[fastest_entropy])
    N.put("kern-entropy-fastest-x", med[fastest_entropy][1] / s1d, "{:.1f}")
    N.put("kern-store-x", sb / s1d, "{:.0f}")

    # the trend of the other codes, pooled over matrices in units of each matrix's floor and output time, extrapolated to
    # .s1's size and to the floor: what the measured codes predict any code of that size decodes at
    pts = _trend_points(per, floor, mats)
    ts, ta = _trend_fit(pts)
    ratio_of = lambda c, s_, a_: float(np.median([per[k][c][1] / _trend_rate(per[k][c][0], floor[k], per[k]["store_bound"][1],
                                                                             s_, a_) for k in mats]))
    ratio = lambda s_, a_, ks: float(np.median([per[k]["pz2_avx2"][1] / _trend_rate(per[k]["pz2_avx2"][0], floor[k],
                                                                                    per[k]["store_bound"][1], s_, a_) for k in ks]))
    r0 = ratio(ts, ta, mats)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        ks = [mats[i] for i in rng.integers(0, len(mats), len(mats))]
        pp = [p for k in ks for p in pts if p[0] == k]
        if len({p[1] for p in pp}) > 2:
            boot.append(ratio(*_trend_fit(pp), ks))
    loo = [ratio(*_trend_fit(_trend_points(per, floor, mats, c)), mats) for c in codecs if c not in ("store_bound", "pz2_avx2", NATIVE)]
    N.put("trend-n", len(pts), "{:d}")
    N.put("trend-slope", -ts, "{:.1f}")
    N.put("trend-s1-pct", r0 * 100, "{:.0f}")
    N.put("trend-s1-x", 1 / r0, "{:.1f}")   # how many times faster the trend predicts a code of .s1's size decodes
    N.put("trend-s1-lo", np.percentile(boot, 2.5) * 100, "{:.0f}")
    N.put("trend-s1-hi", np.percentile(boot, 97.5) * 100, "{:.0f}")
    N.put("trend-s1-loo-lo", min(loo) * 100, "{:.0f}")
    N.put("trend-s1-loo-hi", max(loo) * 100, "{:.0f}")
    at_floor = [_trend_rate(floor[k], floor[k], per[k]["store_bound"][1], ts, ta) for k in mats]
    N.put("trend-floor-mnnzs", float(np.median(at_floor)), "{:.0f}")
    N.put("trend-floor-vs-s1-pct", float(np.median([per[k]["pz2_avx2"][1] / f for k, f in zip(mats, at_floor)])) * 100, "{:.0f}")
    N.put("trend-floor-x", float(np.median([f / per[k]["pz2_avx2"][1] for k, f in zip(mats, at_floor)])), "{:.1f}")
    if all(SOURCE in per[k] for k in mats):
        N.put("trend-source-pct", ratio_of(SOURCE, ts, ta) * 100, "{:.0f}")
    bgrid = np.geomspace(fl, 64, 80)
    _write([{"bits": b, "mnnzs": float(_trend_rate(b, fl, sb, ts, ta))} for b in bgrid], out / "figlim_trend.tsv", ["bits", "mnnzs"])
    _write([{"bits": fl, "mnnzs": float(_trend_rate(fl, fl, sb, ts, ta))}], out / "figlim_trend_floor.tsv", ["bits", "mnnzs"])

    # (c) fraction of the ideal, against the bandwidth per decoding core; medians of per-matrix ratios
    grid = np.logspace(-2, 1, 61)
    trb = {k: np.geomspace(floor[k], 64, 160) for k in mats}
    trd = {k: _trend_rate(trb[k], floor[k], per[k]["store_bound"][1], ts, ta) for k in mats}
    curves, best_other = [], {}
    for x in grid:
        rec = {"x": x}
        ideal = {k: _r(floor[k], per[k]["store_bound"][1], x) for k in mats}
        for c in codecs:
            rec[c] = float(np.median([_r(*per[k][c], x) / ideal[k] for k in mats]))
        # .s1's size at the speed of the fastest entropy-coded stream measured on the same matrix
        rec["practical"] = float(np.median([_r(per[k]["pz2_avx2"][0], max(per[k][e][1] for e in ENTROPY if e in per[k]), x)
                                            / ideal[k] for k in mats]))
        rec["envelope"] = float(np.median([max(_r(*per[k][c], x) for c in codecs) / ideal[k] for k in mats]))
        # the best code on the trend: the size between the floor and 64 bits that delivers most at this bandwidth
        tr = {k: max(_r(b, d, x) for b, d in zip(trb[k], trd[k])) for k in mats}
        rec["trend"] = float(np.median([tr[k] / ideal[k] for k in mats]))
        rec["s1_of_trend"] = float(np.median([_r(*per[k]["pz2_avx2"], x) / tr[k] for k in mats]))
        curves.append(rec)
        best_other[x] = max((rec[c], c) for c in codecs if c not in ("pz2_avx2", NATIVE))
    _write(curves, out / "figlim_frac.tsv", ["x", *codecs, "practical", "envelope", "trend", "s1_of_trend"])
    for x, key in ((0.01, "0.01"), (0.1, "0.1"), (0.25, "0.25"), (1.0, "1")):
        rec = min(curves, key=lambda r: abs(np.log(r["x"] / x)))
        N.put(f"frac-s1-{key}", rec["pz2_avx2"] * 100, "{:.0f}")
        N.put(f"frac-practical-{key}", rec["practical"] * 100, "{:.0f}")
        N.put(f"frac-gain-{key}", (rec["practical"] / rec["pz2_avx2"] - 1) * 100, "{:.0f}")
        N.put(f"frac-trend-{key}", rec["trend"] * 100, "{:.0f}")
        N.put(f"frac-s1oftrend-{key}", rec["s1_of_trend"] * 100, "{:.0f}")
        v, c = best_other[rec["x"]]
        N.put(f"frac-bestother-{key}", v * 100, "{:.0f}")
        N.put(f"frac-bestother-{key}-name", KERNEL_LABELS[c])
    within = [r["x"] for r in curves if r["s1_of_trend"] >= 0.9]
    if within:
        N.put("trend-x90", max(x for x in within if all(r["s1_of_trend"] >= 0.9 for r in curves if r["x"] <= x)), "{:.2g}")
    cross = next((r["x"] for r in curves if best_other[r["x"]][0] > r["pz2_avx2"]), None)
    if cross:
        N.put("frac-cross-gbit", cross, "{:.2f}")
        N.put("frac-cross-name", KERNEL_LABELS[best_other[cross][1]])
    # the per-core bandwidths of familiar set-ups
    _write([{"x": 1 / 32, "label": "1\\,Gbit/s, 32 cores"}, {"x": 0.25, "label": "1\\,Gbit/s, 4 cores"},
            {"x": 2.5, "label": "10\\,Gbit/s, 4 cores"}], out / "figlim_marks.tsv", ["x", "label"])
    rec = min(curves, key=lambda r: abs(np.log(r["x"] * 32)))
    N.put("frac-s1-32c", rec["pz2_avx2"] * 100, "{:.0f}")


def write_panel(cfg: Config, N: Numbers, out: Path) -> None:
    """(d) seconds and extra memory to write the largest matrix of the write-cost panel, fastest first."""
    from .report import FORMAT_LABELS, _group

    rows = [r for r in _stage(cfg, "writecost") if r.get("status") == "ok"]
    if not rows:
        return
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    ds = max(by, key=lambda d: max(_f(r.get("nnz")) or 0 for r in by[d])) if any(_f(r.get("nnz")) for r in rows) else "hlca_core"
    tab = []
    for r in by.get(ds, []):
        if r["format"] in ("s1_pack", "loom"):
            continue
        extra = max(_f(r["peak_rss_mb"]), _f(r.get("peak_child_rss_mb")) or 0) - _f(r["baseline_rss_mb"])
        tab.append({"format": r["format"], "label": FORMAT_LABELS.get(r["format"], r["format"]), "group": _group(r["format"]),
                    "s": _f(r["write_s"]), "gb": max(0.0, extra) / 1024})
    tab.sort(key=lambda t: t["s"])
    for i, t in enumerate(tab):
        t["pos"] = i
        t["note"] = f"{t['gb']:.0f} GB"
    _write(tab, out / "figlim_write.tsv", ["pos", "format", "label", "group", "s", "gb", "note"])
    N.put("figlim-write-dataset", {"hlca_core": "HLCA core"}.get(ds, ds.replace("_", r"\_")))
    s1 = next((t for t in tab if t["format"] == "s1"), None)
    if s1:
        N.put("figlim-write-s1-rank", 1 + [t["format"] for t in tab].index("s1"), "{:d}")
        N.put("figlim-write-n", len(tab), "{:d}")
        gz = next((t for t in tab if t["format"] == "h5ad_gzip"), None)
        if gz:
            N.put("figlim-write-s1-vs-gzip-mem", s1["gb"] / gz["gb"], "{:.1f}")
            N.put("figlim-write-s1-vs-gzip-time", gz["s"] / s1["s"], "{:.1f}")


def redundancy_section(cfg: Config, N: Numbers, out: Path) -> None:
    """How much of each format's file a count model could still remove: 1 - floor / bits, medians over matrices."""
    from .report import FORMAT_LABELS

    floor = {(r["dataset"], r["matrix"]): _best(r) for r in _stage(cfg, "floor")}
    by = defaultdict(list)
    for r in _stage(cfg, "write"):
        fl = floor.get((r["dataset"], r["matrix"]))
        b = _f(r.get("bits_per_nnz"))
        if r.get("status") == "ok" and fl and b and r["format"] != "s1_pack":
            by[r["format"]].append((b, b - fl, (1 - fl / b) * 100))
    tab = []
    for f in sorted(by, key=lambda f: -float(np.median([v[2] for v in by[f]]))):
        v = np.array(by[f])
        tab.append({"format": FORMAT_LABELS.get(f, f), "n": len(v), "bits": float(np.median(v[:, 0])),
                    "excess": float(np.median(v[:, 1])), "pct": float(np.median(v[:, 2])),
                    "pct_lo": float(v[:, 2].min()), "pct_hi": float(v[:, 2].max())})
        key = f.replace("_", "-")
        N.put(f"redund-{key}", tab[-1]["pct"], "{:.0f}")
    _write(tab, out / "tab_redundancy.tsv", ["format", "n", "bits", "excess", "pct", "pct_lo", "pct_hi"])
    # main-text Table 1: the summary table (report.summary_section, written earlier) with the redundant share added
    if (out / "tab_summary.tsv").exists():
        import csv
        share = {t["format"]: t["pct"] for t in tab}
        with open(out / "tab_summary.tsv", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        for r in rows:
            r["redund"] = share.get(r["format"], np.nan)
            if r["format"] in (FORMAT_LABELS["rds_xz"], FORMAT_LABELS["loom"]):
                N.put("sumNr" if r["format"] == FORMAT_LABELS["rds_xz"] else "sumNloom", int(r["n"]), "{:d}")
        _write(rows, out / "tab_main.tsv", ["format", "bits", "rel", "cr", "smallest", "read", "fastest", "read1", "fastest1", "redund", "n"])
    others = [t for t, f in zip(tab, sorted(by, key=lambda f: -float(np.median([v[2] for v in by[f]])))) if not f.startswith("s1")]
    if others:
        low = min(others, key=lambda t: t["pct"])
        N.put("redund-other-min", low["pct"], "{:.0f}")
        N.put("redund-other-min-name", low["format"])
    if "s1" in by:
        N.put("redund-s1-lo", float(min(v[2] for v in by["s1"])), "{:.0f}")
        N.put("redund-s1-hi", float(max(v[2] for v in by["s1"])), "{:.0f}")


def entropy_section(cfg: Config, N: Numbers, out: Path) -> None:
    """What is left to compress inside each code's output (entropy stage): order-0 and order-1 byte entropy, the share of
    the output an order-1 byte coder could remove, and the order-0 entropy of the step and value streams."""
    rows = _stage(cfg, "entropy")
    if not rows:
        return
    floor = {(r["dataset"], r["matrix"]): _best(r) for r in _stage(cfg, "floor")}
    by = defaultdict(list)
    streams = []
    for r in rows:
        k = (r["dataset"], r["matrix"])
        if r["codec"] == "streams":
            if floor.get(k):
                streams.append((_f(r["h0_gap"]) + _f(r["h0_val"]), _f(r["h0_pair"]), floor[k]))
            continue
        b, h0, h1 = _f(r["bits"]), _f(r["h0"]), _f(r["h1"])
        z = _f(r.get("zstd19_bytes"))
        by[r["codec"]].append((b, h0, h1, (1 - h1 / 8) * 100, (z / _f(r["bytes"]) * 100) if z else np.nan))
    tab = []
    for c in sorted(by, key=lambda c: -float(np.median([v[0] for v in by[c]]))):
        v = np.array(by[c], dtype=float)
        tab.append({"codec": KERNEL_LABELS.get(c, c), "n": len(v), "bits": float(np.median(v[:, 0])),
                    "h0": float(np.median(v[:, 1])), "h1": float(np.median(v[:, 2])), "left": float(np.median(v[:, 3]))})
        N.put(f"ent-{c}-h1", tab[-1]["h1"], "{:.2f}")
        N.put(f"ent-{c}-left", tab[-1]["left"], "{:.0f}")
    _write(tab, out / "tab_entropy.tsv", ["codec", "n", "bits", "h0", "h1", "left"])
    if "pz2_avx2" in by:
        v = np.array(by["pz2_avx2"], dtype=float)
        N.put("ent-s1-h0", float(np.median(v[:, 1])), "{:.3f}")
        N.put("ent-s1-h1", float(np.median(v[:, 2])), "{:.3f}")
        N.put("ent-s1-h1-lo", float(v[:, 2].min()), "{:.3f}")
        N.put("ent-s1-zstd-pct", float(np.nanmedian(v[:, 4])), "{:.1f}")
    if streams:
        v = np.array(streams)
        N.put("ent-n", len(v), "{:d}")
        N.put("ent-h0-streams", float(np.median(v[:, 0])), "{:.1f}")
        N.put("ent-h0-pair", float(np.median(v[:, 1])), "{:.1f}")
        N.put("ent-h0-vs-floor", float(np.median(v[:, 0] / v[:, 2])), "{:.1f}")


def run(cfg: Config, N: Numbers, out: Path) -> None:
    kernels_section(cfg, N, out)
    write_panel(cfg, N, out)
    redundancy_section(cfg, N, out)
    entropy_section(cfg, N, out)
