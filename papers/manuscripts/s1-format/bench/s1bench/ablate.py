"""Stage `ablate`: which parts of the .s1 count model earn their bits?

cpp/ablate_bench reads each matrix's canonical CSC and prints one JSON line per variant:

  kind "model"  the ideal static code length of the codec-1 streams with one model component added or removed
                (row order, cell-depth context, position context, value contexts), charged the way the encoder charges it;
  kind "real"   the shipped encoder with one option changed (context configuration, class resolution, lane count, the generic
                zstd codec 2): true payload bytes and the one-thread decode rate, checked against the input.

`aggregate` turns the per-matrix results into generated/tab_ablation.tsv (medians and quartiles over matrices, change relative
to the full model) for the manuscript. Matrices above `ablate.max_nnz` non-zeros (default kernels.max_nnz, else
decode.big_nnz) are left out, as in the kernels stage.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import numpy as np

from .util import BENCH_DIR, Config, Unit, canon_path, load_canon, log, read_stage, result_path, summary, write_tsv

# the model variants in the order of the table, with the label the manuscript shows
MODEL_ORDER = [
    ("m0_order0_source", "order-0 code, source gene order"),
    ("m1_class_order", "+ genes in class order"),
    ("m2_cell_depth_ctx", "+ cell-depth context"),
    ("m3_position_ctx", "+ position context"),
    ("m4_value_ctx", "+ value contexts (full model)"),
]
LOO_ORDER = [
    ("loo_no_class_order", "full model without class order"),
    ("loo_no_cell_depth_ctx", "full model without cell-depth context"),
    ("loo_no_position_ctx", "full model without position context"),
    ("loo_no_value_ctx", "full model without value contexts"),
    ("loo_value_position_only", "value contexts: position only"),
    ("loo_value_depth_only", "value contexts: depth class only"),
]
REAL_ORDER = [
    ("real_default", "shipped encoder, default options"),
    ("real_ctx0", "context configuration 0"), ("real_ctx1", "context configuration 1"), ("real_ctx2", "context configuration 2"),
    ("real_ctx3", "context configuration 3"), ("real_ctx4", "context configuration 4"),
    ("real_cpo0", "exact gene rank"), ("real_cpo1", "1 class per octave"), ("real_cpo2", "2 classes per octave"),
    ("real_cpo8", "8 classes per octave"),
    ("real_lanes16", "16 lanes"), ("real_lanes32", "32 lanes"), ("real_lanes64", "64 lanes"),
    ("real_codec2_zstd", "generic codec (zstd over LEB128 gaps)"),
]


def build() -> str:
    exe = BENCH_DIR / "cpp" / "ablate_bench"
    if not exe.exists():
        raise SystemExit("cpp/ablate_bench not built; run bench/cpp/build.sh")
    return str(exe)


def run(cfg: Config, units: list[Unit], force: bool = False) -> None:
    exe = build()
    kc = cfg.get("ablate", {}) or {}
    cap = int(float(kc.get("max_nnz", (cfg.get("kernels", {}) or {}).get("max_nnz", cfg["decode"]["big_nnz"]))))
    reps, warm = int(kc.get("repeats", 5)), int(kc.get("warmup", 1))
    core = sorted(os.sched_getaffinity(0))[0]
    for u in units:
        out = result_path(cfg, "ablate", u)
        if out.exists() and not force:
            continue
        if int(np.load(canon_path(cfg, u))["indptr"][-1]) > cap:   # reads only indptr
            log(f"{u.key}: above {cap:.0e} non-zeros; skipped")
            continue
        c = load_canon(cfg, u)
        ncol, nnz = c.n, c.nnz
        with tempfile.TemporaryDirectory() as td:
            mat = os.path.join(td, "matrix.bin")
            with open(mat, "wb") as f:
                np.array([c.m, ncol, nnz], dtype=np.uint64).tofile(f)
                c.indptr.astype(np.uint64).tofile(f)
                c.indices.astype(np.uint32).tofile(f)
                c.data.astype(np.uint32).tofile(f)
            del c
            # model variants are timing-free and use up to 8 threads; the real decode is one thread, so pin the whole run to the
            # cores of one node slot rather than to a single core
            r = subprocess.run([exe, mat, str(reps), str(warm)], capture_output=True, text=True)
        if r.returncode != 0:
            log(f"{u.key}: ablate_bench failed: {r.stderr.strip()[-400:]}")
            continue
        rows = []
        for line in r.stdout.strip().splitlines():
            j = json.loads(line)
            row = {"dataset": u.id, "matrix": u.name, "cells": ncol, "nnz": j["nnz"], "kind": j["kind"],
                   "variant": j["variant"], "bits": j["bits"]}
            if j["kind"] == "model":
                for k in ("gap_bits", "res_bits", "extra_bits", "count_bits", "table_bits"):
                    row[k] = j[k]
            else:
                s = summary(j["times"])
                row.update({"payload_bytes": j["payload_bytes"], "total_bytes": j["total_bytes"], "codec": j["codec"],
                            "encode_s": j["encode_s"], "ok": int(j["ok"]), "median_s": s["median_s"],
                            "mnnz_per_s": j["nnz"] / s["median_s"] / 1e6})
            rows.append(row)
        write_tsv(rows, out)
        full = next((x["bits"] for x in rows if x["variant"] == "m4_value_ctx"), float("nan"))
        real = next((x["bits"] for x in rows if x["variant"] == "real_default"), float("nan"))
        log(f"{u.key}: model full {full:.3f} b, real default {real:.3f} b; "
            + ", ".join(f"{x['variant']} {x['bits']:.2f}" for x in rows if x["kind"] == "model"))


def _q(v: list[float]) -> tuple[float, float, float]:
    a = np.asarray(v, dtype=float)
    return float(np.median(a)), float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))


def aggregate(cfg: Config, out_dir) -> None:
    """generated/tab_ablation.tsv: one row per variant with the median, quartiles and the median per-matrix change against
    the full model (model rows) or against the shipped encoder's default options (real rows)."""
    rows = read_stage(cfg, "ablate")
    if not rows:
        log("ablate: no results; table not written")
        return
    by: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        by.setdefault((r["kind"], r["variant"]), {})[r["dataset"] + "/" + r["matrix"]] = r
    out = []
    for kind, order, ref in (("model", MODEL_ORDER + LOO_ORDER, "m4_value_ctx"), ("real", REAL_ORDER, "real_default")):
        refm = by.get((kind, ref), {})
        for variant, label in order:
            d = by.get((kind, variant), {})
            if not d:
                continue
            bits = [float(x["bits"]) for x in d.values()]
            m, q1, q3 = _q(bits)
            rel = [float(x["bits"]) / float(refm[k]["bits"]) for k, x in d.items() if k in refm and float(refm[k]["bits"]) > 0]
            row = {"kind": kind, "variant": variant, "label": label, "n": len(d), "bits_median": m, "bits_q1": q1, "bits_q3": q3,
                   "rel_to_ref_median": _q(rel)[0] if rel else float("nan")}
            if kind == "real":
                rate = [float(x["mnnz_per_s"]) for x in d.values()]
                rm, rq1, rq3 = _q(rate)
                row.update({"mnnz_per_s_median": rm, "mnnz_per_s_q1": rq1, "mnnz_per_s_q3": rq3,
                            "all_ok": int(all(int(x["ok"]) for x in d.values()))})
            out.append(row)
    # how well the model of the full codec reproduces the real payload, per matrix
    full, real = by.get(("model", "m4_value_ctx"), {}), by.get(("real", "real_default"), {})
    ratio = [float(full[k]["bits"]) / float(real[k]["bits"]) for k in full if k in real and float(real[k]["bits"]) > 0]
    if ratio:
        m, q1, q3 = _q(ratio)
        out.append({"kind": "check", "variant": "model_over_real", "label": "full-model estimate / real payload", "n": len(ratio),
                    "bits_median": m, "bits_q1": q1, "bits_q3": q3, "rel_to_ref_median": m})
    from pathlib import Path

    write_tsv(out, Path(out_dir) / "tab_ablation.tsv")
    log(f"ablate: wrote tab_ablation.tsv ({len(out)} rows)")
