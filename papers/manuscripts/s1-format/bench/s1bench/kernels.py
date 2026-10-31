"""Stage `kernels`: how fast can any lossless code decode a count matrix, and how close is .s1?

cpp/kernel_bench encodes each matrix's own row indices and values with the textbook integer codes a format could adopt
(raw arrays, fixed-width bytes, LEB128, Stream VByte, SIMD bit-packing, zstd over gap streams), times decoding them back
into u32 CSC arrays on one pinned core, checks the round trip, and times the .s1 member on the same core. `store_bound`
only writes the output: no decoder that produces CSC arrays can run faster. Every code runs on the whole matrix; matrices
above `kernels.max_nnz` non-zeros (default decode.big_nnz) are left out, which keeps a run to about 20 GB of memory.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import numpy as np

from .util import BENCH_DIR, Config, Unit, canon_path, load_canon, log, result_path, summary, write_tsv
from .write import format_dir


def build() -> str:
    exe = BENCH_DIR / "cpp" / "kernel_bench"
    if not exe.exists():
        raise SystemExit("cpp/kernel_bench not built; run bench/cpp/build.sh")
    return str(exe)


def run(cfg: Config, units: list[Unit], force: bool = False) -> None:
    from singlet.s1.container import S2Reader

    exe = build()
    kc = cfg.get("kernels", {}) or {}
    cap = int(float(kc.get("max_nnz", cfg["decode"]["big_nnz"])))
    reps, warm = int(kc.get("repeats", 7)), int(kc.get("warmup", 2))
    core = sorted(os.sched_getaffinity(0))[0]
    for u in units:
        out = result_path(cfg, "kernels", u)
        if out.exists() and not force:
            continue
        s1 = format_dir(cfg, u) / "s1.s1"
        if not s1.exists():
            log(f"{u.key}: no .s1 file; skipped")
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
            member = os.path.join(td, "counts.pz2")
            with open(member, "wb") as f:
                f.write(S2Reader(str(s1)).read_member("counts"))
            del c
            r = subprocess.run(["taskset", "-c", str(core), exe, mat, member, str(reps), str(warm)], capture_output=True,
                               text=True)
        if r.returncode != 0:
            log(f"{u.key}: kernel_bench failed: {r.stderr.strip()[-400:]}")
            continue
        rows = []
        for line in r.stdout.strip().splitlines():
            j = json.loads(line)
            s = summary(j["times"])
            rows.append({"dataset": u.id, "matrix": u.name, "codec": j["codec"], "nnz": j["nnz"], "cells": ncol,
                         "bits": 8.0 * j["bytes"] / j["nnz"], "encode_s": j["encode_s"], "ok": int(j["ok"]),
                         "median_s": s["median_s"], "q1_s": s["q1_s"], "q3_s": s["q3_s"],
                         "mnnz_per_s": j["nnz"] / s["median_s"] / 1e6, "isa_used": j.get("isa_used", ""),
                         "times_s": ",".join(f"{v:.6f}" for v in j["times"])})
        write_tsv(rows, out)
        log(f"{u.key}: " + ", ".join(f"{x['codec']} {x['bits']:.2f} b {x['mnnz_per_s']:.0f} M/s ok={x['ok']}" for x in rows))
