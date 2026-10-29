"""Stage `eco` (Q10): the calls in Fig. 4, timed against the usual alternative, and checked for identical counts.

Python: singlet.load (.s1 -> AnnData) vs anndata.read_h5ad and scanpy.read_10x_h5. C++: Matrix::open + decode
(cpp/decode_bench). R: R/eco.R (singlet::load as SCE and Seurat vs Seurat::Read10X_h5 and zellkonverter::readH5AD).
PyTorch numbers come from the minibatch stage.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import scipy.sparse as sp

from .util import BENCH_DIR, Config, Unit, evict, load_canon, log, pin_cores, repeat, result_path, summary, write_tsv
from .write import format_dir


def _same(x, c) -> bool:
    """x: cells x genes (AnnData orientation) against the canonical genes x cells matrix."""
    a = sp.csr_matrix(x).astype(np.int64)
    b = c.to_scipy().T.tocsr().astype(np.int64)
    return a.shape == b.shape and (a != b).nnz == 0


def run(cfg: Config, units: list[Unit], force: bool = False) -> None:
    threads = int(cfg["threads"]["headline"])
    reps, warm = int(cfg["repeats"]), int(cfg["warmup"])
    pin_cores(threads)
    for u in units:
        if u.id not in cfg["eco"]["datasets"]:
            continue
        out = result_path(cfg, "eco", u)
        if out.exists() and not force:
            continue
        d = format_dir(cfg, u)
        c = load_canon(cfg, u)
        rows = []

        def py_case(call, path, fn):
            res = {}

            def go():
                res["obj"] = fn()

            ts = repeat(go, reps, warm, before=lambda: evict(path))
            rows.append({"dataset": u.id, "lang": "Python", "call": call, **summary(ts), "equal": _same(res["obj"].X, c)})

        import anndata as ad

        py_case("anndata.read_h5ad", d / "h5ad_gzip.h5ad", lambda: ad.read_h5ad(d / "h5ad_gzip.h5ad"))
        try:
            import scanpy as scp

            py_case("scanpy.read_10x_h5", d / "tenx_h5.h5", lambda: scp.read_10x_h5(d / "tenx_h5.h5"))
        except ImportError:
            pass
        try:
            import singlet

            py_case("singlet.load", d / "s1.s1", lambda: singlet.load(str(d / "s1.s1")))
        except Exception as e:  # the call shown in Fig. 4 must work; a failure here is a finding
            rows.append({"dataset": u.id, "lang": "Python", "call": "singlet.load", "status": "error", "note": str(e)})

        # C++: extract the matrix member once, then time open + decode
        from singlet.s1.container import S2Reader

        member = d / "s1.counts.pz2"
        member.write_bytes(S2Reader(str(d / "s1.s1")).read_member("counts"))
        exe = BENCH_DIR / "cpp" / "decode_bench"
        if exe.exists():
            r = subprocess.run([str(exe), str(member), str(threads), "auto", "native", str(reps), str(warm)], capture_output=True,
                               text=True, check=True)
            j = json.loads(r.stdout.strip().splitlines()[-1])
            rows.append({"dataset": u.id, "lang": "C++", "call": "pz2::Matrix::open + decode", **summary(j["times"]),
                         "equal": ""})

        r = subprocess.run(["Rscript", str(BENCH_DIR / "R" / "eco.R"), str(d), str(threads), str(reps)], capture_output=True, text=True,
                           cwd=BENCH_DIR)
        by_call: dict[str, list] = {}
        for ln in r.stdout.splitlines():
            if ln.strip().startswith("{"):
                j = json.loads(ln)
                if "error" in j:
                    rows.append({"dataset": u.id, "lang": "R", "call": j["call"], "status": "error", "note": j["error"][-500:]})
                    continue
                by_call.setdefault(j["call"], []).append(j)
        for call, js in by_call.items():
            rows.append({"dataset": u.id, "lang": "R", "call": call, **summary([j["seconds"] for j in js]),
                         "equal": all(j["equal_to_reference"] in (True, None) for j in js)})
        if r.returncode != 0:
            rows.append({"dataset": u.id, "lang": "R", "call": "eco.R", "status": "error", "note": r.stderr.strip()[-500:]})
        write_tsv(rows, out)
        log(f"{u.key}: eco {[(x['call'], x.get('median_s')) for x in rows]}")
