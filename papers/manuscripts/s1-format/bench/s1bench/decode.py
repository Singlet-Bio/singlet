"""Stage `decode` (Q2): time from a file on local disk to an in-memory CSC, cold and warm, over thread counts.

For `.s1` it also times the codec alone on a member already in memory, per ISA and row order, which is the number that
separates codec speed from I/O. Each configuration runs in a fresh process pinned to `threads` cores
(`python -m s1bench decode-one ...`), so thread pools and caches from one configuration never help the next.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from . import formats
from .util import BENCH_DIR, Config, Unit, evict, load_canon, log, pin_cores, read_tsv, repeat, result_path, summary, write_tsv
from .write import format_dir


def _one(cfg: Config, unit_key: str, fmt_id: str, threads: int, cold: bool, extra: dict) -> dict:
    """Runs inside the child process."""
    pin_cores(threads)
    f = formats.get([fmt_id])[0]
    ds, mat = unit_key.split(".", 1)
    p = f.path(Path(extra.get("root") or cfg.data_dir) / "formats" / unit_key)
    # a cold read evicts the file before every run, so a warm-up run would only repeat the first timed one
    reps, warm = int(cfg["repeats"]), 0 if cold else int(cfg["warmup"])
    long_s = cfg.get("long_run_s")
    before = (lambda: evict(p)) if cold else None
    # Imports and the library's first-use set-up are not part of a read: a script that reads a matrix has them done. A
    # tiny file of the same format is written and read once, untimed, so the timed runs start with the library loaded
    # and without touching the timed file (decode.prime; R formats load their packages before their own timing loop).
    if (cfg.get("decode") or {}).get("prime", True) and f.lang != "r" and not extra.get("codec_only"):
        _prime(cfg, f, threads, extra)
    # Timed: the library's own read into memory (load), not the benchmark's conversion and checks (to_canon), which the
    # write stage's round trip already covers.
    if f.lang == "r":
        ts = f.time_read(p, threads, reps, warm, cold)
    elif extra.get("codec_only"):
        from singlet.s1.container import S2Reader

        body = S2Reader(str(p)).read_member("counts")
        ts = repeat(lambda: f.decode_only(body, threads, extra["mode"], extra.get("isa")), reps, warm, long_s=long_s)
    elif f.id.startswith("s1"):
        ts = repeat(lambda: f.load(p, threads, mode=extra.get("mode", "canonical"), isa=extra.get("isa")), reps, warm, before,
                    long_s)
    else:
        ts = repeat(lambda: f.load(p, threads), reps, warm, before, long_s)
    res = {**summary(ts), "times": ts}
    if f.id.startswith("s1"):
        res["isa_used"] = _isa_used(extra.get("isa"))
    return res


def _prime(cfg: Config, f, threads: int, extra: dict) -> None:
    import shutil
    import tempfile

    import numpy as np
    import scipy.sparse as sp

    from .util import canon_from_scipy

    rng = np.random.default_rng(0)
    x = sp.random(60, 40, density=0.2, format="csc", random_state=rng, data_rvs=lambda k: rng.integers(1, 9, k))
    c = canon_from_scipy(x, [f"g{i}" for i in range(60)], [f"c{j}" for j in range(40)])
    d = Path(tempfile.mkdtemp(prefix="prime-", dir=cfg.data_dir / "tmp" if (cfg.data_dir / "tmp").is_dir() else None))
    try:
        q = f.path(d)
        f.write(c, q, threads)
        if f.id.startswith("s1"):
            f.load(q, threads, mode=extra.get("mode", "canonical"), isa=extra.get("isa"))
        else:
            f.load(q, threads)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _isa_used(requested) -> str:
    """The kernel the codec actually ran: a request the CPU cannot serve falls back (AVX-512 needs VBMI2)."""
    from singlet import _pz2  # type: ignore

    order = ["scalar", "avx2", "avx512"]
    best = next((t.split("=", 1)[1] for t in _pz2.isa_report().split() if t.startswith("best=")), "scalar")
    if requested in (None, "", "auto"):
        return best
    return order[min(order.index(requested), order.index(best))] if requested in order and best in order else str(requested)


def _child(cfg_path: str, unit_key: str, fmt_id: str, threads: int, cold: bool, extra: dict, overrides) -> dict:
    cmd = [sys.executable, "-m", "s1bench", "--config", cfg_path, *sum((["--set", o] for o in overrides), []),
           "decode-one", unit_key, fmt_id, str(threads), str(int(cold)), json.dumps(extra)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=BENCH_DIR, env=child_env(fmt_id))
    if r.returncode != 0:
        return {"status": "error", "note": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "failed"}
    return json.loads(r.stdout.strip().splitlines()[-1])


def child_env(fmt_id: str) -> dict:
    """The environment of a child process: zarr3 runs with the zarr-python 3 side install first on the path."""
    env = dict(os.environ)
    site = env.get("S1_ZARR3_SITE")
    if fmt_id.startswith("zarr3") and site:
        env["PYTHONPATH"] = site + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _storages(cfg: Config) -> list[dict]:
    """Where the format files are read from. `local`: data_dir (the timed jobs point it at a copy on the node's own disk).
    Others name a root of their own (e.g. the NFS project space) and may restrict threads and cache states."""
    from .util import _expand

    out = []
    for s in cfg.get("storages") or [{"name": "local"}]:
        s = dict(s)
        if s.get("root"):
            s["root"] = _expand(str(s["root"]))
            if not s["root"]:
                continue  # the variable that names this root is not set on this host
        out.append(s)
    return out


def run(cfg: Config, units: list[Unit], fmt_ids: list[str], force: bool = False, sweep: bool = False, overrides=()) -> None:
    for st in _storages(cfg):
        _run(cfg, units, fmt_ids, force, sweep, overrides, st)


def _run(cfg: Config, units: list[Unit], fmt_ids: list[str], force: bool, sweep: bool, overrides, st: dict) -> None:
    H = int(cfg["threads"]["headline"])
    base_threads = list(cfg["threads"]["sweep"]) if sweep else list(cfg["threads"].get("main") or sorted({1, H}))
    thread_list = [t for t in base_threads if t in st["threads"]] if st.get("threads") else base_threads
    caches = [c == "cold" for c in st.get("caches", ["cold", "warm"])]
    warm_threads = set(cfg["threads"].get("warm") or thread_list)  # warm reads only at these thread counts
    local = st["name"] == "local"
    for u in units:
        out = result_path(cfg, "decode", u, "" if local else f".{st['name']}")
        if sweep:
            out = out.with_name(out.name.replace(".tsv", ".sweep.tsv"))
        if out.exists() and not force:
            log(f"{u.key}: decode results exist")
            continue
        written = {r["format"]: r for r in read_tsv(result_path(cfg, "write", u)) if r.get("status") == "ok"}
        nnz = int(next(iter(written.values()))["nnz"]) if written else load_canon(cfg, u).nnz
        # large matrices (decode.big_nnz): the formats in decode.big_formats at the headline thread count from the node's
        # own disk only, and no thread sweep, so that no measurement runs for hours
        big = nnz > float((cfg.get("decode") or {}).get("big_nnz", float("inf")))
        if big and (sweep or not local):
            log(f"{u.key}: {nnz:,} non-zeros, above decode.big_nnz: {'thread sweep' if sweep else st['name']} not timed")
            continue
        unit_fmts = [f for f in fmt_ids if f in cfg["decode"]["big_formats"]] if big else fmt_ids
        unit_threads = [H] if big else thread_list
        rows = []
        configs = []
        for fid in unit_fmts:
            if fid not in written:
                continue
            for t in unit_threads:
                for cold in caches:
                    if not cold and t not in warm_threads:
                        continue
                    if fid.startswith("s1"):
                        for mode in cfg["s1"]["modes"]:
                            configs.append((fid, t, cold, {"mode": mode}))
                    else:
                        configs.append((fid, t, cold, {}))
            if fid == "s1" and local:
                for t in unit_threads:
                    for mode in cfg["s1"]["modes"]:
                        for isa in cfg["s1"]["isas"]:
                            configs.append((fid, t, False, {"codec_only": True, "mode": mode, "isa": isa}))
        for fid, t, cold, extra in configs:
            if st.get("root"):
                extra = {**extra, "root": st["root"]}
            res = _child(str(cfg.path), u.key, fid, t, cold, extra, overrides)
            row = {"dataset": u.id, "matrix": u.name, "run": "sweep" if sweep else "main", "storage": st["name"], "format": fid, "threads": t,
                   "cache": "cold" if cold else "warm",
                   "what": "codec" if extra.get("codec_only") else "file", "mode": extra.get("mode", ""), "isa": extra.get("isa", "auto"),
                   "nnz": nnz}
            if "median_s" in res:
                row.update({"status": "ok", "median_s": res["median_s"], "q1_s": res["q1_s"], "q3_s": res["q3_s"],
                            "mnnz_per_s": nnz / res["median_s"] / 1e6, "n": res.get("n"),
                            "times_s": ",".join(f"{v:.6f}" for v in res.get("times", [])), "isa_used": res.get("isa_used", "")})
            else:
                row.update(res)
            rows.append(row)
            log(f"{u.key} {fid} t={t} {row['cache']} {extra}: {row.get('mnnz_per_s', row.get('note'))}")
        rows += [{**r, "run": "sweep" if sweep else "main"} for r in _cpp_rows(cfg, u, thread_list, nnz)] if "s1" in written and local else []
        write_tsv(rows, out)


def _cpp_rows(cfg: Config, u: Unit, thread_list: list[int], nnz: int) -> list[dict]:
    """Codec speed without Python: cpp/decode_bench on the extracted matrix member, every thread count, ISA and order."""
    exe = BENCH_DIR / "cpp" / "decode_bench"
    if not exe.exists():
        log("cpp/decode_bench not built; run bench/cpp/build.sh to add the C++ rows")
        return []
    from singlet.s1.container import S2Reader

    d = format_dir(cfg, u)
    member = d / "s1.counts.pz2"
    member.write_bytes(S2Reader(str(d / "s1.s1")).read_member("counts"))
    rows = []
    allowed = sorted(os.sched_getaffinity(0))   # the job's cores, which on a shared node need not start at 0
    for t in thread_list:
        cores = ",".join(map(str, allowed[:t]))
        for mode in cfg["s1"]["modes"]:
            for isa in cfg["s1"]["isas"]:
                r = subprocess.run(["taskset", "-c", cores, str(exe), str(member), str(t), isa, mode, str(cfg["repeats"]),
                                    str(cfg["warmup"])], capture_output=True, text=True)
                row = {"dataset": u.id, "matrix": u.name, "storage": "memory", "format": "s1", "threads": t, "cache": "memory",
                       "what": "codec_cpp",
                       "mode": mode, "isa": isa, "nnz": nnz}
                if r.returncode == 0:
                    s = summary(json.loads(r.stdout.strip().splitlines()[-1])["times"])
                    out_j = json.loads(r.stdout.strip().splitlines()[-1])
                    row.update({"status": "ok", "median_s": s["median_s"], "q1_s": s["q1_s"], "q3_s": s["q3_s"],
                                "mnnz_per_s": nnz / s["median_s"] / 1e6, "n": s["n"],
                                "times_s": ",".join(f"{v:.6f}" for v in out_j["times"]), "isa_used": out_j.get("isa_used", "")})
                else:
                    row.update({"status": "error", "note": (r.stderr.strip().splitlines() or ["failed"])[-1]})
                rows.append(row)
    return rows
