"""Stage `decode` (Q2): time from a file on local disk to an in-memory CSC, cold and warm, over thread counts.

For `.s1` it also times the codec alone on a member already in memory, per ISA and row order, which is the number that
separates codec speed from I/O. Each configuration runs in a fresh process pinned to `threads` cores
(`python -m s1bench decode-one ...`), so thread pools and caches from one configuration never help the next.
"""

from __future__ import annotations

import json
import subprocess
import sys
from . import formats
from .util import BENCH_DIR, Config, Unit, evict, load_canon, log, pin_cores, read_tsv, repeat, result_path, summary, write_tsv
from .write import format_dir


def _one(cfg: Config, unit_key: str, fmt_id: str, threads: int, cold: bool, extra: dict) -> dict:
    """Runs inside the child process."""
    pin_cores(threads)
    f = formats.get([fmt_id])[0]
    ds, mat = unit_key.split(".", 1)
    p = f.path(cfg.data_dir / "formats" / unit_key)
    reps, warm = int(cfg["repeats"]), int(cfg["warmup"])
    before = (lambda: evict(p)) if cold else None
    if f.lang == "r":
        ts = f.time_read(p, threads, reps, warm, cold)
    elif extra.get("codec_only"):
        from singlet.s1.container import S2Reader

        body = S2Reader(str(p)).read_member("counts")
        ts = repeat(lambda: f.decode_only(body, threads, extra["mode"], extra.get("isa")), reps, warm)
    elif f.id.startswith("s1"):
        ts = repeat(lambda: f.read(p, threads, mode=extra.get("mode", "canonical"), isa=extra.get("isa")), reps, warm, before)
    else:
        ts = repeat(lambda: f.read(p, threads), reps, warm, before)
    return {**summary(ts), "times": ts}


def _child(cfg_path: str, unit_key: str, fmt_id: str, threads: int, cold: bool, extra: dict, overrides) -> dict:
    cmd = [sys.executable, "-m", "s1bench", "--config", cfg_path, *sum((["--set", o] for o in overrides), []),
           "decode-one", unit_key, fmt_id, str(threads), str(int(cold)), json.dumps(extra)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=BENCH_DIR)
    if r.returncode != 0:
        return {"status": "error", "note": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "failed"}
    return json.loads(r.stdout.strip().splitlines()[-1])


def run(cfg: Config, units: list[Unit], fmt_ids: list[str], force: bool = False, sweep: bool = False, overrides=()) -> None:
    thread_list = list(cfg["threads"]["sweep"]) if sweep else sorted({1, int(cfg["threads"]["headline"])})
    for u in units:
        out = result_path(cfg, "decode", u)
        if out.exists() and not force:
            log(f"{u.key}: decode results exist")
            continue
        written = {r["format"]: r for r in read_tsv(result_path(cfg, "write", u)) if r.get("status") == "ok"}
        nnz = int(next(iter(written.values()))["nnz"]) if written else load_canon(cfg, u).nnz
        rows = []
        configs = []
        for fid in fmt_ids:
            if fid not in written:
                continue
            for t in thread_list:
                for cold in (True, False):
                    if fid.startswith("s1"):
                        for mode in cfg["s1"]["modes"]:
                            configs.append((fid, t, cold, {"mode": mode}))
                    else:
                        configs.append((fid, t, cold, {}))
            if fid == "s1":
                for t in thread_list:
                    for mode in cfg["s1"]["modes"]:
                        for isa in cfg["s1"]["isas"]:
                            configs.append((fid, t, False, {"codec_only": True, "mode": mode, "isa": isa}))
        for fid, t, cold, extra in configs:
            res = _child(str(cfg.path), u.key, fid, t, cold, extra, overrides)
            row = {"dataset": u.id, "matrix": u.name, "format": fid, "threads": t, "cache": "cold" if cold else "warm",
                   "what": "codec" if extra.get("codec_only") else "file", "mode": extra.get("mode", ""), "isa": extra.get("isa", "auto"),
                   "nnz": nnz}
            if "median_s" in res:
                row.update({"status": "ok", "median_s": res["median_s"], "q1_s": res["q1_s"], "q3_s": res["q3_s"],
                            "mnnz_per_s": nnz / res["median_s"] / 1e6})
            else:
                row.update(res)
            rows.append(row)
            log(f"{u.key} {fid} t={t} {row['cache']} {extra}: {row.get('mnnz_per_s', row.get('note'))}")
        rows += _cpp_rows(cfg, u, thread_list, nnz) if "s1" in written else []
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
    for t in thread_list:
        for mode in cfg["s1"]["modes"]:
            for isa in cfg["s1"]["isas"]:
                r = subprocess.run(["taskset", "-c", f"0-{t - 1}", str(exe), str(member), str(t), isa, mode, str(cfg["repeats"]),
                                    str(cfg["warmup"])], capture_output=True, text=True)
                row = {"dataset": u.id, "matrix": u.name, "format": "s1", "threads": t, "cache": "memory", "what": "codec_cpp",
                       "mode": mode, "isa": isa, "nnz": nnz}
                if r.returncode == 0:
                    s = summary(json.loads(r.stdout.strip().splitlines()[-1])["times"])
                    row.update({"status": "ok", "median_s": s["median_s"], "q1_s": s["q1_s"], "q3_s": s["q3_s"],
                                "mnnz_per_s": nnz / s["median_s"] / 1e6})
                else:
                    row.update({"status": "error", "note": (r.stderr.strip().splitlines() or ["failed"])[-1]})
                rows.append(row)
    return rows
