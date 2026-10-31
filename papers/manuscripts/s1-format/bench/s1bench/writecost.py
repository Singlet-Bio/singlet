"""Stage `writecost` (review round 3): seconds and peak memory to write each format from a matrix in memory, on the timing
host. Each (matrix, format) runs in a fresh process pinned to `threads.headline` cores: the canonical matrix is loaded
first (untimed; its resident memory is recorded as the baseline), the format's library is primed on a tiny matrix, then
one write to the node's disk is timed and the file deleted. Peak memory is the process's maximum resident set (children
included, for the R formats), reported with the baseline so that the writer's own share is their difference.
"""

from __future__ import annotations

import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .util import BENCH_DIR, Config, Unit, load_canon, log, pin_cores, read_tsv, result_path, write_tsv


def _rss_mb() -> tuple[float, float]:
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)


def one(cfg: Config, unit_key: str, fmt_id: str, threads: int) -> dict:
    """Runs inside the child process."""
    from . import formats
    from .decode import _prime
    from .util import select_units

    pin_cores(threads)
    f = formats.get([fmt_id])[0]
    u = next(x for x in select_units(cfg, unit_key.split(".", 1)[0]) if x.key == unit_key)
    c = load_canon(cfg, u)
    base_self, _ = _rss_mb()
    if f.lang != "r":
        _prime(cfg, f, threads, {})
    d = Path(tempfile.mkdtemp(prefix="wc-", dir=cfg.data_dir / "tmp"))
    try:
        p = f.path(d)
        t0 = time.perf_counter()
        f.write(c, p, threads)
        dt = time.perf_counter() - t0
        from .util import disk_bytes

        size = disk_bytes(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    peak_self, peak_child = _rss_mb()
    return {"write_s": dt, "bytes": size, "baseline_rss_mb": base_self, "peak_rss_mb": peak_self, "peak_child_rss_mb": peak_child}


def run(cfg: Config, units: list[Unit], fmt_ids: list[str], force: bool = False, overrides=()) -> None:
    from .decode import child_env

    H = int(cfg["threads"]["headline"])
    for u in units:
        out = result_path(cfg, "writecost", u)
        if out.exists() and not force:
            continue
        ok = {r["format"] for r in read_tsv(result_path(cfg, "write", u)) if r.get("status") == "ok"}
        rows = []
        for fid in fmt_ids:
            if fid not in ok:
                continue
            cmd = [sys.executable, "-m", "s1bench", "--config", str(cfg.path), *sum((["--set", o] for o in overrides), []),
                   "writecost-one", u.key, fid, str(H)]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=BENCH_DIR, env=child_env(fid))
            row = {"dataset": u.id, "matrix": u.name, "format": fid, "threads": H}
            if r.returncode == 0:
                row.update({"status": "ok", **json.loads(r.stdout.strip().splitlines()[-1])})
            else:
                row.update({"status": "error", "note": (r.stderr.strip().splitlines() or ["failed"])[-1]})
            rows.append(row)
            log(f"{u.key} {fid}: {row.get('write_s', row.get('note'))} s, peak {row.get('peak_rss_mb', '')} MB")
        write_tsv(rows, out)
