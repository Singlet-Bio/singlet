"""Stage `write` (Q1): write every format from the canonical matrix, measure stored bytes, check the exact round trip."""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path

from . import formats
from .util import Config, Unit, disk_bytes, load_canon, log, now, result_path, write_tsv


def format_dir(cfg: Config, u: Unit) -> Path:
    return cfg.data_dir / "formats" / u.key


def run(cfg: Config, units: list[Unit], fmt_ids: list[str], force: bool = False, threads: int = 0) -> None:
    threads = threads or int(cfg["threads"]["headline"])
    fmts = formats.get(fmt_ids)
    for u in units:
        out = result_path(cfg, "write", u)
        if out.exists() and not force:
            log(f"{u.key}: write results exist")
            continue
        c = load_canon(cfg, u)
        base = format_dir(cfg, u)
        base.mkdir(parents=True, exist_ok=True)
        rows = []
        for f in fmts:
            row = {"dataset": u.id, "matrix": u.name, "format": f.id, "group": f.group, "nnz": c.nnz, "genes": c.m, "cells": c.n}
            why = f.applicable(c, cfg)
            if why:
                rows.append({**row, "status": "not_applicable", "note": why})
                continue
            p = f.path(base)
            if p.exists():
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            try:
                t0 = now()
                f.write(c, p, threads)
                t_write = now() - t0
                size = disk_bytes(p)
                back = f.read(p, threads)
                ok = back.equals(c)
                rows.append({
                    **row,
                    "status": "ok" if ok else "roundtrip_failed",
                    "bytes": size,
                    "bits_per_nnz": 8.0 * size / max(1, c.nnz),
                    "shared_bytes": f.shared_bytes(p),
                    "write_s": t_write,
                    "write_threads": threads,
                })
                log(f"{u.key} {f.id}: {size} B, {8.0 * size / max(1, c.nnz):.2f} bits/nnz, round trip {'ok' if ok else 'FAILED'}")
            except Exception as e:  # a failure is a result, not a reason to stop
                rows.append({**row, "status": "error", "note": f"{type(e).__name__}: {e}"})
                log(f"{u.key} {f.id}: ERROR {e}\n{traceback.format_exc()}")
        write_tsv(rows, out)
