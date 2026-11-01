"""Stage `second` (Q9): compress each .s1 file again with general-purpose compressors; report output / input."""

from __future__ import annotations

import bz2
import lzma
import shutil
import subprocess
import tempfile
from pathlib import Path

from .util import Config, Unit, log, result_path, write_tsv
from .write import format_dir


def _zstd22(b: bytes) -> int:
    import zstandard as zstd

    return len(zstd.ZstdCompressor(level=22).compress(b))


def _xz9e(b: bytes) -> int:
    return len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))


def _brotli11(b: bytes) -> int:
    import brotli

    return len(brotli.compress(b, quality=11, lgwin=24))


def _bzip2(b: bytes) -> int:
    return len(bz2.compress(b, 9))


CODERS = {"zstd22": _zstd22, "xz9e": _xz9e, "brotli11": _brotli11, "bzip2_9": _bzip2}


def _cm(path: Path, exe: str) -> int:
    """cmix or paq8px on the file; both write <name>.cmix / <name>.paq8px* next to the input in a temp dir."""
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / path.name
        shutil.copy(path, src)
        if "cmix" in Path(exe).name:
            dst = Path(t) / "out.cmix"
            subprocess.run([exe, "-c", str(src), str(dst)], check=True, capture_output=True)
        else:
            subprocess.run([exe, "-8", str(src)], check=True, capture_output=True, cwd=t)
            dst = next(p for p in Path(t).iterdir() if p.name.startswith(path.name + ".paq8"))
        return dst.stat().st_size


def run(cfg: Config, units: list[Unit], force: bool = False) -> None:
    sc = cfg["second"]
    for u in units:
        out = result_path(cfg, "second", u)
        if out.exists() and not force:
            continue
        p = format_dir(cfg, u) / "s1.s1"
        if not p.exists():
            log(f"{u.key}: no s1 file; run write first")
            continue
        b = p.read_bytes()
        rows = []
        for name in sc["coders"]:
            n = CODERS[name](b)
            rows.append({"dataset": u.id, "matrix": u.name, "coder": name, "in_bytes": len(b), "out_bytes": n, "ratio": n / len(b)})
        exe = sc.get("cm_coder")
        if exe and len(b) <= int(sc["cm_max_bytes"]):
            n = _cm(p, exe)
            rows.append({"dataset": u.id, "matrix": u.name, "coder": "cm", "cm_exe": Path(exe).name, "in_bytes": len(b), "out_bytes": n,
                         "ratio": n / len(b)})
        write_tsv(rows, out)
        log(f"{u.key}: second pass {[(r['coder'], round(r['ratio'], 4)) for r in rows]}")
