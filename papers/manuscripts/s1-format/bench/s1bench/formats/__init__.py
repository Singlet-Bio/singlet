"""Format registry. Every format writes the canonical matrix (counts, gene ids, barcodes) and reads it back.

A format implements
    write(c: Canon, path: Path, threads: int) -> None
    load(path: Path, threads: int) -> object           (what its library returns: a SciPy matrix and names, or AnnData;
                                                        this alone is what `decode` times)
    to_canon(obj) -> Canon                             (conversion and checks, untimed; read() = to_canon(load()) is the
                                                        exact round trip of the write stage)
    applicable(c: Canon, cfg) -> Optional[str]         (None, or the reason it is skipped for this matrix)
`path` is a file or a directory (`is_dir`). Stored bytes are everything under `path`, names included. Shared side files
that a format downloads once and caches (the .s1 gene vocabulary pack) are reported in `shared_bytes`, not in the size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..util import Canon, Config

GROUP_GENERAL, GROUP_DOMAIN, GROUP_S1 = "general", "domain", "s1"


class Format:
    id: str = ""
    label: str = ""
    ext: str = ""
    is_dir: bool = False
    group: str = GROUP_GENERAL
    lang: str = "py"

    def path(self, base: Path) -> Path:
        return base / f"{self.id}{self.ext}"

    def applicable(self, c: Canon, cfg: Config) -> Optional[str]:
        return None

    def write(self, c: Canon, path: Path, threads: int) -> None:
        raise NotImplementedError

    def load(self, path: Path, threads: int, **kw) -> object:
        raise NotImplementedError

    def to_canon(self, obj) -> Canon:
        raise NotImplementedError

    def read(self, path: Path, threads: int, **kw) -> Canon:
        return self.to_canon(self.load(path, threads, **kw))

    def shared_bytes(self, path: Path) -> int:
        return 0


def registry() -> dict[str, Format]:
    from . import py, rfmt, s1

    fmts: list[Format] = [*py.FORMATS, *rfmt.FORMATS, *s1.FORMATS]
    return {f.id: f for f in fmts}


def get(ids: list[str]) -> list[Format]:
    reg = registry()
    missing = [i for i in ids if i not in reg]
    if missing:
        raise SystemExit(f"unknown format id(s): {', '.join(missing)}")
    return [reg[i] for i in ids]
