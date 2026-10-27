# SPDX-License-Identifier: MIT
"""Where a feature job's cells come from. Counts and ``col`` are read with :mod:`singlet.celllabels.io` (the same reader as the
labels, so labels and features line up on ``col``):

* a per-sample study on data.singlet.bio: :func:`singlet.celllabels.io.read_s1`;
* a legacy ``.singlet`` bundle on disk: :func:`singlet.celllabels.io.read_legacy` (``col`` = position among the columns of
  ``exon_counts.1pz``, the order the .s1 migration gives the cells table).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

DATA_BASE = os.environ.get("SINGLET_DATA_BASE", "https://data.singlet.bio")
# data.singlet.bio refuses urllib's default User-Agent (HTTP 403)
HEADERS = {"User-Agent": "singlet-features (+https://singlet.bio)"}


def _req(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers=HEADERS)


def is_s1(gse: str, base: str = DATA_BASE) -> bool:
    for prefix in ("s1", "store", "v2"):  # DUAL: the old v2/m/ prefix
        req = _req(f"{base}/{prefix}/m/{gse}.json", "HEAD")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
    return False


def legacy_url(gse: str, base: str = DATA_BASE) -> str:
    return f"{base}/data/{gse}/{gse}.singlet"


def fetch_bundle(gse: str, dest_dir: Path, base: str = DATA_BASE) -> Path:
    """Download a legacy bundle (checked against Content-Length) into ``dest_dir``; reuses a complete file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{gse}.singlet"
    url = legacy_url(gse, base)
    with urllib.request.urlopen(_req(url, "HEAD"), timeout=60) as r:
        size = int(r.headers.get("Content-Length") or 0)
    if path.exists() and size and path.stat().st_size == size:
        return path
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(_req(url), timeout=600) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    if size and tmp.stat().st_size != size:
        raise IOError(f"{gse}: downloaded {tmp.stat().st_size} of {size} bytes")
    tmp.replace(path)
    return path


def bundle_gsms(path: Path) -> List[str]:
    from singlet.bundle import SingletBundle

    b = SingletBundle(path)
    try:
        return list(b.manifest.get("gsm_ids") or [])
    finally:
        b.close()


def s1_gsms(gse: str, base: Optional[str] = None) -> List[str]:
    from singlet.s1 import client as C

    st = C.open_study(gse, base=base)
    return list(st.gsm_ids)


def iter_samples(
    gse: str,
    gsms: Optional[List[str]] = None,
    *,
    bundle: Optional[Path] = None,
    s1: bool = False,
    store: Optional[bool] = None,  # DUAL: old names of s1=
    v2: Optional[bool] = None,
) -> Iterator[Tuple[str, object]]:
    """Yield (gsm, SampleInput or the exception raised while reading it)."""
    from singlet.celllabels import io as A

    for old in (store, v2):
        if old is not None:
            s1 = old
    if s1:
        todo = s1_gsms(gse) if gsms is None else gsms
        for g in todo:
            try:
                yield g, A.read_s1(gse, g)
            except Exception as e:  # noqa: BLE001 - reported per sample
                yield g, e
        return
    assert bundle is not None
    todo = bundle_gsms(bundle) if gsms is None else gsms
    for g in todo:
        try:
            yield g, A.read_legacy(bundle, g)
        except Exception as e:  # noqa: BLE001
            yield g, e


is_v2 = is_store = is_s1  # DUAL: old names, removed in the cleanup
v2_gsms = store_gsms = s1_gsms
