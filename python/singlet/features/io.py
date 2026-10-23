# SPDX-License-Identifier: MIT
"""Where a feature job's cells come from. Counts and ``col`` are read with :mod:`singlet.annotate.io` (the same reader as the
labels, so labels and features line up on ``col``):

* a v2 study on data.singlet.bio: :func:`singlet.annotate.io.read_v2`;
* a legacy ``.singlet`` bundle on disk: :func:`singlet.annotate.io.read_legacy` (``col`` = position among the columns of
  ``exon_counts.1pz``, the order the v2 migration gives the cells table).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

DATA_BASE = os.environ.get("SINGLET_DATA_BASE", "https://data.singlet.bio")


def is_v2(gse: str, base: str = DATA_BASE) -> bool:
    req = urllib.request.Request(f"{base}/v2/m/{gse}.json", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:
        return False


def legacy_url(gse: str, base: str = DATA_BASE) -> str:
    return f"{base}/data/{gse}/{gse}.singlet"


def fetch_bundle(gse: str, dest_dir: Path, base: str = DATA_BASE) -> Path:
    """Download a legacy bundle (checked against Content-Length) into ``dest_dir``; reuses a complete file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{gse}.singlet"
    url = legacy_url(gse, base)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        size = int(r.headers.get("Content-Length") or 0)
    if path.exists() and size and path.stat().st_size == size:
        return path
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
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


def v2_gsms(gse: str, base: Optional[str] = None) -> List[str]:
    from singlet.v2 import client as C

    st = C.open_study(gse, base=base)
    return list(st.gsm_ids)


def iter_samples(
    gse: str, gsms: Optional[List[str]] = None, *, bundle: Optional[Path] = None, v2: bool = False
) -> Iterator[Tuple[str, object]]:
    """Yield (gsm, SampleInput or the exception raised while reading it)."""
    from singlet.annotate import io as A

    if v2:
        todo = v2_gsms(gse) if gsms is None else gsms
        for g in todo:
            try:
                yield g, A.read_v2(gse, g)
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
