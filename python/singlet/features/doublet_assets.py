# SPDX-License-Identifier: MIT
"""The two large files of ``doublet/score_v2`` (the cell type network and the atlas centroids), fetched by URL and pinned by sha256.

Both are assets of a GitHub release of Singlet-Bio/singlet (public: any worker can download without credentials). A file is
downloaded once into a cache directory, checked against its pinned sha256 after the download and again every time it is used; a
mismatch deletes the download and raises :class:`AssetHashError`. Nothing else is trusted: the URL may be redirected or mirrored, the
bytes may not differ.

Where things come from, first match wins:

* explicit path argument, or the offline overrides ``SINGLET_CELLTYPE_NN_MODEL`` / ``SINGLET_DOUBLET_CENTROIDS`` (still hash-checked);
* the cache: ``$SINGLET_DOUBLET_CACHE``, else ``$XDG_CACHE_HOME/singlet/doublet_v2`` (``%LOCALAPPDATA%\\singlet\\doublet_v2`` on
  Windows, ``~/.cache/singlet/doublet_v2`` otherwise), one directory per sha256;
* a download from ``$SINGLET_DOUBLET_BASE_URL`` (default: the release ``TAG`` below; ``file://`` works for mirrors).

The release tag is ``doublet-v2-model-<first 12 hex of COMBINED_SHA256>``, where ``COMBINED_SHA256`` is the sha256 of the text
``<sha256>  <name>\\n`` for each asset in ``ASSETS`` order (the format of ``sha256sum``)."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ENV_CACHE = "SINGLET_DOUBLET_CACHE"
ENV_BASE_URL = "SINGLET_DOUBLET_BASE_URL"
REPO = "Singlet-Bio/singlet"


@dataclass(frozen=True)
class Asset:
    name: str
    sha256: str
    size: int


ASSETS = {
    "centroids": Asset(
        "atlas_centroids.npz",
        "ad130c141cf2c3ff43fa6f46df494043bb9e73d4cfa53bd4502ccd0e3b066e84",
        20618845,
    ),
    "encoder": Asset(
        "celltype_nn_m1_notissue_model.pt",
        "b63f5b394de83e8d1523a25a5af55d69a5dbf068381cd3b4ce0f8e4f360ce59b",
        56448814,
    ),
}
COMBINED_SHA256 = hashlib.sha256(
    "".join(f"{a.sha256}  {a.name}\n" for a in ASSETS.values()).encode()
).hexdigest()
TAG = f"doublet-v2-model-{COMBINED_SHA256[:12]}"
DEFAULT_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"
HEADERS = {"User-Agent": "singlet-features (+https://singlet.bio)"}


class AssetHashError(RuntimeError):
    """A model file does not have its pinned sha256."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cache_dir() -> Path:
    if os.environ.get(ENV_CACHE):
        return Path(os.environ[ENV_CACHE])
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]) / "singlet" / "doublet_v2"
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "singlet" / "doublet_v2"
    return Path.home() / ".cache" / "singlet" / "doublet_v2"


def check(path, asset: Asset) -> Path:
    """``path`` if its sha256 is the asset's; otherwise :class:`AssetHashError`."""
    got = sha256_file(path)
    if got != asset.sha256:
        raise AssetHashError(f"{path}: sha256 {got} is not the pinned {asset.sha256} ({asset.name})")
    return Path(path)


def fetch(
    key: str,
    *,
    override: Optional[str] = None,
    cache: Optional[Path] = None,
    base_url: Optional[str] = None,
    assets: Optional[dict] = None,
) -> Path:
    """Local path of asset ``key`` (``centroids`` or ``encoder``), hash-checked. ``override``: a file to use instead of the
    cache and the network (still checked)."""
    asset = (assets or ASSETS)[key]
    if override:
        return check(override, asset)
    d = Path(cache) if cache else cache_dir()
    dest = d / asset.sha256[:16] / asset.name
    if dest.exists():
        try:
            return check(dest, asset)
        except AssetHashError:
            dest.unlink()  # a damaged cache entry is fetched again, never used
    base = (base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
    url = f"{base}/{asset.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".part{os.getpid()}")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            for b in iter(lambda: r.read(1 << 22), b""):
                f.write(b)
        check(tmp, asset)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
