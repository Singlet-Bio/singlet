"""Stage `fetch`: download each dataset's files into data/raw/<id>/ and check their SHA-256."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from .util import Config, Unit, log


def _urls(d: dict) -> list[str]:
    src = d["source"]
    if not src.get("url"):
        raise SystemExit(
            f"{d['id']}: source.url is null. Find the file named in source.file on {src['repository']} and put its direct "
            "URL in datasets.yaml first (see PLAN.md section 2)."
        )
    return [src["url"]] + list(src.get("extra", []))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run(cfg: Config, units: list[Unit], force: bool = False) -> None:
    seen = set()
    for u in units:
        d = u.dataset
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        out = cfg.data_dir / "raw" / d["id"]
        out.mkdir(parents=True, exist_ok=True)
        want = d["source"].get("sha256")
        for i, url in enumerate(_urls(d)):
            dst = out / url.rstrip("/").split("/")[-1]
            if dst.exists() and not force:
                log(f"{d['id']}: have {dst.name}")
            else:
                log(f"{d['id']}: downloading {url}")
                tmp = dst.with_suffix(dst.suffix + ".part")
                with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
                    shutil.copyfileobj(r, f, 1 << 22)
                tmp.replace(dst)
            digest = sha256(dst)
            size = dst.stat().st_size
            if i == 0 and want and digest != want:
                raise SystemExit(f"{d['id']}: sha256 mismatch for {dst.name}: {digest} != {want}")
            print(f"{d['id']}\t{dst.name}\tsha256={digest}\tbytes={size}")
