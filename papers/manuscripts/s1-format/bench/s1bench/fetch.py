"""Stage `fetch`: download each dataset's files into data/raw/<id>/ and check their SHA-256."""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from .util import Config, Unit, log


def local_name(src: dict, url: str, i: int) -> str:
    """File name under data/raw/<id>/: source.filename (main file) or source.extra_names[i-1] when the URL does not end in
    one (figshare ndownloader ids, `?download=1`), else the last path segment."""
    if i == 0 and src.get("filename"):
        return src["filename"]
    names = src.get("extra_names") or []
    if i > 0 and i - 1 < len(names):
        return names[i - 1]
    return url.split("?")[0].rstrip("/").split("/")[-1]


def _urls(d: dict) -> list[str]:
    src = d["source"]
    if not src.get("url"):
        raise SystemExit(
            f"{d['id']}: source.url is null. Find the file named in source.file on {src['repository']} and put its direct "
            "URL in datasets.yaml first (see PLAN.md section 2)."
        )
    return [src["url"]] + list(src.get("extra", []))


def _download(url: str, tmp: Path, tries: int = 5) -> None:
    """Download into tmp, resuming a partial file with a Range request (large files, preemptible jobs)."""
    for attempt in range(tries):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "s1bench/1 (+https://github.com/Singlet-Bio/singlet)"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with tmp.open(mode) as f:
                    shutil.copyfileobj(r, f, 1 << 22)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:  # already complete
                return
            if attempt == tries - 1:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == tries - 1:
                raise
        log(f"retrying {url} from byte {tmp.stat().st_size if tmp.exists() else 0}")


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
            dst = out / local_name(d["source"], url, i)
            if dst.exists() and not force:
                log(f"{d['id']}: have {dst.name}")
            else:
                log(f"{d['id']}: downloading {url}")
                tmp = dst.with_suffix(dst.suffix + ".part")
                _download(url, tmp)
                tmp.replace(dst)
            digest = sha256(dst)
            size = dst.stat().st_size
            rec = (d["source"].get("checksums") or {}).get(dst.name) or ({"sha256": want} if i == 0 and want else {})
            if rec.get("sha256") and digest != rec["sha256"]:
                raise SystemExit(f"{d['id']}: sha256 mismatch for {dst.name}: {digest} != {rec['sha256']}")
            if rec.get("bytes") and size != int(rec["bytes"]):
                raise SystemExit(f"{d['id']}: {dst.name} is {size} bytes, recorded {rec['bytes']}")
            print(f"{d['id']}\t{dst.name}\tsha256={digest}\tbytes={size}")
