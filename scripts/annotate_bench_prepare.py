#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch what the cell type benchmark needs, from public sources, and build the truth tables.

For each study of ``--studies`` (JSON list of {gse, collection_id, dataset_ids}): download the Singlet bundle from
data.singlet.bio, download each CELLxGENE dataset's h5ad (curation API asset URL), keep only its ``obs`` as Parquet, delete the
h5ad, and run ``annotate_truth_cxg.py``. Writes ``<out>/bundles/<GSE>.singlet``, ``<out>/cxg_obs/<dataset>.obs.parquet``,
``<out>/truth/<GSE>/<GSM>.truth.parquet`` and ``<out>/truth/manifest.json`` (the layout ``annotate_bench.py`` reads).

  python scripts/annotate_bench_prepare.py --studies bench_studies.json --out $WORK/data/annotate/bench
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CXG = "https://api.cellxgene.cziscience.com/curation/v1"
UA = {"User-Agent": "singlet-annotate/0.1 (+https://singlet.bio)"}


def _get(url: str, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
        want = int(r.headers.get("Content-Length") or -1)
        while True:
            b = r.read(16 << 20)
            if not b:
                break
            f.write(b)
    if want >= 0 and tmp.stat().st_size != want:
        raise OSError(f"{url}: truncated ({tmp.stat().st_size} of {want} bytes)")
    tmp.replace(dst)


def _h5ad_url(collection_id: str, dataset_id: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(f"{CXG}/collections/{collection_id}/datasets/{dataset_id}", headers=UA), timeout=60) as r:
        d = json.load(r)
    for a in d.get("assets", []):
        if a.get("filetype") == "H5AD":
            return a["url"]
    raise KeyError(f"{dataset_id}: no H5AD asset")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    studies = json.loads(a.studies.read_text())
    for d in ("bundles", "cxg_obs", "truth", "dl"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    man = []
    for st in studies:
        gse = st["gse"]
        bundle = a.out / "bundles" / f"{gse}.singlet"
        if not bundle.exists():
            _get(f"https://data.singlet.bio/data/{gse}/{gse}.singlet", bundle)
        obs = []
        for ds in st["dataset_ids"]:
            p = a.out / "cxg_obs" / f"{ds}.obs.parquet"
            if not p.exists():
                h5 = a.out / "dl" / f"{ds}.h5ad"
                _get(_h5ad_url(st["collection_id"], ds), h5)
                subprocess.run([sys.executable, str(HERE / "annotate_truth_cxg.py"), "--gse", gse, "--bundle", str(bundle), "--h5ad", str(h5),
                                "--save-obs", str(p), "--out", str(a.out / "truth" / "_scratch")], check=False)
                h5.unlink(missing_ok=True)
            obs.append(p)
        cmd = [sys.executable, str(HERE / "annotate_truth_cxg.py"), "--gse", gse, "--bundle", str(bundle), "--out", str(a.out / "truth" / gse)]
        for p in obs:
            cmd += ["--obs", str(p)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        n = len(list((a.out / "truth" / gse).glob("*.truth.parquet")))
        print(f"{gse}: {n} GSMs with truth (exit {r.returncode})", flush=True)
        if r.returncode:
            print(r.stderr[-2000:], flush=True)
        man.append(dict(gse=gse, kept=n > 0, tissue=st.get("tissue", ""), bundle_path=str(bundle)))
    (a.out / "truth" / "manifest.json").write_text(json.dumps({"studies": man}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
