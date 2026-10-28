"""End-to-end smoke test on synthetic counts: no downloads, a few minutes on a laptop.

    cd bench && python tests/smoke.py [--keep]

Runs canon, write (the formats whose packages are installed), decode (1 repeat), ladder, floor, second, minibatch
(local tier, the loaders whose packages are installed) and report into a temporary directory, then checks that every
stage produced rows and that every round trip was exact. Use it after installing the environment on a new machine.
"""

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

BENCH = Path(__file__).resolve().parent.parent

DATASETS = {
    "datasets": [
        {"id": "synth_small", "name": "synthetic", "protocol": "synthetic", "protocol_class": "droplet_3p", "organism": "none",
         "cells": 3000, "features": 2000, "tier": "S", "loader_panel": True,
         "source": {"repository": "generated", "format": "synthetic", "genes": 2000, "cells": 3000, "depth": 2500, "seed": 1},
         "matrices": [{"name": "rna", "value": "umi"}]},
        {"id": "synth_plate", "name": "synthetic deep", "protocol": "synthetic", "protocol_class": "plate_reads", "organism": "none",
         "cells": 800, "features": 3000, "tier": "S",
         "source": {"repository": "generated", "format": "synthetic", "genes": 3000, "cells": 800, "depth": 40000, "theta": 2.0, "seed": 2},
         "matrices": [{"name": "rna", "value": "reads"}]},
    ]
}


def have(*mods):
    try:
        for m in mods:
            importlib.import_module(m)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="s1bench_smoke_"))
    (tmp / "datasets.yaml").write_text(yaml.safe_dump(DATASETS))
    cfg = yaml.safe_load((BENCH / "config.yaml").read_text())
    fmts = ["csc_i32", "csc_zstd19", "mtx_gz", "tenx_h5", "h5ad_none", "h5ad_gzip", "h5ad_lzf", "s1", "s1_pack"]
    fmts += ["zarr"] if have("zarr") else []
    fmts += ["parquet_coo"] if have("pyarrow") else []
    fmts += ["loom"] if have("loompy") else []
    fmts += ["soma"] if have("tiledbsoma") else []
    loaders = [l for l, ok in [("anndata_inmem", True), ("anndata_backed", True), ("anndata_zarr_lazy", have("zarr")),
                               ("s1_stream", True), ("s1_stream_canonical", True)] if ok]
    cfg.update({"data_dir": str(tmp / "data"), "results_dir": str(tmp / "results"), "generated_dir": str(tmp / "generated"),
                "datasets": str(tmp / "datasets.yaml"), "formats": fmts, "repeats": 1, "warmup": 0})
    cfg["threads"]["headline"] = 2
    cfg["floor"]["sampling_cells"] = 100
    cfg["minibatch"].update({"n_batches": 5, "warmup_batches": 1, "seeds": [0], "loaders": loaders, "tiers": [{"name": "nvme"}]})
    cfg["eco"]["datasets"] = []
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg))
    base = [sys.executable, "-m", "s1bench", "--config", str(tmp / "config.yaml")]
    stages = ["canon", "write", "decode", "ladder", "floor", "second"] + (["minibatch"] if have("torch") else [])
    for s in stages:
        print(f"== {s}", flush=True)
        subprocess.run(base + [s], cwd=BENCH, check=True)
    r = subprocess.run(base + ["report"], cwd=BENCH)
    sys.path.insert(0, str(BENCH))
    from s1bench.util import read_tsv

    bad = []
    for p in sorted((tmp / "results").rglob("*.tsv")):
        rows = read_tsv(p)
        if not rows:
            bad.append(f"{p.relative_to(tmp)}: no rows")
        for row in rows:
            if row.get("status") not in (None, "", "ok", "not_applicable"):
                bad.append(f"{p.relative_to(tmp)}: {row.get('format', row.get('loader', ''))} {row.get('status')} {row.get('note', '')}")
    print(f"results under {tmp}")
    print("\n".join(bad) if bad else "all stages produced rows; all round trips exact")
    print((tmp / "generated" / "numbers.tex").read_text()[:3000])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
