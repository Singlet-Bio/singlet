#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Atlas-wide cell type probabilities: one Slurm array task runs the network on a shard of studies (infer_campaign.sbatch).

  plan    the study list and shards, as ``scripts/annotate_campaign.py plan`` (human and mouse studies of /api/manifest)
          python scripts/celltype_nn/infer_campaign.py plan --shards 200 --out plan.tsv
  run     every called cell of every sample of one shard -> <out>/<GSE>/<GSM>.celltype.parquet (+ <GSM>.celltype.json)
          python scripts/celltype_nn/infer_campaign.py run --plan plan.tsv --shard $SLURM_ARRAY_TASK_ID --model model.pt \
              --out $OUT --scratch /tmp/$USER-$SLURM_JOB_ID --ledger $OUT/ledger/shard-$SLURM_ARRAY_TASK_ID.jsonl
  status  summarise the ledgers (same format as the annotation campaign)

Per study: v2 studies are read by range requests through the v2 client, legacy bundles are downloaded to node-local scratch and
deleted after the study. Every table passes ``singlet.celltype_nn.output.assert_public_safe`` before it is written. A sample
whose sidecar exists at the current model version is skipped. Nothing is written to R2 or D1: the cell index ingests ``$OUT``.

``<GSM>.celltype.json`` (internal QC): cells, cells per call, mean call probability, fraction of cells whose call probability is
at least 0.5 below the root, seconds, and the ``barcode_digest`` of :class:`singlet.celllabels.io.SampleInput`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE.parent))

from annotate_campaign import DATA_BASE, _download, _has_v2, plan, status  # noqa: E402


def _samples(gse: str, bundle):
    """Yield SampleInput per sample (v2 when ``bundle`` is None)."""
    from singlet.celllabels import io as IO

    if bundle is None:
        from singlet.v2 import client as C

        for gsm in list(C.open_study(gse).samples):
            yield gsm, (lambda g=gsm: IO.read_v2(gse, g))
    else:
        from singlet.bundle import SingletBundle

        with SingletBundle(bundle) as b:
            gsms = list(b.gsm_ids)
        for gsm in gsms:
            yield gsm, (lambda g=gsm: IO.read_legacy(bundle, g))


def run(a) -> int:
    import numpy as np
    import pandas as pd

    from singlet.celltype_nn import CellTypePredictor, write_sidecar

    pred = CellTypePredictor.load(a.model, version=a.model_version)
    df = pd.read_csv(a.plan, sep="\t")
    mine = df[df["shard"] == a.shard]
    a.out.mkdir(parents=True, exist_ok=True)
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    print(f"shard {a.shard}: {len(mine)} studies, {mine['bytes'].sum() / 1e9:.1f} GB, model {pred.version}", flush=True)
    bad = 0
    for _, row in mine.iterrows():
        gse = row["gse"]
        t0 = time.time()
        rec = dict(gse=gse, shard=a.shard, n_written=0, n_skipped=0, n_errors=0, n_cells=0, s_read=0.0, s_predict=0.0)
        work = a.scratch / gse
        work.mkdir(parents=True, exist_ok=True)
        try:
            v2 = bool(row.get("v2", 0)) or _has_v2(gse)
            rec["source"] = "v2" if v2 else "legacy"
            bundle = None
            if not v2:
                bundle = work / f"{gse}.singlet"
                _download(f"{DATA_BASE}/data/{gse}/{gse}.singlet", bundle)
            odir = a.out / gse
            odir.mkdir(parents=True, exist_ok=True)
            for gsm, load in _samples(gse, bundle):
                dst = odir / f"{gsm}.celltype.parquet"
                meta = odir / f"{gsm}.celltype.json"
                if dst.exists() and meta.exists() and json.loads(meta.read_text()).get("model_version") == pred.version:
                    rec["n_skipped"] += 1
                    continue
                try:
                    t1 = time.time()
                    s = load()
                    rec["s_read"] += time.time() - t1
                    if s.species not in ("human", "mouse") or s.X.shape[0] == 0:
                        continue
                    t1 = time.time()
                    P = pred.predict(s.X, s.gene_ids, s.species, tissue=s.tissue, threads=a.threads)
                    tab = pred.table(P, s.gsm, s.col)
                    rec["s_predict"] += time.time() - t1
                    write_sidecar(tab, dst)
                    called = tab["cell_type_cl"].notna()
                    meta.write_text(json.dumps(dict(
                        gsm=gsm, gse=gse, model_version=pred.version, n_cells=int(len(tab)), species=s.species,
                        called=float(called.mean()), mean_call_prob=float(np.nanmean(tab["cell_type_prob"])) if called.any() else None,
                        calls=tab["cell_type_cl"].value_counts().head(50).to_dict(), barcode_digest=s.barcode_digest,
                        seconds=round(time.time() - t1, 2),
                    ), indent=1))
                    rec["n_written"] += 1
                    rec["n_cells"] += int(len(tab))
                except Exception as e:  # noqa: BLE001
                    rec["n_errors"] += 1
                    (odir / f"{gsm}.celltype.error.txt").write_text(f"{type(e).__name__}: {e}")
            rec["status"] = "ok" if rec["n_errors"] == 0 else f"{rec['n_errors']} sample errors"
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"error {type(e).__name__}: {e}"
        finally:
            shutil.rmtree(work, ignore_errors=True)
        rec["seconds"] = round(time.time() - t0, 1)
        bad += rec["status"] != "ok"
        with open(a.ledger, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--studies", type=Path)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--model-version", required=True, help="e.g. celltype-nn=0.1.0;census=2025-11-08;cl=2026-06-08")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--scratch", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "celltype_nn")
    r.add_argument("--ledger", type=Path, required=True)
    r.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    s = sub.add_parser("status")
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return {"plan": plan, "run": run, "status": status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
