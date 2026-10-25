#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ensemble consensus labels on Singlet's own counts, for training the network on the atlas's own distribution.

The Census corpus covers some tissues thinly and none of it is Singlet's processing. This script runs the ensemble members
(guide with the sample's tissue text, CellTypist with the crosswalk, SingleR-style) on samples of Singlet studies that are not in
the benchmark, keeps the cells on which at least two members agree, and stores their vocabulary counts with the consensus term
(the most specific term the agreeing calls share). ``train.py --singlet-pseudo`` adds them to the training set with a lower
weight. No author label is involved, so these cells teach the network to reproduce the ensemble where the members agree.

  python scripts/celltype_nn/singlet_pseudo.py plan --data $D --exclude-studies bench_studies.json --out $D/pseudo/plan.tsv
  python scripts/celltype_nn/singlet_pseudo.py run  --data $D --plan $D/pseudo/plan.tsv --shard K --out $D/pseudo --scratch $TMP

Plan: human and mouse studies of ``/api/manifest`` (the benchmark studies excluded), stratified by the catalogue's tissue groups
(at most ``--per-group`` studies per group, smaller bundles first, at most ``--max-gb`` each). Run: per study at most
``--samples`` samples with at least 300 called cells; members run on at most ``--cells`` cells per sample; at most ``--keep``
agreed cells per sample are stored. Writes ``<out>/<GSE>.npz`` (CSR vocabulary counts) and ``<out>/<GSE>.parquet``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from annotate_campaign import DATA_BASE, MANIFEST_URL, UA, _download  # noqa: E402


def cmd_plan(a) -> int:
    import pandas as pd

    with urllib.request.urlopen(urllib.request.Request(MANIFEST_URL, headers=UA), timeout=120) as r:
        m = pd.read_csv(io.BytesIO(r.read()), sep="\t", dtype=str)
    org = m["organism"].fillna("").str.lower()
    m = m[org.isin(["human", "mouse", "homo sapiens", "mus musculus"])].copy()
    bench = {s["gse"] for s in json.loads(a.exclude_studies.read_text())}
    m = m[~m["gse_id"].isin(bench)]
    m["bytes"] = pd.to_numeric(m["bytes"], errors="coerce").fillna(0)
    m["n_cells"] = pd.to_numeric(m["n_cells"], errors="coerce").fillna(0)
    m = m[(m["bytes"] > 0) & (m["bytes"] <= a.max_gb * 1e9) & (m["n_cells"] >= 1000)]
    m["group"] = m["tissue_groups"].fillna("").str.split(";").str[0].str.strip().replace("", "none")
    rng = np.random.default_rng(a.seed)
    m["u"] = rng.random(len(m))
    picks = []
    for (g, o), d in m.groupby(["group", "organism"]):
        picks.append(d.sort_values("u").head(a.per_group))
    p = pd.concat(picks).sort_values("bytes", ascending=False).reset_index(drop=True)
    load = [0.0] * a.shards
    shard = []
    for b in p["bytes"]:
        i = int(np.argmin(load))
        shard.append(i)
        load[i] += b
    p["shard"] = shard
    a.out.parent.mkdir(parents=True, exist_ok=True)
    p[["gse_id", "organism", "group", "bytes", "n_cells", "v2_manifest_url", "shard"]].rename(columns={"gse_id": "gse"}).to_csv(
        a.out, sep="\t", index=False
    )
    print(f"{len(p)} studies ({p['organism'].value_counts().to_dict()}), {p['group'].nunique()} tissue groups, "
          f"{p['bytes'].sum() / 1e9:.0f} GB, {a.shards} shards", flush=True)
    return 0


def cmd_run(a) -> int:
    import pandas as pd

    from singlet.bundle import SingletBundle
    from singlet.celllabels import io as IO
    from singlet.celllabels.celltype import lognorm
    from singlet.celllabels.guide import ontology
    from singlet.celltype_nn.ensemble import pool_celltypist, run_celltypist, run_guide, vote
    from singlet.celltype_nn.features import GeneVocab
    from singlet.celltype_nn.ontology import TermSpace
    from ensemble_run import load_singler

    onto = ontology()
    vocab = GeneVocab.from_dict(json.loads((a.data / "vocab.json").read_text()))
    sp = TermSpace(json.loads((a.data / "terms.json").read_text())["terms"])
    xws = json.loads((a.data / "celltypist_crosswalk.json").read_text())
    refs = {}
    plan = pd.read_csv(a.plan, sep="\t")
    mine = plan[plan["shard"] == a.shard]
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.shard)
    min_ic = float(np.log(2.0))
    for _, row in mine.iterrows():
        gse = row["gse"]
        if (a.out / f"{gse}.parquet").exists():
            continue
        t0 = time.time()
        work = a.scratch / gse
        work.mkdir(parents=True, exist_ok=True)
        try:
            v2 = isinstance(row.get("v2_manifest_url"), str) and len(row["v2_manifest_url"]) > 0
            if v2:
                from singlet.v2 import client as C

                gsms = list(C.open_study(gse).samples)
                load = lambda g: IO.read_v2(gse, g)  # noqa: E731
            else:
                bundle = work / f"{gse}.singlet"
                _download(f"{DATA_BASE}/data/{gse}/{gse}.singlet", bundle)
                with SingletBundle(bundle) as b:
                    gsms = list(b.gsm_ids)
                load = lambda g: IO.read_legacy(bundle, g)  # noqa: E731
            rng.shuffle(gsms)
            parts, obs, done = [], [], 0
            for gsm in gsms:
                if done >= a.samples:
                    break
                try:
                    s = load(gsm)
                except Exception as e:  # noqa: BLE001
                    print(f"{gse}/{gsm}: {type(e).__name__}: {e}", flush=True)
                    continue
                if s.species not in ("human", "mouse") or s.X.shape[0] < 300:
                    continue
                take = np.sort(rng.permutation(s.X.shape[0])[: a.cells])
                X = s.X[take]
                g = run_guide(X, s.gene_names, s.species, s.tissue, s.disease)
                ct = pool_celltypist(run_celltypist(X, s.gene_names, s.species, s.tissue, a.celltypist_models), xws, onto)
                if s.species not in refs:
                    refs[s.species] = load_singler(a.data, s.species)
                V = vocab.project(X, s.gene_ids, s.species)
                sr = refs[s.species].score(lognorm(V))
                calls = [g.cl, ct.cl, sr.cl]
                n_called = np.sum([[isinstance(c, str) for c in m] for m in calls], 0)
                cons = vote(calls, onto)
                cons = np.array([sp.map_to_space(c) if isinstance(c, str) and n >= 2 else None for c, n in zip(cons, n_called)], object)
                ok = np.array([isinstance(c, str) and sp.ic[sp.index[c]] >= min_ic for c in cons])
                keep = np.flatnonzero(ok)
                keep = np.sort(rng.permutation(keep)[: a.keep])
                if len(keep) == 0:
                    continue
                parts.append(V[keep])
                obs.append(pd.DataFrame(dict(
                    gse=gse, gsm=gsm, col=s.col[take][keep], species=s.species, tissue=s.tissue, consensus=cons[keep],
                    guide=g.cl[keep], celltypist=ct.cl[keep], singler=sr.cl[keep],
                )))
                done += 1
                print(f"{gse}/{gsm}: {len(take)} cells, {ok.mean():.2f} agreed, kept {len(keep)} ({time.time() - t0:.0f}s)", flush=True)
            if parts:
                sps.save_npz(a.out / f"{gse}.npz", sps.vstack(parts).tocsr())
                pd.concat(obs, ignore_index=True).to_parquet(a.out / f"{gse}.parquet", index=False)
            else:
                pd.DataFrame(columns=["gse"]).to_parquet(a.out / f"{gse}.parquet", index=False)
        except Exception as e:  # noqa: BLE001
            print(f"{gse}: {type(e).__name__}: {e}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return 0


def load_pseudo(d: Path):
    """(CSR counts, obs DataFrame) of every study written by ``run``."""
    import pandas as pd

    Xs, obs = [], []
    for p in sorted(d.glob("*.parquet")):
        o = pd.read_parquet(p)
        z = p.with_suffix(".npz")
        if len(o) == 0 or not z.exists():
            continue
        Xs.append(sps.load_npz(z).tocsr())
        obs.append(o)
    if not Xs:
        return None, None
    return sps.vstack(Xs).tocsr().astype(np.float32), pd.concat(obs, ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--exclude-studies", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-group", type=int, default=12)
    p.add_argument("--max-gb", type=float, default=3.0)
    p.add_argument("--shards", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--scratch", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "ctnn-pseudo")
    r.add_argument("--celltypist-models", type=Path, required=True)
    r.add_argument("--samples", type=int, default=2)
    r.add_argument("--cells", type=int, default=6000)
    r.add_argument("--keep", type=int, default=1500)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
