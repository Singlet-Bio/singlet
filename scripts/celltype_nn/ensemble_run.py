#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the annotator ensemble on the training corpus and turn its calls into training targets.

  python scripts/celltype_nn/ensemble_run.py space   --data $D                 # output terms, ensemble subset, SingleR reference
  python scripts/celltype_nn/ensemble_run.py run     --data $D --chunk K       # members on the subset cells of one chunk
  python scripts/celltype_nn/ensemble_run.py targets --data $D                 # crosswalk, consensus, targets.parquet

The ensemble runs on a subset: a random ``--random-frac`` of the corpus (for the CellTypist crosswalk and the agreement
statistics) plus every cell whose author label has finer terms in the output space (the only cells where the ensemble can add a
refinement). The SingleR-style member is cross-fitted: a corpus cell is never scored against profiles of its own dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def _load_corpus(d: Path):
    import pandas as pd

    return sps.load_npz(d / "corpus_counts.npz").tocsr(), pd.read_parquet(d / "corpus_obs.parquet")


def cmd_space(a) -> int:
    import pandas as pd

    from singlet.celllabels.celltype import lognorm
    from singlet.celltype_nn.ensemble import SingleRRef
    from singlet.celltype_nn.ontology import TermSpace, build_space

    X, obs = _load_corpus(a.data)
    tr = obs["split"].to_numpy() == "train"
    counts = obs.loc[tr, "cell_type_ontology_term_id"].value_counts().to_dict()
    terms, fold = build_space(counts, a.min_cells)
    sp = TermSpace(terms)
    author = obs["cell_type_ontology_term_id"].map(lambda t: fold.get(t) or sp.map_to_space(t)).to_numpy(object)
    has_finer = {t for t in terms if any(t != v and t in sp.ancestors(v) for v in terms)}
    rng = np.random.default_rng(0)
    pick = ((rng.random(len(obs)) < a.random_frac) | np.isin(author, list(has_finer))) & (obs["split"].to_numpy() != "excluded")
    obs["author_term"] = author
    obs["ensemble"] = pick
    obs[["cell_id", "chunk", "author_term", "ensemble"]].to_parquet(a.data / "space_cells.parquet", index=False)
    (a.data / "terms.json").write_text(
        json.dumps(dict(terms=terms, fold=fold, cl_version=sp.o.version, min_cells=a.min_cells), indent=0)
    )
    print(f"{len(terms)} output terms ({len(has_finer)} with finer terms); ensemble subset {pick.sum()} of {len(obs)} cells; "
          f"unmapped author labels {int(pd.isna(author).sum())}", flush=True)
    # SingleR-style reference: pseudobulks per (dataset, author term) of training cells, per species
    vocab = json.loads((a.data / "vocab.json").read_text())
    for species in ("human", "mouse"):
        m = tr & (obs["organism"].to_numpy() == species) & pd.notna(author)
        if m.sum() == 0:
            continue
        idx = np.flatnonzero(m)
        ref = SingleRRef.build(lognorm(X[idx]), author[idx], obs["dataset_id"].to_numpy()[idx], vocab["genes"])
        np.savez_compressed(
            a.data / f"singler_ref_{species}.npz", P=ref.P, labels=ref.labels.astype(str), groups=ref.groups.astype(str)
        )
        print(f"SingleR reference {species}: {len(ref.labels)} profiles, {len(ref.ulabels)} labels, "
              f"{len(ref.union)} marker genes (k={ref.k})", flush=True)
    return 0


def load_singler(d: Path, species: str):
    from singlet.celltype_nn.ensemble import SingleRRef

    p = d / f"singler_ref_{species}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    vocab = json.loads((d / "vocab.json").read_text())
    return SingleRRef(z["P"], z["labels"].astype(object), z["groups"].astype(object), vocab["genes"])


def cmd_run(a) -> int:
    import anndata as ad
    import pandas as pd

    from singlet.celllabels.celltype import lognorm
    from singlet.celltype_nn.ensemble import run_celltypist, run_guide
    from singlet.celltype_nn.features import GeneVocab

    out = a.data / "ensemble" / f"{a.chunk:04d}.parquet"
    if out.exists():
        print(f"{out} exists")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cells = pd.read_parquet(a.data / "space_cells.parquet")
    obs = pd.read_parquet(a.data / "corpus_obs.parquet")
    A = ad.read_h5ad(a.data / "chunks" / f"{a.chunk:04d}.h5ad")
    inchunk = (obs["chunk"] == a.chunk).to_numpy()
    sub = cells[inchunk].reset_index(drop=True)
    o = obs[inchunk].reset_index(drop=True)
    assert (o["soma_joinid"].astype(str).to_numpy() == A.obs_names.to_numpy()).all()
    take = np.flatnonzero(sub["ensemble"].to_numpy())
    meta = o.iloc[take].reset_index(drop=True)
    species = meta["organism"].iloc[0]
    X = sps.csr_matrix(A.X[take], dtype=np.float32)
    gene_ids = A.var["feature_id"].astype(str).to_numpy()
    symbols = A.var["feature_name"].astype(str).to_numpy()
    n = len(take)
    res = pd.DataFrame({"cell_id": meta["cell_id"].to_numpy()})
    print(f"chunk {a.chunk} ({species}): {n} of {A.n_obs} cells in the ensemble subset", flush=True)

    # guide, per (dataset, donor); groups under 50 cells are pooled per dataset
    g_cl = np.empty(n, object)
    g_sc = np.zeros(n, np.float32)
    key = (meta["dataset_id"] + "|" + meta["donor_id"].astype(str)).to_numpy()
    size = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy()
    key = np.where(size >= 50, key, meta["dataset_id"].to_numpy())
    for k in np.unique(key):
        i = np.flatnonzero(key == k)
        if len(i) < 20:
            continue
        try:
            r = run_guide(X[i], symbols, species, str(meta["tissue_general"].iloc[i[0]]))
            g_cl[i], g_sc[i] = r.cl, r.score
        except Exception as e:  # noqa: BLE001
            print(f"  guide {k}: {type(e).__name__}: {e}", flush=True)
    res["guide_cl"], res["guide_score"] = g_cl, g_sc
    print(f"  guide {time.time() - t0:.0f}s", flush=True)

    # CellTypist, per tissue
    ct_lab, ct_sc = {}, {}
    for tissue in meta["tissue_general"].unique():
        i = np.flatnonzero(meta["tissue_general"].to_numpy() == tissue)
        try:
            calls = run_celltypist(X[i], symbols, species, str(tissue), a.celltypist_models)
        except Exception as e:  # noqa: BLE001
            print(f"  celltypist {tissue}: {type(e).__name__}: {e}", flush=True)
            continue
        for m, c in calls.items():
            ct_lab.setdefault(m, np.full(n, None, object))[i] = c.raw
            ct_sc.setdefault(m, np.zeros(n, np.float32))[i] = c.score
    for m in sorted(ct_lab):
        res[f"ct:{m}"] = ct_lab[m]
        res[f"ct_score:{m}"] = ct_sc[m]
    print(f"  celltypist ({len(ct_lab)} models) {time.time() - t0:.0f}s", flush=True)

    # SingleR-style, cross-fitted by dataset
    ref = load_singler(a.data, species)
    if ref is not None:
        vocab = GeneVocab.from_dict(json.loads((a.data / "vocab.json").read_text()))
        L = lognorm(vocab.project(X, gene_ids, species))
        c = ref.score(L, exclude_group=meta["dataset_id"].to_numpy())
        res["singler_cl"], res["singler_score"] = c.cl, c.score
    print(f"  singler {time.time() - t0:.0f}s", flush=True)
    res.to_parquet(out, index=False)
    return 0


def cmd_targets(a) -> int:
    import pandas as pd

    from singlet.celltype_nn.ensemble import MemberCalls, celltypist_crosswalk, pool_celltypist, refinement
    from singlet.celltype_nn.ontology import TermSpace

    cells = pd.read_parquet(a.data / "space_cells.parquet")
    obs = pd.read_parquet(a.data / "corpus_obs.parquet")
    terms = json.loads((a.data / "terms.json").read_text())
    sp = TermSpace(terms["terms"])
    ens = pd.concat([pd.read_parquet(p) for p in sorted((a.data / "ensemble").glob("*.parquet"))], ignore_index=True)
    ens = ens.merge(cells[["cell_id", "author_term"]], on="cell_id", how="left")
    ens = ens.merge(obs[["cell_id", "split", "organism", "tissue_general"]], on="cell_id", how="left")
    tr = (ens["split"] == "train").to_numpy() & ens["author_term"].notna().to_numpy()
    # crosswalk each CellTypist model from training cells, then pool models into one member: the most specific mapped call
    xws = {}
    ct_cols = [c for c in ens.columns if c.startswith("ct:")]
    for c in ct_cols:
        lab = ens[c].to_numpy(object)
        m = tr & pd.notna(lab)
        xws[c[3:]] = celltypist_crosswalk(lab[m], ens["author_term"].to_numpy(object)[m])
    (a.data / "celltypist_crosswalk.json").write_text(json.dumps(xws, indent=1, sort_keys=True))
    n = len(ens)
    raw = {c[3:]: MemberCalls(None, ens[f"ct_score:{c[3:]}"].to_numpy(np.float32), ens[c].to_numpy(object)) for c in ct_cols}
    pooled = pool_celltypist(raw, xws, sp.o, n=len(ens))
    ct, ct_s = pooled.cl, pooled.score
    ens["celltypist_cl"], ens["celltypist_score"] = ct, ct_s
    # refinements of the author term
    members = ["guide_cl", "celltypist_cl", "singler_cl"]
    ref = []
    for row in ens[["author_term"] + members].itertuples(index=False):
        au = row[0]
        r = refinement(au, list(row[1:])) if isinstance(au, str) else None
        ref.append(sp.map_to_space(r) if r else None)
    ens["refine_term"] = ref
    ens["refine_term"] = [r if (r and r != a_ and a_ in sp.ancestors(r)) else None for r, a_ in zip(ens["refine_term"], ens["author_term"])]
    # agreement of each member with the author labels (training split, random part: consistency rate)
    from singlet.celltype_nn.ontology import is_consistent

    stats = {}
    for m in members:
        v = ens.loc[tr, [m, "author_term"]]
        cat = [is_consistent(sp.o, p, t) if isinstance(p, str) else "none" for p, t in zip(v[m], v["author_term"])]
        stats[m] = pd.Series(cat).value_counts(normalize=True).round(4).to_dict()
    stats["refined_cells"] = int(pd.notna(ens["refine_term"]).sum())
    stats["ensemble_cells"] = n
    (a.data / "ensemble_stats.json").write_text(json.dumps(stats, indent=1))
    ens[["cell_id", "author_term", "refine_term", "guide_cl", "guide_score", "celltypist_cl", "celltypist_score", "singler_cl",
         "singler_score"]].to_parquet(a.data / "targets_ensemble.parquet", index=False)
    print(json.dumps(stats, indent=1), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("space")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--min-cells", type=int, default=50)
    s.add_argument("--random-frac", type=float, default=0.15)
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--chunk", type=int, required=True)
    r.add_argument("--celltypist-models", type=Path, required=True)
    t = sub.add_parser("targets")
    t.add_argument("--data", type=Path, required=True)
    a = ap.parse_args(argv)
    return {"space": cmd_space, "run": cmd_run, "targets": cmd_targets}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
