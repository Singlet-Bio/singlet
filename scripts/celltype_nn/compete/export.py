#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inputs shared by every method in the cell type comparison (``scripts/celltype_nn/compete/``).

  python scripts/celltype_nn/compete/export.py reference --data $D --out $D/compete [--per-term 200]
  python scripts/celltype_nn/compete/export.py queries   --data $D --truth <truth dir> --bundles <dir> --out $D/compete

``reference``: per species, the training split of the Census corpus subsampled to at most ``--per-term`` cells per author term
(the folded term the network is trained on), on the network's 6,000-gene vocabulary, as Matrix Market + CSV (for R) and h5ad.
Every reference-based tool sees the same cells and labels, and none of the benchmark collections.

``queries``: per benchmark sample, Singlet's counts (exon + intron, as served) of every called cell on the same vocabulary, the
barcodes, and the truth label where the author kept the cell. Files ``query/<GSE>/<GSM>.{mtx,cells.csv}`` and ``.h5ad``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))


def _write(X: sps.csr_matrix, obs, genes, symbols, stem: Path) -> None:
    import anndata as ad
    import pandas as pd

    stem.parent.mkdir(parents=True, exist_ok=True)
    sio.mmwrite(str(stem) + ".mtx", sps.csr_matrix(X, dtype=np.float32).T.tocoo())  # genes x cells, as R expects
    obs.to_csv(str(stem) + ".cells.csv", index=False)
    pd.DataFrame({"gene_id": genes, "symbol": symbols}).to_csv(stem.parent / "genes.csv", index=False)
    a = ad.AnnData(X=sps.csr_matrix(X, dtype=np.float32), obs=obs.reset_index(drop=True).astype({c: str for c in obs.columns if obs[c].dtype == object}),
                   var=pd.DataFrame({"symbol": symbols}, index=pd.Index(genes)))
    a.obs.index = a.obs.index.astype(str)
    a.write_h5ad(str(stem) + ".h5ad")


def cmd_reference(a) -> int:
    import pandas as pd

    vocab = json.loads((a.data / "vocab.json").read_text())
    X = sps.load_npz(a.data / "corpus_counts.npz").tocsr()
    obs = pd.read_parquet(a.data / "corpus_obs.parquet", columns=["cell_id", "organism", "split", "dataset_id", "tissue_general"])
    cells = pd.read_parquet(a.data / "space_cells.parquet", columns=["author_term"])
    obs["cl"] = cells["author_term"].to_numpy()
    rng = np.random.default_rng(0)
    obs["u"] = rng.random(len(obs))
    for sp in ("human", "mouse"):
        o = obs[(obs["organism"] == sp) & (obs["split"] == "train") & obs["cl"].notna()]
        o = o.sort_values("u")
        o = o[o.groupby("cl").cumcount() < a.per_term]
        idx = np.sort(o.index.to_numpy())
        oo = obs.loc[idx, ["cell_id", "cl", "dataset_id", "tissue_general"]].reset_index(drop=True)
        _write(X[idx], oo, vocab["genes"], vocab["symbols"], a.out / "reference" / f"ref_{sp}")
        print(f"reference {sp}: {len(idx)} cells, {oo['cl'].nunique()} terms, {oo['dataset_id'].nunique()} datasets", flush=True)
    return 0


def cmd_queries(a) -> int:
    import pandas as pd

    from singlet.bundle import SingletBundle
    from singlet.celllabels import io as IO
    from singlet.celltype_nn.features import GeneVocab

    vocab = GeneVocab.from_dict(json.loads((a.data / "vocab.json").read_text()))
    man = json.loads((a.truth / "manifest.json").read_text())
    rows = []
    for st in [x for x in man["studies"] if x.get("kept", True)]:
        gse = st["gse"]
        bundle = a.bundles / f"{gse}.singlet"
        for tf in sorted((a.truth / gse).glob("*.truth.parquet")):
            gsm = tf.name.split(".")[0]
            s = IO.read_legacy(bundle, gsm)
            if s.species is None or len(s.col) > a.max_cells:
                continue
            with SingletBundle(bundle) as b:
                gene_ids = [str(x) for x in b.raw_counts(gsm).var_names]
            V = vocab.project(s.X, gene_ids, s.species)
            t = pd.read_parquet(tf)
            truth = dict(zip(t["barcode"].astype(str), t["cl_id"].astype(str)))
            obs = pd.DataFrame({"gse": gse, "gsm": gsm, "barcode": s.barcodes, "col": s.col, "species": s.species,
                                "tissue_text": s.tissue, "truth": [truth.get(b, "") for b in s.barcodes]})
            _write(V, obs, vocab.genes, vocab.symbols, a.out / "query" / gse / gsm)
            rows.append(dict(gse=gse, gsm=gsm, species=s.species, n_cells=len(obs), n_truth=int((obs["truth"] != "").sum()),
                             cxg_tissue=st.get("tissue", ""), tissue_text=s.tissue))
            print(f"{gse}/{gsm}: {len(obs)} cells, {rows[-1]['n_truth']} with truth", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "query" / "samples.csv", index=False)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--per-term", type=int, default=200)
    q = sub.add_parser("queries")
    q.add_argument("--data", type=Path, required=True)
    q.add_argument("--truth", type=Path, required=True)
    q.add_argument("--bundles", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--max-cells", type=int, default=30000)
    a = ap.parse_args(argv)
    return {"reference": cmd_reference, "queries": cmd_queries}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
