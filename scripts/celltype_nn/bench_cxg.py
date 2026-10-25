#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Score the network on the authors' own matrices of the benchmark cells (CELLxGENE h5ad), to set against its score on Singlet's
counts of the same cells (``bench.py``): the gap is what reprocessing costs.

For each benchmark study the CELLxGENE datasets are downloaded (curation API), raw counts are taken from ``raw.X`` (else ``X``),
and a CELLxGENE cell is scored when its 16-nt barcode core is a truth cell of a sample assigned to that dataset and the curated
label matches the truth label (the same cells ``bench.py`` scores, up to barcode collisions between samples of one dataset).
The h5ad is deleted after use.

  python scripts/celltype_nn/bench_cxg.py --model model.pt --truth <truth dir> --studies bench_studies.json --scratch $TMP --out cxg.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--studies", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)

    import anndata as ad
    import pandas as pd
    import scipy.sparse as sps
    from annotate_bench_prepare import _get, _h5ad_url
    from annotate_truth_cxg import split_barcode
    from bench import calibration, cl_consistency, subtree_correct
    from singlet.celllabels import evaluate as EV
    from singlet.celllabels.guide import Guide, ontology
    from singlet.celltype_nn import CellTypePredictor

    onto = ontology()
    pred = CellTypePredictor.load(a.model)
    a.scratch.mkdir(parents=True, exist_ok=True)
    results = []
    for st in json.loads(a.studies.read_text()):
        gse = st["gse"]
        tfs = sorted((a.truth / gse).glob("*.truth.parquet"))
        if not tfs:
            continue
        T = pd.concat([pd.read_parquet(f) for f in tfs], ignore_index=True)
        T["core"] = [split_barcode(str(b), 16)[0] for b in T["barcode"]]
        for ds in st["dataset_ids"]:
            Td = T[T["cxg_dataset_id"] == ds] if "cxg_dataset_id" in T else T
            if Td.empty:
                continue
            h5 = a.scratch / f"{ds}.h5ad"
            t0 = time.time()
            try:
                if not h5.exists():
                    _get(_h5ad_url(st["collection_id"], ds), h5)
                A = ad.read_h5ad(h5)
            except Exception as e:  # noqa: BLE001
                print(f"{gse}/{ds}: {e}", flush=True)
                continue
            species = "mouse" if str(A.obs.get("organism_ontology_term_id", pd.Series(["NCBITaxon:9606"])).iloc[0]) == "NCBITaxon:10090" else "human"
            core = np.array([split_barcode(str(b), 16)[0] for b in A.obs_names], dtype=object)
            lab = A.obs["cell_type_ontology_term_id"].astype(str).to_numpy()
            want = dict(zip(Td["core"], Td["cl_id"].astype(str)))
            m = np.array([c is not None and want.get(c) == l for c, l in zip(core, lab)])
            if m.sum() < 50:
                print(f"{gse}/{ds}: {m.sum()} matched cells, skipped", flush=True)
                h5.unlink(missing_ok=True)
                continue
            Xr = A.raw.X if A.raw is not None else A.X
            var = A.raw.var if A.raw is not None else A.var
            X = sps.csr_matrix(Xr[np.flatnonzero(m)])
            gene_ids = [str(x) for x in var.index]
            # the dataset's curated tissue: the context the network was trained with
            tissue = str(A.obs["tissue_general"].iloc[0]) if "tissue_general" in A.obs else ""
            P = pred.predict(X, gene_ids, species, tissue=tissue, threads=a.threads)
            s = pred.summarise(P)
            truth = lab[m]
            g = Guide(list(var.get("feature_name", pd.Series(var.index)).astype(str)), species=species)
            tm = EV.TruthMapper(g)
            rec = dict(gse=gse, dataset=ds, species=species, n_truth=int(m.sum()), n_cells=int(m.sum()), integer_counts=bool(np.allclose(X.data, np.round(X.data))),
                       methods={})
            for name, key, pk in (("nn", "call", "call_prob"), ("nn_argmax", "argmax", "argmax_prob")):
                node = np.array([tm(c) if isinstance(c, str) else 0 for c in s[key]], np.int32)
                rep = EV.report(g, truth, node, s[pk])
                rep["cl"] = cl_consistency(onto, s[key], truth)
                rep["calibration_subtree"] = calibration(s[pk].astype(np.float64), subtree_correct(onto, s[key], truth))
                rec["methods"][name] = rep
                bd = " ".join(f"d{d}:{v['accuracy']:.2f}" for d, v in rep["by_depth"].items())
                print(f"{gse}/{ds} {name}: {bd} wrong {rep['fractions']['wrong']:.2f} ({m.sum()} cells, {time.time() - t0:.0f}s)", flush=True)
            results.append(rec)
            h5.unlink(missing_ok=True)
            a.out.write_text(json.dumps(dict(results=results, model=str(a.model)), indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
