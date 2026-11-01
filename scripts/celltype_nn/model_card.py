#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write the model card of a trained cell type network: ``model_card.json`` and ``model_card.md`` next to ``model.pt``.

Training data (Census release, cells, datasets, collections, organisms, tissues, the excluded benchmark collections), the output
space (terms, CL release), the vocabulary, training settings and validation metrics, benchmark results on Singlet's counts
(overall, per study / tissue, against each ensemble member), calibration, coverage and inference cost.

  python scripts/celltype_nn/model_card.py --data $D --model $D/models/m1 --bench $D/bench/m1.summary.json [--cxg $D/bench/m1.cxg.summary.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="model directory (model.pt, train.json)")
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--cxg", type=Path)
    ap.add_argument("--version", default=None)
    a = ap.parse_args(argv)

    import pandas as pd

    from singlet.celltype_nn import __version__

    plan = json.loads((a.data / "plan.json").read_text())
    train = json.loads((a.model / "train.json").read_text())
    terms = json.loads((a.data / "terms.json").read_text())
    vocab = json.loads((a.data / "vocab.json").read_text())
    obs = pd.read_parquet(a.data / "corpus_obs.parquet", columns=["organism", "split", "tissue_general", "dataset_id", "collection_id"])
    bench = json.loads(a.bench.read_text())
    cxg = json.loads(a.cxg.read_text()) if a.cxg and a.cxg.exists() else None
    import re

    m = re.search(r"\d{4}-\d{2}-\d{2}", str(terms.get("cl_version")))
    cl = m.group(0) if m else str(terms.get("cl_version"))
    calib = json.loads((a.model / "calibrate.json").read_text()) if (a.model / "calibrate.json").exists() else None
    temperature = calib["best"]["T"] if calib else train["final"]["temperature"]
    version = a.version or f"celltype-nn={__version__};census={plan['census_version']};cl={cl}"
    used = obs[obs["split"].isin(["train", "val"])]
    card = dict(
        model_version=version,
        task="per-cell probability over Cell Ontology types (softmax; rows sum to 1), calibrated by one temperature",
        training_data=dict(
            source=f"CZ CELLxGENE Census {plan['census_version']} (LTS), primary data, droplet assays",
            licence="CC BY 4.0 (Census); this model and its outputs: CC0",
            cells=int(len(used)),
            train_cells=int((used["split"] == "train").sum()),
            val_cells=int((used["split"] == "val").sum()),
            excluded_copies=int((obs["split"] == "excluded").sum()),
            datasets=int(used["dataset_id"].nunique()),
            collections=int(used["collection_id"].nunique()),
            organisms=used["organism"].value_counts().to_dict(),
            tissues=used["tissue_general"].value_counts().to_dict(),
            excluded_collections=plan.get("excluded_collections"),
            sampling=plan.get("filters"),
        ),
        weak_labels="refinements of the author label agreed by at least two of: Singlet guide, CellTypist 1.7.1 (16 models, CL crosswalk), SingleR-style correlation (rebuilt, cross-fitted by dataset)",
        output_space=dict(n_terms=len(terms["terms"]), cl_version=cl, min_cells=terms.get("min_cells")),
        inputs=dict(n_genes=len(vocab["genes"]), key="human Ensembl gene id; mouse via Ensembl one-to-one orthologs",
                    n_mouse_orthologs=len(vocab.get("mouse") or {}), transform="log1p(1e4 * c / sum over vocabulary)"),
        training=dict(args=train.get("args"), temperature=temperature, closure_nll_temperature=train["final"]["temperature"],
                      calibration=None if calib is None else calib["best"], validation=train["final"]),
        benchmark_singlet_counts={m: dict(mean_over_studies=v["mean_over_studies"], calibration=v["pooled"].get("calibration"),
                                          s_per_million=v["pooled"].get("s_per_million"),
                                          s_per_million_with_read=v["pooled"].get("s_per_million_with_read"))
                                  for m, v in bench.items()},
        per_study={m: {g: dict(d1=p["d1"], d2=p["d2"], d3=p["d3"], d4=p["d4"], wrong=p["wrong"], n=p["n"]) for g, p in v["per_study"].items()}
                   for m, v in bench.items() if m in ("nn", "guide")},
        benchmark_author_matrices=None if cxg is None else {m: v["mean_over_studies"] for m, v in cxg.items()},
    )
    (a.model / "model_card.json").write_text(json.dumps(card, indent=1, default=float))
    nn = bench.get("nn", {}).get("mean_over_studies", {})
    md = [
        f"# Model card: {version}",
        "",
        f"- Task: {card['task']}.",
        f"- Training data: {card['training_data']['source']}; {card['training_data']['cells']:,} cells from "
        f"{card['training_data']['datasets']} datasets in {card['training_data']['collections']} collections "
        f"({', '.join(f'{k} {v:,}' for k, v in card['training_data']['organisms'].items())}), "
        f"{len(card['training_data']['tissues'])} tissues. Licence {card['training_data']['licence']}.",
        f"- Held out: the {len(plan.get('excluded_collections') or [])} CELLxGENE collections of the benchmark studies; "
        f"{card['training_data']['excluded_copies']} copies of benchmark cells found elsewhere were dropped.",
        f"- Weak labels: {card['weak_labels']}.",
        f"- Output: {card['output_space']['n_terms']} CL terms (CL {card['output_space']['cl_version']}); inputs "
        f"{card['inputs']['n_genes']} genes ({card['inputs']['key']}).",
        f"- Temperature {card['training']['temperature']:.3f} (fitted on the reliability of the call, validation collections; "
        f"closure-likelihood fit {card['training']['closure_nll_temperature']:.3f}).",
        f"- Singlet counts, 16 held-out studies, mean over studies: depth 1/2/3/4 = "
        + "/".join(f"{nn.get(f'd{d}', float('nan')):.3f}" for d in (1, 2, 3, 4))
        + f", wrong {nn.get('wrong', float('nan')):.3f}.",
        "",
        "Per study (network): d1/d2/d3/d4, wrong",
        "",
        "| study | d1 | d2 | d3 | d4 | wrong | cells |",
        "|---|---|---|---|---|---|---|",
    ]
    for g, p in sorted(card["per_study"].get("nn", {}).items()):
        md.append(f"| {g} | {p['d1']:.2f} | {p['d2']:.2f} | {p['d3']:.2f} | {p['d4']:.2f} | {p['wrong']:.2f} | {p['n']} |")
    (a.model / "model_card.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
