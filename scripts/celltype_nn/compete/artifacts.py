#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Collect the cell type comparison into the public benchmark artifacts (``bench/public/CONTRACT.md`` section 5):
``celltype_calls.parquet``, ``celltype_truth.parquet`` and ``provenance.json``, read by
``bench/public/capabilities/cell_type.py``.

  python scripts/celltype_nn/compete/artifacts.py --compete $D/compete --data $D --out $WORK/bpub/artifacts/cell-type \
      --commit <40-hex> [--rss rss.tsv] [--status draft]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))

# harness directory -> public tool name
TOOLS = {
    "singlet_nn_argmax": "singlet",  # the forced label (sidecar argmax_cl), the like-for-like output of every competitor
    "singlet_nn": "singlet-call",  # the published call (sidecar cell_type_cl): most specific term with rolled-up probability >= 0.5
    "celltypist": "CellTypist",
    "scanvi": "scANVI",
    "seurat": "Seurat",
    "singler": "SingleR",
    "singlet_guide": "singlet-guide",
}
TOOL_INFO = {
    "singlet": dict(role="singlet", version="1.0.0", url="https://github.com/Singlet-Bio/singlet",
                    params="singlet.celltype_nn model m1_notissue (Census 2025-11-08, CL 2026-06-08): rank inputs over 6,000 genes, "
                           "MLP 2048; label = argmax_cl (the most probable CL term), score = its probability"),
    "singlet-call": dict(role="helper", version="1.0.0",
                         params="the same model's published call cell_type_cl: the most specific CL term with rolled-up probability >= 0.5, "
                                "prob = that rolled-up probability (cell_type_prob)"),
    "CellTypist": dict(role="competitor", version="1.7.1", url="https://github.com/Teichlab/celltypist", citation="Dominguez Conde et al., Science 2022",
                       params="celltypist.train(log1p CP10k, labels = CL term, use_SGD=True, mini_batch=True, balance_cell_type=True, epochs=10) "
                              "on the full Census training split (2.2M human, 0.4M mouse cells); celltypist.annotate(majority_voting=False)"),
    "scANVI": dict(role="competitor", version="scvi-tools 1.4.2", url="https://github.com/scverse/scvi-tools", citation="Xu et al., Mol Syst Biol 2021",
                   params="SCVI(n_latent=30, n_layers=2, nb; batch = dataset; up to 100 epochs, early stopping) -> SCANVI.from_scvi_model (20 epochs); "
                          "query: SCANVI.load_query_data + train(max_epochs=100, weight_decay=0); label = argmax of predict(soft=True)"),
    "Seurat": dict(role="competitor", version="5.0.0", url="https://github.com/satijalab/seurat", citation="Hao et al., Cell 2021 (Azimuth)",
                   params="reference: NormalizeData, FindVariableFeatures(2000), ScaleData, RunPCA(30); query: NormalizeData, "
                          "FindTransferAnchors(reference.reduction='pca', dims=1:30), TransferData; label = predicted.id, prob = prediction.score.max"),
    "SingleR": dict(role="competitor", version="2.0.0", url="https://bioconductor.org/packages/SingleR", citation="Aran et al., Nat Immunol 2019",
                    params="SingleR(test = log1p CP10k, ref = reference log1p CP10k, labels = CL term, aggr.ref=TRUE, de.method='classic'); "
                           "score = delta.next; pruned.labels kept as SingleR's own abstention"),
    "singlet-guide": dict(role="helper", version="1.0.0", params="singlet.celllabels guide (marker hierarchy + kNN smoothing), the atlas's current labels"),
}
METHODS = (
    "**Data.** cxg16: 16 GEO studies (66 samples, 13 human and 3 mouse) that CELLxGENE curators annotated with Cell Ontology "
    "terms. Author labels are moved onto Singlet's called cells by barcode, and only cells that both sides kept are scored "
    "(291,331 cells). Every tool reads Singlet's counts of these cells: exon plus intron, projected on one 6,000-gene vocabulary. "
    "**Reference.** Every tool learns from the same CELLxGENE Census 2025-11-08 training corpus. The corpus is a stratified "
    "sample of 2.6M primary droplet cells, and the 16 studies' collections are excluded. CellTypist and Singlet's network "
    "train on all of it; Seurat, SingleR and scANVI use a stratified subsample of at most 200 cells per term (145k human, "
    "62k mouse cells), the scale they are designed for. Labels are the corpus's curated CL terms, so no tool needs a "
    "crosswalk. **Metrics.** Accuracy at depth d compares the depth-d ancestors of call and truth on the guide's tree (1 "
    "compartment, 2 lineage, 3 type, 4 subtype); a call that stops above depth d counts as wrong there. 'Wrong' is a call "
    "on another branch. Each study pools its samples, and values are means over studies with bootstrap CIs over studies. "
    "Calibration: ECE of each tool's probability against 'the truth lies in the call's subtree'. Abstention: area under "
    "each tool's risk-coverage curve. Robustness: depth-2 accuracy on studies whose tissue is thin in the reference. Runtime: "
    "CPU time per million cells to label a query once the reference is prepared. Singlet gives two labels per cell. The "
    "most probable term is compared with every competitor's label; the calibrated call, which backs off to the parent term "
    "when unsure, is scored for wrong calls and calibration."
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compete", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--rss", type=Path, default=None, help="TSV tool<TAB>max_rss_gb (from sacct)")
    ap.add_argument("--status", default="draft")
    a = ap.parse_args(argv)

    import pandas as pd

    from singlet.celllabels.guide import tissue_groups

    a.out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(a.compete / "query" / "samples.csv")
    truth, parts = [], []
    for r in samples.itertuples():
        q = pd.read_csv(a.compete / "query" / r.gse / f"{r.gsm}.cells.csv", dtype={"truth": str}, keep_default_na=False)
        q = q[q["truth"] != ""]
        truth.append(pd.DataFrame({"gse": r.gse, "gsm": r.gsm, "barcode": q["barcode"].astype(str), "species": r.species,
                                   "truth": q["truth"]}))
    truth = pd.concat(truth, ignore_index=True)
    truth.to_parquet(a.out / "celltype_truth.parquet", index=False)
    runtime, resources = {}, []
    rss = dict(pd.read_csv(a.rss, sep="\t", header=None).values.tolist()) if a.rss and a.rss.exists() else {}
    for d, name in TOOLS.items():
        base = a.compete / "calls" / d
        if not base.exists():
            continue
        for r in samples.itertuples():
            for ext in (".parquet", ".csv"):
                f = base / r.gse / f"{r.gsm}{ext}"
                if f.exists():
                    c = pd.read_parquet(f) if ext == ".parquet" else pd.read_csv(f, dtype={"pred_cl": object, "pruned": object})
                    c = c.assign(tool=name, gse=r.gse, gsm=r.gsm, barcode=c["barcode"].astype(str))
                    for col in ("prob", "pruned"):
                        if col not in c:
                            c[col] = np.nan if col == "prob" else None
                    c = c[["tool", "gse", "gsm", "barcode", "pred_cl", "score", "prob", "pruned"]]
                    c["pred_cl"] = c["pred_cl"].where(c["pred_cl"].astype(str).str.startswith("CL:"), None)
                    c["pruned"] = c["pruned"].where(c["pruned"].astype(str).str.startswith("CL:"), None)
                    parts.append(c.merge(truth[["gse", "gsm", "barcode"]][truth["gsm"] == r.gsm], on=["gse", "gsm", "barcode"]))
        tf = base / "timings_controlled.jsonl"  # timing.py: one exclusive node, same threads, tools back to back
        if not tf.exists():
            tf = base / "timings.jsonl"
        if tf.exists():
            T = {(t["gse"], t["gsm"]): t for t in (json.loads(x) for x in tf.read_text().splitlines() if x.strip())}
            cells = sum(t["n_cells"] for t in T.values())
            wall = sum(t["seconds"] for t in T.values())
            cpu = sum(t["seconds"] * t.get("threads", 1) for t in T.values())
            runtime[name] = cpu / max(cells, 1) * 1e6
            threads = max(t.get("threads", 1) for t in T.values())
            resources.append(dict(tool=name, dataset="cxg16", wall_s=round(wall, 1), cpu_h=round(cpu / 3600, 3), threads=int(threads),
                                  max_rss_gb=float(rss.get(d, rss.get("singlet_nn" if d.startswith("singlet") else name, 0.0)) or 0.0),
                                  node="H100 GPU" if "cuda" in {t.get("device") for t in T.values()} else "Clipper CPU",
                                  notes=f"{cells:,} cells, prediction only (reference prepared)"))
    calls = pd.concat(parts, ignore_index=True)
    calls.to_parquet(a.out / "celltype_calls.parquet", index=False)
    # reference tissue coverage per study (cross-tissue generalisation)
    corpus = pd.read_parquet(a.data / "corpus_obs.parquet", columns=["tissue_general", "split"])
    grp = defaultdict(int)
    for t, n in corpus[corpus["split"] == "train"]["tissue_general"].value_counts().items():
        for g in tissue_groups(str(t)):
            grp[g] += int(n)
    seen = {}
    for g, d in samples.groupby("gse"):
        tissue = str(d["cxg_tissue"].iloc[0])
        groups = tissue_groups(tissue) or tissue_groups(str(d["tissue_text"].iloc[0]))
        n = int(max([grp[x] for x in groups], default=0))
        seen[g] = dict(tissue=tissue, groups=sorted(groups), reference_cells=n, seen=n >= 20000)
    gses = sorted(samples["gse"].unique())
    prov = dict(
        bench_commit=a.commit,
        singlet=dict(version="1.0.0", commit=a.commit, branch="claude/celltype-nn"),
        tools=[dict(name=n, **TOOL_INFO[n]) for n in TOOL_INFO if (calls["tool"] == n).any()],
        datasets=[dict(id="cxg16", title="16 GEO studies with CELLxGENE curated cell types (66 samples)",
                       accessions=dict(gse=";".join(gses), other=[f"CELLxGENE collections: see $WORK/jobs/annotate/bench_studies.json"]),
                       truth="CELLxGENE curators' Cell Ontology terms for the authors' labels, transferred to Singlet's called cells by barcode "
                             "(scripts/annotate_truth_cxg.py)", n_cells=int(len(truth)))],
        resources=resources,
        runtime=runtime,
        reference_tissue=seen,
        seen_cells_threshold=20000,
        methods=METHODS,
        status=a.status,
        caveats=[
            "Singlet's value in the label metrics is its forced label (argmax_cl), the output every competitor gives. This column "
            "was chosen for the comparison after cxg16 was scored; the model and both columns were fixed before that. The published "
            "call (cell_type_cl) is reported separately as singlet-call.",
            "Two training choices of the network followed a first look at cxg16: calibrating the temperature on call reliability (on held-out "
            "Census collections) and trying tissue context (not used). See docs/celltype_nn.md section 4.",
            "Azimuth's own pretrained references cover 3 of the 16 studies' tissues; Seurat reference mapping with the shared Census reference "
            "stands for the Azimuth method on all 16.",
            "Thymic CD34+ progenitors (GSE144870) are the network's largest loss; on the authors' own matrix of these cells it scores 0.98 at "
            "depth 2, so the loss comes from reprocessing.",
            "Ding 2020 (7 protocols) and the 10x / mouse cohort sets listed for cell-type await the harness's Singlet counts.",
        ],
    )
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"{len(calls):,} calls from {calls['tool'].nunique()} tools; {len(truth):,} truth cells; runtime {runtime}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
