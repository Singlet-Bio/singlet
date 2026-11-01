"""v2 mode of the `doublets` capability: score `doublet/score_v2` (python/singlet/features/doublet.py, as landed in prod/campaign)
on the counts a gate binary wrote for a development dataset, against the same truth and competitors as the cached-score mode.

One call per dataset (calls accumulate in OUT):
  v2_gate.py --dataset hgmm1k_v3 --study 10x_hgmm --run RUN_DIR --table hgmm1k_v3_doublets.tsv.gz --out ART_DIR [--truth TEXT]
RUN_DIR   Singlet output directory of the gate binary (filtered_feature_bc_matrix/, doublet_scores.tsv, binary_*.txt)
--table   per-cell table of the ambient/doublets harness (barnyard_artifacts.py / kang_artifacts.py): barcode, singlet_score
          (the engine score = prod), scrublet_score, scdblfinder_score and either label (1 = doublet) or human_umi/mouse_umi
          (barnyard truth: minor species >= 20 percent of the UMIs of the cell)
Writes to OUT the artifacts bench/public/capabilities/doublets.py reads (doublets_cells.parquet, doublets_datasets.tsv,
provenance.json with mode "v2"); `python bench/public/run_capability.py doublets OUT --out OUT/doublets.json` gives the contract
document. Column combo = the v2 score. The stacker was trained on 17 other datasets (no leave-one-out here): run it on
development datasets only, and note which of them are among its training sets (the Kang lanes are).
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--study", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--table", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--truth", default="")
ap.add_argument("--species", default="human")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--minor-frac", type=float, default=0.2)
a = ap.parse_args()
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "python"))
from singlet.features import doublet as D  # noqa: E402

run, out = Path(a.run), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
tab = pd.read_csv(a.table, sep="\t", dtype={"barcode": str})
if "label" not in tab:
    tot = tab.human_umi + tab.mouse_umi
    tab["label"] = (np.minimum(tab.human_umi, tab.mouse_umi) / tot.clip(lower=1) >= a.minor_frac).astype(int)
mx = run / "filtered_feature_bc_matrix"
M = scipy.io.mmread(str(mx / "matrix.mtx.gz")).tocsr()
bc = pd.read_csv(mx / "barcodes.tsv.gz", header=None, sep="\t")[0].astype(str).str.replace(r"-1$", "", regex=True).to_numpy()  # 10x suffix
feat = pd.read_csv(mx / "features.tsv.gz", header=None, sep="\t")
ix = pd.Series(np.arange(len(bc)), index=bc).reindex(tab.barcode)
assert ix.notna().all(), "table barcodes missing from the matrix"
X = sp.csr_matrix(M.T.tocsr()[ix.astype(int).to_numpy()], dtype=np.float32)
prod = D.read_prod_scores(run / "doublet_scores.tsv", tab.barcode)  # the file has the barcodes without suffix
tm = {}
F = D.compute_inputs(X, list(feat[0].astype(str)), a.species, prod, timings=tm)
v2 = D.score_inputs(F)
wall = time.perf_counter() - t0
print(a.dataset, "cells", len(tab), "doublets", int(tab.label.sum()), "finite v2", int(np.isfinite(v2).sum()), tm, flush=True)
if not np.isfinite(v2).all():
    sys.exit(f"{a.dataset}: {int((~np.isfinite(v2)).sum())} cells without a v2 score (fewer than {D.MIN_CELLS} cells, or a non-finite input)")

cells = pd.DataFrame({"dataset": a.dataset, "study": a.study, "i": np.arange(len(tab)), "label": tab.label.astype(int).to_numpy(),
                      "prod": tab.singlet_score.to_numpy(), "scdblfinder": tab.scdblfinder_score.to_numpy(),
                      "scrublet": tab.scrublet_score.to_numpy(), "combo": v2.astype(float)})
dsrow = {"dataset": a.dataset, "study": a.study, "cells": len(tab), "doublets": int(tab.label.sum()), "truth": a.truth or "see table"}
pq, tsv = out / "doublets_cells.parquet", out / "doublets_datasets.tsv"
if pq.exists():
    cells = pd.concat([pd.read_parquet(pq).query("dataset != @a.dataset"), cells], ignore_index=True)
    ds = pd.read_csv(tsv, sep="\t")
    ds = pd.concat([ds[ds.dataset != a.dataset], pd.DataFrame([dsrow])], ignore_index=True)
else:
    ds = pd.DataFrame([dsrow])
cells.to_parquet(pq, index=False)
ds.to_csv(tsv, sep="\t", index=False)

# provenance.json, rewritten from all datasets present
from sklearn.metrics import average_precision_score  # noqa: E402

TOOLS = ("combo", "prod", "scdblfinder", "scrublet")
ap_ = {t: np.mean([average_precision_score(g.label, g[t]) for _, g in cells.groupby("dataset")]) for t in TOOLS}
n = len(ds)
sha = os.environ.get("SHA") or subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"


def rd(f):
    return (run / f).read_text().split()[0] if (run / f).exists() else "unknown"


prov = dict(
    mode="v2", bench_commit=sha, status="draft", generated_at=pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
    tool_version="doublet/score_v2",
    singlet=dict(version="1.0.0", commit=sha, branch="prod/campaign",
                 binary=f"gate binary {rd('binary_commit.txt')} (sha256 {rd('binary_sha256.txt')[:12]}); python: singlet.features.doublet score_v2"),
    singlet_params="singlet.features.doublet score_v2: gradient boosting (300 trees, 15 leaves) on within-sample ranks of log UMI, "
                   "log genes, the doublet_scores.tsv of the engine, two artificial-doublet scorers (sdfl, sdflc) and two features of the "
                   "distilled atlas encoder; frozen, trained on 17 other datasets; counts from the filtered matrix of the gate binary",
    methods=("Mode v2 of the doublets capability. Counts: the filtered_feature_bc_matrix and doublet_scores.tsv written by the gate binary "
             "on the development dataset(s) listed; truth and the competitor scores (Scrublet, scDblFinder, the engine score) from the "
             "ambient/doublets harness table of the same run. Singlet: doublet/score_v2, the frozen stacker of python/singlet/features/"
             "doublet.py. Metrics, tolerances and gates as in the cached-score mode; AUPRC and AUROC per dataset, pooled means over "
             "datasets. Barnyard truth: minor species >= 20 percent of the UMIs of a cell; only heterotypic doublets are visible."),
    conclusion=(f"On {n} development dataset(s) with the counts of the gate binary, doublet/score_v2 has mean AUPRC {ap_[TOOLS[0]]:.3f} "
                f"against {ap_[TOOLS[1]]:.3f} for the engine score, {ap_[TOOLS[2]]:.3f} for scDblFinder and {ap_[TOOLS[3]]:.3f} for Scrublet."),
    cost=dict(singlet=round(wall / len(tab) * 1e4, 1), prod=2.0, scdblfinder=43.6, scrublet=7.7),
    resources=[dict(tool="singlet", dataset="pooled", wall_s=round(wall, 1),
                    max_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2), threads=a.threads,
                    notes=f"v2 scoring of the last dataset added ({a.dataset}, {len(tab)} cells) from the filtered matrix of the run; "
                          "the engine run is not included")],
    caveats=["v2 mode: the stacker was trained on 17 datasets; a dataset among them (the Kang lanes) is in-sample for it. hgmm1k_v3 is not.",
             "Barnyard datasets have human and mouse genes in one matrix; the atlas encoder reads the human genes only."],
)
json.dump(prov, open(out / "provenance.json", "w"), indent=1)
print("v2 mean AUPRC", {k: round(float(v), 4) for k, v in ap_.items()})
