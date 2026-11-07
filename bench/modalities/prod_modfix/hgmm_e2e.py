# SPDX-License-Identifier: MIT
"""hgmm1k_v3 end to end with the pipeline binary: ambient rho vs cross-species truth, corrected-matrix residual
cross-species counts, barnyard and doublet calls on the called cells.

    python -P hgmm_e2e.py RUN_DIR OUT.json      (truth: modality-bench/ambient/work/cells.csv, bench/modalities/ambient/prep.py)
"""
import gzip
import json
import sys

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import pearsonr, spearmanr

run, out = sys.argv[1:3]
W = "/mnt/projects/debruinz_project/debruinz/singlet/modality-bench/ambient/work"
meta = pd.read_csv(f"{W}/cells.csv").set_index("barcode")
R = {}
a = pd.read_csv(f"{run}/ambient_contamination.tsv", sep="\t")
a.index = [b.split("-")[0] for b in a.barcode]
m = meta.join(a[["rho"]], how="inner", rsuffix="_new")
pure = m.species.isin(["human", "mouse"])
x, t = m.loc[pure, "rho"].values, m.loc[pure, "rho_truth"].values
R["ambient"] = dict(cells=int(pure.sum()), mean_new=float(x.mean()), mean_truth=float(t.mean()),
                    ratio=float(x.mean() / t.mean()), spearman=float(spearmanr(x, t)[0]), pearson=float(pearsonr(x, t)[0]),
                    reference=dict(old_singlet=dict(mean=0.0027, spearman=0.41), soupx=0.49, decontx_bg=0.69, cellbender=0.60))
R["ambient_json"] = json.load(open(f"{run}/ambient_contamination.json"))
# corrected matrix: cross-species counts left in pure cells
try:
    def lines(p):
        with gzip.open(p, "rt") as f:
            return [l.rstrip("\n") for l in f]
    C = scipy.io.mmread(f"{run}/ambient_corrected_counts.mtx.gz").tocsc()
    feats = lines(f"{run}/ambient_corrected_counts_features.tsv.gz")
    bcs = [b.split("-")[0] for b in lines(f"{run}/ambient_corrected_counts_barcodes.tsv.gz")]
    mouse_gene = np.array(["ENSMUSG" in f or f.startswith("GRCm39") or "mm10" in f for f in feats])
    bi = {b: i for i, b in enumerate(bcs)}
    hum = [bi[b] for b in m.index[m.species == "human"] if b in bi]
    mou = [bi[b] for b in m.index[m.species == "mouse"] if b in bi]
    cross_after = C[mouse_gene][:, hum].sum() + C[~mouse_gene][:, mou].sum()
    cross_before = float((m.loc[m.species == "human", "M"]).sum() + (m.loc[m.species == "mouse", "H"]).sum())
    R["correction"] = dict(cross_species_before=cross_before, cross_species_after=float(cross_after),
                           left_fraction=float(cross_after / max(cross_before, 1)),
                           reference_left=dict(soupx=0.231, decontx_bg=0.231, cellbender=0.115))
except Exception as e:  # noqa: BLE001
    R["correction_error"] = repr(e)
try:
    by = pd.read_csv(f"{run}/barnyard_classification.tsv", sep="\t")
    R["barnyard"] = dict(rows=len(by), calls=by.species_call.value_counts().to_dict())
except Exception as e:  # noqa: BLE001
    R["barnyard_error"] = repr(e)
d = pd.read_csv(f"{run}/doublet_scores.tsv", sep="\t")
d.index = [b.split("-")[0] for b in d.barcode]
dj = meta.join(d[["doublet_score", "is_doublet"]], how="inner")
truth = dj.minor_frac >= 0.2
calls = dj.is_doublet.astype(str).isin(["1", "True", "true"])
R["doublets"] = dict(cells=len(dj), called=int(calls.sum()), true_heterotypic=int(truth.sum()),
                     tp=int((calls & truth).sum()), fp=int((calls & ~truth).sum()),
                     reference=dict(old_singlet_called=220, scrublet_called=1, scdblfinder_called=17))
print(json.dumps(R, indent=1, default=float))
json.dump(R, open(out, "w"), indent=1, default=float)
