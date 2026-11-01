# SPDX-License-Identifier: MIT
"""Where Singlet's unspliced molecules are missing compared with STARsolo-Velocyto and alevin-fry: per-gene U totals of the release-gate spliced-unspliced artifact.
usage: s8_ugap.py GENES.tsv.gz GTF OUT.json"""
import gzip
import json
import re
import sys

import numpy as np
import pandas as pd

g = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
S, V, A = g["unspliced:singlet_U"], g["unspliced:starsolo_U"], g["unspliced:alevinfry_U"]
rows = {}
with open(sys.argv[2]) as f:
    for ln in f:
        if ln[0] == "#":
            continue
        p = ln.rstrip("\n").split("\t")
        if p[2] != "gene":
            continue
        a = p[8]
        gid = re.search(r'gene_id "([^"]+)"', a).group(1).split(".")[0]
        bt = re.search(r'gene_type "([^"]+)"', a) or re.search(r'gene_biotype "([^"]+)"', a)
        nm = re.search(r'gene_name "([^"]+)"', a)
        rows[gid] = (p[0], int(p[3]), int(p[4]), p[6], bt.group(1) if bt else "?", nm.group(1) if nm else gid)
ann = pd.DataFrame.from_dict(rows, orient="index", columns=["chrom", "start", "end", "strand", "biotype", "name"])
# overlap with another gene: same strand / opposite strand
ov_same, ov_anti = {}, {}
for chrom, d in ann.groupby("chrom"):
    d = d.sort_values("start")
    st, en, sd, ids = d.start.to_numpy(), d.end.to_numpy(), d.strand.to_numpy(), d.index.to_numpy()
    for i in range(len(d)):
        j = np.flatnonzero((st < en[i]) & (en > st[i]))
        j = j[j != i]
        ov_same[ids[i]] = bool((sd[j] == sd[i]).any()); ov_anti[ids[i]] = bool((sd[j] != sd[i]).any())
ann["ov_same"], ann["ov_anti"] = pd.Series(ov_same), pd.Series(ov_anti)
d = g.join(ann, how="left")
d["ref"] = (V + A) / 2
d["gap"] = d.ref - S                      # molecules the two competitors call U beyond Singlet
tot = {"singlet_U": int(S.sum()), "starsolo_U": int(V.sum()), "alevinfry_U": int(A.sum()), "competitor_mean": float(d.ref.sum()), "net_gap": float(d.gap.sum()),
       "gross_deficit": float(d.gap[d.gap > 0].sum()), "gross_surplus": float(-d.gap[d.gap < 0].sum())}
out = {"totals": tot}
pos = d.gap.clip(lower=0).sort_values(ascending=False)
out["deficit_share_top"] = {str(k): float(pos.head(k).sum() / pos.sum()) for k in (10, 50, 200, 1000)}
out["genes_with_deficit"] = int((d.gap > 0.5).sum())
top = d.loc[pos.head(25).index, ["name", "biotype", "chrom", "strand", "ov_same", "ov_anti", "unspliced:singlet_U", "unspliced:starsolo_U", "unspliced:alevinfry_U", "gap"]]
out["top25"] = top.reset_index().to_dict("records")
cat = {}
for nm, m in (("overlapped_same_strand", d.ov_same), ("overlapped_antisense_only", d.ov_anti & ~d.ov_same), ("isolated", ~d.ov_same & ~d.ov_anti),
              ("mito", d.chrom.isin(["chrM", "MT"]))):
    cat[nm] = {"genes": int(m.sum()), "singlet": int(S[m].sum()), "competitors": float(d.ref[m].sum()), "gap": float(d.gap[m].sum())}
out["by_overlap"] = cat
bt = d.groupby("biotype")[["unspliced:singlet_U", "ref", "gap"]].sum().sort_values("gap", ascending=False).head(8)
out["by_biotype"] = bt.reset_index().to_dict("records")
# per-gene ratio by expression decile (reference U)
e = d[d.ref >= 20].copy(); e["ratio"] = e["unspliced:singlet_U"] / e.ref
e["bin"] = pd.qcut(e.ref, 5, duplicates="drop")
out["ratio_by_expression"] = [{"ref_range": str(k), "n": int(len(x)), "median_ratio": float(x.ratio.median()), "q10": float(x.ratio.quantile(.1)), "q90": float(x.ratio.quantile(.9)),
                               "weighted_ratio": float(x["unspliced:singlet_U"].sum() / x.ref.sum())} for k, x in e.groupby("bin", observed=True)]
out["starsolo_vs_alevinfry"] = {"weighted_ratio": float(V.sum() / A.sum())}
json.dump(out, open(sys.argv[3], "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
