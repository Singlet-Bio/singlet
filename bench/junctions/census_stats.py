# SPDX-License-Identifier: MIT
"""Diagnostics on a finished junction census (scripts/store_junction_table.py), read-only:

* chromosome names of the census vs the GTF, and the coordinate offset that best matches annotated introns (mouse check);
* for unannotated junctions in >= 2 samples, how many pass each (n_studies, umi_called) threshold, and what share of called UMIs the
  annotated junctions alone and annotated + passing junctions hold.

    python bench/junctions/census_stats.py <run dir> <build> <gtf> > census_stats_<build>.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from singlet.store import junctions as J  # noqa: E402

run, build, gtf = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cdir = run / f"census_{build}"
cen = pd.read_parquet(cdir, filters=[("n_samples", ">=", 2)], columns=["chrom", "start", "end", "n_studies", "n_samples", "umi_called", "umi_uncalled"])
ann = J.read_gtf(gtf)
gtf_chroms = Counter(c for c, _s, _e in ann.introns)
cen_chroms = cen["chrom"].value_counts()
out = {
    "build": build,
    "census_rows_in_2plus_samples": int(len(cen)),
    "census_top_chroms": {str(k): int(v) for k, v in cen_chroms.head(12).items()},
    "gtf_top_chroms": dict(gtf_chroms.most_common(12)),
}
# best coordinate offset on a sample of the census rows of the most common chromosome names shared with the GTF
keys = set(ann.introns)
samp = cen.sort_values("umi_called", ascending=False).head(200_000)
best = []
for rename in ("as_is", "add_chr", "strip_chr"):
    ch = samp["chrom"].astype(str)
    if rename == "add_chr":
        ch = "chr" + ch
    elif rename == "strip_chr":
        ch = ch.str.replace(r"^chr", "", regex=True)
    for ds in (-1, 0, 1):
        for de in (-1, 0, 1):
            hit = sum((c, s + ds, e + de) in keys for c, s, e in zip(ch, samp["start"], samp["end"]))
            best.append((hit, rename, ds, de))
best.sort(reverse=True)
out["offset_match_top200k"] = [{"hits": h, "chrom": r, "d_start": ds, "d_end": de} for h, r, ds, de in best[:5]]

# threshold grid with the convention that matches best
_h, rename, ds, de = best[0]
ch = cen["chrom"].astype(str)
if rename == "add_chr":
    ch = "chr" + ch
elif rename == "strip_chr":
    ch = ch.str.replace(r"^chr", "", regex=True)
ia = pd.MultiIndex.from_arrays([ch, cen["start"] + ds, cen["end"] + de]).isin(pd.MultiIndex.from_tuples(list(keys)))
umi = cen["umi_called"].to_numpy(np.float64)
amb = cen["umi_uncalled"].to_numpy(np.float64)
famb = np.where(umi + amb > 0, amb / np.maximum(umi + amb, 1), 0)
tot = umi.sum()
out["annotated_rows"] = int(ia.sum())
out["annotated_umi_share"] = float(umi[ia].sum() / tot)
grid = []
for st in (3, 5, 10, 20, 50, 100):
    for u in (10, 30, 100, 300, 1000):
        ok = ~ia & (cen["n_studies"].to_numpy() >= st) & (umi >= u) & (famb <= 0.5)
        grid.append({"min_studies": st, "min_umi": u, "novel": int(ok.sum()), "umi_share_ann_plus_novel": float((umi[ia].sum() + umi[ok].sum()) / tot)})
out["novel_grid"] = grid
st2 = cen["n_studies"].to_numpy() >= 2
out["unannotated_in_2plus_studies"] = int((~ia & st2).sum())
out["unannotated_in_2plus_samples_1_study"] = int((~ia & ~st2).sum())
print(json.dumps(out, indent=1))
