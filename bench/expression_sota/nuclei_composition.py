# SPDX-License-Identifier: MIT
"""Nuclei over-call: a hashtag truth and a pre-registered composition test.
  python nuclei_composition.py RUN_DIR OUT.json [HASHTAG_COUNTS.tsv]

Composition test (parameters frozen 2026-10-31 BEFORE scoring on the hashtag truth):
  p0       ambient intronic fraction = intron UMI / (exon + intron UMI) pooled over uncalled barcodes with 1 <= total <= lower
           (lower = the called_v2 ambient bound; here the 10th percentile of called_v2 totals / 5 is NOT used: the pool is totals 1..100)
  active   only when the library is intron rich: median intronic fraction of the knee-called barcodes minus p0 >= 0.15
  keep     an added barcode (called_v2 and not knee) stays called when the one-sided binomial p-value that its intronic fraction
           exceeds p0 is <= 0.01 after BH over the added barcodes; knee-called barcodes are never removed.
"""
import gzip, json, sys
import numpy as np, pandas as pd, scipy.io as sio
from scipy.stats import binom
run, out = sys.argv[1:3]; hf = sys.argv[3] if len(sys.argv) > 3 else None
bare = lambda b: b.split("-")[0]
cc = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t", dtype={"barcode": str}); cc["b"] = cc.barcode.map(bare); cc = cc.set_index("b")
knee = cc["called_knee"] if "called_knee" in cc else cc["called"]; v2 = cc["called_v2"]
def colsum(prefix):
    M = sio.mmread(gzip.open(f"{run}/{prefix}.mtx.gz", "rb")).tocsc()
    bc = [bare(l.split("\t")[0].strip()) for l in gzip.open(f"{run}/{prefix}_barcodes.tsv.gz", "rt")]
    return pd.Series(np.asarray(M.sum(0)).ravel(), index=bc)
ex, it = colsum("exon_counts"), colsum("intron_counts")
d = pd.DataFrame({"exon": ex, "intron": it}).fillna(0); d["tot"] = d.exon + d.intron; d = d[d.tot > 0]
d["frac"] = d.intron / d.tot
d["knee"] = knee.reindex(d.index).fillna(0).astype(int); d["v2"] = v2.reindex(d.index).fillna(0).astype(int)
called = d[(d.knee == 1) | (d.v2 == 1)]
pool = d[(d.knee == 0) & (d.v2 == 0) & (d.tot >= 1) & (d.tot <= 100)]
p0 = pool.intron.sum() / pool.tot.sum()
med_knee = d[d.knee == 1].frac.median()
active = bool(med_knee - p0 >= 0.15)
added = d[(d.v2 == 1) & (d.knee == 0)].copy()
pv = binom.sf(added.intron.values - 1, added.tot.values.astype(int), p0)
o = np.argsort(pv); m = len(pv); adj = np.empty(m); adj[o] = np.minimum.accumulate((pv[o] * m / (np.arange(m) + 1))[::-1])[::-1]
added["keep"] = (adj <= 0.01) if active else True
res = {"p0": float(p0), "median_intronic_knee": float(med_knee), "active": active, "n_knee": int(d.knee.sum()), "n_v2": int(d.v2.sum()),
       "n_added": int(len(added)), "n_added_kept": int(added.keep.sum()), "n_added_removed": int((~added.keep).sum()),
       "median_umi": {"knee": float(d[d.knee == 1].tot.median()), "added": float(added.tot.median())},
       "median_intronic": {"added": float(added.frac.median()), "pool": float(pool.frac.median())}}
if hf:
    h = pd.read_csv(hf, sep="\t"); h["b"] = h.barcode.map(bare); h = h.set_index("b").drop(columns="barcode")
    d["htot"] = h.sum(1).reindex(d.index).fillna(0); d["hmax"] = h.max(1).reindex(d.index).fillna(0)
    d["hfrac"] = np.where(d.htot > 0, d.hmax / d.htot.clip(lower=1), 0)
    emp = d[(d.knee == 0) & (d.v2 == 0) & (d.tot <= 20)]
    T = max(float(np.percentile(emp.htot, 99)), 20.0)   # hashtag UMI a background droplet exceeds < 1% of the time
    pos = lambda x: (x.htot >= T) & (x.hfrac >= 0.6)
    added = added.join(d[["htot", "hfrac"]]); added["pos"] = pos(added)
    res["hashtag"] = {"threshold_umi": T, "empty_droplet_p99": float(np.percentile(emp.htot, 99)),
                      "knee_called_positive": float(pos(d[d.knee == 1]).mean()), "added_positive": float(added.pos.mean()),
                      "added_kept_positive": float(added[added.keep].pos.mean()) if added.keep.any() else None,
                      "added_removed_positive": float(added[~added.keep].pos.mean()) if (~added.keep).any() else None}
    ctl = {}
    for lo, hi in ((20, 50), (50, 100), (100, 200), (200, 400)):
        x = d[(d.knee == 0) & (d.v2 == 0) & (d.tot >= lo) & (d.tot < hi)]
        ctl[f"{lo}-{hi}"] = {"n": int(len(x)), "positive": float(pos(x).mean()), "median_htot": float(x.htot.median())}
    res["hashtag"]["uncalled_by_rna_umi"] = ctl
    ad = added
    res["hashtag"]["added_by_rna_umi"] = {f"{lo}-{hi}": {"n": int(((ad.tot >= lo) & (ad.tot < hi)).sum()), "positive": float(ad[(ad.tot >= lo) & (ad.tot < hi)].pos.mean()) if ((ad.tot >= lo) & (ad.tot < hi)).any() else None}
                                         for lo, hi in ((20, 100), (100, 200), (200, 400), (400, 2000))}
    res["hashtag"]["added_median_htot"] = float(ad.htot.median()); res["hashtag"]["knee_median_htot"] = float(d[d.knee == 1].htot.median())
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
