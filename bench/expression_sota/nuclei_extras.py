# SPDX-License-Identifier: MIT
"""nuclei2k_v2: who are the barcodes the pin's called_v2 adds over STARsolo EmptyDrops_CR (Gene and GeneFull)? QC profile by group."""
import pandas as pd, numpy as np
R = "/mnt/projects/debruinz_project/debruinz/singlet/pipeline-benchmark/runs/nuclei2k_v2"
bare = lambda s: s.split("-")[0]
cc = pd.read_csv(f"{R}/singlet-public/cell_calls.tsv", sep="\t", dtype={"barcode": str})
qc = pd.read_csv(f"{R}/singlet-public/cell_qc_metrics.tsv", sep="\t", dtype={"barcode": str}, usecols=["barcode", "n_umi", "n_genes", "pct_mt", "intronic_pct"]).set_index("barcode")
rd = lambda p: {bare(x.strip()) for x in open(p)}
ed = rd(f"{R}/starsolo-public-cr/Solo.out/Gene/filtered/barcodes.tsv")
edf = rd(f"{R}/starsolo-public-cr/Solo.out/GeneFull/filtered/barcodes.tsv")
v2 = set(cc.barcode[cc.called_v2 == 1]); kn = set(cc.barcode[cc.called == 1])
print("calls: v2", len(v2), "knee", len(kn), "EDCR Gene", len(ed), "EDCR GeneFull", len(edf))
print("v2&ED", len(v2 & ed), "v2&EDfull", len(v2 & edf), "P vs ED %.3f R %.3f; P vs EDfull %.3f R %.3f" % (len(v2 & ed) / len(v2), len(v2 & ed) / len(ed), len(v2 & edf) / len(v2), len(v2 & edf) / len(edf)))
cc["fdr"] = pd.to_numeric(cc.fdr, errors="coerce")
q = qc.copy(); q.index = [bare(i) for i in q.index]
cc["b"] = cc.barcode.map(bare); cc = cc.set_index("b")
groups = {"v2 & ED": v2 & ed, "v2 only (not ED Gene)": v2 - ed, "v2 only (not ED GeneFull)": v2 - edf, "ED only": ed - v2, "empty-ish (ranks 3000-6000)": None}
for name, s in groups.items():
    if s is None: continue
    d = q.reindex(list(s)); print(f"{name:30s} n={len(s):5d} umi_med={d.n_umi.median():7.0f} genes_med={d.n_genes.median():6.0f} mt%_med={d.pct_mt.median():5.2f} intronic%_med={d.intronic_pct.median():5.1f}  umi q10={d.n_umi.quantile(.1):.0f}")
amb = cc[(cc.umi_total.between(20, 100)) & ~cc.index.isin(v2 | ed)].index[:20000]
d = q.reindex(list(amb)); print(f"{'ambient 20-100 UMI':30s} n={len(amb):5d} umi_med={d.n_umi.median():7.0f} mt%_med={d.pct_mt.median():5.2f} intronic%_med={d.intronic_pct.median():5.1f}")
x = cc.loc[list(v2 - ed)]; print("v2-only fdr quantiles", x.fdr.quantile([.1, .5, .9]).round(6).to_dict(), "umi_total", x.umi_total.describe()[["min", "50%", "max"]].to_dict())
# intact filter: thresholds from the barcodes both callers agree on (q99 mito, q01 intronic), as in bench/cellcall_truth "intact"
core = q.reindex(list(v2 & ed)); mt_hi, in_lo = core.pct_mt.quantile(.99), core.intronic_pct.quantile(.01)
for name, s in (("shared", v2 & ed), ("v2-only", v2 - ed)):
    d = q.reindex(list(s)); ok = (d.pct_mt <= mt_hi) & (d.intronic_pct >= in_lo)
    print(f"intact filter (mt<={mt_hi:.1f}%, intronic>={in_lo:.1f}%): {name} pass {int(ok.sum())} of {len(s)}")
d = q.reindex(list(amb)); print("ambient pass", int(((d.pct_mt <= mt_hi) & (d.intronic_pct >= in_lo)).sum()), "of", len(amb))
