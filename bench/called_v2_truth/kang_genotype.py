# SPDX-License-Identifier: MIT
"""Kang 2018 (GSE96583 batch 2 ctrl): are the barcodes called_v2 adds beyond the demuxlet table real single cells?

Independent of Singlet's own demux: cellsnp-lite pileup on the 19,743 informative SNPs of the reference run (ref_k8), then
vireo with the 8 donor genotypes that run estimated (``-d GT_donors.vireo.vcf.gz``), for

* ``extra``: barcodes production called_v2 calls that are not in the demuxlet table;
* ``truth_low``: truth cells at the extras' depth (total <= 90th percentile of the extras), ``truth_rand``: random truth cells;
* ``ambient``: uncalled barcodes with 20-50 UMIs (real empty droplets);
* ``synthetic_ambient``: sums of random ``ambient`` barcodes with the extras' UMI totals. An empty droplet at 300 UMIs looks like
  this: a mixture of all donors in ambient proportions. With few informative SNP UMIs any barcode can look like one donor, so
  this group gives the rate at which pure ambient gets a confident singlet call at the same depth;
* ``truth_downsampled``: the random truth cells with their SNP UMIs thinned to the extras' totals (sensitivity at that depth).

  prep  RUN_DIR DATASETS_TSV OUT   -> OUT/barcodes.txt, OUT/groups.tsv
  synth OUT                        -> OUT/cellsnp_aug (cellsnp dir + synthetic columns)
  summ  OUT                        -> OUT/kang_genotype.json
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd


def prep(run, ds_tsv, out):
    rng = np.random.default_rng(11)
    os.makedirs(out, exist_ok=True)
    t = pd.read_csv(ds_tsv, sep="\t", dtype={"barcode": str})
    prod = t["v:prod (knee, fdr .001)"].astype(bool)
    bk = t["v:bounded knee, fdr .001"].astype(bool)
    tr = t["truth_kang"].astype(bool)
    extra = t[prod & ~tr]
    hi = np.quantile(extra["total"], 0.9)
    truth = t[tr]
    tl = truth[truth["total"] <= hi]
    tl = tl.sample(n=min(3000, len(tl)), random_state=1)
    trr = truth.sample(n=1500, random_state=2)
    cc = pd.read_csv(os.path.join(run, "cell_calls.tsv"), sep="\t", dtype={"barcode": str})
    amb = cc[(cc["called_v2"] == 0) & (cc["umi_total"] >= 20) & (cc["umi_total"] <= 50)]
    amb = amb.sample(n=min(8000, len(amb)), random_state=3)
    rows = [(b, "extra_kept_bounded" if k else "extra_dropped_bounded", n) for b, k, n in zip(extra["barcode"], bk[prod & ~tr], extra["total"])]
    rows += [(b, "truth_low", n) for b, n in zip(tl["barcode"], tl["total"])]
    rows += [(b, "truth_rand", n) for b, n in zip(trr["barcode"], trr["total"])]
    rows += [(b, "ambient", n) for b, n in zip(amb["barcode"], amb["umi_total"])]
    g = pd.DataFrame(rows, columns=["barcode", "group", "total"]).drop_duplicates("barcode")
    g.to_csv(os.path.join(out, "groups.tsv"), sep="\t", index=False)
    with open(os.path.join(out, "barcodes.txt"), "w") as fh:
        for b in g["barcode"]:
            fh.write(f"{b}-1\n")
    print(g["group"].value_counts().to_dict(), "extra total 90th pct", hi)


def synth(out, n_synth=2000):
    from scipy.io import mmread, mmwrite
    import scipy.sparse as sp

    rng = np.random.default_rng(5)
    src = os.path.join(out, "cellsnp")
    dst = os.path.join(out, "cellsnp_aug")
    os.makedirs(dst, exist_ok=True)
    samples = [ln.strip() for ln in open(os.path.join(src, "cellSNP.samples.tsv")) if ln.strip()]
    g = pd.read_csv(os.path.join(out, "groups.tsv"), sep="\t", dtype={"barcode": str}).set_index("barcode")
    bare = [s.split("-")[0] for s in samples]
    grp = g["group"].reindex(bare).to_numpy()
    tot = g["total"].reindex(bare).to_numpy()
    amb_idx = np.nonzero(grp == "ambient")[0]
    ext_tot = tot[grp == "extra_kept_bounded"]  # depth of the extras that survive the recommended fix
    groups = []
    for _ in range(n_synth):
        target = rng.choice(ext_tot)
        perm = rng.permutation(amb_idx)
        cs = np.cumsum(tot[perm])
        k = int(np.searchsorted(cs, target) + 1)
        groups.append(perm[:k])
    S = sp.csc_matrix((np.ones(sum(len(x) for x in groups)), (np.concatenate(groups),
                       np.repeat(np.arange(n_synth), [len(x) for x in groups]))), shape=(len(samples), n_synth))
    mats = {tag: sp.csc_matrix(mmread(os.path.join(src, f"cellSNP.tag.{tag}.mtx"))).astype(np.int64)
            for tag in ("AD", "DP", "OTH") if os.path.exists(os.path.join(src, f"cellSNP.tag.{tag}.mtx"))}
    # truth cells thinned to the extras' totals: each molecule kept with prob f; DP' ~ Binom(DP, f), AD' ~ Hypergeom
    tr_idx = np.nonzero(grp == "truth_rand")[0]
    ds_t = rng.choice(ext_tot, size=len(tr_idx))
    frac = np.minimum(1.0, ds_t / tot[tr_idx])
    DPd = mats["DP"][:, tr_idx].toarray()
    ADd = np.minimum(mats["AD"][:, tr_idx].toarray(), DPd)
    F = np.broadcast_to(frac[None, :], DPd.shape)
    DPn = rng.binomial(DPd, F)
    ADn = rng.hypergeometric(ADd, DPd - ADd, DPn)
    thinned = {"DP": sp.csc_matrix(DPn), "AD": sp.csc_matrix(ADn)}
    if "OTH" in mats:
        O = mats["OTH"][:, tr_idx].toarray()
        thinned["OTH"] = sp.csc_matrix(rng.binomial(O, np.broadcast_to(frac[None, :], O.shape)))
    for tag, M in mats.items():
        A = sp.hstack([M, (M @ S).tocsc(), thinned[tag]]).tocsc()
        A.data = A.data.astype(np.int64)
        A.eliminate_zeros()
        mmwrite(os.path.join(dst, f"cellSNP.tag.{tag}.mtx"), A, field="integer")
    shutil.copy(os.path.join(src, "cellSNP.base.vcf.gz"), os.path.join(dst, "cellSNP.base.vcf.gz"))
    with open(os.path.join(dst, "cellSNP.samples.tsv"), "w") as fh:
        for s in samples:
            fh.write(s + "\n")
        for i in range(n_synth):
            fh.write(f"SYNTH{i:05d}-1\n")
        for i in range(len(tr_idx)):
            fh.write(f"TRUTHDS{i:05d}-1\n")
    st = np.array([tot[x].sum() for x in groups])
    pd.concat([
        pd.DataFrame({"barcode": [f"SYNTH{i:05d}" for i in range(n_synth)], "group": "synthetic_ambient", "total": st}),
        pd.DataFrame({"barcode": [f"TRUTHDS{i:05d}" for i in range(len(tr_idx))], "group": "truth_downsampled",
                      "total": np.minimum(ds_t, tot[tr_idx])}),
    ]).to_csv(os.path.join(out, "groups_synth.tsv"), sep="\t", index=False)
    print("synthetic ambient: n", n_synth, "totals median", float(np.median(st)), "barcodes per synth median",
          float(np.median([len(x) for x in groups])))


def summ(out, singlet_donor=None):
    g = pd.concat([pd.read_csv(os.path.join(out, f), sep="\t", dtype={"barcode": str}) for f in ("groups.tsv", "groups_synth.tsv")])
    g = g.set_index("barcode")
    v = pd.read_csv(os.path.join(out, "vireo", "donor_ids.tsv"), sep="\t")
    v["barcode"] = v["cell"].str.split("-").str[0]
    v = v.set_index("barcode").join(g, how="inner")
    v["status"] = np.where(v["donor_id"].isin(["doublet", "unassigned"]), v["donor_id"], "singlet")
    v["confident_singlet"] = (v["status"] == "singlet") & (v["prob_max"] >= 0.9)
    R = {}
    bins = [0, 5, 10, 20, 50, 100, 1e9]
    for name, d in v.groupby("group"):
        r = {"n": int(len(d)), "total_median": float(d["total"].median()),
             "n_vars": {str(qq): float(d["n_vars"].quantile(qq)) for qq in (0.1, 0.25, 0.5, 0.75, 0.9)},
             "status": d["status"].value_counts().to_dict(), "confident_singlet": int(d["confident_singlet"].sum()),
             "confident_singlet_frac": float(d["confident_singlet"].mean())}
        cut = pd.cut(d["n_vars"], bins, right=False)
        r["by_n_vars"] = {str(k): {"n": int(len(x)), "confident_singlet_frac": float(x["confident_singlet"].mean()) if len(x) else None,
                                   "doublet_frac": float((x["status"] == "doublet").mean()) if len(x) else None}
                          for k, x in d.groupby(cut, observed=False)}
        R[name] = r
    if singlet_donor:
        sd = pd.read_csv(singlet_donor, sep="\t", dtype={"cell": str}).set_index("cell")
        ex = v[v["group"].str.startswith("extra") & v["confident_singlet"]]
        s = sd.reindex(ex.index)
        ok = s["donor_id"].notna() & ~s["donor_id"].isin(["doublet", "unassigned"])
        # map Singlet donors to vireo donors through the truth cells
        tr = v[v["group"].str.startswith("truth") & v["confident_singlet"]]
        st = sd.reindex(tr.index)
        m = st["donor_id"].notna() & ~st["donor_id"].isin(["doublet", "unassigned"])
        mp = pd.crosstab(st["donor_id"][m], tr["donor_id"][m]).idxmax(axis=1).to_dict()
        agree = (s["donor_id"][ok].map(mp) == ex["donor_id"][ok])
        R["extra_confident_vireo_vs_singlet_demux"] = {"n_both_singlet": int(ok.sum()), "agree": float(agree.mean()) if ok.any() else None}
    json.dump(R, open(os.path.join(out, "kang_genotype.json"), "w"), indent=1, default=str)
    for k, r in R.items():
        if "n" in r:
            print(f"{k:24s} n={r['n']:5d} umi={r['total_median']:.0f} n_vars_med={r['n_vars']['0.5']:.0f} "
                  f"conf_singlet={r['confident_singlet_frac']:.3f} status={r['status']}")
    print(R.get("extra_confident_vireo_vs_singlet_demux"))


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "prep":
        prep(*sys.argv[2:5])
    elif m == "synth":
        synth(sys.argv[2])
    else:
        summ(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
