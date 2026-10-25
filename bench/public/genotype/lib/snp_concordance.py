"""PRIVATE (runs on Clipper next to the genotype files). Per-donor genotype concordance of Singlet's, vireo's and
souporcell's donor genotypes against replicate-lane truth. Writes ONLY counts: a 3x3 genotype-class table per
(pool, tool, donor, stratum). No site, allele count or genotype is written anywhere.

Usage: snp_concordance.py RUNS TRUTH_DIR OUT.tsv POOL [POOL ...]
  RUNS      $WORK/bench-public/runs  (RUNS/<pool>/{singlet_auto,vireo,souporcell}/...)
Truth for pool P with replicate lane(s) R: for each truth donor d of P, the pseudobulk of d's demuxlet-singlet cells in the
lane of R that contains d (cellsnp-lite counts at the shared 1000G panel), genotyped with genotypes.genotype_from_counts.
Strata: `all` = every site where the tool emits a call and the truth is confident; `dp10` = those where the tool's own
donor-level depth is >= 10.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "capabilities"))
import pools as P  # noqa: E402
import genotypes as G  # noqa: E402
import demux_eval as DE  # noqa: E402
from _gcommon import contingency, hungarian  # noqa: E402


def truth_genotypes(runs: Path, truth_dir: Path, pool: str):
    """{truth donor: (site key list, genotype array)} from the replicate lane(s)."""
    out = {}
    for rep in P.REPLICATES[pool]:
        sites, ad, dp, bcs = G.read_cellsnp(P.rundir(runs, rep) / "vireo" / "cellsnp")
        t = P.load_truth(truth_dir, rep)
        lab = {b: d for b, d in t["ind"].items() if t.loc[b, "multiplets"] == "singlet" and d != "<NA>"}
        pb = G.pseudobulk(ad, dp, bcs, lab)
        keys = [G.site_key(*r) for r in sites[["chrom", "pos", "ref", "alt"]].itertuples(index=False)]
        for d, (a, n) in pb.items():
            out[d] = (keys, G.genotype_from_counts(a, n))
    return out


def tool_labels_and_vcfs(runs: Path, pool: str):
    """tool name -> (labels Series, vcf path, column order of clusters in that vcf)."""
    d = P.rundir(runs, pool)
    res = {}
    s = d / "singlet_auto"
    if (s / "donor").exists() or (s / "donor_assignments.tsv").exists():
        res["singlet"] = (DE.read_singlet(s), s / "private" / "donor", "singlet")
    v = d / "vireo" / "told_s1"
    if (v / "donor_ids.tsv").exists():
        res["vireo"] = (DE.read_vireo(v), v / "GT_donors.vireo.vcf.gz", "vireo")
    so = d / "souporcell" / "out"
    if (so / "clusters.tsv").exists():
        res["souporcell"] = (DE.read_souporcell(so), so / "cluster_genotypes.vcf", "souporcell")
    return res


def tool_donor_gts(tool: str, path: Path):
    """cluster label -> (site key list, GT array, DP array)."""
    if tool == "singlet":
        out = {}
        for f in sorted(path.glob("donor*.vcf")):
            keys, gts, dps = G.read_donor_vcf(f)
            out[f.stem] = (keys, gts[0], dps[0])
        return out
    keys, gts, dps = G.read_donor_vcf(path)
    names = [f"donor{j}" for j in range(len(gts))] if tool == "vireo" else [f"cluster{j}" for j in range(len(gts))]
    return {n: (keys, gts[j], dps[j]) for j, n in enumerate(names)}


def main():
    runs, truth_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = []
    for pool in sys.argv[4:]:
        T = P.load_truth(truth_dir, pool)
        tg = truth_genotypes(runs, truth_dir, pool)
        for tool, (lab, path, kind) in tool_labels_and_vcfs(runs, pool).items():
            tsing = T[T["multiplets"] == "singlet"]
            common = tsing.index.intersection(lab.index)
            pred = lab.loc[common, "label"]
            ok = ~pred.isin(DE.UNASSIGNED)
            if ok.sum() == 0:
                continue
            ct = contingency(tsing.loc[common[ok], "ind"].values, pred[ok].values)
            tr = np.unique(tsing.loc[common[ok], "ind"].values.astype(str))
            pr = np.unique(pred[ok].values.astype(str))
            mapping = {pr[j]: tr[i] for i, j in hungarian(ct)}
            gts = tool_donor_gts(tool, path)
            for cl, td in mapping.items():
                if td not in tg or cl not in gts:
                    continue
                tkeys, tgt = tg[td]
                ukeys, ugt, udp = gts[cl]
                tool_idx = {k: i for i, k in enumerate(ukeys)}
                sel = np.array([tool_idx.get(k, -1) for k in tkeys])
                have = sel >= 0
                tool_gt = np.full(len(tkeys), -1, dtype=np.int8); tool_dp = np.zeros(len(tkeys), dtype=np.int64)
                tool_gt[have] = ugt[sel[have]]; tool_dp[have] = udp[sel[have]]
                n_truth_conf = int((tgt >= 0).sum())
                for stratum, mask in (("all", None), ("dp10", tool_dp >= 10)):
                    c = G.confusion(tgt, tool_gt, mask)
                    rows.append(dict(pool=pool, tool=tool, donor=td, cluster=cl, stratum=stratum,
                                     n_truth_conf=n_truth_conf, n_truth_nonref=int((tgt > 0).sum()),
                                     **{f"t{t}_c{u}": int(c[t, u]) for t in range(3) for u in range(3)}))
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(pd.DataFrame(rows).groupby(["pool", "tool", "stratum"])[["n_truth_conf"] + [f"t{t}_c{t}" for t in range(3)]].sum())


if __name__ == "__main__":
    main()
