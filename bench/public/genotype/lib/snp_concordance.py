"""PRIVATE (runs on Clipper next to the genotype files). Per-donor genotype concordance of Singlet's, vireo's, souporcell's
and a plain pileup caller's donor genotypes against truth. Writes ONLY counts: a 3x3 genotype-class table per
(pool, tool, donor, stratum). No site, allele count or genotype is written anywhere.

Usage: snp_concordance.py RUNS TRUTH_DIR OUT.tsv POOL [POOL ...]
  RUNS      $WORK/bench-public/runs  (RUNS/<pool>/{singlet_auto,vireo,souporcell}/...)

Truth, two kinds (both real):
 * Kang lanes: for truth donor d of pool P, the pseudobulk of d's demuxlet-singlet cells in the REPLICATE lane (another
   library of the same donors; cellsnp-lite counts at the shared 1000G panel), genotyped with a fixed binomial model
   (genotypes.genotype_from_counts: error 0.01, flat prior, posterior >= 0.99 = confident).
 * 1000 Genomes individuals (lcl_hg00362, lcl_hg01277: single-donor libraries of HG00362 / HG01277; osorio2019_lcl_mix:
   GM12878 + GM18502 = NA12878 + NA18502): the genotypes of the 1000 Genomes high-coverage callset, read from
   $WORK/bench-public/private/truth_gt (private, built by fetch_1kg_truth.sbatch).
Strata: `all` = every site where the tool emits a call and the truth is available; `dp10` = those where the tool's own
donor-level depth is >= 10. Clusters are matched to truth donors by cell overlap (Kang) or by genotype agreement (1000G).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "capabilities"))
import pools as P  # noqa: E402
import genotypes as G  # noqa: E402
import demux_eval as DE  # noqa: E402
from _gcommon import contingency, hungarian  # noqa: E402

KG_POOLS = {"lcl_hg00362": ["HG00362"], "lcl_hg01277": ["HG01277"], "osorio2019_lcl_mix": ["NA12878", "NA18502"]}
_GT = {"0|0": 0, "0/0": 0, "0|1": 1, "1|0": 1, "0/1": 1, "1/0": 1, "1|1": 2, "1/1": 2}


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
            if (t["ind"] == d).sum() >= DE.MIN_TRUTH_DONOR_CELLS:
                out[d] = (keys, G.genotype_from_counts(a, n))
    return out


def kg_truth(runs: Path, pool: str):
    """{individual: (site keys, GT array)} from the private 1000 Genomes table (never written out)."""
    d = runs.parent / "private" / "truth_gt"
    t = pd.concat([pd.read_csv(f, sep="\t", dtype=str) for f in sorted(d.glob("chr*.tsv.gz")) if f.stat().st_size > 200])  # skip chromosomes whose extraction came out empty
    keys = (t["chrom"] + ":" + t["pos"] + ":" + t["ref"] + ":" + t["alt"]).tolist()
    return {ind: (keys, t[ind].map(_GT).fillna(-1).astype(np.int8).values) for ind in KG_POOLS[pool]}


def tool_labels_and_vcfs(runs: Path, pool: str):
    """tool name -> (labels DataFrame or None, vcf path)."""
    d = P.rundir(runs, pool)
    res = {}
    s = d / os.environ.get("SINGLET_RUN", "singlet_auto")  # SINGLET_RUN: score a variant run dir (genotype-sota)
    if (s / "private" / "donor").exists():
        res["singlet"] = (DE.read_singlet(s), s / "private" / "donor")
    v = d / "vireo" / "told_s1"
    if (v / "GT_donors.vireo.vcf.gz").exists():
        res["vireo"] = (DE.read_vireo(v), v / "GT_donors.vireo.vcf.gz")
    so = d / "souporcell" / "out"
    if (so / "cluster_genotypes.vcf").exists():
        res["souporcell"] = (DE.read_souporcell(so), so / "cluster_genotypes.vcf")
    return res


def tool_donor_gts(tool: str, path: Path):
    """cluster label -> (site key list, GT array, DP array)."""
    if tool == "singlet":
        out = {}
        post = float(os.environ.get("SINGLET_POST", "0"))   # re-genotype the aggregated AD,DP at a stricter posterior (genotype-sota)
        for f in sorted(path.glob("donor*.vcf")):
            keys, gts, dps = G.read_donor_vcf(f)
            if post:
                alt = []
                for line in open(f):
                    if line[0] != "#":
                        c = line.split("	")[9].split(":")[2].split(","); alt.append(int(c[1]))
                gts[0] = G.genotype_from_counts(np.array(alt), dps[0], post=post)
            out[f.stem] = (keys, gts[0], dps[0])
        return out
    keys, gts, dps = G.read_donor_vcf(path)
    names = [f"donor{j}" for j in range(len(gts))] if tool == "vireo" else [f"cluster{j}" for j in range(len(gts))]
    return {n: (keys, gts[j], dps[j]) for j, n in enumerate(names)}


def pileup_tool(runs: Path, pool: str, labels: pd.Series | None):
    """cellsnp-lite pseudobulk genotyped with the same fixed binomial model as the truth (a plain pileup caller)."""
    sites, ad, dp, bcs = G.read_cellsnp(P.rundir(runs, pool) / "vireo" / "cellsnp")
    keys = [G.site_key(*r) for r in sites[["chrom", "pos", "ref", "alt"]].itertuples(index=False)]
    lab = {G.norm_bc(b): "all" for b in bcs} if labels is None else {b: l for b, l in labels.items() if l not in DE.UNASSIGNED}
    pb = G.pseudobulk(ad, dp, bcs, lab)
    return {cl: (keys, G.genotype_from_counts(a, n), n.astype(np.int64)) for cl, (a, n) in pb.items()}


PAIRED = {}   # (pool, donor) -> {tool: (truth GT, tool GT, tool DP)} for the paired-site stratum (genotype-sota)


def compare(rows, pool, tool, cl, name, tg, ukeys, ugt, udp):
    tkeys, tgt = tg
    tool_idx = {k: i for i, k in enumerate(ukeys)}
    sel = np.array([tool_idx.get(k, -1) for k in tkeys])
    have = sel >= 0
    tool_gt = np.full(len(tkeys), -1, dtype=np.int8); tool_dp = np.zeros(len(tkeys), dtype=np.int64)
    tool_gt[have] = ugt[sel[have]]; tool_dp[have] = udp[sel[have]]
    if rows is not None and tool != "m":
        PAIRED.setdefault((pool, name), {})[tool] = (tgt, tool_gt, tool_dp)
    for stratum, mask in (("all", None), ("dp10", tool_dp >= 10)):
        c = G.confusion(tgt, tool_gt, mask)
        rows.append(dict(pool=pool, tool=tool, donor=name, cluster=cl, stratum=stratum, n_truth_conf=int((tgt >= 0).sum()),
                         n_truth_nonref=int((tgt > 0).sum()),
                         **{f"t{t}_c{u}": int(c[t, u]) for t in range(3) for u in range(3)}))
    return G.confusion(tgt, tool_gt)


def match_by_genotype(pool, tg, gts):
    inds = list(tg); cls = list(gts)
    mat = np.zeros((len(cls), len(inds)))
    for i, cl in enumerate(cls):
        for j, ind in enumerate(inds):
            c = compare([], pool, "m", cl, ind, tg[ind], *gts[cl])
            mat[i, j] = c.trace() / max(c.sum(), 1)
    r, c_ = linear_sum_assignment(-mat)
    return {cls[i]: inds[j] for i, j in zip(r, c_)}


def match_by_cells(T, lab):
    tsing = T[T["multiplets"] == "singlet"]
    n_by = tsing["ind"].map(tsing["ind"].value_counts())
    tsing = tsing[n_by >= DE.MIN_TRUTH_DONOR_CELLS]
    common = tsing.index.intersection(lab.index)
    pred = lab.loc[common]
    ok = ~pred.isin(DE.UNASSIGNED)
    if ok.sum() == 0:
        return {}
    ct = contingency(tsing.loc[common[ok], "ind"].values, pred[ok].values)
    tr = np.unique(tsing.loc[common[ok], "ind"].values.astype(str)); pr = np.unique(pred[ok].values.astype(str))
    return {pr[j]: tr[i] for i, j in hungarian(ct)}


def main():
    runs, truth_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = []
    for pool in sys.argv[4:]:
        kg = pool in KG_POOLS
        T = None if kg else P.load_truth(truth_dir, pool)
        tg = kg_truth(runs, pool) if kg else truth_genotypes(runs, truth_dir, pool)
        tools = tool_labels_and_vcfs(runs, pool)
        for tool, (lab, path) in tools.items():
            gts = tool_donor_gts(tool, path)
            mapping = match_by_genotype(pool, tg, gts) if kg else match_by_cells(T, lab["label"])
            for cl, td in mapping.items():
                if td in tg and cl in gts:
                    compare(rows, pool, tool, cl, td, tg[td], *gts[cl])
        # plain pileup baseline, clustered like the competitor pipeline (vireo labels; all cells when the pool has one donor)
        if (P.rundir(runs, pool) / "vireo" / "cellsnp").exists():
            vlab = tools["vireo"][0]["label"] if "vireo" in tools else None
            single = kg and len(KG_POOLS[pool]) == 1
            if single or vlab is not None:
                gts = pileup_tool(runs, pool, None if single else vlab)
                mapping = match_by_genotype(pool, tg, gts) if kg else match_by_cells(T, vlab)
                for cl, td in mapping.items():
                    if td in tg and cl in gts:
                        compare(rows, pool, "cellsnp-pileup", cl, td, tg[td], *gts[cl])
        print("snps", pool, flush=True)
    # paired10: only truth sites where every tool has DP >= 10 for the matched donor, so tools are scored on the same sites
    for (pool, name), d in PAIRED.items():
        if len(d) < 2:
            continue
        mask = np.logical_and.reduce([dp >= 10 for (_, _, dp) in d.values()])
        for tool, (tgt, tgt_tool, _) in d.items():
            c = G.confusion(tgt, tgt_tool, mask)
            rows.append(dict(pool=pool, tool=tool, donor=name, cluster="", stratum="paired10", n_truth_conf=int((tgt >= 0).sum()),
                             n_truth_nonref=int((tgt > 0).sum()), **{f"t{t}_c{u}": int(c[t, u]) for t in range(3) for u in range(3)}))
    g = pd.DataFrame(rows)
    g.to_csv(out, sep="\t", index=False)
    print(g.groupby(["pool", "tool", "stratum"])[["n_truth_conf"] + [f"t{t}_c{t}" for t in range(3)]].sum())


if __name__ == "__main__":
    main()
