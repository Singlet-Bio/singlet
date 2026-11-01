"""PRIVATE (runs on Clipper next to the chrM allele counts). mtDNA benchmark evaluations that write COUNTS ONLY: no site,
allele count or haplotype leaves this machine.

Usage: mt_eval.py RUNS KANG_TRUTH_DIR OUT_DIR

Three analyses, all on real data with real truth:

 haplotype  Kang lanes. Truth haplotype of a donor = homoplasmic alt alleles (AF >= 0.9, depth >= 20, against rCRS) in the
            donor's pseudobulk in the REPLICATE lane (another library of the same donor, cells labelled by demuxlet),
            counted by mgatk. Predictions: Singlet's per-donor mt VCFs (genotype-free clusters), and mgatk counts pooled by
            vireo's clusters (the pipeline a user would run). Output: per donor TP / predicted / truth counts.
 voi        MAESTER BPDCN712. 26 clone-defining mt variants with author-validated positive cells (from the paired
            MAESTER library of the same cells). Per variant, AUROC of each tool's per-cell VAF from the standard 3' library
            for truth-positive vs the other truth cells; Singlet's own variant list and per-cell calls are scored too.
 linemix    MAESTER K562 + BT142 mixture. Variants with high-coverage per-line allele fractions from MAESTER; agreement
            of each tool's 3' per-line AF with them (sufficient statistics only).
"""
from __future__ import annotations

import gzip
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "capabilities"))
import pools as P  # noqa: E402
import demux_eval as DE  # noqa: E402
from _gcommon import contingency, hungarian  # noqa: E402

L = 16569
BASES = "ACGT"
MIN_DEPTH = 20
HOM_AF = 0.9


def nbc(b):
    return str(b).split("-")[0]


# ------------------------------------------------------------------------------------------------ count loaders
class Counts:
    """cells x (pos0*4 + base) sparse counts; cell barcodes without -1."""

    def __init__(self, cells, mat):
        self.cells = list(cells); self.idx = {c: i for i, c in enumerate(self.cells)}; self.m = mat.tocsr()

    def pseudobulk(self, cell_ids) -> np.ndarray:
        rows = [self.idx[c] for c in cell_ids if c in self.idx]
        if not rows:
            return np.zeros((L, 4))
        return np.asarray(self.m[rows].sum(axis=0)).reshape(L, 4)

    def cell_site(self, pos0: int):
        """(depth, per-base counts) per cell at a position: arrays n_cells x 4."""
        return np.asarray(self.m[:, pos0 * 4: pos0 * 4 + 4].todense())


def load_mgatk(run_dir: Path, pool: str) -> Counts:
    fin = run_dir / "mgatk" / "out" / "final"
    frames = []
    for bi, b in enumerate(BASES):
        d = pd.read_csv(fin / f"{pool_name(run_dir)}.{b}.txt.gz", header=None, names=["pos", "bc", "f", "r"])
        d["base"] = bi
        frames.append(d)
    d = pd.concat(frames)
    d["bc"] = d["bc"].map(nbc)
    cells = sorted(d["bc"].unique()); ci = {c: i for i, c in enumerate(cells)}
    col = (d["pos"].values - 1) * 4 + d["base"].values
    m = sp.coo_matrix(((d["f"] + d["r"]).values.astype(np.float64), (d["bc"].map(ci).values, col)), shape=(len(cells), L * 4))
    return Counts(cells, m)


def pool_name(run_dir: Path) -> str:
    return run_dir.name


def load_singlet_counts(sdir: Path, keep) -> Counts:
    """Singlet's raw chrM allele counts (private/mt_allele_counts) for the barcodes in `keep` (no -1)."""
    pv = sdir / "private"
    feats = [l.strip().split("	")[0] for l in gzip.open(pv / "mt_allele_counts_features.tsv.gz", "rt")]
    bcs = [nbc(l.strip().split("	")[0]) for l in gzip.open(pv / "mt_allele_counts_barcodes.tsv.gz", "rt")]
    col = np.empty(len(feats), dtype=np.int64)
    for i, f in enumerate(feats):
        _, pos, b = f.split(":")
        col[i] = (int(pos) - 1) * 4 + BASES.index(b)
    keep_idx = np.array([i for i, c in enumerate(bcs) if c in keep], dtype=np.int64)
    new_idx = np.full(len(bcs), -1, dtype=np.int64); new_idx[keep_idx] = np.arange(len(keep_idx))
    parts = []
    with gzip.open(pv / "mt_allele_counts.mtx.gz", "rt") as fh:
        for line in fh:
            if not line.startswith("%"):
                break
        for chunk in pd.read_csv(fh, sep=" ", header=None, names=["f", "c", "v"], chunksize=20_000_000):
            c = new_idx[chunk["c"].values - 1]
            m = c >= 0
            parts.append((c[m], col[chunk["f"].values[m] - 1], chunk["v"].values[m]))
    r = np.concatenate([x[0] for x in parts]); cc = np.concatenate([x[1] for x in parts]); v = np.concatenate([x[2] for x in parts])
    m = sp.coo_matrix((v.astype(np.float64), (r, cc)), shape=(len(keep_idx), L * 4))
    return Counts([bcs[i] for i in keep_idx], m)


def read_ref(sdir: Path) -> str:
    return "".join(l.strip() for l in open(sdir / "mt_reference.fa") if not l.startswith(">")).upper()


def haplotype(pb: np.ndarray, ref: str):
    """(set of (pos0, alt) homoplasmic vs ref, set of covered pos0) from a pseudobulk."""
    depth = pb.sum(axis=1)
    cov = set(np.where(depth >= MIN_DEPTH)[0].tolist())
    hs = set()
    for p in cov:
        for b in range(4):
            if BASES[b] != ref[p] and pb[p, b] / depth[p] >= HOM_AF:
                hs.add((p, b))
    return hs, cov


def read_vcf_alts(path: Path):
    s = set()
    if path.exists():
        for l in open(path):
            if l[0] == "#":
                continue
            f = l.split("\t")
            s.add((int(f[1]) - 1, BASES.index(f[4].strip()[0])))
    return s


def cluster_map(truth: pd.DataFrame, lab: pd.Series):
    tsing = truth[truth["multiplets"] == "singlet"]
    n_by = tsing["ind"].map(tsing["ind"].value_counts())
    tsing = tsing[n_by >= DE.MIN_TRUTH_DONOR_CELLS]
    common = tsing.index.intersection(lab.index)
    pred = lab.loc[common]
    ok = ~pred.isin(DE.UNASSIGNED)
    ct = contingency(tsing.loc[common[ok], "ind"].values, pred[ok].values)
    tr = np.unique(tsing.loc[common[ok], "ind"].values.astype(str)); pr = np.unique(pred[ok].values.astype(str))
    return {pr[j]: tr[i] for i, j in hungarian(ct)}


# ------------------------------------------------------------------------------------------------ haplotype (Kang)
def run_haplotype(runs: Path, truth_dir: Path, out: Path):
    rows, pairs = [], []
    mg_cache = {}

    def mg(pool):
        if pool not in mg_cache:
            mg_cache[pool] = load_mgatk(P.rundir(runs, pool), pool)
        return mg_cache[pool]

    for pool in P.KANG:
        rd = P.rundir(runs, pool)
        if not (rd / "mgatk" / "out" / "final").exists() or not (rd / "singlet_auto" / "private" / "mt").exists():
            continue
        ref = read_ref(rd / "singlet_auto")
        T = P.load_truth(truth_dir, pool)
        # truth haplotypes from replicate lanes
        truth_h = {}
        for rep in P.REPLICATES[pool]:
            Tr = P.load_truth(truth_dir, rep)
            cm = mg(rep)
            for d, g in Tr[Tr["multiplets"] == "singlet"].groupby("ind"):
                if d == "<NA>" or len(g) < DE.MIN_TRUTH_DONOR_CELLS:
                    continue
                truth_h[d] = haplotype(cm.pseudobulk(g.index.tolist()), ref)
        tools = {}
        lab_s = DE.read_singlet(rd / "singlet_auto")["label"]
        tools["singlet"] = (lab_s, lambda cl: read_vcf_alts(rd / "singlet_auto" / "private" / "mt" / f"{cl}_mt_variants.vcf"))
        vdir = rd / "vireo" / "told_s1"
        if (vdir / "donor_ids.tsv").exists():
            lab_v = DE.read_vireo(vdir)["label"]
            cm = mg(pool)
            vh = {}
            for cl, g in lab_v[~lab_v.isin(DE.UNASSIGNED)].groupby(lab_v):
                vh[cl] = haplotype(cm.pseudobulk(g.index.tolist()), ref)[0]
            tools["mgatk+vireo"] = (lab_v, lambda cl, vh=vh: vh.get(cl, set()))
        for tool, (lab, getter) in tools.items():
            mp = cluster_map(T, lab)
            pred = {}
            for cl, td in mp.items():
                if td not in truth_h:
                    continue
                Hh, cov = truth_h[td]
                S = getter(cl)
                Sc = {x for x in S if x[0] in cov}
                pred[td] = S
                rows.append(dict(pool=pool, tool=tool, donor=td, n_pred=len(S), n_pred_covered=len(Sc), n_truth=len(Hh),
                                 tp=len(Sc & Hh)))
            ds = sorted(pred)
            for a, b in itertools.combinations(ds, 2):
                tsep = len(truth_h[a][0] ^ truth_h[b][0]) > 0
                psep = len(pred[a] ^ pred[b]) > 0
                pairs.append(dict(pool=pool, tool=tool, truth_separable=int(tsep), tool_separable=int(psep), agree=int(tsep == psep)))
        print("haplotype", pool, "done", flush=True)
    pd.DataFrame(rows).to_csv(out / "mt_haplotype_counts.tsv", sep="\t", index=False)
    pd.DataFrame(pairs).to_csv(out / "mt_haplotype_pairs.tsv", sep="\t", index=False)


# ------------------------------------------------------------------------------------------------ voi (BPDCN712)
def auroc(pos_scores, neg_scores):
    from scipy.stats import rankdata
    x = np.concatenate([pos_scores, neg_scores]); r = rankdata(x)
    n1, n0 = len(pos_scores), len(neg_scores)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_voi(runs: Path, truth_dir: Path, out: Path):
    pool = "maester_bpdcn712_3p"
    rd = P.rundir(runs, pool)
    if not (rd / "mgatk" / "out" / "final").exists():
        return
    tdir = Path(truth_dir).parent / "maester_truth"
    pos = pd.read_csv(tdir / "bpdcn712_positive_cells.tsv", sep="\t")
    pos["bc"] = pos["cell"].map(nbc)
    pos["p0"] = pos["variant"].str.extract(r"^(\d+)_")[0].astype(int) - 1
    pos["alt"] = pos["variant"].str.extract(r">([ACGT])$")[0]
    universe = sorted(set(pos["bc"]))
    sc = load_singlet_counts(rd / "singlet_auto", set(universe))
    mgc = load_mgatk(rd, pool)
    sdir = rd / "singlet_auto"
    mv = pd.read_csv(sdir / "mt_variants.tsv", sep="\t")
    listed = set(zip(mv["pos"] - 1, mv["alt"]))
    # Singlet's per-cell heteroplasmy calls (public matrix)
    het = None
    hf = sdir / "mt_heteroplasmy.mtx.gz"
    if hf.exists():
        feats = [l.rstrip("\n").split("\t")[0] for l in gzip.open(sdir / "mt_heteroplasmy_features.tsv.gz", "rt")]
        hb = [nbc(l.strip().split("\t")[0]) for l in gzip.open(sdir / "mt_heteroplasmy_barcodes.tsv.gz", "rt")]
        hm = mmread(hf).tocsr()
        if hm.shape == (len(hb), len(feats)):
            hm = hm.T.tocsr()
        het = (feats, {c: i for i, c in enumerate(hb)}, hm, hb)
    rows = []
    for k, (v, g) in enumerate(pos.groupby("variant"), start=1):
        p0, alt = int(g["p0"].iloc[0]), "ACGT".index(g["alt"].iloc[0])
        posc = set(g["bc"])
        negc = [c for c in universe if c not in posc]
        r = dict(voi=f"voi{k:02d}", n_pos_truth=len(posc), n_neg_truth=len(negc), singlet_listed=int((p0, "ACGT"[alt]) in listed))
        scores = {}
        for tool, cnt in (("singlet", sc), ("mgatk", mgc)):
            site = cnt.cell_site(p0)
            depth = site.sum(axis=1)
            vaf = np.divide(site[:, alt], depth, out=np.zeros_like(depth, dtype=float), where=depth > 0)
            scores[tool] = (depth, vaf, cnt.idx)
        # cells with depth >= 2 in both tools
        okc = [c for c in universe if scores["singlet"][2].get(c) is not None and scores["mgatk"][2].get(c) is not None
               and scores["singlet"][0][scores["singlet"][2][c]] >= 2 and scores["mgatk"][0][scores["mgatk"][2][c]] >= 2]
        okp = [c for c in okc if c in posc]; okn = [c for c in okc if c not in posc]
        r["n_pos_eval"], r["n_neg_eval"] = len(okp), len(okn)
        for tool in ("singlet", "mgatk"):
            d, vf, ix = scores[tool]
            r[f"auroc_{tool}"] = auroc(vf[[ix[c] for c in okp]], vf[[ix[c] for c in okn]]) if len(okp) >= 5 and len(okn) >= 5 else np.nan
        # Singlet per-cell heteroplasmy calls at this site
        if het is not None:
            fi = [i for i, f in enumerate(het[0]) if f.startswith(f"chrM:{p0 + 1}:") and f.endswith(">" + "ACGT"[alt])]
            called = set()
            for i in fi:
                called |= {het[3][j] for j in het[2][i].nonzero()[1]}
            r["singlet_call_tp"] = len(called & set(okp)); r["singlet_call_fp"] = len(called & set(okn))
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "mt_voi_summary.tsv", sep="\t", index=False)
    print("voi done", len(rows), flush=True)


# ------------------------------------------------------------------------------------------------ line mix
def run_linemix(runs: Path, truth_dir: Path, out: Path):
    pool = "maester_k562_bt142_3p"
    rd = P.rundir(runs, pool)
    if not (rd / "mgatk" / "out" / "final").exists():
        return
    tdir = Path(truth_dir).parent / "maester_truth"
    T = P.load_truth(truth_dir, pool)
    V = pd.read_csv(tdir / "celllinemix_variants.tsv", sep="\t")
    V["p0"] = V["var"].str.extract(r"^(\d+)_")[0].astype(int) - 1
    V["alt"] = V["var"].str.extract(r">([ACGT])$")[0].map({b: i for i, b in enumerate(BASES)})
    sc = load_singlet_counts(rd / "singlet_auto", set(T.index)); mgc = load_mgatk(rd, pool)
    stats = []
    for line in ("K562", "BT142"):
        # cells every tool saw: truth cells of the line that are in both count tables
        cells = [c for c in T.index[T["ind"] == line] if c in sc.idx and c in mgc.idx]
        pbs = {"singlet": sc.pseudobulk(cells), "mgatk": mgc.pseudobulk(cells)}
        dsite = {t: b.sum(axis=1) for t, b in pbs.items()}
        col = f"mean_af.{line}"
        sel = V[V[col] >= 2.0]       # MAESTER reports mean_af in percent; keep variants with AF >= 2%
        keep = [(p0, int(alt), taf / 100.0) for p0, alt, taf in zip(sel["p0"], sel["alt"], sel[col])
                if dsite["singlet"][p0] >= 100 and dsite["mgatk"][p0] >= 100]
        for tool, pb in pbs.items():
            depth = pb.sum(axis=1)
            for blk in range(10):      # genome blocks of 1,700 positions: the unit of the bootstrap in the capability module
                it = [(p0, alt, taf) for p0, alt, taf in keep if p0 // 1700 == blk]
                x = np.array([t for _, _, t in it]); y = np.array([pb[p0, alt] / depth[p0] for p0, alt, _ in it])
                stats.append(dict(line=line, tool=tool, block=blk, n=len(x), sx=x.sum(), sy=y.sum(), sxx=(x * x).sum(),
                                  syy=(y * y).sum(), sxy=(x * y).sum(), sabs=np.abs(x - y).sum(),
                                  n_within_005=int((np.abs(x - y) <= 0.05).sum())))
    pd.DataFrame(stats).to_csv(out / "mt_linemix_stats.tsv", sep="\t", index=False)
    print("linemix done", flush=True)


def main():
    runs, truth_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    what = sys.argv[4:] or ["haplotype", "voi", "linemix"]
    if "haplotype" in what: run_haplotype(runs, truth_dir, out)
    if "voi" in what: run_voi(runs, truth_dir, out)
    if "linemix" in what: run_linemix(runs, truth_dir, out)


if __name__ == "__main__":
    main()
