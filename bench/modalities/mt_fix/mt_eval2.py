"""mtDNA evaluation of the caller harness output (adapted from bench/modalities/genotype_pool/mt_eval.py).

Usage: mt_eval2.py TRUTH SINGLET_DIR REF_PILEUP.npz HARNESS_OUT CELLS_ALL.txt VCF_DIR OUT.json [--label NAME]

  SINGLET_DIR   the original pipeline output (mt_reference.fa, donor/donor_assignments.tsv)
  REF_PILEUP    ref_k8/mt_pileup.npz: independent pysam pileup, called cells, MAPQ>=20, BQ>=10 (no strand)
  HARNESS_OUT   output dir of mt_caller_bench (mt_variants.tsv, mt_heteroplasmy_entries.tsv, features)
  CELLS_ALL.txt barcode order of the harness matrices
  VCF_DIR       directory with donorN_mt_variants.vcf (old: HARNESS_OUT/mt, new: HARNESS_OUT/private/mt)

Metrics (same definitions and thresholds as the original, so numbers are comparable):
  A  per-cell VAF fidelity vs the independent pileup
       A_orig:     original definition (best non-reference allele over depth, pool-majority reference)
       A_same_alt: VAF of the entry's own called alt allele (alt / depth) vs the pileup
  B  variant list vs the pooled pileup: precision (>=3 reads, >=1% AF), recall of strong variants (>=10 reads, >=5% AF),
     for the public list, and for the public list + haplotype-like sites (private/mt_sites.tsv) + donor VCF sites
  C  per-donor VCF precision / recall vs truth haplotypes (pseudobulk AF>=0.9, depth>=20 vs rCRS), covered positions only
  D  credibility of heteroplasmic sites: variants with >=50 het cells; donor concentration of carrier cells
"""
import sys, os, json, re, itertools, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "genotype_pool"))
from common import load_truth, norm_bc, best_match_acc

ap = argparse.ArgumentParser()
ap.add_argument("truth"); ap.add_argument("sdir"); ap.add_argument("npz"); ap.add_argument("hdir")
ap.add_argument("cells_all"); ap.add_argument("vcf_dir"); ap.add_argument("out")
ap.add_argument("--label", default="")
a = ap.parse_args()

T = load_truth(a.truth)
z = np.load(a.npz, allow_pickle=True)
arr = z["arr"]
cells = [norm_bc(c) for c in z["cells"]]
all_cells = [norm_bc(l.strip()) for l in open(a.cells_all) if l.strip()]
res = {"label": a.label}
rcrs = "".join(l.strip() for l in open(os.path.join(a.sdir, "mt_reference.fa")) if not l.startswith(">")).upper()

ref = pd.DataFrame(arr, columns=["ci", "pos0", "A", "C", "G", "T"])
ref["cov"] = ref[list("ACGT")].sum(axis=1)
ref["bc"] = [cells[i] for i in ref.ci]
glob_c = ref.groupby("pos0")[list("ACGT")].sum()
maj = glob_c.idxmax(axis=1)
res["pileup_reads_used"] = int(z["n_reads"])

V = pd.read_csv(os.path.join(a.hdir, "mt_variants.tsv"), sep="\t")
res["n_variants_public"] = int(len(V))
V["pos0"] = V.pos - 1
if "n_cells_het" in V:
    res["D_variants_ge50_het_cells"] = int((V.n_cells_het >= 50).sum())
    res["D_variants_ge50_het_cells_frac"] = float((V.n_cells_het >= 50).mean()) if len(V) else None

# ---------- entries
E = pd.read_csv(os.path.join(a.hdir, "mt_heteroplasmy_entries.tsv"), sep="\t")
feats = [l.strip() for l in open(os.path.join(a.hdir, "mt_heteroplasmy_features.tsv"))]
E["feat"] = [feats[i] for i in E.variant]
E["bc"] = [all_cells[i] for i in E.cell]
E["pos0"] = [int(re.findall(r"(\d+)", f)[0]) - 1 for f in E.feat]
E["alt"] = [f.split(">")[1] for f in E.feat]
res["het_matrix_nnz"] = int(len(E))
m = E.merge(ref[["bc", "pos0", "A", "C", "G", "T", "cov"]], on=["bc", "pos0"], how="left")


def vaf_orig(r):
    if np.isnan(r["cov"]) or r["cov"] == 0:
        return np.nan
    return max(r[b] for b in "ACGT" if b != maj[r["pos0"]]) / r["cov"]


def vaf_alt(r):
    if np.isnan(r["cov"]) or r["cov"] == 0:
        return np.nan
    return r[r["alt"]] / r["cov"]


# vectorised same-alt VAF
mm = m.dropna(subset=["cov"]).copy()
mm = mm[mm["cov"] > 0]
altc = np.zeros(len(mm))
for b in "ACGT":
    sel = (mm.alt == b).values
    altc[sel] = mm.loc[sel, b].values
mm["vaf_alt"] = altc / mm["cov"].values
majv = mm.pos0.map(maj).values
non = np.zeros(len(mm))
for b in "ACGT":
    cnt = mm[b].values.astype(float)
    cnt[majv == b] = 0
    non = np.maximum(non, cnt)
mm["vaf_orig"] = non / mm["cov"].values
res["A_entries"] = int(len(m)); res["A_entries_with_pileup"] = int(len(mm))
for name in ("orig", "same_alt"):
    col = mm["vaf_orig"] if name == "orig" else mm["vaf_alt"]
    if len(mm) > 2:
        d = (mm.vaf - col).abs()
        res["A_%s_pearson" % name] = float(np.corrcoef(mm.vaf, col)[0, 1])
        res["A_%s_mean_abs_diff" % name] = float(d.mean())
        res["A_%s_frac_within_0.01" % name] = float((d < 0.01).mean())

# ---------- B
V["pos0"] = V.pos - 1


def supported(min_alt=3, min_af=0.01, min_dp=10):
    s = set()
    for p, row in glob_c.iterrows():
        dp = row.sum()
        if dp < min_dp:
            continue
        for b in "ACGT":
            if b != rcrs[p] and row[b] >= min_alt and row[b] / dp >= min_af:
                s.add((p, b))
    return s


def read_vcf(p):
    s = set()
    if not os.path.exists(p):
        return s
    for l in open(p):
        if l[0] == "#":
            continue
        f = l.split("\t")
        s.add((int(f[1]) - 1, f[4].strip()))
    return s


pool_sup = supported()
strong = supported(10, 0.05)
pub = set(zip(V.pos0, V.alt))
res["B_pool_pileup_variants_alt3_af1pct"] = len(pool_sup)
res["B_strong_pileup_variants_alt10_af5pct"] = len(strong)
res["B_public_variants"] = len(pub)
res["B_public_supported_by_pileup"] = len(pub & pool_sup)
res["B_public_precision_vs_pileup"] = len(pub & pool_sup) / max(len(pub), 1)
res["B_public_supported_alt3_any_af"] = len(pub & supported(3, 0.0))
res["B_public_in_strong"] = len(pub & strong)
res["B_strong_recall_public"] = len(pub & strong) / max(len(strong), 1)
# union with private haplotype-like sites and donor VCF variants (new layout only)
hap_sites = set()
sp = os.path.join(a.hdir, "private", "mt_sites.tsv")
if os.path.exists(sp):
    S = pd.read_csv(sp, sep="\t")
    res["B_sites_evaluated"] = int(len(S))
    res["B_sites_class_counts"] = S["class"].value_counts().to_dict()
    hap_sites = set(zip(S[S["class"] == "haplotype"].pos - 1, S[S["class"] == "haplotype"].alt))
donor_sites = set()
for k in range(32):
    donor_sites |= read_vcf(os.path.join(a.vcf_dir, "donor%d_mt_variants.vcf" % k))
union = pub | hap_sites | donor_sites
res["B_union_variants"] = len(union)
res["B_union_precision_vs_pileup"] = len(union & pool_sup) / max(len(union), 1)
res["B_union_in_strong"] = len(union & strong)
res["B_strong_recall_union"] = len(union & strong) / max(len(strong), 1)
res["B_haplotype_like_sites"] = len(hap_sites)

# ---------- C
da = pd.read_csv(os.path.join(a.sdir, "donor", "donor_assignments.tsv"), sep="\t")
da = da[da.cell_status == "cell"]
da["bc"] = da.cell.map(norm_bc)
da = da.set_index("bc")
com = da.index.intersection(T.index)
tt = T.loc[com]
sing = tt.multiplets == "singlet"
acc, mp = best_match_acc(tt.ind[sing], da.loc[com, "donor_id"][sing])
refc = ref.copy()
refc["ind"] = refc.bc.map(T.ind)
refc = refc[refc.bc.map(T.multiplets) == "singlet"].dropna(subset=["ind"])
hap, hap75 = {}, {}
for dn, g in refc.groupby("ind"):
    s = g.groupby("pos0")[list("ACGT")].sum()
    hs, depth, hs75 = set(), set(), set()
    for p, row in s.iterrows():
        dp = row.sum()
        if dp < 20:
            continue
        depth.add(p)
        for b in "ACGT":
            if b != rcrs[p] and row[b] / dp >= 0.9:
                hs.add((p, b))
            if b != rcrs[p] and row[b] / dp >= 0.75:
                hs75.add((p, b))
    hap[dn] = (hs, depth)
    hap75[dn] = hs75
res["C_truth_donor_haplotype_sizes"] = {d: len(h[0]) for d, h in hap.items()}
rows = []
for sl, td in mp.items():
    vf = os.path.join(a.vcf_dir, sl + "_mt_variants.vcf")
    if not os.path.exists(vf) or td not in hap:
        continue
    S_ = read_vcf(vf)
    Rh, cov = hap[td]
    Sc = {x for x in S_ if x[0] in cov}
    rows.append(dict(singlet_donor=sl, truth_donor=td, n_singlet=len(S_), n_singlet_in_covered=len(Sc), n_ref_haplotype=len(Rh),
                     tp=len(Sc & Rh), precision=len(Sc & Rh) / max(len(Sc), 1), recall=len(Sc & Rh) / max(len(Rh), 1),
                     precision_vs_af75=len(Sc & hap75[td]) / max(len(Sc), 1), recall_vs_af75=len(Sc & hap75[td]) / max(len({x for x in hap75[td] if x[0] in cov}), 1),
                     missed=sorted("%d%s" % (p_ + 1, b_) for p_, b_ in Rh - Sc),
                     false_pos=sorted("%d%s" % (p_ + 1, b_) for p_, b_ in Sc - Rh)))
dbg_f = os.path.join(a.hdir, "private", "hap_debug.tsv")
if os.path.exists(dbg_f):
    dbg = pd.read_csv(dbg_f, sep="\t")
    for r_ in rows:
        k = int(r_["singlet_donor"].replace("donor", ""))
        reasons = {}
        for m_ in r_["missed"]:
            pos_ = int(m_[:-1])
            q = dbg[(dbg.donor == k) & (dbg.pos == pos_)]
            reasons[m_] = (q.iloc[0].result + ("(masked)" if q.iloc[0].masked else "") + " af=%.2f dp=%d fwd/rev=%d/%d prof=%d/%d" %
                           (q.iloc[0].af, q.iloc[0].dp, q.iloc[0].alt_fwd, q.iloc[0].alt_rev, q.iloc[0].prof_fwd,
                            q.iloc[0].prof_rev)) if len(q) else "not_in_donor_at_af>=0.5_dp>=10"
        r_["missed_reasons"] = reasons
res["C_per_donor"] = rows
if rows:
    res["C_mean_precision"] = float(np.mean([r["precision"] for r in rows]))
    res["C_mean_recall"] = float(np.mean([r["recall"] for r in rows]))
    res["C_mean_precision_vs_truth_af75"] = float(np.mean([r["precision_vs_af75"] for r in rows]))
    res["C_mean_recall_vs_truth_af75"] = float(np.mean([r["recall_vs_af75"] for r in rows]))
pairs = list(itertools.combinations(sorted(hap), 2))
ref_sep = {"%s-%s" % (x, y): len(hap[x][0] ^ hap[y][0]) for x, y in pairs}
inv = {v: k for k, v in mp.items()}
sd = {}
for td in hap:
    p = os.path.join(a.vcf_dir, inv.get(td, "x") + "_mt_variants.vcf")
    if os.path.exists(p):
        sd[td] = read_vcf(p)
res["C_ref_pairs_distinguishable"] = sum(v > 0 for v in ref_sep.values())
res["C_singlet_pairs_distinguishable"] = sum(len(sd[x] ^ sd[y]) > 0 for x, y in pairs if x in sd and y in sd)
agree = [(len(sd[x] ^ sd[y]) > 0) == (ref_sep["%s-%s" % (x, y)] > 0) for x, y in pairs if x in sd and y in sd]
res["C_pair_call_agreement"] = float(np.mean(agree)) if agree else None
res["C_n_pairs"] = len(pairs)

# ---------- D: do carrier cells of a heteroplasmic site concentrate in one donor?
E["ind"] = E.bc.map(T.ind)
E = E.dropna(subset=["ind"])
shares = []
for v, g in E.groupby("variant"):
    if len(g) >= 5:
        shares.append(g.ind.value_counts(normalize=True).iloc[0])
if shares:
    res["D_variants_with_ge5_carriers"] = len(shares)
    res["D_median_top_donor_share"] = float(np.median(shares))
    res["D_frac_variants_top_donor_ge_0.8"] = float(np.mean(np.array(shares) >= 0.8))
hd = os.path.join(a.vcf_dir, "het_site_donor_af.tsv")
if os.path.exists(hd):
    H = pd.read_csv(hd, sep="\t")
    res["D_donor_af_table_sites"] = int(len(H))
    res["D_donor_specific_sites"] = int(H.donor_specific.sum())
    res["D_max_top_af"] = float(H.top_af.max()) if len(H) else None
    res["D_sites_top_af_ge_0.05"] = int((H.top_af >= 0.05).sum())
res["D_expected_top_share_if_random"] = float(T.ind[T.multiplets == "singlet"].value_counts(normalize=True).iloc[0])
json.dump(res, open(a.out, "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
