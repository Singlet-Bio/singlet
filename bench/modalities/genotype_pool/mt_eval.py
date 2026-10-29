"""mtDNA evaluation. Usage: mt_eval.py TRUTH SINGLET_DIR PILEUP.npz OUT.json
A) per-cell VAF fidelity: Singlet heteroplasmy entries vs the pysam read-level pileup (same filters), using Singlet's own
   definition of ref (pool-majority base) and VAF (best non-ref allele / depth).
B) variant calling: Singlet mt_variants.tsv vs variants supported in the pseudobulk pileup.
C) donors: haplotype variants per truth donor (pseudobulk AF>=0.9, depth>=20 vs rCRS) vs Singlet per-donor mt VCFs, after
   mapping Singlet donors to truth donors by their cell assignments."""
import sys, os, json, gzip, re, itertools
import numpy as np, pandas as pd, scipy.io
from common import *

truth_f, sdir, npz, out = sys.argv[1:5]
T = load_truth(truth_f)
z = np.load(npz, allow_pickle=True)
arr = z["arr"]
cells = [norm_bc(c) for c in z["cells"]]


def opn(p):
    with open(p, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    return gzip.open(p, "rt") if gz else open(p)


res = {}
rcrs = "".join(l.strip() for l in open(os.path.join(sdir, "mt_reference.fa")) if not l.startswith(">")).upper()
ref = pd.DataFrame(arr, columns=["ci", "pos0", "A", "C", "G", "T"])
ref["cov"] = ref[list("ACGT")].sum(axis=1)
ref["bc"] = [cells[i] for i in ref.ci]
res["pileup_reads_used"] = int(z["n_reads"])
res["cells_with_chrM"] = int(ref.ci.nunique())
res["cells_total"] = len(cells)
cell_cov = ref.groupby("bc")["cov"].sum()
res["median_chrM_bases_per_cell"] = float(cell_cov.median())
glob_c = ref.groupby("pos0")[list("ACGT")].sum()
maj = glob_c.idxmax(axis=1)
res["rcrs_vs_pool_majority_mismatch_positions"] = int(sum(1 for p, b in maj.items() if rcrs[p] != b and glob_c.loc[p].sum() >= 100))

# ---------- A
V = pd.read_csv(os.path.join(sdir, "mt_variants.tsv"), sep="\t")
res["singlet_n_variants"] = int(len(V))
M = scipy.io.mmread(opn(os.path.join(sdir, "mt_heteroplasmy.mtx"))).tocsr()
feats = [l.strip() for l in opn(os.path.join(sdir, "mt_heteroplasmy_features.tsv"))]
bcs = [norm_bc(l.strip()) for l in opn(os.path.join(sdir, "mt_heteroplasmy_barcodes.tsv"))]
res["het_matrix_shape"] = list(M.shape)
res["het_matrix_nnz"] = int(M.nnz)
res["het_feature_examples"] = feats[:3]
coo = M.tocoo()
sv = pd.DataFrame({"feat": [feats[i] for i in coo.row], "bc": [bcs[j] for j in coo.col], "vaf_singlet": coo.data})
sv["pos0"] = [int(re.findall(r"(\d+)", f)[0]) - 1 for f in sv.feat]
m = sv.merge(ref[["bc", "pos0", "A", "C", "G", "T", "cov"]], on=["bc", "pos0"], how="left")


def vaf_ref(r):
    if np.isnan(r["cov"]) or r["cov"] == 0:
        return np.nan
    non = [r[b] for b in "ACGT" if b != maj[r["pos0"]]]
    return max(non) / r["cov"]


m["vaf_ref"] = m.apply(vaf_ref, axis=1)
ok = m.dropna(subset=["vaf_ref"])
res["A_entries"] = int(len(m))
res["A_entries_with_pileup"] = int(len(ok))
if len(ok) > 2:
    d = (ok.vaf_singlet - ok.vaf_ref).abs()
    res["A_vaf_pearson"] = float(np.corrcoef(ok.vaf_singlet, ok.vaf_ref)[0, 1])
    res["A_vaf_mean_abs_diff"] = float(d.mean())
    res["A_frac_within_0.01"] = float((d < 0.01).mean())

# ---------- B
V["pos0"] = V.pos - 1


def supported(min_alt=3, min_af=0.01):
    s = set()
    for p, row in glob_c.iterrows():
        dp = row.sum()
        if dp < 10:
            continue
        for b in "ACGT":
            if b != rcrs[p] and row[b] >= min_alt and row[b] / dp >= min_af:
                s.add((p, b))
    return s


pool_sup = supported()
sing_set = set(zip(V.pos0, V.alt))
res["B_singlet_variants"] = len(sing_set)
res["B_pool_pileup_variants_alt3_af1pct"] = len(pool_sup)
res["B_singlet_variants_supported_by_pileup"] = len(sing_set & pool_sup)
res["B_singlet_precision_vs_pileup"] = len(sing_set & pool_sup) / max(len(sing_set), 1)
strong = supported(10, 0.05)
res["B_strong_pileup_variants_alt10_af5pct"] = len(strong)
res["B_strong_recall_by_singlet"] = len(strong & sing_set) / max(len(strong), 1)
res["B_singlet_variants_in_strong"] = len(sing_set & strong)

# ---------- C
da = pd.read_csv(os.path.join(sdir, "donor", "donor_assignments.tsv"), sep="\t")
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
hap = {}
for dn, g in refc.groupby("ind"):
    s = g.groupby("pos0")[list("ACGT")].sum()
    hs = set()
    depth = set()
    for p, row in s.iterrows():
        dp = row.sum()
        if dp < 20:
            continue
        depth.add(p)
        for b in "ACGT":
            if b != rcrs[p] and row[b] / dp >= 0.9:
                hs.add((p, b))
    hap[dn] = (hs, depth)
res["C_truth_donor_haplotype_sizes"] = {d: len(h[0]) for d, h in hap.items()}
res["C_truth_donor_covered_positions"] = {d: len(h[1]) for d, h in hap.items()}


def read_vcf(p):
    s = set()
    for l in open(p):
        if l[0] == "#":
            continue
        f = l.split("\t")
        s.add((int(f[1]) - 1, f[4].strip()))
    return s


rows = []
for sl, td in mp.items():
    vf = os.path.join(sdir, "mt", sl + "_mt_variants.vcf")
    if not os.path.exists(vf) or td not in hap:
        continue
    S = read_vcf(vf)
    Rh, cov = hap[td]
    Sc = {x for x in S if x[0] in cov}
    rows.append(dict(singlet_donor=sl, truth_donor=td, n_singlet=len(S), n_singlet_in_covered=len(Sc), n_ref_haplotype=len(Rh),
                     tp=len(Sc & Rh), precision=len(Sc & Rh) / max(len(Sc), 1), recall=len(Sc & Rh) / max(len(Rh), 1)))
res["C_per_donor"] = rows
pairs = list(itertools.combinations(sorted(hap), 2))
ref_sep = {f"{a}-{b}": len(hap[a][0] ^ hap[b][0]) for a, b in pairs}
res["C_ref_pairs_distinguishable"] = sum(v > 0 for v in ref_sep.values())
res["C_n_pairs"] = len(pairs)
inv = {v: k for k, v in mp.items()}
sd = {}
for td in hap:
    p = os.path.join(sdir, "mt", inv.get(td, "x") + "_mt_variants.vcf")
    if os.path.exists(p):
        sd[td] = read_vcf(p)
res["C_singlet_pairs_distinguishable"] = sum(len(sd[a] ^ sd[b]) > 0 for a, b in pairs if a in sd and b in sd)
agree = [(len(sd[a] ^ sd[b]) > 0) == (ref_sep[f"{a}-{b}"] > 0) for a, b in pairs if a in sd and b in sd]
res["C_pair_call_agreement"] = float(np.mean(agree)) if agree else None
res["C_ref_pair_diffs"] = ref_sep
json.dump(res, open(out, "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
