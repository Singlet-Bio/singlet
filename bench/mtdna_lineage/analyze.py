"""mtDNA lineage benchmark: MAESTER (Miller et al. 2022) paired standard 10x 3' + MAESTER libraries of the same cells.

  python analyze.py lines  ROOT OUT.json     K562 / BT142 cell-line mix: line ID from homoplasmic variants (haplotype
                                             deconvolution), K562 subclonal variants, VAF fidelity, privacy split
  python analyze.py bpdcn  ROOT OUT.json     BPDCN712 bone marrow: 23 MAESTER clones (author calls) as truth; variant
                                             detection, clone calling, downsampling, phasing / deconvolution tests
  python analyze.py size   ROOT OUT.json     storage cost of the representations

ROOT = $WORK/out/mtdna_lineage (pile/<name>/mt.{read,umi1,umi3}.bin from mtpile, <name>/ from Singlet, truth/).
Molecule-level counts (umi1 for 3', umi3 for MAESTER as maegatk -mr 3) are the default unit.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtlib as M  # noqa: E402

W = "/mnt/projects/debruinz_project/debruinz/singlet"
TRUTH = os.environ.get("MTL_TRUTH", f"{W}/data/maester/truth")
MQUAD = os.environ.get("MTL_MQUAD", f"{W}/envs/mtl/bin/mquad")


import time as _time
_T0 = _time.time()


def log(msg):
    print(f"[{_time.time() - _T0:7.0f}s] {msg}", file=sys.stderr, flush=True)


def bc(s):
    return s.split("-")[0]


def load_cells(pdir):
    return [bc(x.strip()) for x in open(f"{pdir}/cells.txt") if x.strip()]


def load(root, name, level):
    pdir = f"{root}/pile/{name}"
    return M.Pile.load(f"{pdir}/mt.{level}.bin"), load_cells(pdir)


def reference(root):
    for p in glob.glob(f"{root}/*/mt_reference.fa") + glob.glob(f"{root}/*/private/mt_reference.fa"):
        return M.read_fasta(p)
    raise SystemExit("no mt_reference.fa")


def align(p, cells, want):
    """Subset pile p (columns = cells) to barcodes `want` (missing barcodes become empty cells)."""
    idx = {c: i for i, c in enumerate(cells)}
    present = [w for w in want if w in idx]
    sub = p.subset_cells(np.array([idx[w] for w in present], np.int64))
    if len(present) == len(want):
        return sub
    pos_new = {w: i for i, w in enumerate(want)}
    remap = np.array([pos_new[w] for w in present], np.uint32)
    return M.Pile(len(want), remap[sub.cell], sub.pos, sub.c)


def vname(ps, b, ref):
    return f"{ps + 1}{ref[ps]}>{M.BASES[b]}"


def parse_v(v):
    """'2593_G>A' or '2593G>A' -> (pos0, alt_idx)."""
    v = v.replace("_", "")
    i = 0
    while v[i].isdigit():
        i += 1
    return int(v[:i]) - 1, M.BASES.index(v[-1])


def coverage_summary(p):
    cov = p.cov_matrix()
    per_cell = np.asarray(cov.sum(1)).ravel() / M.L
    covered1 = np.asarray((cov >= 1).sum(1)).ravel() / M.L
    covered5 = np.asarray((cov >= 5).sum(1)).ravel() / M.L
    q = lambda x: {k: float(np.quantile(x, v)) for k, v in (("p10", .1), ("median", .5), ("p90", .9))} | {"mean": float(x.mean())}
    prof = np.asarray(cov.sum(0)).ravel() / max(p.n_cells, 1)
    bins = prof[: M.L // 100 * 100].reshape(-1, 100).mean(1)
    return dict(mean_depth_per_cell=q(per_cell), frac_positions_ge1=q(covered1), frac_positions_ge5=q(covered5),
                pooled_profile_100bp=[round(float(x), 3) for x in bins],
                frac_positions_mean_depth_ge1=float((prof >= 1).mean()), frac_positions_mean_depth_ge10=float((prof >= 10).mean()))


def run_mquad(AD, DP, names, out, minDP=5, extra=(), tenx_cutoff=10.0):
    """MQuad on cells x variants AD/DP; returns the passed variant names (ordered by MQuad's deltaBIC)."""
    os.makedirs(out, exist_ok=True)
    M.write_mtx_pair(AD, DP, f"{out}/in")
    cmd = [MQUAD, "-m", f"{out}/in.AD.mtx,{out}/in.DP.mtx", "-o", out, "-p", "8", "--minDP", str(minDP), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(f"{out}/mquad.log", "w").write(r.stdout + r.stderr)
    # MQuad's own selection step can crash when the knee locator finds no knee (it plots before checking), so the
    # deltaBIC table it writes first is read back and its selection rule (knee of the deltaBIC distribution, minor
    # component >= 2 cells) is applied here; tenx_cutoff is MQuad's fallback for low-depth data.
    f = f"{out}/BIC_params.csv" if os.path.exists(f"{out}/BIC_params.csv") else f"{out}/debug_unsorted_BIC_params.csv"
    if not os.path.exists(f):
        return [], None, None
    bic = pd.read_csv(f).sort_values("deltaBIC", ascending=False)
    cutoff = None
    try:
        from mquad.mquad_utils import findKnee
        cutoff = float(findKnee(bic.deltaBIC)[3])
    except Exception:
        pass
    if cutoff is None or not np.isfinite(cutoff):
        cutoff = tenx_cutoff
    ok = (bic.deltaBIC >= cutoff) & (bic.num_cells_minor_cpt >= 2)
    passed = [names[int(str(v)[3:]) - 1] for v in bic.variant_name[ok]]
    bic["variant"] = [names[int(str(v)[3:]) - 1] for v in bic.variant_name]
    return passed, bic, cutoff


def clone_calls(AD, DP, ks, n_init=50, seed=0):
    fits = M.clone_fit(AD, DP, ks=ks, n_init=n_init, seed=seed)
    elbo = {k: v[1] for k, v in fits.items()}
    # choose K by the elbow of the ELBO gain (vireo/MQuad practice): smallest K whose next gain < 2% of the total span
    ksort = sorted(elbo)
    span = max(elbo.values()) - min(elbo.values())
    best = ksort[-1]
    for a, b in zip(ksort, ksort[1:]):
        if elbo[b] - elbo[a] < 0.02 * span:
            best = a
            break
    return fits, elbo, best


# ---------------------------------------------------------------------------------------------------------------------
# K562 / BT142
# ---------------------------------------------------------------------------------------------------------------------

def lines(root, out):
    import pyreadr
    ref = reference(root)
    cells_tib = list(pyreadr.read_r(f"{TRUTH}/TenX_CellLineMix_Cells.rds").values())[0]
    truth = dict(zip(cells_tib.cell.map(bc), cells_tib.CellType_RNA.astype(str)))
    want = list(truth)
    res = {"n_truth_cells": len(want)}
    m3, c3 = load(root, "k562_3p", "umi1")
    r3, _ = load(root, "k562_3p", "read")
    mm, cm = load(root, "k562_maester", "umi3")
    P3, R3, PM = align(m3, c3, want), align(r3, c3, want), align(mm, cm, want)
    lab = np.array([truth[w] for w in want])
    log("coverage")
    res["coverage"] = {"3p_umi": coverage_summary(P3), "3p_read": coverage_summary(R3), "maester_umi3": coverage_summary(PM)}

    log("haplotype panel")
    # line haplotypes from MAESTER pseudobulk per RNA label (truth-informed reference panel)
    pbK, pbB = PM.pseudobulk(np.nonzero(lab == "K562")[0]), PM.pseudobulk(np.nonzero(lab == "BT142")[0])
    dK, dB = pbK.sum(1), pbB.sum(1)
    refidx = np.array([M.BASES.find(x) for x in ref])
    hap = []
    for ps in range(M.L):
        if dK[ps] < 50 or dB[ps] < 50 or refidx[ps] < 0:
            continue
        aK = pbK[ps, :4] + pbK[ps, 4:]; aB = pbB[ps, :4] + pbB[ps, 4:]
        bK, bB = int(aK.argmax()), int(aB.argmax())
        if bK != bB and aK[bK] / dK[ps] >= 0.9 and aB[bB] / dB[ps] >= 0.9:
            hap.append((ps, bK, bB))
    res["line_defining_sites"] = len(hap)
    hv = {}
    for g in ("K562", "BT142"):
        gi = np.nonzero(lab == g)[0]
        for nm, P in (("3p", P3), ("maester", PM)):
            hv[f"{g}_{nm}"] = write_hom_vcf(f"{root}/private_vcf/k562mix_{g}_{nm}.vcf", P.pseudobulk(gi), ref)
    res["homoplasmic_calls_written"] = hv
    res["line_defining_examples"] = [f"{ps + 1} K562:{M.BASES[a]} BT142:{M.BASES[b]} ref:{ref[ps]}" for ps, a, b in hap[:40]]

    def deconvolve(P, eps=0.01):
        """Per-cell K562 fraction w by ML over line-defining sites: P(K-allele) = w(1-eps) + (1-w)eps."""
        sitesK = [(ps, a) for ps, a, b in hap]; sitesB = [(ps, b) for ps, a, b in hap]
        AK, _, _, _ = M.site_matrices(P, sitesK); AB, _, _, _ = M.site_matrices(P, sitesB)
        k = np.asarray(AK.sum(1)).ravel(); b = np.asarray(AB.sum(1)).ravel()
        n = k + b
        w = np.where(n > 0, (k / np.maximum(n, 1) - eps) / (1 - 2 * eps), np.nan)
        return np.clip(w, 0, 1), n

    dec = {}
    for nm, P in (("3p_umi", P3), ("3p_read", R3), ("maester_umi3", PM)):
        w, n = deconvolve(P)
        call = np.where(np.isnan(w), "none", np.where(w >= 0.5, "K562", "BT142"))
        ok = call != "none"
        dec[nm] = dict(frac_cells_with_informative_molecule=float(ok.mean()),
                       accuracy_vs_rna=float((call[ok] == lab[ok]).mean()) if ok.any() else None,
                       median_informative_molecules=float(np.median(n)),
                       frac_cells_mixed=float(((w > 0.1) & (w < 0.9))[ok].mean()) if ok.any() else None,
                       median_minor_fraction=float(np.nanmedian(np.minimum(w, 1 - w))) if ok.any() else None)
    res["haplotype_deconvolution"] = dec

    log("k562 subclonal vois")
    # K562 subclonal variants (authors' rule, 3.4_TenX_K562_clones.R, on MAESTER umi3): mean cov > 100, q01 K562 < 1%,
    # q99 K562 > 2%, mean AF K562 > 5 x BT142.
    cand = M.candidate_sites(PM, ref, min_cells=3, min_alt=2)
    AD, DP, _, _ = M.site_matrices(PM, cand)
    af = (AD.multiply(1 / np.maximum(DP.toarray(), 1))).toarray() if AD.shape[1] < 30000 else None
    iK, iB = lab == "K562", lab == "BT142"
    meancov = np.asarray(DP.mean(0)).ravel()
    vois = []
    for j, (ps, b) in enumerate(cand):
        a = af[:, j] * 100
        if meancov[j] > 100 and np.quantile(a[iK], .01) < 1 and np.quantile(a[iK], .99) > 2 and a[iK].mean() > 5 * a[iB].mean():
            vois.append((ps, b))
    vois = [v for v in vois if vname(*v, ref) not in ("7693C>T", "8251G>A")]  # not confirmed by bulk ATAC (authors)
    res["k562_subclonal_vois_maester"] = [vname(*v, ref) for v in vois]
    log("vaf fidelity")
    res["vaf_fidelity"] = vaf_fidelity(P3, PM, vois, mask=iK)
    log("detection")
    det = detection(root, "k562_3p", P3, ref, vois, iK)
    res["variant_detection"] = {k: v for k, v in det.items() if not k.startswith("_")}
    groups = donor_groups(root, "k562_3p", want)
    from sklearn.metrics import adjusted_rand_score
    ok = groups != "none"
    res["singlet_demux_vs_rna"] = dict(frac_assigned=float(ok.mean()), ari=float(adjusted_rand_score(lab[ok], groups[ok])),
                                       table=pd.crosstab(lab, groups).to_dict())
    res["singlet_outputs"] = singlet_outputs(root, "k562_3p", want, R3, PM, ref)
    for nm, (P, tool) in (("k562_3p", (P3, "mgatk")), ("k562_maester", (PM, "maegatk"))):
        fd = f"{root}/pile/{nm}/{tool}/final"
        if glob.glob(f"{fd}/*.A.txt.gz"):
            res[f"{nm}_vs_{tool}"] = tool_agreement(P, M.load_mgatk(fd, want), ref)
    log("privacy")
    res["privacy"] = privacy(P3, ref, groups, vois, det)
    M.dump(res, out)


def write_hom_vcf(path, pb, ref, min_dp=10, min_af=0.9):
    """Homoplasmic calls (pseudobulk AF >= min_af vs rCRS) as a VCF for Haplogrep. PRIVATE: haplogroup-defining."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = pb.sum(1)
    n = 0
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n##contig=<ID=chrM,length=16569>\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n")
        for ps in range(M.L):
            if d[ps] < min_dp or ref[ps] not in "ACGT":
                continue
            a = pb[ps, :4] + pb[ps, 4:]
            b = int(a.argmax())
            if M.BASES[b] != ref[ps] and a[b] / d[ps] >= min_af:
                f.write(f"chrM\t{ps + 1}\t.\t{ref[ps]}\t{M.BASES[b]}\t.\tPASS\t.\tGT\t1\n"); n += 1
    return n


def vaf_fidelity(P3, PM, sites, mask=None, min_dp_m=20):
    """3' molecule VAF vs MAESTER VAF in the same cells at the given sites."""
    if not sites:
        return {}
    A3, D3, _, _ = M.site_matrices(P3, sites); Am, Dm, _, _ = M.site_matrices(PM, sites)
    A3, D3, Am, Dm = (x.toarray() for x in (A3, D3, Am, Dm))
    sel = (D3 >= 1) & (Dm >= min_dp_m)
    if mask is not None:
        sel &= mask[:, None]
    v3, vm = A3[sel] / D3[sel], Am[sel] / Dm[sel]
    out = dict(n_entries=int(sel.sum()), pearson=float(np.corrcoef(v3, vm)[0, 1]) if sel.sum() > 2 else None,
               mean_abs_diff=float(np.abs(v3 - vm).mean()) if sel.sum() else None)
    # binomial consistency of the 3' alt count with the MAESTER VAF (two-sided, 1% level)
    from scipy.stats import binom
    k, n, pm = A3[sel], D3[sel], np.clip(vm, 1e-3, 1 - 1e-3)
    lo, hi = binom.cdf(k, n, pm), binom.sf(k - 1, n, pm)
    out["frac_binomially_consistent"] = float((np.minimum(lo, hi) > 0.005).mean()) if sel.sum() else None
    out["median_3p_depth_at_site"] = float(np.median(D3[sel])) if sel.sum() else None
    cov3 = (D3 >= 1)
    out["frac_cell_site_covered_3p"] = float(cov3[mask].mean() if mask is not None else cov3.mean())
    out["frac_cell_site_ge20_maester"] = float((Dm >= 20)[mask].mean() if mask is not None else (Dm >= 20).mean())
    return out


def singlet_outputs(root, name, want, P_read, PM, ref):
    """Score Singlet's own outputs (singlet_export.py -> <name>/singlet_mt.npz): (a) its private read-level counts vs
    mtpile read level (same counting rules) on the evaluation cells; (b) its public per-cell heteroplasmy values vs
    the MAESTER VAF of the same cell and site."""
    f = f"{root}/{name}/singlet_mt.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    out = {}
    bcs = [bc(x) for x in z["counts_barcodes"]]
    col_of = {b: i for i, b in enumerate(bcs)}
    keep = [col_of.get(w, -1) for w in want]
    n_bc = int(z["counts_shape"][1])
    out["private_counts_barcodes"] = n_bc
    out["private_counts_nnz_all_barcodes"] = int(len(z["counts_data"]))
    # rebuild a Pile for the evaluation cells
    def to_pile(prefix):
        ip, ix, dt = z[f"{prefix}_indptr"], z[f"{prefix}_indices"], z[f"{prefix}_data"]
        cells, feats, vals = [], [], []
        for new, old in enumerate(keep):
            if old < 0:
                continue
            sl = slice(ip[old], ip[old + 1])
            cells.append(np.full(sl.stop - sl.start, new)); feats.append(ix[sl]); vals.append(dt[sl])
        return np.concatenate(cells), np.concatenate(feats), np.concatenate(vals).astype(np.int64)
    ct, ft, vt = to_pile("counts")
    out["private_counts_nnz_eval_cells"] = int(len(vt))
    cr, fr, vr = to_pile("rev")
    key_t = ct * M.L * 4 + ft; key_r = cr * M.L * 4 + fr
    o = np.argsort(key_t, kind="stable"); key_t, ct, ft, vt = key_t[o], ct[o], ft[o], vt[o]
    rev = np.zeros(len(key_t), np.int64)
    j = np.searchsorted(key_t, key_r); ok = (j < len(key_t)) & (key_t[np.minimum(j, len(key_t) - 1)] == key_r)
    rev[j[ok]] = vr[ok]
    pos, base = ft // 4, ft % 4
    k2 = ct * M.L + pos
    uk, inv = np.unique(k2, return_inverse=True)
    c = np.zeros((len(uk), 8), np.int64)
    np.add.at(c, (inv, base), vt - rev); np.add.at(c, (inv, base + 4), rev)
    PS = M.Pile(len(want), (uk // M.L).astype(np.uint32), (uk % M.L).astype(np.uint32), c)
    ag = tool_agreement(P_read, PS, ref, calls=False)
    out["private_counts_vs_mtpile_read"] = ag
    # public heteroplasmy matrix
    if "het_data" in z:
        sites = [str(x) for x in z["het_sites"]]
        hbc = {bc(x): i for i, x in enumerate(z["het_barcodes"])}
        ip, ix, dt = z["het_indptr"], z["het_indices"], z["het_data"].astype(np.float64) / 10000.0
        sl = []
        for t in sites:   # chrM:<pos>:<ref>><alt>
            pp, ra = t.split(":")[1], t.split(":")[2]
            sl.append((int(pp) - 1, M.BASES.index(ra[-1])))
        Am, Dm, _, _ = M.site_matrices(PM, sl); Am, Dm = Am.toarray(), Dm.toarray()
        A3, D3, _, _ = M.site_matrices(P_read, sl); D3 = D3.toarray()
        v_s, v_m, zero, cov3 = [], [], 0, 0
        for new, w in enumerate(want):
            old = hbc.get(w)
            if old is None:
                continue
            for k in range(ip[old], ip[old + 1]):
                j_ = ix[k]
                zero += dt[k] == 0
                if Dm[new, j_] >= 20:
                    v_s.append(dt[k]); v_m.append(Am[new, j_] / Dm[new, j_])
        n_entries = int(sum(ip[hbc[w] + 1] - ip[hbc[w]] for w in want if w in hbc))
        cov_pairs = int((D3 >= 1).sum())
        v_s, v_m = np.array(v_s), np.array(v_m)
        out["public_heteroplasmy"] = dict(
            n_sites=len(sites), n_entries_eval_cells=n_entries, n_zero_entries=int(zero),
            n_cell_site_pairs_covered_3p=cov_pairs,
            frac_covered_pairs_stored=n_entries / max(cov_pairs, 1),
            n_with_maester_dp20=int(len(v_s)),
            pearson_vs_maester=float(np.corrcoef(v_s, v_m)[0, 1]) if len(v_s) > 2 else None,
            mean_abs_diff_vs_maester=float(np.abs(v_s - v_m).mean()) if len(v_s) else None,
            frac_maester_vaf_lt_001=float((v_m < 0.01).mean()) if len(v_m) else None)
        # are Singlet's public sites supported by MAESTER pseudobulk?
        pb = PM.pseudobulk(); d = pb.sum(1)
        afm = [float((pb[p, b] + pb[p, b + 4]) / max(d[p], 1)) for p, b in sl]
        out["public_sites_maester_pool_af"] = dict(zip(sites, afm))
        out["public_sites_frac_maester_af_ge_0.005"] = float(np.mean(np.array(afm) >= 0.005))
    return out


def tool_agreement(p_ours, p_tool, ref, calls=True):
    """Per (cell, position, base) agreement of two count tensors over the same cells; and their mgatk-rule calls."""
    def keyed(p):
        k = p.cell.astype(np.int64) * M.L + p.pos
        return k, p.c[:, :4] + p.c[:, 4:]
    ka, ca = keyed(p_ours); kb, cb = keyed(p_tool)
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    da, db = ca[ia].sum(1), cb[ib].sum(1)
    refidx = np.array([M.BASES.find(x) for x in ref])
    rb = refidx[common % M.L]
    alt_a = da - ca[ia, rb]; alt_b = db - cb[ib, rb]
    out = dict(n_cell_pos_ours=int(len(ka)), n_cell_pos_tool=int(len(kb)), n_common=int(len(common)),
               depth_pearson=float(np.corrcoef(da, db)[0, 1]), depth_ratio_median=float(np.median(da / np.maximum(db, 1))),
               nonref_pearson=float(np.corrcoef(alt_a, alt_b)[0, 1]))
    out["exact_cell_pos_base_match"] = float((ca[ia] == cb[ib]).all(1).mean())
    if not calls:
        return out
    sa, _ = M.mgatk_stats(p_ours, ref, M.candidate_sites(p_ours, ref, 3, 1))
    sb, _ = M.mgatk_stats(p_tool, ref, M.candidate_sites(p_tool, ref, 3, 1))
    A, B = set(sa.variant[sa.mgatk_pass]), set(sb.variant[sb.mgatk_pass])
    out.update(mgatk_rule_calls_ours=len(A), mgatk_rule_calls_tool=len(B), mgatk_rule_calls_both=len(A & B))
    return out


def detection(root, name, P3, ref, truth_sites, mask=None):
    """Which truth variants each caller finds on the 3' library: Singlet public list, mgatk thresholds (on the same
    molecule counts), MQuad."""
    tset = {vname(*v, ref) for v in truth_sites}
    out = {"n_truth": len(tset)}
    sing = f"{root}/{name}/mt_variants.tsv"
    if os.path.exists(sing):
        sv = pd.read_csv(sing, sep="\t")
        cols = {c.lower(): c for c in sv.columns}
        if "pos" in cols and "alt" in cols:
            names = {f"{int(r[cols['pos']])}{ref[int(r[cols['pos']]) - 1]}>{r[cols['alt']]}" for _, r in sv.iterrows()}
        else:
            names = set(sv.iloc[:, 0].astype(str))
        out["singlet_public"] = dict(n_calls=len(names), recall=len(names & tset) / max(len(tset), 1),
                                     found=sorted(names & tset), _names=names)
        priv = f"{root}/{name}/private/mt_sites.tsv"
        if os.path.exists(priv):
            ps = pd.read_csv(priv, sep="\t")
            pc = {c.lower(): c for c in ps.columns}
            ps["v"] = [f"{int(r[pc['pos']])}{ref[int(r[pc['pos']]) - 1]}>{r[pc['alt']]}" for _, r in ps.iterrows()]
            ev = ps[ps.v.isin(tset)]
            out["singlet_truth_sites_evaluated"] = ev.drop(columns=["v"]).assign(v=ev.v).to_dict("records")
    st, (AD, DP) = M.mgatk_stats(P3, ref, M.candidate_sites(P3, ref, min_cells=2, min_alt=1))
    log(f"  mgatk stats on {len(st)} candidates")
    mg = set(st.variant[st.mgatk_pass])
    out["mgatk_thresholds"] = dict(n_candidates=len(st), n_calls=len(mg), recall=len(mg & tset) / max(len(tset), 1),
                                   found=sorted(mg & tset),
                                   truth_stats=st[st.variant.isin(tset)][["variant", "mean_af", "vmr", "strand_correlation",
                                                                         "n_cells_conf_detected", "n_cells_alt"]].to_dict("records"))
    # MQuad on candidates with >= 2 cells carrying >= 2 alt molecules (keeps the run tractable)
    # MQuad input: candidates with >= 5 carrier cells and >= 10 alt molecules (all 30k candidates take ~1 h, and
    # MQuad's minor-component rule needs several carrier cells anyway)
    keep = st[(st.n_cells_alt >= 5) & ((st.alt_fwd + st.alt_rev) >= 10)].index.values
    names = list(st.variant.values[keep])
    if os.environ.get("MTL_SKIP_MQUAD"):
        out["mquad"] = dict(skipped=True)
        out["_stats"], out["_mquad"] = st, set()
        out["_singlet"] = out.get("singlet_public", {}).pop("_names", set())
        return out
    with tempfile.TemporaryDirectory() as td:
        log(f"  mquad on {len(names)} variants")
        passed, bic, cut = run_mquad(AD[:, keep], DP[:, keep], names, td, minDP=2)
    tb = bic[bic.variant.isin(tset)][["variant", "deltaBIC", "num_cells_minor_cpt"]].to_dict("records") if bic is not None else []
    out["mquad"] = dict(n_input=len(names), n_calls=len(passed), deltaBIC_cutoff=cut, truth_deltaBIC=tb, recall=len(set(passed) & tset) / max(len(tset), 1),
                        found=sorted(set(passed) & tset), top=passed[:30])
    out["_stats"], out["_mquad"] = st, set(passed)
    out["_singlet"] = out.get("singlet_public", {}).pop("_names", set())
    return out


def donor_groups(root, name, want):
    """Singlet's own donor demux (donor_assignments.tsv): donor id per evaluation cell, 'none' otherwise."""
    f = f"{root}/{name}/donor_assignments.tsv"
    if not os.path.exists(f):
        return np.array(["none"] * len(want))
    d = pd.read_csv(f, sep="	")
    d = d[(d.cell_status == "cell") & d.donor_id.astype(str).str.startswith("donor")]
    m = dict(zip(d.cell.map(bc), d.donor_id))
    return np.array([m.get(w, "none") for w in want])


def privacy(P3, ref, groups, somatic, det):
    """Public / private split, evaluated on the variable sites a caller could publish: Singlet's public list, MQuad's
    calls, and candidates with pooled AF >= 0.5% in >= 5 cells. groups = Singlet's donor demux (per-donor
    pseudobulks). Rules (public if):
      R1  sample pseudobulk AF < 0.9
      R2  AF < 0.9 in every donor pseudobulk
      R3  R2 and not a common homoplasmic population variant (gnomAD v3.1 AF_hom >= 0.001)
      R4  R2 and absent from gnomAD as homoplasmic and as heteroplasmic (AF_hom = 0, AF_het <= 0.001)"""
    gn = gnomad()
    st = det["_stats"].copy()
    pb = P3.pseudobulk(); dpb = pb.sum(1)
    bi = st.alt.map(M.BASES.index).values; p0 = st.pos.values - 1
    st["pb_af"] = (pb[p0, bi] + pb[p0, bi + 4]) / np.maximum(dpb[p0], 1)
    gmax = np.zeros(len(st))
    for g in sorted(set(groups) - {"none"}):
        pg = P3.pseudobulk(np.nonzero(groups == g)[0]); dg = pg.sum(1)
        afg = np.where(dg[p0] >= 10, (pg[p0, bi] + pg[p0, bi + 4]) / np.maximum(dg[p0], 1), 0)
        gmax = np.maximum(gmax, afg)
    st["group_max_af"] = gmax
    st["gnomad_af_hom"] = [gn.get(v, (0, 0))[0] for v in st.variant]
    st["gnomad_af_het"] = [gn.get(v, (0, 0))[1] for v in st.variant]
    plaus = st.variant.isin(det["_mquad"] | det["_singlet"]) | ((st.pb_af >= 0.005) & (st.n_cells_alt >= 5))
    st = st[plaus]
    germ = st.group_max_af >= 0.9
    common = st.gnomad_af_hom >= 0.001
    inpop = (st.gnomad_af_hom > 0) | (st.gnomad_af_het > 0.001)
    som = {vname(*v, ref) for v in somatic}
    rules = {"R1_pseudobulk_lt_0.9": st.pb_af < 0.9, "R2_every_donor_lt_0.9": st.group_max_af < 0.9,
             "R3_R2_and_not_common_homoplasmic": (st.group_max_af < 0.9) & ~common,
             "R4_R2_and_absent_from_gnomad": (st.group_max_af < 0.9) & ~inpop}
    out = {"n_plausible_sites": int(len(st)), "n_donor_groups": int(len(set(groups) - {"none"})),
           "n_germline_in_a_donor": int(germ.sum()), "n_common_homoplasmic_gnomad": int(common.sum())}
    for nm, r in rules.items():
        pub = st[r]
        out[nm] = dict(n_public=int(r.sum()), leaked_donor_germline=int((germ & r).sum()),
                       leaked_common_homoplasmic=int((common & r).sum()),
                       somatic_truth_public=len(set(pub.variant) & som), somatic_truth_in_plausible=len(set(st.variant) & som),
                       leaked_examples=list(pub[(germ | common)[r]].variant[:15]))
    for nm, names in (("singlet_public_today", det["_singlet"]), ("mquad_calls", det["_mquad"])):
        sub = st[st.variant.isin(names)]
        out[nm] = dict(n=len(names), donor_germline=int((sub.group_max_af >= 0.9).sum()),
                       common_homoplasmic=int((sub.gnomad_af_hom >= 0.001).sum()),
                       examples=list(sub[(sub.group_max_af >= 0.9) | (sub.gnomad_af_hom >= 0.001)].variant[:15]))
    return out


_GN = None


def gnomad():
    """gnomAD v3.1 chrM: variant name -> (AF_hom, AF_het)."""
    global _GN
    if _GN is None:
        import gzip
        _GN = {}
        with gzip.open(f"{TRUTH}/gnomad.genomes.v3.1.sites.chrM.vcf.bgz", "rt") as f:
            for line in f:
                if line[0] == "#":
                    continue
                t = line.split("\t", 8)
                if len(t[3]) != 1 or len(t[4]) != 1:
                    continue
                info = dict(x.split("=", 1) for x in t[7].split(";") if "=" in x)
                _GN[f"{t[1]}{t[3]}>{t[4]}"] = (float(info.get("AF_hom", 0) or 0), float(info.get("AF_het", 0) or 0))
    return _GN


# ---------------------------------------------------------------------------------------------------------------------
# BPDCN712 clonal hematopoiesis sample
# ---------------------------------------------------------------------------------------------------------------------

def bpdcn(root, out):
    ref = reference(root)
    pos = pd.read_csv(f"{TRUTH}/4.4_positive_cells.txt", sep="\t")
    pos["bc"] = pos.cell.map(bc)
    vois = [parse_v(x.strip()) for x in open(f"{TRUTH}/4.2_vois.txt") if x.strip()]
    vnames = [vname(*v, ref) for v in vois]
    # clone label per positive cell: the authors' clone column (variant or variant pair)
    clone_of = pos.drop_duplicates("bc").set_index("bc").clone.to_dict()
    m3, c3 = load(root, "bpdcn_3p", "umi1")
    r3, _ = load(root, "bpdcn_3p", "read")
    mm, cm = load(root, "bpdcn_maester", "umi3")
    # evaluation universe: cells called in the 3' library that MAESTER also saw (>= 100 molecules on chrM)
    PMall = mm
    depth_m = np.bincount(PMall.cell, weights=PMall.depth(), minlength=PMall.n_cells)
    seen_m = {cm[i] for i in np.nonzero(depth_m >= 100)[0]}
    want = [c for c in c3 if c in seen_m]
    res = {"n_cells_3p": len(c3), "n_cells_eval": len(want), "n_truth_positive_cells": len(clone_of),
           "n_truth_positive_in_eval": sum(c in clone_of for c in want), "vois": vnames}
    P3, R3, PM = align(m3, c3, want), align(r3, c3, want), align(mm, cm, want)
    truth = np.array([clone_of.get(c, "none") for c in want])
    res["coverage"] = {"3p_umi": coverage_summary(P3), "3p_read": coverage_summary(R3), "maester_umi3": coverage_summary(PM)}

    res["homoplasmic_calls_written"] = {nm: write_hom_vcf(f"{root}/private_vcf/bpdcn_{nm}.vcf", P.pseudobulk(), ref)
                                        for nm, P in (("3p", P3), ("maester", PM))}

    log("per-variant")
    # 1. per-variant sensitivity / specificity of the 3' library for the truth clone variants
    A3, D3, _, _ = M.site_matrices(P3, vois); Am, Dm, _, _ = M.site_matrices(PM, vois)
    A3, D3, Am, Dm = (x.toarray() for x in (A3, D3, Am, Dm))
    per = []
    for j, v in enumerate(vnames):
        posc = np.array([v in [x.replace("_", "") for x in str(t).split("-")] for t in truth])
        negc = (Dm[:, j] >= 20) & (Am[:, j] == 0)
        per.append(dict(variant=v, n_pos=int(posc.sum()), pos_covered_3p=float((D3[posc, j] >= 1).mean()) if posc.any() else None,
                        pos_alt_3p=float((A3[posc, j] >= 1).mean()) if posc.any() else None,
                        mean_depth_pos_3p=float(D3[posc, j].mean()) if posc.any() else None,
                        maester_mean_depth_pos=float(Dm[posc, j].mean()) if posc.any() else None,
                        neg_n=int(negc.sum()), neg_alt_3p=float((A3[negc, j] >= 1).mean()) if negc.any() else None,
                        frac_all_cells_covered_3p=float((D3[:, j] >= 1).mean())))
    res["per_variant_3p"] = per
    res["vaf_fidelity"] = vaf_fidelity(P3, PM, vois)

    for nm, (P, tool) in (("bpdcn_3p", (P3, "mgatk")), ("bpdcn_maester", (PM, "maegatk"))):
        fd = f"{root}/pile/{nm}/{tool}/final"
        if glob.glob(f"{fd}/*.A.txt.gz"):
            res[f"{nm}_vs_{tool}"] = tool_agreement(P, M.load_mgatk(fd, want), ref)

    res["singlet_outputs"] = singlet_outputs(root, "bpdcn_3p", want, R3, PM, ref)

    log("detection")
    # 2. variant discovery on the 3' library
    det = detection(root, "bpdcn_3p", P3, ref, vois)
    res["variant_detection"] = {k: v for k, v in det.items() if not k.startswith("_")}
    res["privacy"] = privacy(P3, ref, donor_groups(root, "bpdcn_3p", want), vois, det)

    log("clones")
    # 3. clone calling: vireoSNP binomial mixture on (a) MAESTER, truth variants; (b) 3', truth variants (oracle
    #    selection: isolates the coverage limit); (c) 3', MQuad-selected variants (end to end)
    is_pos = truth != "none"
    res["clones"] = {}
    def score(lbl, mask):
        from sklearn.metrics import adjusted_rand_score
        t = truth[mask]; p = lbl[mask]
        return dict(ari_positive_cells=float(adjusted_rand_score(t[t != "none"], p[t != "none"])),
                    matched=M.matched_f1(t[t != "none"], p[t != "none"]))
    for nm, (A, D) in (("maester_truth_variants", (Am, Dm)), ("3p_truth_variants", (A3, D3))):
        As, Ds = sp.csc_matrix(A), sp.csc_matrix(D)
        log(f"  vireo {nm}")
        fits, elbo, best = clone_calls(As, Ds, ks=range(10, 31, 4), n_init=20)
        lbl = fits[best][0]
        res["clones"][nm] = dict(k=best, elbo=elbo, **score(lbl, np.ones(len(lbl), bool)))
    # per-cell assignability: cells with >= 1 molecule at any truth variant
    res["frac_positive_cells_with_any_truth_site_covered_3p"] = float((D3[is_pos].sum(1) > 0).mean())
    res["frac_positive_cells_with_any_alt_3p"] = float((A3[is_pos].sum(1) > 0).mean())
    res["frac_positive_cells_with_own_alt_3p"] = own_alt(truth, A3, vnames)

    log("thinning")
    # 4. MAESTER molecules thinned towards 3'-like depth (how much enrichment the clone calls need)
    ds = {}
    for frac in (1.0, 0.3, 0.1, 0.03, 0.01):
        P = PM if frac == 1.0 else PM.thin(frac, seed=1)
        A, D, _, _ = M.site_matrices(P, vois)
        A, D = A.toarray(), D.toarray()
        ds[f"maester_x{frac}"] = dict(mean_depth=float(P.depth().sum() / P.n_cells / M.L),
                                      frac_pos_own_alt=own_alt(truth, A, vnames),
                                      frac_pos_any_site_covered=float((D[is_pos].sum(1) > 0).mean()))
    res["downsampling"] = ds

    log("phasing")
    # 5. phasing and whole-genome deconvolution tests
    res["phasing"] = phasing(root, vois, vnames, truth, Am, Dm, want)
    M.dump(res, out)


def own_alt(truth, A, vnames):
    """Fraction of truth-positive cells with >= 1 alt molecule at one of their own clone's variants."""
    idx = {v: j for j, v in enumerate(vnames)}
    hit = []
    for i, t in enumerate(truth):
        if t == "none":
            continue
        js = [idx[x.replace("_", "")] for x in str(t).split("-") if x.replace("_", "") in idx]
        hit.append(bool(js) and A[i, js].sum() > 0)
    return float(np.mean(hit)) if hit else None


def phase2(root, out):
    """Within-cell VAF equality of variant pairs that define one clone (a single phased haplotype predicts the same VAF
    for both in every cell), BPDCN MAESTER molecules (umi3). Cells with >= MIN molecules at both sites."""
    from scipy.stats import binomtest, chi2
    ref = reference(root)
    pos = pd.read_csv(f"{TRUTH}/4.4_positive_cells.txt", sep="	"); pos["bc"] = pos.cell.map(bc)
    clone_of = pos.drop_duplicates("bc").set_index("bc").clone.to_dict()
    mm, cm = load(root, "bpdcn_maester", "umi3")
    want = [c for c in cm if c in clone_of]
    PM = align(mm, cm, want)
    truth = np.array([clone_of[c] for c in want])
    res = {}
    for t in sorted({t for t in truth if "-" in str(t)}):
        a, b = [parse_v(x) for x in t.split("-")]
        A, D, _, _ = M.site_matrices(PM, [a, b]); A, D = A.toarray(), D.toarray()
        for mn in (5, 10):
            cells = (truth == t) & (D[:, 0] >= mn) & (D[:, 1] >= mn)
            if cells.sum() < 5:
                continue
            ka, na, kb, nb = A[cells, 0], D[cells, 0], A[cells, 1], D[cells, 1]
            pa, pb = ka / na, kb / nb
            pp = (ka + kb) / (na + nb)
            var = pp * (1 - pp) * (1 / na + 1 / nb)
            z2 = np.where(var > 0, (pa - pb) ** 2 / np.maximum(var, 1e-12), 0)
            p = chi2.sf(z2, 1)
            # pooled over cells: heterogeneity of the per-cell VAF difference
            res[f"{t} (>= {mn} molecules)"] = dict(
                n_cells=int(cells.sum()), vaf_a_median=float(np.median(pa)), vaf_b_median=float(np.median(pb)),
                pearson_across_cells=float(np.corrcoef(pa, pb)[0, 1]) if cells.sum() > 2 and pa.std() > 0 and pb.std() > 0 else None,
                frac_cells_unequal_p001=float((p < 0.001).mean()), frac_cells_unequal_bonf=float((p < 0.05 / cells.sum()).mean()),
                median_abs_diff=float(np.median(np.abs(pa - pb))), pooled_chi2_p=float(chi2.sf(z2.sum(), int(cells.sum()))))
    M.dump(res, out)


def facts(root, out):
    """Small numbers quoted in the doc: strand share of chrM molecules per library, median clone VAF per BPDCN clone
    (MAESTER), reference bases per molecule (covered histogram), gnomAD common homoplasmic site count."""
    res = {}
    for nm, lv in (("k562_3p", "umi1"), ("bpdcn_3p", "umi1"), ("k562_maester", "umi3"), ("bpdcn_maester", "umi3")):
        p, _ = load(root, nm, lv)
        fwd, rev = p.c[:, :4].sum(), p.c[:, 4:].sum()
        # per position majority-strand share, depth-weighted
        pb = p.pseudobulk(); f, r = pb[:, :4].sum(1), pb[:, 4:].sum(1)
        maj = np.maximum(f, r).sum() / max((f + r).sum(), 1)
        sp_ = pd.read_csv(f"{root}/pile/{nm}/mt.span.tsv", sep="	")
        cv = sp_[sp_.kind == "covered"]
        med_cov = None
        if len(cv):
            cdf = np.cumsum(cv.n_molecules.values) / cv.n_molecules.sum()
            med_cov = float(cv.bin.values[np.searchsorted(cdf, 0.5)])
        res[nm] = dict(rev_share=float(rev / (fwd + rev)), per_position_majority_strand_share=float(maj),
                       median_covered_bases_per_molecule=med_cov)
    ref = reference(root)
    pos = pd.read_csv(f"{TRUTH}/4.4_positive_cells.txt", sep="	"); pos["bc"] = pos.cell.map(bc)
    clone_of = pos.drop_duplicates("bc").set_index("bc").clone.to_dict()
    mm, cm = load(root, "bpdcn_maester", "umi3")
    want = [c for c in cm if c in clone_of]
    PM = align(mm, cm, want); truth = np.array([clone_of[c] for c in want])
    med = {}
    for t in sorted(set(truth)):
        vs = [parse_v(x) for x in str(t).split("-")]
        A, D, _, _ = M.site_matrices(PM, vs); A, D = A.toarray(), D.toarray()
        m = truth == t
        cell_vaf = A[m].sum(1) / np.maximum(D[m].sum(1), 1)
        med[t] = float(np.median(cell_vaf[D[m].sum(1) >= 3])) if (D[m].sum(1) >= 3).any() else None
    res["bpdcn_clone_median_vaf_maester"] = med
    res["n_clones_median_vaf_ge_0.9"] = int(sum(v is not None and v >= 0.9 for v in med.values()))
    gn = gnomad()
    res["gnomad_af_hom_ge_0.001"] = int(sum(1 for v in gn.values() if v[0] >= 0.001))
    res["gnomad_snvs"] = len(gn)
    import importlib.metadata as im
    res["versions"] = {k: im.version(k) for k in ("mquad", "vireoSNP", "mgatk") if _has(k)}
    M.dump(res, out)


def _has(k):
    import importlib.metadata as im
    try:
        im.version(k); return True
    except Exception:
        return False


def phasing(root, vois, vnames, truth, Am, Dm, want):
    """(a) Within-cell VAF equality of two variants that define one clone (a single 'haplotype' predicts equal VAF);
    (b) molecules that cover >= 2 informative sites (read-level phasing), from mtpile --sites output;
    (c) molecule span distributions (3' vs MAESTER)."""
    out = {}
    from scipy.stats import chi2
    pairs = sorted({t for t in truth if "-" in str(t)})
    idx = {v: j for j, v in enumerate(vnames)}
    for t in pairs:
        a, b = [x.replace("_", "") for x in t.split("-")]
        if a not in idx or b not in idx:
            continue
        ja, jb = idx[a], idx[b]
        cells = (truth == t) & (Dm[:, ja] >= 20) & (Dm[:, jb] >= 20)
        if cells.sum() < 5:
            continue
        ka, na, kb, nb = Am[cells, ja], Dm[cells, ja], Am[cells, jb], Dm[cells, jb]
        pa, pb = ka / na, kb / nb
        pp = (ka + kb) / (na + nb)
        # per-cell 2x2 chi-square (equal VAF), Bonferroni over cells
        var = pp * (1 - pp) * (1 / na + 1 / nb)
        z2 = np.where(var > 0, (pa - pb) ** 2 / np.maximum(var, 1e-12), 0)
        p = chi2.sf(z2, 1)
        out[t] = dict(n_cells=int(cells.sum()), vaf_a_median=float(np.median(pa)), vaf_b_median=float(np.median(pb)),
                      corr=float(np.corrcoef(pa, pb)[0, 1]) if cells.sum() > 2 else None,
                      frac_cells_unequal_bonf=float((p < 0.05 / cells.sum()).mean()),
                      median_abs_diff=float(np.median(np.abs(pa - pb))))
    res = {"clone_pairs_vaf_equality": out}
    for name in ("bpdcn_3p", "bpdcn_maester"):
        sp_ = f"{root}/pile/{name}/mt.span.tsv"
        if os.path.exists(sp_):
            d = pd.read_csv(sp_, sep="\t")
            s = d[d.kind == "span"]
            tot = s.n_molecules.sum()
            cdf = np.cumsum(s.n_molecules.values) / tot
            res[f"{name}_molecule_span"] = dict(median=float(s.bin.values[np.searchsorted(cdf, 0.5)]),
                                                p90=float(s.bin.values[np.searchsorted(cdf, 0.9)]),
                                                frac_ge_500=float(s[s.bin >= 500].n_molecules.sum() / tot),
                                                frac_ge_1000=float(s[s.bin >= 1000].n_molecules.sum() / tot))
        sb = f"{root}/pile/{name}/mt.sites.bin"
        if os.path.exists(sb):
            res[f"{name}_co_observation"] = co_observation(sb, f"{root}/pile/{name}/sites.txt")
    return res


def co_observation(path, sites_path):
    """Molecules that report a consensus base at >= 2 of the listed sites, by site pair distance."""
    sites = [int(x) for x in open(sites_path) if x.strip()]
    with open(path, "rb") as f:
        assert f.read(4) == b"MTS1"
        n = int(np.frombuffer(f.read(8), "<u8")[0])
        r = np.frombuffer(f.read(n * 12), np.dtype([("cell", "<u4"), ("umi", "<u4"), ("site", "<u2"), ("base", "u1"), ("reads", "u1")]))
    key = r["cell"].astype(np.uint64) << np.uint64(32) | r["umi"].astype(np.uint64)
    order = np.argsort(key, kind="stable")
    key, site = key[order], r["site"][order]
    _, start, cnt = np.unique(key, return_index=True, return_counts=True)
    multi = cnt >= 2
    pairs = {}
    for s0, c in zip(start[multi], cnt[multi]):
        ss = sorted(set(site[s0:s0 + c].tolist()))
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                d = abs(sites[ss[j]] - sites[ss[i]])
                pairs[d] = pairs.get(d, 0) + 1
    return dict(molecules_with_any_site=int(len(cnt)), molecules_with_ge2_sites=int(multi.sum()),
                frac_ge2=float(multi.mean()) if len(cnt) else None,
                pair_distance_counts={str(k): v for k, v in sorted(pairs.items())[:50]})


def depthcurve(root, out):
    """Real-data depth curve: truth-positive BPDCN cells binned by their own 3' chrM depth (reads per cell, the unit of
    the atlas survey), with the fraction that show >= 1 alt molecule at their clone's variant(s), and the fraction of
    all cells covered at each truth variant. No thinning or simulation."""
    ref = reference(root)
    pos = pd.read_csv(f"{TRUTH}/4.4_positive_cells.txt", sep="	"); pos["bc"] = pos.cell.map(bc)
    clone_of = pos.drop_duplicates("bc").set_index("bc").clone.to_dict()
    vois = [parse_v(x.strip()) for x in open(f"{TRUTH}/4.2_vois.txt") if x.strip()]
    vnames = [vname(*v, ref) for v in vois]
    m3, c3 = load(root, "bpdcn_3p", "umi1"); r3, _ = load(root, "bpdcn_3p", "read")
    want = [c for c in c3 if c in clone_of]
    P3, R3 = align(m3, c3, want), align(r3, c3, want)
    truth = np.array([clone_of[c] for c in want])
    reads = np.bincount(R3.cell, weights=R3.depth(), minlength=R3.n_cells) / 90.0   # aligned bases / read length
    mol = np.bincount(P3.cell, weights=P3.depth(), minlength=P3.n_cells) / M.L        # mean molecule depth
    A3, D3, _, _ = M.site_matrices(P3, vois); A3, D3 = A3.toarray(), D3.toarray()
    idx = {v: j for j, v in enumerate(vnames)}
    own = []
    for i, t in enumerate(truth):
        js = [idx[x.replace("_", "")] for x in str(t).split("-") if x.replace("_", "") in idx]
        own.append((A3[i, js].sum() > 0, D3[i, js].sum() > 0))
    own = np.array(own)
    edges = np.quantile(reads, np.linspace(0, 1, 11))
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (reads >= a) & (reads <= b)
        rows.append(dict(chrM_reads_lo=float(a), chrM_reads_hi=float(b), n=int(m.sum()), mean_mol_depth=float(mol[m].mean()),
                         frac_own_site_covered=float(own[m, 1].mean()), frac_own_alt=float(own[m, 0].mean())))
    M.dump(dict(n_positive=len(want), bins=rows,
                overall_frac_own_alt=float(own[:, 0].mean()), overall_frac_own_covered=float(own[:, 1].mean()),
                median_chrM_reads=float(np.median(reads))), out)


# ---------------------------------------------------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------------------------------------------------

def varint(a):
    """LEB128 bytes of non-negative integers (vectorised)."""
    a = np.asarray(a, np.uint64).ravel()
    if a.size == 0:
        return b""
    nb = np.ones(a.size, np.int64)
    for k in range(1, 10):
        nb += a >= (np.uint64(1) << np.uint64(7 * k))
    out = np.zeros(int(nb.sum()), np.uint8)
    start = np.concatenate([[0], np.cumsum(nb)[:-1]])
    for k in range(int(nb.max())):
        m = nb > k
        byte = ((a[m] >> np.uint64(7 * k)) & np.uint64(0x7F)).astype(np.uint8)
        byte[nb[m] > k + 1] |= 0x80
        out[start[m] + k] = byte
    return out.tobytes()


def encode_sizes(p: M.Pile, ref: str, n_var_sites=None, sites=None):
    """Bytes of several encodings of one sample's chrM counts, zstd-19 compressed."""
    import zstandard
    z = zstandard.ZstdCompressor(level=19)
    refidx = np.array([M.BASES.find(x) for x in ref])
    order = np.lexsort((p.pos, p.cell))
    cell, pos, c = p.cell[order], p.pos[order], p.c[order]
    out = {"n_cells": p.n_cells, "n_cell_pos": int(len(cell)), "n_nonzero_base_strand": int((c > 0).sum())}
    # E1 full tensor, COO with per-cell delta positions, 8 counts as varints
    dpos = np.diff(np.concatenate([[0], pos.astype(np.int64)]))
    newcell = np.concatenate([[True], cell[1:] != cell[:-1]])
    dpos[newcell] = pos[newcell]
    ncell = np.bincount(cell, minlength=p.n_cells)
    e1 = varint(ncell) + varint(dpos) + varint(c.ravel())
    out["E1_full_tensor_strand"] = len(z.compress(e1))
    # E2 full tensor without strand (4 counts)
    c4 = c[:, :4] + c[:, 4:]
    out["E2_full_tensor_nostrand"] = len(z.compress(varint(ncell) + varint(dpos) + varint(c4.ravel())))
    # E3 reference-compressed: depth by strand per (cell,pos) + non-reference counts only (pos-in-record, base, f, r)
    rb = refidx[pos]
    dfw, drv = c[:, :4].sum(1), c[:, 4:].sum(1)
    alt_mask = np.zeros_like(c[:, :4], bool)
    for b in range(4):
        alt_mask[:, b] = (rb != b) & (c4[:, b] > 0)
    rec_i, base_i = np.nonzero(alt_mask)
    nalt = np.bincount(rec_i, minlength=len(cell))
    e3 = varint(ncell) + varint(dpos) + varint(dfw) + varint(drv) + varint(nalt) + bytes(base_i.astype(np.uint8)) + \
        varint(c[rec_i, base_i]) + varint(c[rec_i, base_i + 4])
    out["E3_depth_strand_plus_nonref"] = len(z.compress(e3))
    out["n_nonref_entries"] = int(len(rec_i))
    # E4 depth only (no strand) + nonref without strand
    e4 = varint(ncell) + varint(dpos) + varint(c4.sum(1)) + varint(nalt) + bytes(base_i.astype(np.uint8)) + varint(c4[rec_i, base_i])
    out["E4_depth_plus_nonref_nostrand"] = len(z.compress(e4))
    # E5 curated cell x variant AD/DP for the given sites
    if sites:
        AD, DP, _, _ = M.site_matrices(p, sites)
        DPc = sp.csc_matrix(DP); ADc = sp.csc_matrix(AD)
        e5 = varint(np.diff(DPc.indptr)) + varint(np.diff(np.concatenate([[0], DPc.indices]))) + varint(DPc.data) + \
            varint(np.diff(ADc.indptr)) + varint(ADc.indices) + varint(ADc.data)
        out["E5_cell_x_variant_AD_DP"] = len(z.compress(e5))
        out["E5_n_sites"] = len(sites)
        out["E5_nnz_dp"] = int(DPc.nnz)
    return out


def size(root, out):
    ref = reference(root)
    res = {}
    for name, levels in (("k562_3p", ("read", "umi1")), ("bpdcn_3p", ("read", "umi1")),
                         ("k562_maester", ("read", "umi1", "umi3")), ("bpdcn_maester", ("read", "umi1", "umi3"))):
        if not os.path.exists(f"{root}/pile/{name}/mt.umi1.bin"):
            continue
        for lv in levels:
            p, _ = load(root, name, lv)
            st, _ = M.mgatk_stats(p, ref, M.candidate_sites(p, ref, min_cells=5, min_alt=2)) if lv != "read" else (None, None)
            sites = None
            if st is not None and len(st):
                top = st[(st.n_cells_alt >= 5)].sort_values("vmr", ascending=False).head(500)
                sites = [(int(r.pos) - 1, M.BASES.index(r.alt)) for r in top.itertuples()]
            res[f"{name}/{lv}"] = encode_sizes(p, ref, sites=sites)
        # what Singlet writes today
        for fn in ("private/mt_allele_counts.1pz", "private/mt_allele_counts_rev.1pz", "mt_heteroplasmy.1pz", "mt_variants.tsv",
                   "private/mt_sites.tsv", "exon_counts.1pz"):
            fp = f"{root}/{name}/{fn}"
            if os.path.exists(fp):
                res.setdefault(f"{name}/singlet_files", {})[fn] = os.path.getsize(fp)
    M.dump(res, out)


if __name__ == "__main__":
    cmd, root, out = sys.argv[1:4]
    {"lines": lines, "bpdcn": bpdcn, "size": size, "depthcurve": depthcurve, "phase2": phase2, "facts": facts}[cmd](root, out)
