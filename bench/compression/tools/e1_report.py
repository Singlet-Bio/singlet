#!/usr/bin/env python
"""E1 report: cross-fitted ladder tables, depth tables, effective k and floor bracket from the per-cell npz files.

    python -P tools/e1_report.py [--res interval|gene]      -> results/e1_report[_gene].md , results/e1_summary[_gene].tsv

Hyper-parameters of each level (k, w, alpha, prior) are chosen by 2-fold cross-fitting over scored cells (parity of the
random cell order): parameters picked on one half of the cells are applied to the other half.  The depth tables
cross-fit within each depth bucket (the decoder may pick hyper-parameters as a function of the depth it already knows).
"""
import os, sys, glob, argparse, csv, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1_split as E

RES = E.RES


def labels_L2():
    return [f"k={k},w={w}" for k in E.KS for w in E.WS]


def xfit(M, par, groups=None):
    """M [n, c] candidate costs. Returns (sel_idx [n], chosen description counts) choosing, for cells of parity p, the
    candidate with the lowest summed cost on cells of parity 1-p (within the same group if groups is given)."""
    n, c = M.shape
    sel = np.zeros(n, dtype=np.int64)
    glist = [np.arange(n)] if groups is None else [np.flatnonzero(groups == g) for g in np.unique(groups)]
    gl_all = np.zeros(c)
    for idx in glist:
        for p in (0, 1):
            tgt = idx[par[idx] == p]
            trn = idx[par[idx] != p]
            if len(trn) < 5:
                trn = np.arange(n)[par != p]
            j = int(np.argmin(M[trn].sum(0)))
            sel[tgt] = j
    return sel


def pick(arr, sel):
    return np.take_along_axis(arr, sel[:, None], 1)[:, 0]


class FileData:
    def __init__(self, path):
        z = E.load_percell(path)
        v = ~np.isnan(z["N"])
        self.z = {k: (a[v] if a.ndim >= 1 and a.shape[0] == len(v) else a) for k, a in z.items()}
        self.n = int(v.sum())
        self.par = (np.arange(self.n) % 2).astype(int)
        self.use_nbr = "ms_L2" in z
        self.na = len(E.ALPHAS)
        self.npri = self.z["ms_L3"].shape[2]
        self.labs = E.prior_labels(self.use_nbr)

    def flat(self, name):
        a = self.z[name]
        return a.reshape(self.n, -1)

    def cand_label(self, kind, j):
        if kind == "L2":
            return labels_L2()[j]
        nl2 = len(self.labs) - 1
        if kind in ("L3seq", "L3dmseq"):
            return "a=inf" if j == 0 else f"a={E.ALPHAS[j - 1]:g}"
        if kind in ("L3", "L3dm"):
            if j == 0:
                return "L1, a=inf"
            if j <= nl2:
                return f"{self.labs[j]}, a=inf"
            j -= 1 + nl2
            return f"{self.labs[j % self.npri]}, a={E.ALPHAS[j // self.npri]:g}"
        return str(j)

    def level_arrays(self):
        """Candidate matrices per family: kind -> (ms, pres, cnt, basis) with shape [n, ncand].
        The L3 / L3dm families include the alpha = infinity candidates (prior used unchanged: L1 and every L2)."""
        z = self.z
        n = self.n
        c1 = lambda a: a[:, None]
        fam = {}
        fam["L0"] = (c1(z["ms_L0"]), c1(z["pres_L0"]), c1(z["cnt_L0"]), "B")
        fam["L1"] = (c1(z["ms_L1"]), c1(z["pres_L1"]), c1(z["cnt_L1"]), "B")
        if self.use_nbr:
            fam["L2"] = (z["ms_L2"], z["pres_L2"], z["cnt_L2"], "B")
        hs = lambda *a: np.hstack(a)
        fam["L3seq"] = (hs(c1(z["ms_L1"]), z["ms_L3"][:, :, 0]), hs(c1(z["pres_L1"]), z["pres_L3"][:, :, 0]),
                        hs(c1(z["cnt_L1"]), z["cnt_L3"][:, :, 0]), "B")
        if self.use_nbr:
            fam["L3"] = (hs(c1(z["ms_L1"]), z["ms_L2"], z["ms_L3"].reshape(n, -1)),
                         hs(c1(z["pres_L1"]), z["pres_L2"], z["pres_L3"].reshape(n, -1)),
                         hs(c1(z["cnt_L1"]), z["cnt_L2"], z["cnt_L3"].reshape(n, -1)), "B")
            fam["L3dm"] = (hs(c1(z["ms_L1"]), z["ms_L2"], z["ms_L3dm"].reshape(n, -1)), None, None, "B")
        else:
            fam["L3"] = fam["L3seq"]
            fam["L3dm"] = (hs(c1(z["ms_L1"]), z["ms_L3dm"][:, :, 0]), None, None, "B")
        fam["L3dmseq"] = (hs(c1(z["ms_L1"]), z["ms_L3dm"][:, :, 0]), None, None, "B")
        if self.use_nbr:
            fam["L4"] = (z["ms_L4"], z["pres_L4"], z["cnt_L4"], "B")
        fam["F1"] = (c1(z["F1"]), None, None, "full")
        fam["F1dm"] = (z["F1dm"], None, None, "full")
        return fam

    def select(self, groups=None):
        """Cross-fitted per-cell selected costs for every family: kind -> dict(ms, pres, cnt, sel, basis)."""
        out = {}
        for kind, (ms, pres, cnt, basis) in self.level_arrays().items():
            if ms.shape[1] == 1:
                sel = np.zeros(self.n, dtype=np.int64)
            elif kind in ("L4", "F1dm"):  # oracle / reference rows: best pooled candidate (not cross-fitted)
                sel = np.full(self.n, int(np.argmin(ms.sum(0))), dtype=np.int64)
            else:
                sel = xfit(ms, self.par, groups)
            out[kind] = dict(ms=pick(ms, sel), pres=None if pres is None else pick(pres, sel),
                             cnt=None if cnt is None else pick(cnt, sel), sel=sel, basis=basis)
        return out


def fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ladder_table(fd, sel, mask=None):
    z = fd.z
    m = np.ones(fd.n, dtype=bool) if mask is None else mask
    UB, nB, nC = z["NB"][m].sum(), z["nnzB"][m].sum(), z["nnz"][m].sum()
    UF = z["N"][m].sum()
    base = np.sum(sel["L1"]["ms"][m])
    rows = []
    names = [("L0", "L0 population"), ("L1", "L1 pop x depth"), ("L2", "L2 neighbours"), ("L3seq", "L3 sequential (prior L1)"),
             ("L3", "L3 best (nbr prior + own half)"), ("L3dmseq", "L3dm sequential (exact DM, prior L1)"),
             ("L3dm", "L3dm best (exact DM)"), ("L4", "L4 in-sample oracle")]
    for kind, name in names:
        if kind not in sel:
            continue
        s = sel[kind]
        ms = np.sum(s["ms"][m])
        sp = None if s["pres"] is None else np.sum(s["pres"][m]) + np.sum(s["cnt"][m])
        gain = base - ms
        rows.append((name, ms / UB, ms / nB, None if sp is None else sp / nB, gain / nB, gain / m.sum(), gain / nC))
    return rows, dict(UB=UB, nB=nB, nC=nC, ncell=int(m.sum()))


def md_ladder(rows, meta):
    s = ["| level | ms b/UMI | ms b/nnzB | sparse b/nnzB | gain vs L1 b/nnzB | gain bits/cell | gain b/nnz_cell |", "|---|---|---|---|---|---|---|"]
    for name, a, b, c, g, gc, gn in rows:
        s.append(f"| {name} | {a:.3f} | {b:.3f} | {fmt(c)} | {g:+.3f} | {gc:+.1f} | {gn:+.3f} |")
    return "\n".join(s)


def depth_groups(fd, min_cells=40):
    """Merge the 16 log buckets greedily into groups of >= min_cells cells; returns group id per cell and descriptions."""
    b = fd.z["bucket"].astype(int)
    edges = fd.z["edges"]
    counts = np.bincount(b, minlength=E.NB_BUCKETS)
    gid = np.zeros(E.NB_BUCKETS, dtype=int)
    g, acc = 0, 0
    for i in range(E.NB_BUCKETS):
        gid[i] = g
        acc += counts[i]
        if acc >= min_cells:
            g += 1
            acc = 0
    # fold a short tail into the previous group
    last = gid.max()
    if (gid == last).any() and counts[gid == last].sum() < min_cells and last > 0:
        gid[gid == last] = last - 1
    # drop empty trailing ids
    cell_g = gid[b]
    return cell_g


def depth_table(fd, min_cells=40):
    cg = depth_groups(fd, min_cells)
    sel = fd.select(groups=cg)
    z = fd.z
    rows = []
    for g in np.unique(cg):
        m = cg == g
        if m.sum() < 10:
            continue
        lo, hi = z["N"][m].min(), z["N"][m].max()
        UB, nB, nC = z["NB"][m].sum(), z["nnzB"][m].sum(), z["nnz"][m].sum()
        L1 = sel["L1"]["ms"][m].sum()
        gains = {k: (L1 - sel[k]["ms"][m].sum()) for k in ("L2", "L3seq", "L3", "L3dm") if k in sel}
        rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), medN=float(np.median(z["N"][m])), nnz=float(z["nnz"][m].mean()),
                         L1=L1 / nB, gains_nnzB={k: v / nB for k, v in gains.items()},
                         gains_cell={k: v / m.sum() for k, v in gains.items()},
                         gains_nnzC={k: v / nC for k, v in gains.items()},
                         log2N=float(np.mean(np.log2(z["N"][m]))), mask=m))
    return rows, sel


def keff(rows, key, sel, fd):
    """slope of per-cell gain (bits) vs log2 N across depth groups (weighted OLS) and at cell level."""
    x = np.array([r["log2N"] for r in rows])
    y = np.array([r["gains_cell"][key] for r in rows])
    w = np.array([r["n"] for r in rows], dtype=float)
    if len(rows) < 3:
        return None
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    sxx = np.sum(w * (x - xm) ** 2)
    slope = np.sum(w * (x - xm) * (y - ym)) / sxx if sxx > 0 else np.nan
    pred = ym + slope * (x - xm)
    ss_res, ss_tot = np.sum(w * (y - pred) ** 2), np.sum(w * (y - ym) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # cell-level OLS
    z = fd.z
    g = sel["L1"]["ms"] - sel[key]["ms"]
    lx = np.log2(z["N"])
    s2 = np.cov(lx, g)[0, 1] / np.var(lx, ddof=1) if len(lx) > 3 else np.nan
    icpt = ym - slope * xm                                # gain_bits(log2 N) = icpt + slope * log2 N
    n0 = 2.0 ** (-icpt / slope) if slope > 0 else float("nan")   # depth at which the fitted gain crosses zero
    med = float(np.median(z["N"]))
    bound = max(slope, 0.0) * math.log2(med)              # (k_eff/2) log2 N_med: the brief's zero-intercept bound, bits/cell
    return dict(slope=slope, keff=2 * slope, r2=r2, keff_cell=2 * s2, N0=n0, bound_bits=bound,
                bound_per_nnz=bound / float(np.median(z["nnz"])))


CLASS_EDGES = [0, 150, 500, 1500, 1e9]
CLASS_NAMES = ["N<150", "150-500", "500-1500", ">=1500"]


def class_table(fd):
    """gains by coarse depth class (hyper-parameters cross-fitted within each class)."""
    z = fd.z
    cid = np.searchsorted(CLASS_EDGES, z["N"], side="right") - 1
    sel = fd.select(groups=cid)
    rows = []
    for g, name in enumerate(CLASS_NAMES):
        m = cid == g
        if m.sum() < 30:
            continue
        nB, nC = z["nnzB"][m].sum(), z["nnz"][m].sum()
        L1 = sel["L1"]["ms"][m].sum()
        gains = {k: L1 - sel[k]["ms"][m].sum() for k in ("L2", "L3seq", "L3", "L3dm") if k in sel}
        rows.append(dict(cls=name, n=int(m.sum()), L1=L1 / nB, nnz=float(z["nnz"][m].mean()),
                         **{k: v / nB for k, v in gains.items()}, **{k + "_cell": v / m.sum() for k, v in gains.items()},
                         **{k + "_nnzC": v / nC for k, v in gains.items()}))
    return rows


def report(res):
    paths = sorted(glob.glob(os.path.join(RES, f"e1_percell_*__{res}__cells.npz")))
    epaths = {os.path.basename(p).replace("__empties", "__cells"): p
              for p in glob.glob(os.path.join(RES, f"e1_percell_*__{res}__empties.npz"))}
    out = [f"# E1 report ({res} resolution) -- generated by tools/e1_report.py\n"]
    summ = []
    allclass = []
    hdr = ["res", "file", "subset", "n_cells", "median_N", "median_nnz", "F1_b_nnz", "F1dm_gain_b_nnz", "L1_b_nnzB", "L1_sp_b_nnzB",
           "gain_L2", "gain_L3seq", "gain_L3", "gain_L3dm", "gain_L3dmseq", "gain_L3_bits_cell", "gain_L3_b_nnz_cell",
           "keff_L3", "keff_L3dm", "floor_cs_b_nnzB", "floor_post_b_nnzB", "L4_b_nnzB"]
    for p in paths:
        base = os.path.basename(p)[len("e1_percell_"):-len("__cells.npz")]
        sample, member, _ = base.split("__")
        fname = f"{sample}/{member}"
        fd = FileData(p)
        if fd.n < 30:
            out.append(f"## {fname}\n\nonly {fd.n} scored cells; skipped.\n")
            continue
        z = fd.z
        sel = fd.select()
        rows, meta = ladder_table(fd, sel)
        out.append(f"## {fname} ({res}) -- called cells N>=50: {fd.n} scored of {int(z['npop'][0])}, F={int(z['F'][0])}, "
                   f"median N={np.median(z['N']):.0f}, median nnz={np.median(z['nnz']):.0f}, median nnz_B={np.median(z['nnzB']):.0f}\n")
        out.append(f"tuned delta0={z['delta0'][0]:g}, delta1={z['delta1'][0]:g}, tau={z['tau'][0]:g}. "
                   f"Cross-fitted choices (most common): "
                   + "; ".join(f"{k}: {fd.cand_label(k, int(np.bincount(sel[k]['sel']).argmax()))}" for k in ("L2", "L3seq", "L3", "L3dm") if k in sel)
                   + "\n")
        out.append(md_ladder(rows, meta) + "\n")
        f1 = sel["F1"]["ms"].sum() / z["nnz"].sum()
        f1dm = sel["F1dm"]["ms"].sum() / z["nnz"].sum()
        out.append(f"Full-cell reference (bits per nnz of the whole cell): F1 static L1 multiset = {f1:.3f}; "
                   f"F1dm Dirichlet-multinomial centred on L1 (best alpha={E.ALPHAS[int(sel['F1dm']['sel'][0])]:g}) = {f1dm:.3f} "
                   f"(within-cell burstiness gain {f1 - f1dm:+.3f}).\n")
        # floor bracket in B units
        cs_b = z["NB"] * z["Hcs"] - z["lfNB"] + z["sumlfB"]
        floor_cs = cs_b.sum() / z["nnzB"].sum()
        floor_post = np.nan
        if "Hpost" in z:
            k = len(z["Hpost"])
            pb = z["NB"][:k] * z["Hpost"] - z["lfNB"][:k] + z["sumlfB"][:k]
            floor_post = pb.sum() / z["nnzB"][:k].sum()
            l1_k = sel["L1"]["ms"][:k].sum() / z["nnzB"][:k].sum()
        else:
            l1_k = np.nan
        l4 = sel["L4"]["ms"].sum() / z["nnzB"].sum() if "L4" in sel else np.nan
        l1 = sel["L1"]["ms"].sum() / z["nnzB"].sum()
        out.append(f"Floor bracket for the held-out half (multiset bits per nnz_B): L1 = {l1:.3f}; Chao-Shen implied "
                   f"H(Mult(N_B, pi)) = {floor_cs:.3f}; posterior-entropy implied = {floor_post:.3f} "
                   f"(first {len(z['Hpost']) if 'Hpost' in z else 0} cells; L1 on those cells = {l1_k:.3f}); "
                   f"in-sample L4 oracle = {l4:.3f}.\n")
        # depth table
        drows, dsel = depth_table(fd)
        out.append("By depth (hyper-parameters cross-fitted within each depth group), gains vs L1 in bits per cell [b/nnzB]:\n")
        out.append("| N range | cells | mean nnz | L1 b/nnzB | L2 | L3 seq | L3 best | L3dm best |\n|---|---|---|---|---|---|---|---|")
        for r in drows:
            g = r["gains_cell"]; gn = r["gains_nnzB"]
            out.append(f"| {r['lo']:.0f}-{r['hi']:.0f} | {r['n']} | {r['nnz']:.0f} | {r['L1']:.3f} | "
                       + " | ".join(f"{g[k]:+.1f} [{gn[k]:+.3f}]" if k in g else "" for k in ("L2", "L3seq", "L3", "L3dm")) + " |")
        out.append("")
        ke3 = keff(drows, "L3", dsel, fd) if "L3" in dsel else None
        ke3d = keff(drows, "L3dm", dsel, fd)
        for nm, ke in (("L3", ke3), ("L3dm", ke3d)):
            if ke:
                out.append(f"effective k ({nm}; 2 x slope of gain-bits/cell vs log2 N over depth groups): {ke['keff']:.1f} "
                           f"(R2={ke['r2']:.2f}; cell-level OLS {ke['keff_cell']:.1f}); fitted gain crosses zero at N0={ke['N0']:.0f}; "
                           f"zero-intercept bound (k_eff/2) log2 N_med = {ke['bound_bits']:.0f} bits/cell = {ke['bound_per_nnz']:.3f} b/nnz")
        out.append("")
        crow = class_table(fd)
        out.append("By coarse depth class (cross-fitted within class), gain vs L1 in b/nnzB (bits/cell):\n")
        out.append("| class | cells | mean nnz | L1 b/nnzB | L2 | L3 seq | L3 best | L3dm best |\n|---|---|---|---|---|---|---|---|")
        for r in crow:
            out.append(f"| {r['cls']} | {r['n']} | {r['nnz']:.0f} | {r['L1']:.3f} | " + " | ".join(
                f"{r[k]:+.3f} ({r[k + '_cell']:+.1f})" if k in r else "" for k in ("L2", "L3seq", "L3", "L3dm")) + " |")
        out.append("")
        for r in crow:
            allclass.append((fname, r))
        # empties control
        ep = epaths.get(os.path.basename(p))
        e_gain = np.nan
        if ep:
            fe = FileData(ep)
            if fe.n >= 30:
                se = fe.select()
                re, me = ladder_table(fe, se)
                out.append(f"Empties control ({fe.n} droplets with N>=20, population = empties):\n")
                out.append(md_ladder([r for r in re if not r[0].startswith(("L2", "L4"))], me) + "\n")
        hdr_row = [res, fname, "cells", fd.n, f"{np.median(z['N']):.0f}", f"{np.median(z['nnz']):.0f}", f"{f1:.4f}", f"{f1 - f1dm:.4f}",
                   f"{l1:.4f}", fmt(rows[1][3], 4)]
        g = {n: r for n, r in zip([x[0] for x in rows], rows)}
        def gv(nm):
            for r in rows:
                if r[0].startswith(nm):
                    return r[4]
            return np.nan
        hdr_row += [fmt(gv("L2"), 4), fmt(gv("L3 seq"), 4), fmt(gv("L3 best"), 4), fmt(gv("L3dm best"), 4), fmt(gv("L3dm seq"), 4),
                    fmt([r[5] for r in rows if r[0].startswith("L3 best")][0], 2), fmt([r[6] for r in rows if r[0].startswith("L3 best")][0], 4),
                    fmt(ke3["keff"] if ke3 else np.nan, 2), fmt(ke3d["keff"] if ke3d else np.nan, 2),
                    fmt(floor_cs, 3), fmt(floor_post, 3), fmt(l4, 3)]
        summ.append(hdr_row)
        out.append("\n")
    out.append("## Cross-file summary by depth class (gain vs L1 in b/nnzB; b/nnz of the whole cell in brackets for L3 best)\n")
    out.append("| file | class | cells | mean nnz | L1 b/nnzB | L2 | L3 seq | L3 best | L3dm best |\n|---|---|---|---|---|---|---|---|---|")
    for fname, r in allclass:
        out.append(f"| {fname} | {r['cls']} | {r['n']} | {r['nnz']:.0f} | {r['L1']:.3f} | {r['L2']:+.3f} | {r['L3seq']:+.3f} | "
                   f"{r['L3']:+.3f} [{r['L3_nnzC']:+.3f}] | {r['L3dm']:+.3f} |")
    out.append("")
    suffix = "" if res == "interval" else f"_{res}"
    with open(os.path.join(RES, f"e1_report{suffix}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    with open(os.path.join(RES, f"e1_summary{suffix}.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(hdr)
        w.writerows(summ)
    print("wrote", os.path.join(RES, f"e1_report{suffix}.md"), len(summ), "files")




def afrac_report():
    """learning curve: gain vs the fraction of the cell's UMIs given as context (A)."""
    lines = ["# E1 learning curve: gain of the held-out part B given a context part A holding a fraction f of the UMIs\n",
             "Gains vs L1 on B (cross-fitted). b/nnzB = per held-out nonzero; bits/cell = per cell; I(A;B) is symmetric in A and B for the "
             "true process, so compare f and 1-f at equal bits/cell. mean N_B/N = 1-f.\n",
             "| file | f (context share) | cells | L1 b/nnzB | L2 gain | L3 seq gain | L3 best gain | L3dm best gain | L3 best bits/cell | L2 bits/cell | seq(L3seq) bits/cell |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    paths = {}
    for p in glob.glob(os.path.join(RES, "e1_afrac_*__interval__f*.npz")):
        b = os.path.basename(p)[len("e1_afrac_"):-4]
        sample, member, res, f = b.split("__")
        paths.setdefault((sample, member), []).append((int(f[1:]), p))
    for (sample, member), lst in sorted(paths.items()):
        mp = os.path.join(RES, f"e1_percell_{sample}__{member}__interval__cells.npz")
        if os.path.exists(mp):
            lst = lst + [(50, mp)]
        for f, p in sorted(lst):
            fd = FileData(p)
            sel = fd.select()
            z = fd.z
            nB = z["nnzB"].sum()
            L1 = sel["L1"]["ms"].sum()
            g = lambda k: (L1 - sel[k]["ms"].sum())
            lines.append(f"| {sample}/{member} | {f / 100:.2f} | {fd.n} | {L1 / nB:.3f} | {g('L2') / nB:+.3f} | {g('L3seq') / nB:+.3f} | "
                         f"{g('L3') / nB:+.3f} | {g('L3dm') / nB:+.3f} | {g('L3') / fd.n:+.1f} | {g('L2') / fd.n:+.1f} | {g('L3seq') / fd.n:+.1f} |")
    with open(os.path.join(RES, "e1_afrac_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="interval")
    ap.add_argument("--afrac", action="store_true", help="learning-curve report from e1_afrac_*.npz")
    a = ap.parse_args()
    if a.afrac:
        afrac_report()
    else:
        report(a.res)
