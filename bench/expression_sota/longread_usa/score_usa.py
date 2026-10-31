# SPDX-License-Identifier: MIT
"""Score Singlet / STARsolo Velocyto / alevin-fry USA against the long-read truth (plan: TRUTH_RULE.md, "Scoring plan (frozen)").

  score_usa.py X_DIR OUT.json   X_DIR = .../expression-sota/longread_usa (truth/molecules.parquet, singlet/, starsolo/, alevinfry/)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "extract"))
sys.path.insert(0, str(HERE))
import runio as R  # noqa: E402
from truth_usa import enc  # noqa: E402

U_, S_, A_ = 1, 0, 2
NAME = {S_: "S", U_: "U", A_: "A"}
Q = 0.9


def pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def truth_mats(mol: pd.DataFrame, col: str, cells: list[str], genes: list[str]):
    ci = {c: i for i, c in enumerate(cells)}
    gi = {g: i for i, g in enumerate(genes)}
    m = mol[mol[col] >= 0]
    r = m.CB.map(ci).to_numpy()
    c = m.gene.map(gi).to_numpy()
    ok = ~(pd.isna(r) | pd.isna(c))
    r, c, k = r[ok].astype(np.int64), c[ok].astype(np.int64), m[col].to_numpy()[ok]
    out = {}
    for name, v in (("S", S_), ("U", U_), ("A", A_)):
        s = k == v
        out[name] = sp.csr_matrix((np.ones(s.sum()), (r[s], c[s])), shape=(len(cells), len(genes)))
    return out


def at(M: sp.csr_matrix, r, c):
    return np.asarray(M[r, c]).ravel()


def tool_mats(name, X: Path, cells, genes):
    if name == "singlet":
        run = X / "singlet"
        s = R.load_singlet(run, cells)
        universe = genes
        E, U, A = (s[k].pad(universe) for k in ("E", "U", "A"))
        S = E.X - A.X
        S.data = np.maximum(S.data, 0)
        S.eliminate_zeros()
        return {"S": S, "U": U.X, "A": A.X}, E.present
    if name == "starsolo":
        v = {k: R.load_starsolo(X / "starsolo", "Velocyto", cells, layer=k).pad(genes) for k in ("spliced", "unspliced", "ambiguous")}
        return {"S": v["spliced"].X, "U": v["unspliced"].X, "A": v["ambiguous"].X}, v["unspliced"].present
    af = R.load_alevinfry(X / "alevinfry", cells)
    af = {k: m.pad(genes) for k, m in af.items()}
    return {"S": af["S"].X, "U": af["U"].X, "A": af["A"].X}, af["U"].present


def score_tool(T: dict, TT: dict, keep_cells: np.ndarray, genes_n: int) -> dict:
    """T: tool S/U/A csr (cells x genes); TT: truth S/U/A; keep_cells: row indices."""
    T = {k: v[keep_cells].tocsr() for k, v in T.items()}
    TT = {k: v[keep_cells].tocsr() for k, v in TT.items()}
    Nt = sum(TT.values()).tocsr()
    Nm = sum(T.values()).tocsr()
    res = {"cells": int(len(keep_cells)), "truth_molecules": int(Nt.sum()), "tool_molecules": int(Nm.sum())}
    res["share_all_molecules"] = {"truth_U": float(TT["U"].sum() / max(Nt.sum(), 1)), "tool_U": float(T["U"].sum() / max(Nm.sum(), 1)),
                                  "tool_S": float(T["S"].sum() / max(Nm.sum(), 1)), "tool_A": float(T["A"].sum() / max(Nm.sum(), 1)),
                                  "truth_S": float(TT["S"].sum() / max(Nt.sum(), 1)), "truth_A": float(TT["A"].sum() / max(Nt.sum(), 1))}
    co = Nt.tocoo()
    r, c = co.row.astype(np.int64), co.col.astype(np.int64)
    nt = co.data
    nm = at(Nm, r, c)
    ut, um = at(TT["U"], r, c), at(T["U"], r, c)
    st, sm = at(TT["S"], r, c), at(T["S"], r, c)
    at_, am = at(TT["A"], r, c), at(T["A"], r, c)
    cov = np.where(nm > 0, nt / np.maximum(nm, 1), 0.0)
    res["pairs_with_truth"] = int(len(r))
    res["share_on_truth_pairs"] = {"truth_U": float(ut.sum() / nt.sum()), "tool_U": float(um.sum() / max(nm.sum(), 1))}
    for tag, sel in (("q90", (cov >= Q) & (nm > 0)), ("q100", (cov >= 1.0) & (nm > 0)), ("truth_pairs_all", np.ones(len(r), bool))):
        d = {"pairs": int(sel.sum())}
        if sel.sum() > 0:
            d["share"] = {"truth_U": float(ut[sel].sum() / nt[sel].sum()), "tool_U": float(um[sel].sum() / max(nm[sel].sum(), 1))}
            d["pair_pearson_log1p_U"] = pearson(np.log1p(ut[sel]), np.log1p(um[sel]))
            gsum_t = np.bincount(c[sel], weights=ut[sel], minlength=genes_n)
            gsum_m = np.bincount(c[sel], weights=um[sel], minlength=genes_n)
            g = (gsum_t > 0) | (gsum_m > 0)
            d["gene_pooled_pearson_log1p_U"] = pearson(np.log1p(gsum_t[g]), np.log1p(gsum_m[g]))
            d["genes"] = int(g.sum())
            # per-cell median
            order = np.argsort(r[sel], kind="stable")
            rr, a, b = r[sel][order], np.log1p(ut[sel][order]), np.log1p(um[sel][order])
            cuts = np.r_[0, np.flatnonzero(np.diff(rr)) + 1, len(rr)]
            rs = [pearson(a[i:j], b[i:j]) for i, j in zip(cuts[:-1], cuts[1:]) if j - i >= 20]
            rs = [x for x in rs if x == x]
            d["per_cell_median_pearson"] = float(np.median(rs)) if rs else float("nan")
            d["cells_with_ge20_pairs"] = len(rs)
            tp = int(((ut > 0) & (um > 0) & sel).sum())
            fp = int(((ut == 0) & (um > 0) & sel).sum())
            fn = int(((ut > 0) & (um == 0) & sel).sum())
            tn = int(((ut == 0) & (um == 0) & sel).sum())
            d["has_unspliced"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                                  "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
        res[tag] = d
    # single-molecule pairs: confusion truth class x tool class
    one = (nt == 1) & (nm == 1)
    conf = {}
    for a, av in (("S", st), ("U", ut), ("A", at_)):
        for b, bv in (("S", sm), ("U", um), ("A", am)):
            conf[f"truth_{a}->tool_{b}"] = int(((av == 1) & (bv == 1) & one).sum())
    res["single_molecule_pairs"] = {"n": int(one.sum()), "confusion": conf}
    return res


def starsolo_umi(X: Path, mol: pd.DataFrame, col: str) -> dict:
    """Per-UMI Velocyto classes of STAR-velodump joined to the truth molecules on (CB, UMI) and gene."""
    mk = (enc(mol.CB.tolist(), 16) << 22) | enc(mol.UB.tolist(), 11)
    ok = (mk >= 0) & (mol[col].to_numpy() >= 0)
    mk, tg, tc = mk[ok], mol.gene.to_numpy()[ok], mol[col].to_numpy()[ok]
    o = np.argsort(mk)
    mk, tg, tc = mk[o], tg[o], tc[o]
    code = {"S": S_, "U": U_, "A": A_}
    conf = np.zeros((3, 3), np.int64)
    cellkeys = np.unique(mk >> 22)
    rows_c = {"S": 0, "U": 0, "A": 0}   # STARsolo dump rows in truth cells, by STAR class
    hit_c = {"S": 0, "U": 0, "A": 0}    # ... of which (CB, UMI) is a T1 truth molecule (capture by the long reads, any gene)
    joined = same_gene = other = 0
    seen = np.zeros(len(mk), bool)
    n_dump = 0
    with gzip.open(X / "starsolo" / "velo_umis.tsv.gz", "rt") as f:
        for ch in pd.read_csv(f, sep="\t", header=None, names=["cb", "umi", "gene", "cls"], chunksize=5_000_000, dtype=str):
            n_dump += len(ch)
            k = (enc(ch.cb.tolist(), 16) << 22) | enc(ch.umi.tolist(), 11)
            j = np.minimum(np.searchsorted(mk, k), len(mk) - 1)
            hit = (k >= 0) & (mk[j] == k)
            joined += int(hit.sum())
            cbk = enc(ch.cb.tolist(), 16)
            incell = np.isin(cbk, cellkeys)
            for nm_ in ("S", "U", "A"):
                m_ = (ch.cls.to_numpy() == nm_)
                rows_c[nm_] += int((m_ & incell).sum())
                hit_c[nm_] += int((m_ & hit).sum())
            sub = ch[hit]
            jj = j[hit]
            sg = (sub.gene.to_numpy() == tg[jj])
            same_gene += int(sg.sum())
            seen[jj[sg]] = True
            tcls = tc[jj[sg]]
            scls = sub.cls.map(code).to_numpy()[sg]     # E / M (exon-only / multi-gene) rows have no S/U/A class: counted apart
            known = ~pd.isna(scls)
            other += int((~known).sum())
            np.add.at(conf, (tcls[known], scls[known].astype(int)), 1)
    names = ["S", "U", "A"]
    out = {"truth_molecules": int(len(mk)), "dump_rows": int(n_dump), "joined_on_cb_umi": joined, "joined_same_gene": same_gene, "joined_other_class_E_M": other,
           "truth_molecules_recovered": int(seen.sum()),
           "confusion_truth_rows_tool_cols": {f"{names[i]}->{names[j]}": int(conf[i, j]) for i in range(3) for j in range(3)}}
    out["capture_by_star_class"] = {k: {"star_umis_in_truth_cells": rows_c[k], "in_truth": hit_c[k],
                                        "fraction": hit_c[k] / max(rows_c[k], 1)} for k in rows_c}
    tot = conf.sum()
    out["accuracy"] = float(np.trace(conf) / max(tot, 1))
    out["share_U"] = {"truth": float(conf[1].sum() / max(tot, 1)), "starsolo": float(conf[:, 1].sum() / max(tot, 1))}
    return out


def main():
    X = Path(sys.argv[1])
    mol = pd.read_parquet(X / "truth" / "molecules.parquet")
    cells_all = sorted(mol.CB.unique())
    genes = sorted(mol.gene.unique())
    out = {"truth_cells": len(cells_all), "truth_genes": len(genes), "molecules": int(len(mol))}
    tools = {}
    for name in ("singlet", "starsolo", "alevinfry"):
        try:
            tools[name] = tool_mats(name, X, cells_all, genes)
        except Exception as e:  # noqa: BLE001
            out[f"{name}_load_error"] = repr(e)
    present = np.ones(len(cells_all), bool)
    for n, (_, p) in tools.items():
        present &= p
        out[f"{n}_cells_reported"] = int(p.sum())
    keep_common = np.flatnonzero(present)
    out["common_cells"] = int(len(keep_common))
    for col in ("clsLR", "clsR2", "clsLR5", "clsLR20", "clsLR_pb", "clsLR_ont"):
        TT = truth_mats(mol, col, cells_all, genes)
        d = {"common": {}, "own_cells": {}}
        for n, (T, p) in tools.items():
            d["common"][n] = score_tool(T, TT, keep_common, len(genes))
            if col == "clsLR":
                d["own_cells"][n] = score_tool(T, TT, np.flatnonzero(p), len(genes))
        out[col] = d
        print(f"[score] {col} done", file=sys.stderr, flush=True)
    if (X / "starsolo" / "velo_umis.tsv.gz").exists():
        out["starsolo_per_umi"] = {c: starsolo_umi(X, mol, c) for c in ("clsLR", "clsR2")}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
