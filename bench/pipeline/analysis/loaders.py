"""Load every tool's output for one dataset as gene-level cells x genes matrices.

All loaders return a dict of name -> anndata.AnnData (cells x genes, integer counts,
obs_names = bare 16-mer barcodes without the "-1" suffix, var_names = Ensembl gene ids
without version). Layers follow the velocyto vocabulary: spliced / unspliced / ambiguous.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def _bare(bc: str) -> str:
    return bc.split("-")[0]


def _gid(g: str) -> str:
    return g.split(".")[0]


def _read_lines(p: Path) -> list[str]:
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as f:
        return [ln.rstrip("\n") for ln in f]


def _mtx(p: Path) -> sp.csr_matrix:
    return sp.csr_matrix(scipy.io.mmread(str(p)))


def _collapse(features: list[str], m: sp.csr_matrix, gene_of) -> tuple[list[str], sp.csr_matrix]:
    """Sum feature rows (features x cells) into gene rows; returns (genes, genes x cells)."""
    genes = [gene_of(f) for f in features]
    uniq = sorted(set(genes))
    idx = {g: i for i, g in enumerate(uniq)}
    rows = np.array([idx[g] for g in genes])
    agg = sp.csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(len(uniq), len(rows)))
    return uniq, sp.csr_matrix(agg @ m)


def _feature_gene_resolver(gene_ids: list[str]):
    """Singlet feature names are '<gene_id>_<gene_name>_[intron_]<chrom>:<s>-<e>'; gene ids can
    themselves contain '_' (barnyard prefixes), so match the longest known gene-id prefix."""
    known = set(gene_ids)

    def gene_of(name: str) -> str:
        parts = name.split("_")
        for k in range(len(parts) - 1, 0, -1):
            cand = "_".join(parts[:k])
            if cand in known:
                return cand
        return parts[0]

    return gene_of


def load_singlet(out: Path) -> ad.AnnData | None:
    """Singlet --output-format mtx: exon_counts / intron_counts (feature level, all barcodes
    passed to the pileup), ambiguous_counts (gene level, fixed engine only), cell_calls.tsv."""
    if not (out / "exon_counts.mtx.gz").exists():
        return None
    raw_dir = out / "raw_feature_bc_matrix"
    if not raw_dir.exists():
        raw_dir = out / "filtered_feature_bc_matrix"
    feats = pd.read_csv(raw_dir / "features.tsv.gz", sep="\t", header=None)
    gene_ids = feats[0].astype(str).tolist()
    gene_of = _feature_gene_resolver(gene_ids)

    def part(prefix: str):
        m = _mtx(out / f"{prefix}.mtx.gz")
        f = _read_lines(out / f"{prefix}_features.tsv.gz")
        b = _read_lines(out / f"{prefix}_barcodes.tsv.gz")
        return f, m, b

    ef, em, eb = part("exon_counts")
    genes_e, ge = _collapse(ef, em, gene_of)
    layers = {}
    all_genes = sorted(set(gene_ids) | set(genes_e))
    gidx = {g: i for i, g in enumerate(all_genes)}

    def to_full(genes, m):
        rows = np.array([gidx[g] for g in genes])
        c = m.tocoo()
        return sp.csr_matrix((c.data, (rows[c.row], c.col)), shape=(len(all_genes), m.shape[1]))

    exon = to_full(genes_e, ge)
    intron = sp.csr_matrix(exon.shape)
    if (out / "intron_counts.mtx.gz").exists():
        f, m, b = part("intron_counts")
        assert b == eb, "intron and exon barcodes differ"
        gi, mi = _collapse(f, m, gene_of)
        intron = to_full(gi, mi)
    amb = sp.csr_matrix(exon.shape)
    if (out / "ambiguous_counts.mtx.gz").exists():
        f, m, b = part("ambiguous_counts")
        ga = [gene_of(x) for x in f]  # '<gene_id>_<gene_name>'
        amb = to_full(ga, m)
    spliced = exon - amb
    a = ad.AnnData(
        X=sp.csr_matrix((exon + intron).T),
        obs=pd.DataFrame(index=[_bare(x) for x in eb]),
        var=pd.DataFrame(index=[_gid(g) for g in all_genes]),
    )
    a.layers["exon"] = sp.csr_matrix(exon.T)
    a.layers["spliced"] = sp.csr_matrix(spliced.T)
    a.layers["unspliced"] = sp.csr_matrix(intron.T)
    a.layers["ambiguous"] = sp.csr_matrix(amb.T)
    a.var_names_make_unique()
    calls = out / "cell_calls.tsv"
    a.obs["is_cell"] = False
    if calls.exists():
        cc = pd.read_csv(calls, sep="\t")
        col = "is_cell" if "is_cell" in cc.columns else cc.columns[-1]
        cells = {_bare(b) for b, v in zip(cc["barcode"], cc[col]) if str(v).lower() in ("1", "true")}
        a.obs["is_cell"] = a.obs_names.isin(cells)
    stats = out / "pileup_stats.json"
    if stats.exists():
        a.uns["pileup_stats"] = json.loads(stats.read_text())
    return a


def _starsolo_matrix(d: Path, name="matrix.mtx") -> tuple[sp.csr_matrix, list[str], list[str]]:
    m = _mtx(d / name)
    feats = pd.read_csv(d / "features.tsv", sep="\t", header=None)[0].astype(str).tolist()
    bcs = _read_lines(d / "barcodes.tsv")
    return m, feats, bcs


def load_starsolo(out: Path) -> dict[str, ad.AnnData]:
    """STARsolo Solo.out/{Gene,GeneFull}/raw (+ filtered barcodes) and Velocyto/raw S/U/A."""
    res = {}
    solo = out / "Solo.out"
    for feat in ("Gene", "GeneFull"):
        d = solo / feat / "raw"
        if not d.exists():
            continue
        m, feats, bcs = _starsolo_matrix(d)
        a = ad.AnnData(X=sp.csr_matrix(m.T), obs=pd.DataFrame(index=[_bare(b) for b in bcs]),
                       var=pd.DataFrame(index=[_gid(g) for g in feats]))
        fb = solo / feat / "filtered" / "barcodes.tsv"
        cells = {_bare(b) for b in _read_lines(fb)} if fb.exists() else set()
        a.obs["is_cell"] = a.obs_names.isin(cells)
        res[f"starsolo_{feat}"] = a
    v = solo / "Velocyto" / "raw"
    if v.exists():
        parts = {k: _starsolo_matrix(v, f"{k}.mtx") for k in ("spliced", "unspliced", "ambiguous")}
        _, feats, bcs = parts["spliced"]
        X = sum(sp.csr_matrix(parts[k][0].T) for k in parts)
        a = ad.AnnData(X=X, obs=pd.DataFrame(index=[_bare(b) for b in bcs]),
                       var=pd.DataFrame(index=[_gid(g) for g in feats]))
        for k in parts:
            a.layers[k] = sp.csr_matrix(parts[k][0].T)
        if "starsolo_Gene" in res:
            a.obs["is_cell"] = a.obs_names.isin(res["starsolo_Gene"].obs_names[res["starsolo_Gene"].obs["is_cell"]])
        res["starsolo_Velocyto"] = a
    return res


def load_alevinfry(out: Path) -> ad.AnnData | None:
    """simpleaf quant USA mode: alevin/quants_mat.mtx is cells x (S genes | U genes | A genes)."""
    q = next(iter(out.glob("af/af_quant/alevin")), None)
    if q is None or not (q / "quants_mat.mtx").exists():
        return None
    m = sp.csr_matrix(scipy.io.mmread(str(q / "quants_mat.mtx")))
    cols = _read_lines(q / "quants_mat_cols.txt")
    rows = _read_lines(q / "quants_mat_rows.txt")
    n = len(cols) // 3 if len(cols) % 3 == 0 and m.shape[1] == len(cols) else None
    if m.shape[1] == 3 * len(cols):  # newer layout: cols lists genes once
        genes = cols
        n = len(cols)
    else:
        genes = cols[:n]
    S, U, A = m[:, :n], m[:, n:2 * n], m[:, 2 * n:3 * n]
    a = ad.AnnData(X=sp.csr_matrix(S + U + A), obs=pd.DataFrame(index=[_bare(b) for b in rows]),
                   var=pd.DataFrame(index=[_gid(g) for g in genes]))
    a.layers["spliced"], a.layers["unspliced"], a.layers["ambiguous"] = (sp.csr_matrix(x) for x in (S, U, A))
    a.obs["is_cell"] = False
    a.var_names_make_unique()
    return a


def load_cellranger_h5(p: Path) -> tuple[ad.AnnData | None, ad.AnnData | None]:
    """10x-published Cell Ranger filtered matrix: (Gene Expression, Antibody Capture)."""
    if not p.exists():
        return None, None
    import h5py

    with h5py.File(p, "r") as f:
        g = f["matrix"]
        m = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
        bcs = [_bare(b.decode()) for b in g["barcodes"][:]]
        ids = [x.decode() for x in g["features"]["id"][:]]
        ftype = [x.decode() for x in g["features"]["feature_type"][:]]
    ftype = np.array(ftype)
    out = []
    for t in ("Gene Expression", "Antibody Capture"):
        sel = np.where(ftype == t)[0]
        if len(sel) == 0:
            out.append(None)
            continue
        a = ad.AnnData(X=sp.csr_matrix(m[sel].T), obs=pd.DataFrame(index=bcs),
                       var=pd.DataFrame(index=[_gid(ids[i]) for i in sel]))
        a.obs["is_cell"] = True
        a.var_names_make_unique()
        out.append(a)
    return out[0], out[1]


def load_time(p: Path) -> dict:
    """GNU time -v output -> wall seconds, max RSS GB, CPU seconds."""
    if not p.exists():
        return {}
    d = {}
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("Elapsed (wall clock) time"):
            t = ln.split(": ", 1)[1].split(":")
            secs = sum(float(x) * 60 ** i for i, x in enumerate(reversed(t)))
            d["wall_s"] = secs
        elif ln.startswith("Maximum resident set size"):
            d["max_rss_gb"] = int(ln.split(": ")[1]) / 1024**2
        elif ln.startswith("User time"):
            d["user_s"] = float(ln.split(": ")[1])
        elif ln.startswith("System time"):
            d["sys_s"] = float(ln.split(": ")[1])
    if "user_s" in d:
        d["cpu_s"] = d["user_s"] + d.get("sys_s", 0.0)
    return d


def cross_worker_exon_hits(run_log: Path) -> int | None:
    """Atlas engine only: '[parallel-cross-worker] N cross-worker entries -> M exon hits, M UMI
    unique'. Those M molecules were deduplicated apart from the workers' molecules."""
    import re

    if not run_log.exists():
        return None
    for ln in run_log.read_text(errors="replace").splitlines():
        m = re.search(r"\[parallel-cross-worker\].*?(\d+) exon hits", ln)
        if m:
            return int(m.group(1))
    return None
