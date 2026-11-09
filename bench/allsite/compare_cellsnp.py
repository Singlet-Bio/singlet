#!/usr/bin/env python3
"""D-101 allele counts against cellsnp-lite on the same reads (counts only; no site, genotype or count leaves this script).

  compare_cellsnp.py <singlet out dir> <cellsnp-lite dir> <out.json>

The cellsnp-lite run is mode 1 on the engine's own tagged BAM, at the engine's all-site sites (private/allsite/sites.tsv.gz as -R),
over the engine's called cells (-b), with --minCOUNT 0 --minMAF 0 so every given site is reported, --cellTAG CB --UMItag UB.
Matched on (chrom, pos, alt) and barcode. The engine's entries are the stored ones (informative against the donor genotype), so cellsnp-lite
entries the store dropped are counted apart and the per-site sums cover stored entries only. Reports per-entry and per-site agreement of DP (ref + alt UMIs) and AD (alt UMIs).
"""
import gzip, json, os, sys
import numpy as np
import scipy.io, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_reader import read_delta


def lines(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return [l.rstrip("\n") for l in f]


def main(sdir, cdir, out):
    barcodes, sites, er, ec, ead, edp = read_delta(os.path.join(sdir, "private/allsite"))
    feats = [f"{c}:{p}:{r}>{a}" for c, p, r, a in sites]
    bcs = barcodes
    sad = sp.coo_matrix((ead, (er, ec)), shape=(len(feats), len(bcs))).tocsr()   # sites x barcodes, the stored entries
    sdp = sp.coo_matrix((edp, (er, ec)), shape=(len(feats), len(bcs))).tocsr()
    key = {}
    for i, n in enumerate(feats):
        c, p, ra = n.split(":"); key[(c, int(p), ra[2])] = i
    cbase = [l.split("\t") for l in lines(os.path.join(cdir, "cellSNP.base.vcf.gz")) if not l.startswith("#")]
    csam = lines(os.path.join(cdir, "cellSNP.samples.tsv"))
    cad = scipy.io.mmread(os.path.join(cdir, "cellSNP.tag.AD.mtx")).tocsr()
    cdp = scipy.io.mmread(os.path.join(cdir, "cellSNP.tag.DP.mtx")).tocsr()
    # rows: cellsnp sites that match an engine site (same alt)
    rows_c, rows_s = [], []
    for j, t in enumerate(cbase):
        i = key.get((t[0], int(t[1]), t[4]))
        if i is not None: rows_c.append(j); rows_s.append(i)
    bidx = {b: k for k, b in enumerate(bcs)}
    strip = lambda b: b[:-2] if b.endswith("-1") and b[:-2] in bidx else b
    cols_c = [k for k, b in enumerate(csam) if strip(b) in bidx]
    cols_s = [bidx[strip(csam[k])] for k in cols_c]
    A_s = sad[rows_s][:, cols_s].tocsr(); D_s = sdp[rows_s][:, cols_s].tocsr()
    A_c = cad[rows_c][:, cols_c].tocsr(); D_c = cdp[rows_c][:, cols_c].tocsr()
    # entries: union of nonzero DP
    U = ((D_s != 0) + (D_c != 0)).tocoo()
    ds = np.asarray(D_s[U.row, U.col]).ravel(); dc = np.asarray(D_c[U.row, U.col]).ravel()
    as_ = np.asarray(A_s[U.row, U.col]).ravel(); ac = np.asarray(A_c[U.row, U.col]).ravel()
    r = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and x.std() > 0 and y.std() > 0 else float("nan")
    psd = np.asarray(D_s.sum(1)).ravel(); pcd = np.asarray(D_c.sum(1)).ravel()
    psa = np.asarray(A_s.sum(1)).ravel(); pca = np.asarray(A_c.sum(1)).ravel()
    res = {
        "engine_sites": len(feats), "cellsnp_sites": len(cbase), "matched_sites": len(rows_s),
        "engine_cells": len(bcs), "cellsnp_cells": len(csam), "matched_cells": len(cols_s),
        "entries_union": int(len(ds)),
        "entries_dp_equal": float(np.mean(ds == dc)) if len(ds) else None,
        "entries_ad_equal": float(np.mean(as_ == ac)) if len(ds) else None,
        "entries_both_equal": float(np.mean((ds == dc) & (as_ == ac))) if len(ds) else None,
        "entries_engine_only": int(np.sum((ds > 0) & (dc == 0))), "entries_cellsnp_only": int(np.sum((ds == 0) & (dc > 0))),
        "stored_entries": int(len(ead)),
        "entries_dp_equal_where_stored": float(np.mean((ds == dc)[ds > 0])) if np.any(ds > 0) else None,
        "entries_ad_equal_where_stored": float(np.mean((as_ == ac)[ds > 0])) if np.any(ds > 0) else None,
        "cellsnp_entries_not_stored": int(np.sum((ds == 0) & (dc > 0))), "cellsnp_entries_not_stored_with_alt": int(np.sum((ds == 0) & (dc > 0) & (ac > 0))),
        "r_entry_dp": r(ds, dc), "r_entry_ad": r(as_, ac),
        "r_site_dp": r(psd, pcd), "r_site_ad": r(psa, pca),
        "sum_dp_engine": int(psd.sum()), "sum_dp_cellsnp": int(pcd.sum()),
        "sum_ad_engine": int(psa.sum()), "sum_ad_cellsnp": int(pca.sum()),
        "site_af_abs_diff_median_dp10": float(np.median(np.abs(psa / np.maximum(psd, 1) - pca / np.maximum(pcd, 1))[(psd >= 10) & (pcd >= 10)]))
        if np.any((psd >= 10) & (pcd >= 10)) else None,
    }
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
