# SPDX-License-Identifier: MIT
"""Reads per UMI on matched molecule sets: the engine's per-molecule dump against Cell Ranger's molecule_info.h5.

The engine's ``reads_per_umi_hist.tsv.gz`` covers every molecule it emits (exonic, intronic and mixed); Cell Ranger 3.0's molecule_info
holds only molecules it assigns to a gene from transcriptome-compatible (exonic) reads. To compare like with like, this joins the two
molecule tables on (barcode, gene, UMI) over the cells both callers keep, and reports

* the molecule ratio per cell for every engine label (S exon-only, A exon and intron, U intron-only) against Cell Ranger,
* how many molecules each side has that the other lacks, by engine label,
* on the matched molecules: pooled reads-per-UMI histograms (16 bins, 16+ in the last), their total variation distance, the
  saturation each implies, the fraction of molecules with the same read count, and the per-cell mean reads per UMI correlation.

The engine dump comes from ``SINGLET_MOLECULE_DUMP=PATH`` (columns barcode, gene_id, umi, label, flags, diag, reads); label D molecules
(dropped by the multi-gene rules) are not in the engine histogram and are left out here too.

    python rpu_matched.py --dump molecules.tsv.gz --hist OUT/reads_per_umi_hist.tsv.gz --molinfo pbmc_1k_v3_molecule_info.h5 \
        --cr-filtered filtered_feature_bc_matrix.h5 --calls OUT/cell_calls.tsv --out result.json
"""

from __future__ import annotations

import argparse
import gzip
import json

import h5py
import numpy as np
import pandas as pd

BINS = 16
CODE = np.array(list("ACGT"))


def hist(reads: np.ndarray) -> np.ndarray:
    return np.bincount(np.minimum(reads, BINS) - 1, minlength=BINS)


def sat(h: np.ndarray) -> float:
    return float(1 - h.sum() / (h * np.arange(1, BINS + 1)).sum())


def tvd(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def decode_umis(u: np.ndarray, length: int, msb_first: bool) -> np.ndarray:
    i = np.arange(length)
    shifts = 2 * (length - 1 - i) if msb_first else 2 * i
    chars = np.ascontiguousarray(CODE[(u[:, None] >> shifts[None, :]) & 3].astype("<U1"))
    return chars.view(f"<U{length}").ravel()


def norm(bc: str) -> str:
    return bc.split("-")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--molinfo", required=True)
    ap.add_argument("--cr-filtered", required=True)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with h5py.File(a.cr_filtered, "r") as h:
        cr_cells = {norm(b.decode()) for b in h["matrix/barcodes"][:]}
    calls = pd.read_csv(a.calls, sep="\t", usecols=["barcode", "is_cell"])
    eng_cells = {norm(b) for b, c in zip(calls.barcode, calls.is_cell.astype(str)) if c in ("1", "true", "True", "TRUE")}
    cells = cr_cells & eng_cells

    # Cell Ranger molecules in the shared cells
    with h5py.File(a.molinfo, "r") as h:
        bcs = np.array([norm(b.decode()) for b in h["barcodes"][:]])
        bidx = h["barcode_idx"][:]
        keep = np.isin(bcs, sorted(cells))[bidx]
        umi = h["umi"][:][keep].astype(np.int64)
        fid = np.array([f.decode() for f in h["features/id"][:]])[h["feature_idx"][:][keep]]
        cnt = h["count"][:][keep].astype(np.int64)
        bc_cr = bcs[bidx[keep]]
    # Engine molecules in the shared cells
    d = pd.read_csv(a.dump, sep="\t", dtype={"barcode": str, "gene_id": str, "umi": str, "label": str, "flags": np.int64,
                                              "diag": np.int64, "reads": np.int64})
    d["barcode"] = d.barcode.map(norm)
    d = d[d.barcode.isin(cells) & (d.label != "D")]
    umi_len = int(d.umi.str.len().mode()[0])

    # The UMI bit order of molecule_info is checked, not assumed: keep the order that matches more engine molecules
    best = None
    for msb in (True, False):
        cr = pd.DataFrame({"barcode": bc_cr, "gene_id": fid, "umi": decode_umis(umi, umi_len, msb), "cr_reads": cnt})
        m = d.merge(cr, on=["barcode", "gene_id", "umi"], how="inner")
        if best is None or len(m) > len(best[1]):
            best = (cr, m, msb)
    cr, m, msb = best

    # engine histogram file: same cells (sanity: the dump reproduces the histogram)
    eh = np.zeros(BINS, np.int64)
    with gzip.open(a.hist, "rt") as f:
        f.readline()
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if norm(p[0]) in cells:
                eh += np.array(p[1:], np.int64)

    per_cell_cr = cr.groupby("barcode").size()
    res = {
        "cells_shared": len(cells),
        "umi_len": umi_len,
        "molinfo_umi_msb_first": msb,
        "molecules_cr": int(len(cr)),
        "molecules_engine": int(len(d)),
        "engine_hist_file_molecules": int(eh.sum()),
        "dump_hist_tvd_vs_hist_file": round(tvd(hist(d.reads.to_numpy()), eh), 6),
        "engine_by_label": {k: int(v) for k, v in d.label.value_counts().items()},
        "molecules_per_cell_median_ratio": {},
        "matched": {},
    }
    for name, sel in {"all": d, "S": d[d.label == "S"], "S+A": d[d.label.isin(["S", "A"])], "U": d[d.label == "U"]}.items():
        pc = sel.groupby("barcode").size().reindex(per_cell_cr.index, fill_value=0)
        res["molecules_per_cell_median_ratio"][name] = round(float(np.median(pc / per_cell_cr)), 4)
    mm = m[["barcode", "gene_id", "umi"]]
    res["matched"]["molecules"] = int(len(m))
    res["matched"]["fraction_of_cr"] = round(len(m) / len(cr), 5)
    res["matched"]["fraction_of_engine"] = round(len(m) / len(d), 5)
    res["matched"]["engine_unmatched_by_label"] = {k: int(v) for k, v in
                                                  d.merge(mm, how="left", indicator=True).query("_merge == 'left_only'").label.value_counts().items()}
    res["matched"]["fraction_of_engine_by_label"] = {k: round(float((m.label == k).sum() / max((d.label == k).sum(), 1)), 5) for k in ("S", "A", "U")}
    he, hc = hist(m.reads.to_numpy()), hist(m.cr_reads.to_numpy())
    res["matched"]["pooled_fraction_engine"] = [round(float(x), 5) for x in he / he.sum()]
    res["matched"]["pooled_fraction_cr"] = [round(float(x), 5) for x in hc / hc.sum()]
    res["matched"]["pooled_tvd"] = round(tvd(he, hc), 5)
    res["matched"]["saturation_engine"] = round(sat(he), 5)
    res["matched"]["saturation_cr"] = round(sat(hc), 5)
    res["matched"]["same_read_count"] = round(float((np.minimum(m.reads, BINS) == np.minimum(m.cr_reads, BINS)).mean()), 5)
    res["matched"]["engine_reads_minus_cr_median"] = float(np.median(m.reads - m.cr_reads))
    g = m.assign(re=np.minimum(m.reads, BINS), rc=np.minimum(m.cr_reads, BINS)).groupby("barcode")[["re", "rc"]].mean()
    res["matched"]["mean_rpu_pearson"] = round(float(np.corrcoef(g.re, g.rc)[0, 1]), 5)
    # all engine exonic-compatible molecules (S+A) against all Cell Ranger molecules
    sa = d[d.label.isin(["S", "A"])]
    hs, hall = hist(sa.reads.to_numpy()), hist(cr.cr_reads.to_numpy())
    res["engine_SA_vs_cr_all"] = {"pooled_tvd": round(tvd(hs, hall), 5), "saturation_engine": round(sat(hs), 5), "saturation_cr": round(sat(hall), 5)}
    hd = hist(d.reads.to_numpy())
    res["engine_all_vs_cr_all"] = {"pooled_tvd": round(tvd(hd, hall), 5), "saturation_engine": round(sat(hd), 5), "saturation_cr": round(sat(hall), 5)}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
