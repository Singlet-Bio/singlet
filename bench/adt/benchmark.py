# SPDX-License-Identifier: MIT
"""Method benchmark for ADT counting against Cell Ranger's own output (10x pbmc_1k_protein_v3, TotalSeq-B, 3' v3).

Methods, all on the same antibody FASTQs and scored on Cell Ranger's filtered barcodes:

* ``singlet``: :func:`singlet.adt.count.count_library` (Cell Ranger feature-barcoding logic, panel from the h5).
* ``singlet_discovered``: the same with the panel identified from the reads against the TotalSeq catalog.
* ``kite``: kallisto | bustools via ``kb count --workflow kite:10xFB`` (Hamming-1 feature map, bustools whitelist
  correction and 10xFB barcode translation).
* ``adt_counter_h``: ``bench/adt/adt_counter_harness.cpp`` (the pipeline's ``adt_matcher.h`` + ``adt_counter.h``),
  once with the pipeline's offsets 0..5 and once with 0..10.
* ``cellranger``: the truth, ``filtered_feature_bc_matrix.h5`` (Cell Ranger 3.0.0).

Writes ``benchmark.json`` (per method: total UMIs, ratio to Cell Ranger, per-cell / per-entry Pearson and Spearman of
log1p counts, per-antibody totals ratio, run time) and ``benchmark_entries.csv.gz`` (subsample of matrix entries).

    python -P bench/adt/benchmark.py --fastq-dir DIR --h5 H5 --out OUT --work SCRATCH [--threads 8]
"""

import argparse
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse, stats


def read_cr_h5(path):
    import h5py

    with h5py.File(path, "r") as f:
        g = f["matrix"]
        M = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
        feat = {k: [x.decode() for x in g["features"][k][:]] for k in g["features"].keys()
                if g["features"][k].dtype.kind in "SO"}
        bcs = [x.decode() for x in g["barcodes"][:]]
    ft = pd.DataFrame({k: v for k, v in feat.items() if len(v) == M.shape[0]})
    return M, ft, bcs


def fetch(url, dest):
    if not os.path.exists(dest):
        subprocess.run(["curl", "-sSfL", "--retry", "5", "-o", dest + ".part", url], check=True)
        os.replace(dest + ".part", dest)
    return dest


def score(name, M, features, cr, cr_features, seconds, extra=None):
    """M: features x CR barcodes (same order), rows named by ``features``; cr: CR antibody matrix."""
    common = [f for f in cr_features if f in features]
    ri = [features.index(f) for f in common]
    ci = [cr_features.index(f) for f in common]
    A = M[ri].toarray().astype(float)
    T = cr[ci].toarray().astype(float)
    la, lt = np.log1p(A), np.log1p(T)
    per_cell = [stats.pearsonr(la[:, j], lt[:, j])[0] for j in range(A.shape[1]) if lt[:, j].std() > 0 and la[:, j].std() > 0]
    out = {
        "method": name, "seconds": round(seconds, 1), "n_features_common": len(common),
        "umis": float(A.sum()), "umis_cellranger": float(T.sum()), "ratio": float(A.sum() / max(T.sum(), 1)),
        "pearson_log_entries": float(stats.pearsonr(la.ravel(), lt.ravel())[0]),
        "spearman_entries": float(stats.spearmanr(A.ravel(), T.ravel())[0]),
        "pearson_cell_totals": float(stats.pearsonr(np.log1p(A.sum(0)), np.log1p(T.sum(0)))[0]),
        "median_per_cell_pearson": float(np.median(per_cell)) if per_cell else None,
        "exact_entries": float((A == T).mean()),
        "abs_diff_share": float(np.abs(A - T).sum() / max(T.sum(), 1)),
        "per_feature_ratio": {f: float(A[i].sum() / max(T[i].sum(), 1)) for i, f in enumerate(common)},
    }
    if extra:
        out.update(extra)
    return out


def run_singlet(r1, r2, panel_df, wl_url, tr_url, cr_bcs, work, discovered=False):
    from singlet.adt import count as C
    from singlet.adt import panel as P
    from singlet.adt import seq as S

    t0 = time.time()
    wl_path = fetch(wl_url, os.path.join(work, os.path.basename(wl_url)))
    tr_path = fetch(tr_url, os.path.join(work, os.path.basename(tr_url)))
    # decide offset / translation / panel from the first 500k reads
    head = []
    for i, (a, b) in enumerate(C.iter_pairs(r1, r2)):
        head.append((a, b))
        if i >= 500_000:
            break
    cat = P.load_catalog()
    offs = P.detect_offset([b for _, b in head], list(cat.sequence))
    off = max(offs, key=offs.get)
    gex_codes = np.sort(S.encode_strings([b.split("-")[0] for b in cr_bcs], 16)[0])
    top_cb, _ = S.encode_strings([a[:16].decode() for a, _ in head[:200_000] if b"N" not in a[:16]], 16)
    table, ch = C.best_translation(top_cb, gex_codes, tr_path)
    notes = {"offsets": {k: round(v, 4) for k, v in offs.items() if v > 0.01}, "offset": off, "translation": ch}
    if discovered:
        tcb = C.translate(top_cb, table)[0] if ch["use"] else top_cb
        incell = S.lookup(gex_codes, tcb) >= 0
        sel = [b for a, b in head[:200_000] if b"N" not in a[:16]]  # same reads, same order as top_cb
        feats = [b[off:off + 15] for b, ok in zip(sel, incell) if ok]
        pan = P.discover(feats, cat)
        panel_df = pan.features
        notes["discovery"] = {"status": pan.status, **{k: v for k, v in pan.notes.items() if k != "unexplained_top"},
                              "unexplained_top": pan.notes["unexplained_top"][:5]}
    wl = np.sort(table[0]) if ch["use"] else np.sort(S.encode_strings(C.load_barcode_list(wl_path), 16)[0])
    lc = C.count_library(r1, r2, cb_len=16, umi_len=12, offset=off, panel_sequences=list(panel_df.sequence),
                         whitelist_codes=wl, gex_barcodes=cr_bcs, translation=table if ch["use"] else None)
    return lc, list(panel_df.name), time.time() - t0, {**notes, "stats": lc.stats}, panel_df


def run_kite(r1, r2, panel_df, cr_bcs, work, threads):
    t0 = time.time()
    d = os.path.join(work, "kite")
    os.makedirs(d, exist_ok=True)
    ft = os.path.join(d, "features.tsv")
    panel_df[["sequence", "name"]].to_csv(ft, sep="\t", header=False, index=False)
    subprocess.run(["kb", "ref", "-i", f"{d}/idx", "-g", f"{d}/t2g.txt", "-f1", f"{d}/fa.fa", "--workflow", "kite:10xFB",
                    ft], check=True, capture_output=True)
    fq = [x for pair in zip(r1, r2) for x in pair]
    subprocess.run(["kb", "count", "-i", f"{d}/idx", "-g", f"{d}/t2g.txt", "-x", "10xv3", "--workflow", "kite:10xFB",
                    "-t", str(threads), "-o", f"{d}/out", "--h5ad"] + fq, check=True, capture_output=True)
    import anndata

    h = glob.glob(f"{d}/out/counts_unfiltered/*.h5ad")[0]
    ad = anndata.read_h5ad(h)
    names = list(ad.var_names)
    bc_index = {b: i for i, b in enumerate(ad.obs_names)}
    X = sparse.csr_matrix(ad.X)
    cols = []
    for b in cr_bcs:
        i = bc_index.get(b.split("-")[0])
        cols.append(X[i].toarray().ravel() if i is not None else np.zeros(X.shape[1]))
    M = sparse.csc_matrix(np.array(cols).T)
    return M, names, time.time() - t0


def run_harness(r1, r2, panel_df, cr_bcs, work, max_off, src):
    t0 = time.time()
    exe = os.path.join(work, "adt_counter_harness")
    if not os.path.exists(exe):
        cxx = os.environ.get("CXX", "g++")
        subprocess.run([cxx, "-O2", "-std=c++17", "-I", os.path.join(src, "include"),
                        os.path.join(src, "bench/adt/adt_counter_harness.cpp"), "-lz", "-o", exe], check=True)
    ref = os.path.join(work, "harness_ref.csv")
    panel_df[["name", "sequence"]].to_csv(ref, index=False, header=False)
    bcf = os.path.join(work, "harness_bcs.txt")
    with open(bcf, "w") as f:
        f.write("\n".join(b.split("-")[0] for b in cr_bcs) + "\n")
    # lanes of one library share molecules: concatenate (gzip members) so the exact dedup sees them together
    cat1, cat2 = os.path.join(work, "h_R1.fastq.gz"), os.path.join(work, "h_R2.fastq.gz")
    for src_list, dst in ((r1, cat1), (r2, cat2)):
        if not os.path.exists(dst):
            with open(dst, "wb") as fo:
                for x in src_list:
                    with open(x, "rb") as fi:
                        shutil.copyfileobj(fi, fo)
    out = os.path.join(work, f"harness_{max_off}")
    subprocess.run([exe, cat1, cat2, ref, bcf, "16", "12", str(max_off), out], check=True)
    total = sparse.csc_matrix(_read_mtx(out + ".mtx"))
    js_all = [json.load(open(out + ".json"))]
    return total, list(panel_df.name), time.time() - t0, {"lanes": js_all}


def _read_mtx(p):
    from scipy.io import mmread

    return mmread(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq-dir", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--src", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--methods", default="singlet,singlet_discovered,kite,harness5,harness10")
    a = ap.parse_args()
    from singlet.adt.chemistry import WL_BASE

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    r1 = sorted(glob.glob(os.path.join(a.fastq_dir, "*_R1_*.fastq.gz")))
    r2 = [x.replace("_R1_", "_R2_") for x in r1]
    M, ft, bcs = read_cr_h5(a.h5)
    ab = (ft.feature_type == "Antibody Capture").values
    cr = M[np.flatnonzero(ab)]
    crf = ft[ab].reset_index(drop=True)
    panel_df = pd.DataFrame({"name": crf["name"], "sequence": crf["sequence"], "id": crf["id"]})
    print(f"Cell Ranger: {len(bcs)} cells, {ab.sum()} antibodies, {int(cr.sum())} UMIs; patterns {set(crf.get('pattern', []))}",
          flush=True)
    res = []
    meth = a.methods.split(",")
    wl, tr = WL_BASE + "3M-february-2018.txt.gz", WL_BASE + "translation_3M-february-2018.txt.gz"
    if "singlet" in meth:
        lc, names, secs, notes, _ = run_singlet(r1, r2, panel_df, wl, tr, bcs, a.work)
        res.append(score("singlet", lc.matrix, names, cr, list(crf["name"]), secs, {"notes": notes}))
        print(json.dumps(res[-1])[:600], flush=True)
    if "singlet_discovered" in meth:
        lc, names, secs, notes, pdf = run_singlet(r1, r2, panel_df, wl, tr, bcs, a.work, discovered=True)
        # discovered names are catalog names: map back to CR names by sequence for scoring
        seq2cr = dict(zip(crf["sequence"], crf["name"]))
        names = [seq2cr.get(s, n) for s, n in zip(pdf.sequence, pdf.name)]
        notes["panel_vs_cellranger"] = {"cr_only": sorted(set(crf["sequence"]) - set(pdf.sequence)),
                                       "discovered_only": sorted(set(pdf.sequence) - set(crf["sequence"]))}
        res.append(score("singlet_discovered", lc.matrix, names, cr, list(crf["name"]), secs, {"notes": notes}))
        print(json.dumps(res[-1])[:600], flush=True)
    if "kite" in meth and shutil.which("kb"):
        try:
            Mk, names, secs = run_kite(r1, r2, panel_df, bcs, a.work, a.threads)
            res.append(score("kite", Mk, names, cr, list(crf["name"]), secs))
        except subprocess.CalledProcessError as e:
            res.append({"method": "kite", "error": (e.stderr or b"").decode()[-2000:]})
        print(json.dumps(res[-1])[:600], flush=True)
    for mo in (5, 10):
        if f"harness{mo}" in meth:
            Mh, names, secs, notes = run_harness(r1, r2, panel_df, bcs, a.work, mo, a.src)
            res.append(score(f"adt_counter_h_offset0-{mo}", Mh, names, cr, list(crf["name"]), secs, {"notes": notes}))
            print(json.dumps(res[-1])[:600], flush=True)
    with open(os.path.join(a.out, "benchmark.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
