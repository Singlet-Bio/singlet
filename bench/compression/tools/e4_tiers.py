"""E4 Part B (bytes): called-cells member and empties member under three accounting variants.

For every corpus sample and each of exon / intron (and, for the bundle sample GSM4317429 = tumor_3p, also sj and vdj):
  subset 'orig'    the file as stored today
  subset 'all'     all columns re-encoded (sanity: should match 'orig' up to the zstd build)
  subset 'cells'   called cells only            (member 1)
  subset 'empties' empty droplets only          (member 2)
and, for each subset, the bytes under
  (i)   today's writer (singlet::pz::write_1pz through the compiled python binding): names, permutation, ptr and
        colsums all still inside the member
  (ii)  names + permutation + ptr + colsums removed (Track A's measured costs: rownames -> 8 B vocabulary id,
        permutation -> per-row nnz counts, ptr/colsums derived, barcodes -> whitelist index); chunk payload unchanged
  (iii) (ii) with the chunk payload replaced by the best payload code length available in results/ (see below)

Payload sources for (iii), all from results/ at the time of the run:
  realized all-column codecs  track_b_codecs.tsv (b_rans16 payload bits/nnz) and track_c_cv.tsv (out-of-sample c_* rows)
  per-subset code lengths     track_d_entropy.tsv: M2 (per-feature x depth presence + rate-context values, 2-fold CV)
                              for cells; min(M2, amb_loo) for empties; side_H0_kj_mdl (per-column UMI totals) as side
                              information. These are the only per-subset numbers on disk; on the 'all' subset their
                              nnz-weighted sum agrees with the best realized codec within a few percent (printed).
  'iii_uniform'               best realized all-column bits/nnz applied to every subset (sensitivity)

    python -P tools/e4_tiers.py [--samples a,b] [--no-bundle-extra]
Output: results/e4_tiers.tsv   (one row per sample x member x subset)
"""
import argparse
import csv
import math
import os
import sys
import time
import zlib
import zipfile
import json

import numpy as np
import zstandard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_lib as L  # noqa: E402
from e4_audit import rowcounts_cost  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------------------------------------------------
# results/ lookups
# ---------------------------------------------------------------------------------------------------------------------
def load_sources():
    R = L.RESULTS
    src = {}
    # Track A names: whitelist-index and canonical sorted-set bytes for the full barcode list of each sample
    wl, canon = {}, {}
    for r in read_tsv(os.path.join(R, "track_a_names.tsv")):
        if r["field"] != "colnames" or not r["member"].startswith("exon"):
            continue
        if r["scheme"].startswith("whitelist_index"):
            wl[r["sample"]] = int(r["bytes"])
        elif r["scheme"].startswith("sorted_set_bound"):
            canon[r["sample"]] = int(r["bytes"])
    src["wl"], src["canon"] = wl, canon
    # Track B / C realized all-column payload bits per nnz
    best = {}
    for r in read_tsv(os.path.join(R, "track_b_codecs.tsv")):
        if r["matrix_ok"] != "1" or r["codec"] != "b_rans16":
            continue
        k = (r["sample"], r["member"].split("_")[0])
        v = float(r["payload_bits_nnz"])
        if k not in best or v < best[k][0]:
            best[k] = (v, "track_b:b_rans16")
    p = os.path.join(R, "track_c_cv.tsv")
    for r in read_tsv(p):
        if r["matrix_ok"] != "1" or r["codec"] in ("c_r1_insample", "c_r2_insample", "c_r2p"):
            continue
        if r["model_tag"].endswith("_none"):  # in-sample trained
            continue
        k = (r["sample"], r["member"].split("_")[0])
        v = float(r["payload_bits_per_nnz"])
        if k not in best or v < best[k][0]:
            best[k] = (v, f"track_c:{r['codec']}")
    src["realized"] = best
    # Track D per-subset code lengths
    d = {}
    for r in read_tsv(os.path.join(R, "track_d_entropy.tsv")):
        d[(r["sample"], r["member"], r["subset"], r["quantity"])] = float(r["bits"])
    src["d"] = d
    # newer sources if present (state them in the output)
    extra = []
    for fn in sorted(os.listdir(R)):
        if fn.startswith(("e2_", "e1_")) and fn.endswith(".tsv"):
            extra.append(fn)
    src["extra_files"] = extra
    return src


def log2_binom(W, k):
    return (math.lgamma(W + 1) - math.lgamma(k + 1) - math.lgamma(W - k + 1)) / math.log(2)


def solve_w(k, nbytes):
    lo, hi = float(k) + 1, 1e10
    for _ in range(200):
        mid = (lo + hi) / 2
        if log2_binom(mid, k) / 8 < nbytes:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------------------------------------------------
# matrix helpers
# ---------------------------------------------------------------------------------------------------------------------
def subset_csc(M, keep):
    cols = np.flatnonzero(keep)
    ip = M["indptr"].astype(np.int64)
    lens = (ip[1:] - ip[:-1])[cols]
    nip = np.zeros(len(cols) + 1, np.int64)
    nip[1:] = np.cumsum(lens)
    starts = ip[cols]
    idx = np.repeat(starts - nip[:-1], lens) + np.arange(int(lens.sum()))
    return dict(indptr=nip.astype(np.int32), indices=M["indices"][idx], data=M["data"][idx], n=len(cols), cols=cols)


def names_z3(names):
    raw = b"\0".join(s.encode() for s in names) + b"\0"
    return len(zstandard.ZstdCompressor(level=3).compress(raw))


def comp_of(buf):
    h, o = L.parse_pz(buf)
    return dict(
        n_chunks=h["num_chunks"], perm_z=h["perm_z_sz"], ptr_z=h["ptr_z_sz"], ctable=4 * h["num_chunks"], chunks_bytes=o["chunks_bytes"],
        first_chunk=int(o["ctable"][0]) if h["num_chunks"] else 0, colsums_z=h["colsums_z_sz"], meta_z=h["metadata_z_sz"],
        file_bytes=len(buf), hdr_ftr=96 + 16, ctab=o["ctable"],
    )


COLS = [
    "sample", "member", "subset", "ncols", "nnz", "m_rows", "n_chunks", "file_bytes_i", "perm_z", "ptr_z", "ctable", "chunks_bytes", "first_chunk_blob",
    "colsums_z", "meta_z", "rown_z3", "coln_z3", "rowcounts_bytes", "barcodes_wl_bytes", "barcodes_canon_bytes", "bytes_ii", "bytes_iii", "bytes_iii_uniform",
    "bytes_ii_canon", "bytes_iii_canon", "bpn_today_chunks", "bpn_iii", "bpn_uniform", "side_kj_bytes", "iii_source", "prefix_bytes_today",
    "first_cells_n_chunks", "first_cells_chunk_bytes", "n_called_first_chunks", "note",
]


def analyze(pz, sample, kind, M, orig_buf, called, src, rows, dsample=None, with_iii=True, subset_names=("all", "cells", "empties")):
    """Append rows (orig, all, cells, empties) for one matrix."""
    dsample = dsample or sample
    n = M["n"]
    m = M["m"]
    nnz_col = np.diff(M["indptr"]).astype(np.int64)
    wl_all = src["wl"].get(dsample)
    canon_all = src["canon"].get(dsample)
    bpb = wl_all / n if wl_all else float("nan")
    Weff = solve_w(n, canon_all) if canon_all else None
    realized = src["realized"].get((dsample, kind))
    dkey = (dsample, kind)
    d = src["d"]
    rown = names_z3(M["rownames"])
    if orig_buf is not None:
        base = comp_of(orig_buf)
        coln_all = names_z3(M["colnames"])
        # first-1024-cells prefix in today's layout
        cc = np.add.reduceat(called.astype(np.int64), np.arange(0, n, 1024)) if n else np.array([0])
        cum = np.cumsum(cc)
        want = min(1024, int(called.sum()))
        k = int(np.searchsorted(cum, want)) + 1 if want else 0
        k = min(k, base["n_chunks"])
        first_bytes = int(base["ctab"][:k].sum())
        rows.append(dict(
            sample=sample, member=kind, subset="orig", ncols=n, nnz=int(M["nnz"]), m_rows=m, n_chunks=base["n_chunks"], file_bytes_i=base["file_bytes"],
            perm_z=base["perm_z"], ptr_z=base["ptr_z"], ctable=base["ctable"], chunks_bytes=base["chunks_bytes"], first_chunk_blob=base["first_chunk"],
            colsums_z=base["colsums_z"], meta_z=base["meta_z"], rown_z3=rown, coln_z3=coln_all,
            bpn_today_chunks=round(base["chunks_bytes"] * 8 / max(1, M["nnz"]), 4),
            prefix_bytes_today=96 + base["perm_z"] + base["ptr_z"] + base["ctable"], first_cells_n_chunks=k, first_cells_chunk_bytes=first_bytes,
            n_called_first_chunks=int(cum[k - 1]) if k else 0,
            note="file as stored",
        ))
    all_subsets = {"all": np.ones(n, bool), "cells": called, "empties": ~called}
    for name in subset_names:
        keep = all_subsets[name]
        sub = subset_csc(M, keep)
        names_c = [M["colnames"][i] for i in sub["cols"]]
        tmp = os.path.join(L.SCRATCH, "tier_tmp.1pz")
        ok = pz.write_1pz(tmp, sub["indptr"], sub["indices"], sub["data"], m, sub["n"], rownames=M["rownames"], colnames=names_c, zstd_level=3, chunk_cols=1024, threads=1)
        assert ok
        buf = open(tmp, "rb").read()
        c = comp_of(buf)
        rnnz = np.bincount(sub["indices"], minlength=m)
        rc = rowcounts_cost(rnnz)
        nsub = sub["n"]
        nnz_sub = len(sub["data"])
        bwl = int(round(bpb * nsub)) if wl_all else 0
        bcanon = int(round(log2_binom(Weff, nsub) / 8)) if Weff and nsub <= Weff else 0
        fixed = 96 + c["ctable"] + 16 + 8  # header + chunk index + footer + vocabulary id
        bytes_ii = fixed + c["chunks_bytes"] + rc + bwl
        bytes_ii_canon = fixed + c["chunks_bytes"] + rc + bcanon
        # (iii) payload
        bpn_iii = bpn_uni = None
        side_kj = 0
        src_txt = ""
        if with_iii and realized and dkey[1] in ("exon", "intron"):
            mem = kind
            key_all = (dsample, mem, "all", "M2_featdepth_cv_total")
            if key_all in d:
                nn = {"all": int(M["nnz"]), "cells": nnz_sub, "empties": nnz_sub}
                sname = name
                m2 = d.get((dsample, mem, sname, "M2_featdepth_cv_total"))
                if sname == "empties":
                    amb = d.get((dsample, mem, "empties", "amb_loo_bits"))
                    cand = [x for x in (m2, amb) if x is not None]
                    bits = min(cand) if cand else None
                    src_txt = "track_d:min(M2_cv,amb_loo)"
                else:
                    bits = m2
                    src_txt = "track_d:M2_cv"
                if bits is not None and nnz_sub:
                    bpn_iii = bits / nnz_sub
                    kjb = d.get((dsample, mem, sname, "side_H0_kj_mdl"))
                    side_kj = int(math.ceil(kjb / 8)) if kjb else 0
                bpn_uni = realized[0]
        if bpn_iii is not None:
            payload = int(math.ceil(bpn_iii * nnz_sub / 8))
            bytes_iii = fixed + payload + side_kj + rc + bwl
            bytes_iii_canon = fixed + payload + side_kj + rc + bcanon
            bytes_iii_uni = fixed + int(math.ceil(bpn_uni * nnz_sub / 8)) + side_kj + rc + bwl
        else:
            bytes_iii = bytes_iii_canon = bytes_iii_uni = ""
        rows.append(dict(
            sample=sample, member=kind, subset=name, ncols=nsub, nnz=nnz_sub, m_rows=m, n_chunks=c["n_chunks"], file_bytes_i=c["file_bytes"], perm_z=c["perm_z"],
            ptr_z=c["ptr_z"], ctable=c["ctable"], chunks_bytes=c["chunks_bytes"], first_chunk_blob=c["first_chunk"], colsums_z=c["colsums_z"], meta_z=c["meta_z"],
            rown_z3=rown, coln_z3=names_z3(names_c), rowcounts_bytes=rc, barcodes_wl_bytes=bwl, barcodes_canon_bytes=bcanon, bytes_ii=bytes_ii, bytes_iii=bytes_iii,
            bytes_iii_uniform=bytes_iii_uni, bytes_ii_canon=bytes_ii_canon, bytes_iii_canon=bytes_iii_canon,
            bpn_today_chunks=round(c["chunks_bytes"] * 8 / max(1, nnz_sub), 4), bpn_iii=round(bpn_iii, 4) if bpn_iii is not None else "",
            bpn_uniform=round(bpn_uni, 4) if bpn_uni is not None else "", side_kj_bytes=side_kj, iii_source=src_txt + ("" if not realized else f"; uniform={realized[1]}") if bpn_iii is not None else "",
            prefix_bytes_today=96 + c["perm_z"] + c["ptr_z"] + c["ctable"], note="re-encoded with the production writer (python binding), level 3, chunk_cols 1024",
        ))
        log(f"{sample:22s} {kind:7s} {name:8s} ncols={nsub:6d} nnz={nnz_sub:8d} (i)={c['file_bytes']:9d} chunks={c['chunks_bytes']:9d} (ii)={bytes_ii:9d} (iii)={bytes_iii}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=",".join(L.CORPUS))
    ap.add_argument("--no-bundle-extra", action="store_true")
    ap.add_argument("--no-gene", action="store_true")
    args = ap.parse_args()
    os.makedirs(L.SCRATCH, exist_ok=True)
    pz = L.load_pz()
    src = load_sources()
    log("extra result files seen (not used unless listed in iii_source):", src["extra_files"])
    rows = []
    sample_info = {}
    for sid in args.samples.split(","):
        d = os.path.join(L.BC, "corpus", sid)
        calls_raw = open(os.path.join(d, "cell_calls.tsv"), "rb").read()
        called_bcs = set(ln.split("\t")[0] for ln in calls_raw.decode().split("\n")[1:] if ln)
        mats = {}
        for kind in ("exon", "intron"):
            path = os.path.join(d, f"{kind}_counts.1pz")
            buf = open(path, "rb").read()
            M = pz.read_1pz(path)
            called = np.fromiter((b in called_bcs for b in M["colnames"]), bool, M["n"])
            assert called.sum() == len(called_bcs), (sid, called.sum(), len(called_bcs))
            analyze(pz, sid, kind, M, buf, called, src, rows)
            sample_info[sid] = dict(n_called=int(called.sum()), n_cols=M["n"], cell_calls_raw=len(calls_raw), cell_calls_deflated=len(zlib.compress(calls_raw, 6)))
            mats[kind] = (M, called)
        if not args.no_gene:
            gene_level(pz, sid, mats["exon"][0], mats["intron"][0], mats["exon"][1], src, rows)
    # bundle sample with all modalities
    if not args.no_bundle_extra:
        z = zipfile.ZipFile(L.BUNDLE)
        g = "GSM4317429"
        calls_raw = z.read(f"samples/{g}/cell_calls.tsv")
        called_bcs = set(ln.split("\t")[0] for ln in calls_raw.decode().split("\n")[1:] if ln)
        for kind in ("sj", "vdj"):
            fn = {"sj": "sj_counts", "vdj": "vdj_gene_usage"}[kind]
            buf = z.read(f"samples/{g}/{fn}.1pz")
            p = os.path.join(L.SCRATCH, g)
            os.makedirs(p, exist_ok=True)
            with open(os.path.join(p, fn + ".1pz"), "wb") as f:
                f.write(buf)
            M = pz.read_1pz(os.path.join(p, fn + ".1pz"))
            called = np.fromiter((b in called_bcs for b in M["colnames"]), bool, M["n"])
            analyze(pz, "tumor_3p", kind, M, buf, called, src, rows, dsample="tumor_3p", with_iii=False)
        sample_info["tumor_3p"]["cell_calls_deflated_bundle"] = z.getinfo(f"samples/{g}/cell_calls.tsv").compress_size
        sample_info["tumor_3p"]["psi_bytes"] = z.getinfo(f"samples/{g}/splice_psi.1pz").file_size
        sample_info["tumor_3p"]["json_bytes"] = sum(z.getinfo(f"samples/{g}/{x}").compress_size for x in ("summary.json", "pileup_stats.json", "provenance.json"))
        sample_info["tumor_3p"]["mt_bytes"] = z.getinfo(f"samples/{g}/mt_heteroplasmy.1pz").file_size
    # write
    with open(os.path.join(L.RESULTS, "e4_tiers.tsv"), "w", newline="\n") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in COLS) + "\n")
    with open(os.path.join(L.RESULTS, "e4_tiers_samples.json"), "w") as f:
        json.dump(sample_info, f, indent=1)
    # sanity: realized vs per-subset sum (the 'all' row)
    log("wrote results/e4_tiers.tsv", len(rows))


def gene_level(pz, sid, Mx, Mi, called, src, rows):
    """Informational: the gene-level (exon + intron summed per ENSG id) matrix of the called cells, today's writer.
    This is what SingletBundle.raw_counts(gene_level=True, cells='called') builds on the client from the two interval files."""
    from scipy.sparse import csc_matrix

    toks_x = [r.split("_")[0] for r in Mx["rownames"]]
    toks_i = [r.split("_")[0] for r in Mi["rownames"]]
    gid = {g: i for i, g in enumerate(dict.fromkeys(toks_x + toks_i))}
    ng = len(gid)
    n = Mx["n"]
    assert Mi["colnames"] == Mx["colnames"]
    parts = []
    for M, toks in ((Mx, toks_x), (Mi, toks_i)):
        g_of_row = np.array([gid[t] for t in toks], np.int64)
        col = np.repeat(np.arange(n), np.diff(M["indptr"]))
        parts.append(csc_matrix((M["data"].astype(np.int64), (g_of_row[M["indices"]], col)), shape=(ng, n)))
    G = (parts[0] + parts[1]).tocsc()
    G.sort_indices()
    Gd = dict(indptr=G.indptr.astype(np.int32), indices=G.indices.astype(np.int32), data=G.data.astype(np.uint32), n=n, m=ng, nnz=G.nnz,
              rownames=list(gid.keys()), colnames=Mx["colnames"])
    # rowcounts etc. via the generic path: only the 'cells' subset; no payload model for gene-level
    analyze(pz, sid, "gene", Gd, None, called, src, rows, with_iii=False, subset_names=("cells",))


if __name__ == "__main__":
    main()
