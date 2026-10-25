"""Reduce the ATAC / Multiome / Visium benchmark runs to the small artifacts the public contract publishes.

    python make_artifacts.py atac   DS --tenx-dir D --prefix P --kind arc|cratac --singlet RUN --chromap RUN --out ART
    python make_artifacts.py spatial DS --tenx-dir D --prefix P --singlet RUN --starsolo RUN --out ART

Everything a metric needs goes into per-barcode / per-spot / per-gene tables and peak BEDs (CSV/TSV, gzip). The only
quantities computed here rather than in bench/public/capabilities/*.py are the ones that need the multi-GB inputs: exact
fragment concordance (one chromosome) and per-spot expression-profile correlations. They are written as `derived`
artifacts.
"""
import argparse, gzip, io, json, os, re, tarfile
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["atac", "spatial"]); ap.add_argument("ds")
ap.add_argument("--tenx-dir", required=True); ap.add_argument("--prefix", required=True)
ap.add_argument("--kind", default="arc"); ap.add_argument("--singlet", required=True)
ap.add_argument("--chromap"); ap.add_argument("--starsolo"); ap.add_argument("--out", required=True)
ap.add_argument("--gex", help="Multiome: Singlet GEX run dir (cell_calls.tsv) for singlet_gex_called")
ap.add_argument("--chrom", default="chr21")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T = lambda x: f"{a.tenx_dir}/{a.prefix}_{x}"
strip = lambda s: s.astype(str).str.replace("-1", "", regex=False)


def time_of(p):
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    w = re.search(r"Elapsed \(wall clock\) time.*: (.*)", txt).group(1)
    parts = [float(x) for x in w.split(":")]
    sec = parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) == 3 else 0)
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt).group(1))
    cpu = float(re.search(r"User time \(seconds\): ([\d.]+)", txt).group(1)) + \
        float(re.search(r"System time \(seconds\): ([\d.]+)", txt).group(1))
    return {"wall_s": round(sec, 1), "max_rss_gb": round(rss / 1048576, 2), "cpu_h": round(cpu / 3600, 3)}


def write_bed(src, dst):
    d = pd.read_csv(src, sep="\t", header=None, comment="#", usecols=[0, 1, 2])
    d.to_csv(dst, sep="\t", header=False, index=False, compression="gzip")


if a.what == "atac":
    ds = a.ds
    # 10x per-barcode
    if a.kind == "arc":
        t = pd.read_csv(T("per_barcode_metrics.csv"))
        t["barcode"] = strip(t.gex_barcode)
        t = t.rename(columns={"atac_fragments": "fragments", "atac_TSS_fragments": "tss_fragments",
                              "atac_peak_region_fragments": "peak_region_fragments", "gex_umis_count": "gex_umis"})
        t["atac_barcode"] = strip(t.atac_barcode)
        frag10, peak10 = T("atac_fragments.tsv.gz"), T("atac_peaks.bed")
    else:
        t = pd.read_csv(T("singlecell.csv"))
        t = t[t.barcode != "NO_BARCODE"].copy()
        t["barcode"] = strip(t.barcode)
        t["is_cell"] = t.is__cell_barcode
        t = t.rename(columns={"passed_filters": "fragments", "TSS_fragments": "tss_fragments"})
        frag10, peak10 = T("fragments.tsv.gz"), T("peaks.bed")
    keep10 = ["barcode", "is_cell", "fragments", "tss_fragments", "peak_region_fragments"] + \
        (["atac_barcode", "gex_umis"] if a.kind == "arc" else [])
    t = t[[c for c in keep10 if c in t]].set_index("barcode").add_prefix("tenx_")
    s = pd.read_csv(f"{a.singlet}/atac_barcode_metrics.tsv", sep="\t")
    s["barcode"] = strip(s.barcode)
    s = s.set_index("barcode")[["is_cell", "fragments", "tss_fragments", "peak_region_fragments", "frip",
                                "tss_enrichment", "nucleosome_signal", "duplicate_read_pairs"]].add_prefix("singlet_")
    if os.path.exists(f"{a.singlet}/atac_cells.tsv"):
        ac = pd.read_csv(f"{a.singlet}/atac_cells.tsv", sep="\t", dtype=str)
        s["singlet_atac_barcode"] = pd.Series(strip(ac.atac_barcode).values, index=strip(ac.barcode)).reindex(s.index)
    c = None
    if a.chromap:
        q = pd.read_csv(f"{a.chromap}/comp/snapatac2_qc.tsv", sep="\t")
        q["barcode"] = strip(q.barcode)
        q = q.set_index("barcode")
        c = pd.DataFrame({"chromap_fragments": q.n_fragment, "chromap_tsse": q.tsse,
                          "chromap_frac_mito": q.get("frac_mito"),
                          "chromap_nucleosome_signal": q.nucleosome_signal_signac})
    j = t.join(s, how="outer")
    if c is not None:
        j = j.join(c, how="outer")
    if a.gex:
        cc = pd.read_csv(f"{a.gex}/cell_calls.tsv", sep="	", dtype={"barcode": str})
        col = "called_v2" if "called_v2" in cc else "called"
        j["singlet_gex_called"] = pd.Series(cc[col].astype(int).values, index=strip(cc.barcode)).reindex(j.index).fillna(0).astype(int)
    for col in ("tenx_is_cell", "singlet_is_cell"):
        j[col] = j[col].fillna(0).astype(int)
    big = (j.tenx_is_cell == 1) | (j.singlet_is_cell == 1) | (j.tenx_fragments.fillna(0) >= 100) | \
        (j.singlet_fragments.fillna(0) >= 100) | (j.get("chromap_fragments", pd.Series(0, index=j.index)).fillna(0) >= 100)
    j = j[big]
    j.index.name = "barcode"
    j.to_csv(f"{a.out}/atac_{ds}_barcodes.tsv.gz", sep="\t", compression="gzip")

    write_bed(peak10, f"{a.out}/atac_{ds}_peaks_tenx.bed.gz")
    write_bed(f"{a.singlet}/peaks.bed", f"{a.out}/atac_{ds}_peaks_singlet.bed.gz")
    if a.chromap and os.path.exists(f"{a.chromap}/comp/macs3_peaks.narrowPeak"):
        write_bed(f"{a.chromap}/comp/macs3_peaks.narrowPeak", f"{a.out}/atac_{ds}_peaks_chromap_macs3.bed.gz")

    # Fragment concordance on one chromosome over 10x cells (barcodes in the 10x cell-barcode space)
    cells = set(j.index[j.tenx_is_cell == 1])
    tr = dict(zip(j.tenx_atac_barcode.dropna(), j.tenx_atac_barcode.dropna().index)) if "tenx_atac_barcode" in j else {}

    def frags(path, translate):
        out = set(); lens = []
        with gzip.open(path, "rt") as fh:
            started = False
            for line in fh:
                if not line.startswith(a.chrom + "\t"):
                    if started and not line.startswith("#"):
                        break
                    continue
                started = True
                ch, st, en, b = line.split("\t")[:4]
                b = b.replace("-1", "")
                if b not in cells:
                    b = translate.get(b, b)
                if b in cells:
                    out.add((int(st), int(en), b)); lens.append(int(en) - int(st))
        return out, np.array(lens)

    F10, L10 = frags(frag10, tr)
    conc = {"chromosome": a.chrom, "cells": len(cells), "tenx": len(F10),
            "tenx_length_hist": np.histogram(L10, bins=[0, 50, 100, 147, 200, 294, 400, 600, 1000, 5000])[0].tolist()}
    for tool, path in (("singlet", f"{a.singlet}/fragments.tsv.gz"),
                       ("chromap", f"{a.chromap}/fragments.tsv.gz" if a.chromap else None)):
        if not path or not os.path.exists(path):
            continue
        Fx, Lx = frags(path, tr if tool == "chromap" else {})
        both = F10 & Fx
        near = 0
        Fx1 = {(s0 + d1, e0 + d2, b) for (s0, e0, b) in Fx for d1 in (-1, 0, 1) for d2 in (-1, 0, 1)}
        near = len(F10 & Fx1)
        conc[tool] = {"fragments": len(Fx), "exact_shared": len(both), "within_1bp_shared": near,
                      "length_hist": np.histogram(Lx, bins=[0, 50, 100, 147, 200, 294, 400, 600, 1000, 5000])[0].tolist()}
    json.dump(conc, open(f"{a.out}/atac_{ds}_fragment_concordance.json", "w"), indent=1)

    prov = {"dataset": ds, "singlet": {"summary": json.load(open(f"{a.singlet}/atac_summary.json")),
                                       "time": time_of(f"{a.singlet}/time.txt"),
                                       "time_encode": time_of(f"{a.singlet}/time_encode.txt")}}
    if a.chromap:
        prov["chromap"] = {"time": time_of(f"{a.chromap}/time.txt"), "comparators_time": time_of(f"{a.chromap}/time_comp.txt"),
                           "steps": json.load(open(f"{a.chromap}/comp/competitors_time.json"))
                           if os.path.exists(f"{a.chromap}/comp/competitors_time.json") else None}
    json.dump(prov, open(f"{a.out}/atac_{ds}_runs.json", "w"), indent=1)

else:
    import h5py, scipy.io, scipy.sparse as sp
    ds = a.ds

    def read_h5(p):
        with h5py.File(p, "r") as f:
            m = f["matrix"]
            X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=m["shape"][:])
            bc = [b.decode().replace("-1", "") for b in m["barcodes"][:]]
            ids = [i.decode().split(".")[0] for i in m["features"]["id"][:]]
        return X, bc, ids

    Xt, bt, gt = read_h5(T("raw_feature_bc_matrix.h5"))
    with tarfile.open(T("spatial.tar.gz")) as tf:
        mem = [m for m in tf.getmembers() if "tissue_positions" in m.name][0]
        txt = tf.extractfile(mem).read().decode()
    pos = pd.read_csv(io.StringIO(txt), header=0 if txt.startswith("barcode") else None)
    pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    pos["barcode"] = strip(pos.barcode)
    pos = pos.set_index("barcode")
    spots = list(pos.index)

    def gene_id(f):
        m = re.match(r"(ENS[A-Z]*G\d+)", f)
        return m.group(1) if m else f

    # Singlet exon counts, columns summed onto spots
    S = a.singlet
    Xs = scipy.io.mmread(f"{S}/exon_counts.mtx.gz").tocsc()
    bs = [l.strip().replace("-1", "") for l in gzip.open(f"{S}/exon_counts_barcodes.tsv.gz", "rt")]
    gs = [gene_id(l.strip().split("\t")[0]) for l in gzip.open(f"{S}/exon_counts_features.tsv.gz", "rt")]
    spt = pd.read_csv(f"{S}/spatial_positions.tsv", sep="\t", dtype={"spot": str})
    spt["barcode"] = strip(spt.barcode)
    spt = spt.set_index("barcode")
    col_spot = spt.spot.reindex(bs)
    sidx = {b: i for i, b in enumerate(spots)}
    rows = [i for i, b in enumerate(col_spot) if isinstance(b, str) and b in sidx]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, [sidx[col_spot.iloc[i]] for i in rows])), shape=(len(bs), len(spots)))
    Ms = (Xs @ A).tocsc()
    # STARsolo raw Gene matrix
    tools = {"singlet": (Ms, gs)}
    bt_idx = {b: i for i, b in enumerate(bt)}
    sel = [bt_idx.get(b, -1) for b in spots]
    Mt = sp.hstack([Xt[:, [i]] if i >= 0 else sp.csc_matrix((Xt.shape[0], 1)) for i in sel]).tocsc()
    tools["spaceranger"] = (Mt, gt)
    if a.starsolo:
        R = f"{a.starsolo}/Solo.out/Gene/raw"
        Xo = scipy.io.mmread(f"{R}/matrix.mtx").tocsc()
        bo = [l.strip() for l in open(f"{R}/barcodes.tsv")]
        go = [gene_id(l.split("\t")[0]) for l in open(f"{R}/features.tsv")]
        oi = {b: i for i, b in enumerate(bo)}
        so = [oi.get(b, -1) for b in spots]
        Mo = sp.hstack([Xo[:, [i]] if i >= 0 else sp.csc_matrix((Xo.shape[0], 1)) for i in so]).tocsc()
        tools["starsolo"] = (Mo, go)
    # common genes (present in every tool's feature list)
    common = None
    for M, g in tools.values():
        common = set(g) if common is None else common & set(g)
    common = sorted(common)
    sub = {}
    for k, (M, g) in tools.items():
        gi = {}
        for i, x in enumerate(g):
            gi.setdefault(x, i)
        sub[k] = M[[gi[x] for x in common]].tocsc()
    out = pd.DataFrame(index=pd.Index(spots, name="spot"))
    out["array_row"] = pos.array_row; out["array_col"] = pos.array_col
    out["spaceranger_in_tissue"] = pos.in_tissue
    sp_t = spt.dropna(subset=["spot"]).groupby("spot").agg(in_tissue=("in_tissue", "max"), tissue_prob=("tissue_prob", "max"))
    out["singlet_in_tissue"] = sp_t.in_tissue.reindex(spots).fillna(0).astype(int).values
    out["singlet_tissue_prob"] = sp_t.tissue_prob.reindex(spots).fillna(0).values
    for k, M in sub.items():
        out[f"{k}_umis"] = np.asarray(M.sum(0)).ravel()
        out[f"{k}_genes"] = np.asarray((M > 0).sum(0)).ravel()

    def lcpm(M):
        M = M.astype(float).tocsc(); tot = np.asarray(M.sum(0)).ravel(); tot[tot == 0] = 1
        return M.multiply(1e4 / tot).tocsc().log1p()

    L = {k: lcpm(M) for k, M in sub.items()}
    for k in sub:
        if k == "spaceranger":
            continue
        r = np.full(len(spots), np.nan)
        for jj in range(len(spots)):
            x = L[k][:, jj].toarray().ravel(); y = L["spaceranger"][:, jj].toarray().ravel()
            if x.any() and y.any():
                r[jj] = np.corrcoef(x, y)[0, 1]
        out[f"{k}_profile_r_vs_spaceranger"] = r
    out.to_csv(f"{a.out}/spatial_{ds}_spots.tsv.gz", sep="\t", compression="gzip")
    tis = (pos.in_tissue.reindex(spots) == 1).to_numpy()
    g = pd.DataFrame(index=pd.Index(common, name="gene_id"))
    for k, M in sub.items():
        g[f"{k}_tissue_total"] = np.asarray(M[:, np.flatnonzero(tis)].sum(1)).ravel()
    g = g[g.sum(1) > 0]
    g.to_csv(f"{a.out}/spatial_{ds}_genes.tsv.gz", sep="\t", compression="gzip")
    prov = {"dataset": ds, "singlet": {"summary": json.load(open(f"{S}/spatial_summary.json")),
                                       "time": time_of(f"{S}/time.txt"), "time_encode": time_of(f"{S}/time_encode.txt")},
            "starsolo": {"time": time_of(f"{a.starsolo}/time.txt")} if a.starsolo else None,
            "columns": {"singlet": len(bs), "singlet_on_slide": len(rows)}}
    json.dump(prov, open(f"{a.out}/spatial_{ds}_runs.json", "w"), indent=1)
print("artifacts:", sorted(os.listdir(a.out)))
