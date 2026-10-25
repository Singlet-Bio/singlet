"""Singlet ATAC vs Cell Ranger ARC / Cell Ranger ATAC published outputs (and the MACS3 / SnapATAC2 comparators).

    python compare_atac.py --singlet RUN_DIR --tenx-dir DIR --prefix NAME --kind arc|cratac --out result.json
        [--comp-10x COMP_DIR_ON_10X_FRAGMENTS] [--comp-singlet COMP_DIR_ON_SINGLET_FRAGMENTS] [--chrom chr1]

Barcodes are compared in the space the 10x tool reports cells in: GEX barcodes for ARC (Singlet translates ATAC
barcodes with the ARC pair list), ATAC barcodes for Cell Ranger ATAC. Suffixes "-1" are dropped.
"""
import argparse, gzip, json, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

ap = argparse.ArgumentParser()
ap.add_argument("--singlet", required=True); ap.add_argument("--tenx-dir", required=True)
ap.add_argument("--prefix", required=True); ap.add_argument("--kind", choices=["arc", "cratac"], required=True)
ap.add_argument("--out", required=True); ap.add_argument("--comp-10x"); ap.add_argument("--comp-singlet")
ap.add_argument("--chrom", default="chr1")
a = ap.parse_args()
T = lambda x: f"{a.tenx_dir}/{a.prefix}_{x}"
strip = lambda s: s.str.replace("-1", "", regex=False)
R = {"dataset": a.prefix, "reference_tool": "Cell Ranger ARC 2.0.0" if a.kind == "arc" else "Cell Ranger ATAC 1.2.0"}

# ── 10x per-barcode table ──
if a.kind == "arc":
    t = pd.read_csv(T("per_barcode_metrics.csv"))
    t["barcode"] = strip(t.gex_barcode)
    t = t.rename(columns={"atac_fragments": "fragments", "atac_TSS_fragments": "tss_fragments",
                          "atac_peak_region_fragments": "peak_region_fragments", "atac_raw_reads": "raw_read_pairs",
                          "atac_dup_reads": "dup_read_pairs", "atac_mitochondrial_reads": "mito_read_pairs"})
    frag_path, peak_path = T("atac_fragments.tsv.gz"), T("atac_peaks.bed")
else:
    t = pd.read_csv(T("singlecell.csv"))
    t = t[t.barcode != "NO_BARCODE"].copy()
    t["barcode"] = strip(t.barcode)
    t["is_cell"] = t.is__cell_barcode
    t = t.rename(columns={"passed_filters": "fragments", "TSS_fragments": "tss_fragments", "total": "raw_read_pairs",
                          "duplicate": "dup_read_pairs", "mitochondrial": "mito_read_pairs"})
    frag_path, peak_path = T("fragments.tsv.gz"), T("peaks.bed")
t = t.set_index("barcode")
tenx_cells = set(t.index[t.is_cell == 1])

# ── Singlet ──
s = pd.read_csv(f"{a.singlet}/atac_barcode_metrics.tsv", sep="\t").set_index("barcode")
s.index = s.index.str.replace("-1", "", regex=False)
sing_cells = set(s.index[s.is_cell == 1])
summ = json.load(open(f"{a.singlet}/atac_summary.json"))
R["singlet_summary"] = {k: summ[k] for k in ("barcode_layout", "cells", "peaks", "unique_fragments", "qc") if k in summ}

inter = tenx_cells & sing_cells
R["cell_calls"] = {"tenx": len(tenx_cells), "singlet": len(sing_cells), "both": len(inter),
                   "jaccard": len(inter) / max(1, len(tenx_cells | sing_cells)),
                   "recall_of_tenx": len(inter) / max(1, len(tenx_cells)),
                   "precision_vs_tenx": len(inter) / max(1, len(sing_cells))}

def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return None
    return {"n": int(m.sum()), "spearman": float(spearmanr(x[m], y[m])[0]),
            "pearson_log": float(pearsonr(np.log1p(x[m]), np.log1p(y[m]))[0]),
            "median_ratio_singlet_over_tenx": float(np.median((x[m] + 1) / (y[m] + 1)))}

j = s.join(t, how="inner", lsuffix="_s", rsuffix="_t")
jc = j.loc[j.index.isin(tenx_cells | sing_cells)]
R["per_cell"] = {"cells_union_matched": int(len(jc))}
for col in ("fragments", "tss_fragments", "peak_region_fragments"):
    if f"{col}_s" in jc and f"{col}_t" in jc:
        R["per_cell"][col] = corr(jc[f"{col}_s"].to_numpy(float), jc[f"{col}_t"].to_numpy(float))
jc_t_frip = jc["peak_region_fragments_t"] / jc["fragments_t"].replace(0, np.nan)
R["per_cell"]["frip"] = corr(jc["frip"].to_numpy(float), jc_t_frip.to_numpy(float))
R["per_cell"]["frip_median"] = {"singlet": float(jc.loc[jc.index.isin(inter), "frip"].median()),
                                "tenx": float(jc_t_frip[jc.index.isin(inter)].median())}
# all barcodes with >= 100 10x fragments (ranking agreement beyond cells)
jb = j[(j.fragments_t >= 100) | (j.fragments_s >= 100)]
R["per_barcode_ge100"] = corr(jb.fragments_s.to_numpy(float), jb.fragments_t.to_numpy(float))

# ── Fragment-level concordance on one chromosome, shared cells ──
pairs = None
if a.kind == "arc":
    # 10x fragments may carry ATAC or GEX barcodes: map ATAC -> GEX with the per-barcode table
    pairs = dict(zip(strip(t.atac_barcode), t.index))

def load_frags(path, chrom, cells, translate=None):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith(chrom + "\t"):
                if rows and not line.startswith("#"):
                    break  # sorted by chrom: past the block
                continue
            c, st, en, b, n = line.rstrip("\n").split("\t")[:5]
            b = b.replace("-1", "")
            if translate is not None and b not in cells:
                b = translate.get(b, b)
            if b in cells:
                rows.append((int(st), int(en), b, int(n)))
    return pd.DataFrame(rows, columns=["start", "end", "barcode", "n"])

ft = load_frags(frag_path, a.chrom, inter, pairs)
fs = load_frags(f"{a.singlet}/fragments.tsv.gz", a.chrom, inter)
kt = set(zip(ft.start, ft.end, ft.barcode)); ks = set(zip(fs.start, fs.end, fs.barcode))
both = kt & ks
# +-1 bp tolerance on each end
ks1 = set()
for st, en, b in ks:
    for d1 in (-1, 0, 1):
        for d2 in (-1, 0, 1):
            ks1.add((st + d1, en + d2, b))
R["fragments_" + a.chrom] = {"cells": len(inter), "tenx": len(kt), "singlet": len(ks), "exact_both": len(both),
                             "tenx_found_exact": len(both) / max(1, len(kt)),
                             "singlet_found_exact": len(both) / max(1, len(ks)),
                             "tenx_found_within_1bp": len(kt & ks1) / max(1, len(kt)),
                             "read_pairs_per_fragment_tenx": float(ft.n.mean()) if len(ft) else None,
                             "read_pairs_per_fragment_singlet": float(fs.n.mean()) if len(fs) else None}

# ── Peaks ──
def read_bed(p):
    d = pd.read_csv(p, sep="\t", header=None, comment="#", usecols=[0, 1, 2])
    d.columns = ["chrom", "start", "end"]
    return d[~d.chrom.isin(["chrM", "MT"])]

def overlap_frac(A, B):
    """fraction of intervals in A overlapping >= 1 interval of B; bp jaccard."""
    hit, inter_bp = 0, 0
    for c, ga in A.groupby("chrom"):
        gb = B[B.chrom == c].sort_values("start")
        if gb.empty:
            continue
        bs, be = gb.start.to_numpy(), gb.end.to_numpy()
        bmax = np.maximum.accumulate(be)
        for st, en in zip(ga.start.to_numpy(), ga.end.to_numpy()):
            k = np.searchsorted(bs, en)  # intervals starting before en
            if k > 0 and bmax[k - 1] > st:
                hit += 1
    def bp(D):
        tot = 0
        for c, g in D.groupby("chrom"):
            g = g.sort_values("start"); cs, ce = -1, -1
            for st, en in zip(g.start, g.end):
                if st > ce:
                    tot += max(0, ce - cs); cs, ce = st, en
                else:
                    ce = max(ce, en)
            tot += max(0, ce - cs)
        return tot
    return hit / max(1, len(A)), bp

def peak_cmp(Pa, Pb):
    fa, bp = overlap_frac(Pa, Pb)
    fb, _ = overlap_frac(Pb, Pa)
    ua, ub, un = bp(Pa), bp(Pb), bp(pd.concat([Pa, Pb]))
    return {"n_a": int(len(Pa)), "n_b": int(len(Pb)), "a_overlapped_by_b": fa, "b_overlapped_by_a": fb,
            "bp_jaccard": (ua + ub - un) / max(1, un), "median_width_a": float((Pa.end - Pa.start).median()),
            "median_width_b": float((Pb.end - Pb.start).median())}

P10 = read_bed(peak_path); PS = read_bed(f"{a.singlet}/peaks.bed")
R["peaks"] = {"singlet_vs_tenx": peak_cmp(PS, P10)}
if a.comp_10x and os.path.exists(f"{a.comp_10x}/macs3_peaks.narrowPeak"):
    PM10 = read_bed(f"{a.comp_10x}/macs3_peaks.narrowPeak")
    R["peaks"]["macs3_on_tenx_fragments_vs_tenx"] = peak_cmp(PM10, P10)
    R["peaks"]["singlet_vs_macs3_on_tenx_fragments"] = peak_cmp(PS, PM10)
if a.comp_singlet and os.path.exists(f"{a.comp_singlet}/macs3_peaks.narrowPeak"):
    PMS = read_bed(f"{a.comp_singlet}/macs3_peaks.narrowPeak")
    R["peaks"]["singlet_vs_macs3_on_singlet_fragments"] = peak_cmp(PS, PMS)

# ── TSS enrichment / nucleosome signal vs SnapATAC2 + Signac definitions on the 10x fragments ──
if a.comp_10x and os.path.exists(f"{a.comp_10x}/snapatac2_qc.tsv"):
    q = pd.read_csv(f"{a.comp_10x}/snapatac2_qc.tsv", sep="\t").set_index("barcode")
    if pairs is not None:
        q.index = [pairs.get(b, b) for b in q.index]
    qq = s.join(q, how="inner")
    qq = qq[qq.index.isin(inter)]
    R["qc_vs_competitors_on_tenx_fragments"] = {
        "tss_enrichment_vs_snapatac2_tsse": corr(qq.tss_enrichment.to_numpy(float), qq.tsse.to_numpy(float)),
        "nucleosome_signal_vs_signac": corr(qq.nucleosome_signal.to_numpy(float),
                                            qq.nucleosome_signal_signac.to_numpy(float)),
        "median_tsse": {"singlet": float(qq.tss_enrichment.median()), "snapatac2": float(qq.tsse.median())}}

# ── Runtime ──
def rt(p):
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    w = re.search(r"Elapsed \(wall clock\) time.*: (.*)", txt).group(1)
    parts = [float(x) for x in w.split(":")]
    sec = parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) == 3 else 0)
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt).group(1))
    cpu = float(re.search(r"User time \(seconds\): ([\d.]+)", txt).group(1)) + \
        float(re.search(r"System time \(seconds\): ([\d.]+)", txt).group(1))
    return {"wall_s": sec, "max_rss_gb": rss / 1048576, "cpu_h": cpu / 3600}
R["runtime"] = {"singlet_process": rt(f"{a.singlet}/time.txt"), "singlet_encode": rt(f"{a.singlet}/time_encode.txt")}
if a.comp_10x and os.path.exists(f"{a.comp_10x}/competitors_time.json"):
    R["runtime"]["comparators_on_tenx_fragments"] = json.load(open(f"{a.comp_10x}/competitors_time.json"))
json.dump(R, open(a.out, "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
