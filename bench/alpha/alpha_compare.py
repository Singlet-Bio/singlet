"""Score one published Research Alpha sample against the authors' deposited Cell Ranger matrix.

    python -P alpha_compare.py PICKS_TSV ROW_INDEX OUT_DIR
      env EDROP   = a singlet checkout with bench/empty_droplets/fetch.py and singlet._pz built in place
      env CATALOG = optional catalog_map.parquet (gsm, srx_accession, srr_accessions, read_count)

What is compared:
  * Singlet = the sample exactly as published on data.singlet.bio (legacy .singlet bundle): exon_counts.1pz,
    intron_counts.1pz, cell_calls.tsv, summary.json, pileup_stats.json. Members are read by exact HTTP ranges (the zip
    central directory, then each member's local header and data), so no request touches a restricted member.
  * Authors = the filtered matrix in the GEO sample's supplementary folder (10x .h5 or matrix/barcodes/features).

Metrics (all per sample, written to OUT/<GSM>.json; per-barcode table to OUT/<GSM>.cells.csv.gz):
  * cell recovery = |Singlet called & author| / |author|; precision = |Singlet called & author| / |Singlet called|;
    also the share of the authors' cells that are in Singlet's barcode matrix at all (>= 100 reads).
  * per-cell UMI ratio, Singlet / author, on cells both called (median, p10, p90), for Singlet exon and exon+intron.
  * the Singlet layer that matches the authors' counting mode (exon only vs exon+intron, as Cell Ranger <7 / >=7) is
    inferred from which one's gene pseudobulk correlates better (log Pearson) with the authors'; both are reported.
  * per-cell gene-vector correlation (log1p Pearson, median over up to 2,000 shared cells), raw and depth-matched:
    both profiles of a cell are binomially thinned to the smaller of the two totals; the noise floor is the same
    correlation between two independent thinnings of the authors' profile to that depth.
  * pseudobulk Spearman over common genes (shared cells).
  * reads_used / reads_available: reads Singlet processed (summary.json) over all SRA spots of the sample (NCBI runinfo,
    every run of every SRX of the GSM), plus the first run's spots (the campaign used only the first SRR).
Genes are matched on Ensembl ID without version; if the authors' features are not Ensembl IDs, on gene symbol.
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import json
import os
import re
import struct
import sys
import time
import urllib.request
import zipfile

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

PICKS, IDX, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(OUT, exist_ok=True)
EDROP = os.environ["EDROP"]
_spec = importlib.util.spec_from_file_location("edfetch", os.path.join(EDROP, "bench/empty_droplets/fetch.py"))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
UA = {"User-Agent": "singlet-alpha-bench/1.0 (+https://singlet.bio)"}
BASE = os.environ.get("SINGLET_DATA_BASE", "https://data.singlet.bio")
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/samples"
BC_RE = re.compile(r"[ACGT]{16}|[ACGT]{14}|[ACGT]{12}")
rows = list(csv.DictReader(open(PICKS, encoding="utf-8"), delimiter="\t"))
P = rows[IDX]
GSE, GSM = P["gse"], P["gsm"]
res = {"gse": GSE, "gsm": GSM, "stratum": P["stratum"], "species": P["species"], "chem": P["chem"], "size": P["size"],
       "protocol_label": P["protocol"], "assay_family": P["assay_family"], "engine": "alpha (published atlas)",
       "dataset_scope": "Research Alpha public sample (not withheld, not excluded, downloadable, QC tier 0)"}
t0 = time.time()


def http(url, rng=None, tries=6, timeout=300):
    h = dict(UA)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                b = r.read()
            if rng and len(b) != rng[1] - rng[0] + 1:
                raise IOError("short read")
            return b
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            if i == tries - 1:
                raise
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
        time.sleep(3 * (i + 1))


class ExactRangeFile(io.RawIOBase):
    """Seekable file over HTTP ranges that requests exactly the bytes asked for (no read-ahead), so reads never
    overlap a member the server refuses to serve."""

    def __init__(self, url):
        self.url, self.pos, self.n_req, self.n_bytes = url, 0, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        b = http(self.url, (self.pos, self.pos + n - 1))
        self.pos += n
        self.n_req += 1
        self.n_bytes += n
        return b

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def norm_bc(s):
    m = BC_RE.search(str(s))
    return m.group(0) if m else None


def strip_ver(g):
    return str(g).split(".")[0]


def dump(final=True):
    res["seconds"] = round(time.time() - t0, 1)
    res["complete"] = final
    json.dump(res, open(os.path.join(OUT, f"{GSM}.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ------------------------------------------------------------------ Singlet (published bundle)
try:
    rf = ExactRangeFile(f"{BASE}/data/{GSE}/{GSE}.singlet")
    zf = zipfile.ZipFile(rf)
    pre = f"samples/{GSM}/"
    names = {n[len(pre):]: n for n in zf.namelist() if n.startswith(pre)}
    pz = F.load_pz()
    mats = {}
    for kind in ("exon", "intron"):
        if f"{kind}_counts.1pz" in names:
            mats[kind] = F.read_member_matrix(zf, names[f"{kind}_counts.1pz"], pz)
    meta = {}
    for j in ("summary.json", "pileup_stats.json", "provenance.json"):
        if j in names:
            meta[j[:-5]] = json.loads(zf.read(names[j]))
    cc_header, cset, methods = F.parse_cell_calls(zf.read(names["cell_calls.tsv"]).decode())
except Exception as e:  # noqa: BLE001
    res["error"] = f"singlet fetch: {type(e).__name__}: {e}"
    dump()
    sys.exit(0)
cols = mats["exon"][2]
genes = sorted({F.gene_key(n) for (_, rn, _, _) in mats.values() for n in rn})
gi = {g: i for i, g in enumerate(genes)}
sym = {}
for (_, rn, _, _) in mats.values():
    for nme in rn:
        g = F.gene_key(nme)
        if g not in sym:
            parts = nme.split("_")
            sym[g] = parts[1] if len(parts) > 1 else g
S_ex = F.aggregate_rows(mats["exon"][0], mats["exon"][1], gi).tocsc()
S_in = F.aggregate_rows(mats["intron"][0], mats["intron"][1], gi).tocsc() if "intron" in mats else sp.csc_matrix(S_ex.shape, dtype=np.int32)
s_genes = np.array([strip_ver(g) for g in genes])
s_sym = np.array([str(sym[g]).upper() for g in genes])
s_bc = np.array([norm_bc(b) for b in cols])
s_called = np.array([b in cset for b in cols])
summ, ps = meta.get("summary", {}), meta.get("pileup_stats", {})
ws, eh = ps.get("wrong_strand"), ps.get("exon_hits")
res["singlet"] = {
    "n_barcodes": int(len(s_bc)), "n_called": int(s_called.sum()), "call_method": methods,
    "protocol_name": summ.get("protocol_name"), "reference_build": summ.get("reference_build"),
    "git_sha": summ.get("git_sha"), "singlet_version": summ.get("singlet_version"),
    "reads_used": summ.get("total_reads") or summ.get("n_input_reads"), "n_input_reads": summ.get("n_input_reads"),
    "mapping_rate": summ.get("mapping_rate"), "median_umi_per_cell": summ.get("median_umi_per_cell"),
    "wrong_strand_frac": (ws / (ws + eh)) if (ws is not None and eh) else None,
    "reverse_strand": ps.get("reverse_strand"), "bytes_read": rf.n_bytes, "http_requests": rf.n_req,
}

# ------------------------------------------------------------------ reads available (NCBI runinfo)
def runinfo(gsm):
    eu = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ids = re.findall(r"<Id>(\d+)</Id>", http(f"{eu}/esearch.fcgi?db=sra&term={gsm}&retmax=50").decode())
    if not ids:
        return None
    time.sleep(0.4)
    txt = http(f"{eu}/efetch.fcgi?db=sra&id={','.join(ids)}&rettype=runinfo&retmode=text").decode()
    df = pd.read_csv(io.StringIO(txt))
    df = df[df["Run"].astype(str).str.match(r"^[SED]RR\d+")]
    if "SampleName" in df and (df["SampleName"].astype(str) == gsm).any():
        df = df[df["SampleName"].astype(str) == gsm]
    df = df.sort_values("Run", key=lambda s: s.str[3:].astype(int))
    return df


try:
    time.sleep((IDX % 8) * 0.5)
    ri = runinfo(GSM)
    if ri is not None and len(ri):
        spots = pd.to_numeric(ri["spots"], errors="coerce").fillna(0)
        res["sra"] = {"n_runs": int(len(ri)), "n_experiments": int(ri["Experiment"].nunique()),
                      "spots_all_runs": int(spots.sum()), "spots_first_run": int(spots.iloc[0]),
                      "first_run": str(ri["Run"].iloc[0]), "library_strategies": sorted(ri["LibraryStrategy"].astype(str).unique().tolist())}
except Exception as e:  # noqa: BLE001
    res.setdefault("warnings", []).append(f"runinfo: {type(e).__name__}: {e}")
if os.environ.get("CATALOG") and os.path.exists(os.environ["CATALOG"]):
    try:
        cat = pd.read_parquet(os.environ["CATALOG"])
        r = cat[cat.gsm_id == GSM]
        if len(r):
            res["catalog"] = {"srx": str(r.srx_accession.iloc[0]), "srr": str(r.srr_accessions.iloc[0]),
                              "read_count": None if pd.isna(r.read_count.iloc[0]) else int(r.read_count.iloc[0])}
    except Exception as e:  # noqa: BLE001
        res.setdefault("warnings", []).append(f"catalog: {e}")
used = res["singlet"]["reads_used"]
avail = (res.get("sra") or {}).get("spots_all_runs")
res["reads_used_over_available"] = (used / avail) if (used and avail) else None
res["reads_used_over_first_run"] = (used / res["sra"]["spots_first_run"]) if (used and res.get("sra", {}).get("spots_first_run")) else None

# ------------------------------------------------------------------ authors' filtered matrix (GEO sample supplement)
def read_10x_h5(data):
    with h5py.File(io.BytesIO(data), "r") as f:
        g = f["matrix"] if "matrix" in f else f[list(f.keys())[0]]
        M = sp.csc_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=tuple(g["shape"][()]))
        bcs = [x.decode() for x in g["barcodes"][()]]
        feat = g["features"] if "features" in g else g
        ids = [x.decode() for x in (feat["id"][()] if "id" in feat else feat["genes"][()])]
        nm = [x.decode() for x in (feat["name"][()] if "name" in feat else (feat["gene_names"][()] if "gene_names" in feat else feat["id"][()]))]
        ftype = [x.decode() for x in feat["feature_type"][()]] if "feature_type" in feat else ["Gene Expression"] * len(ids)
    keep = np.array([t == "Gene Expression" for t in ftype])
    if len(keep) != M.shape[0] or len(ids) != M.shape[0]:
        raise ValueError(f"h5 features ({len(ids)}) do not match the matrix rows ({M.shape[0]})")
    return M[keep], bcs, [i for i, k in zip(ids, keep) if k], [n for n, k in zip(nm, keep) if k]


def read_mtx(mtx, bcs, feats):
    df = pd.read_csv(io.BytesIO(mtx), sep=r"\s+", comment="%", header=None, dtype=np.float64)
    nr, nc = int(df.iloc[0, 0]), int(df.iloc[0, 1])
    df = df.iloc[1:]
    if df.shape[1] < 3:
        df[2] = 1.0
    M = sp.csc_matrix((df[2].values, (df[0].values.astype(np.int64) - 1, df[1].values.astype(np.int64) - 1)), shape=(nr, nc))
    bl = bcs.decode().split()
    fl = [l.split("\t") for l in feats.decode().splitlines() if l.strip()]
    if M.shape[0] != len(fl) and M.shape[1] == len(fl):
        M = M.T.tocsc()
    ids = [f[0] for f in fl]
    nm = [f[1] if len(f) > 1 else f[0] for f in fl]
    if len(fl[0]) >= 3:
        keep = np.array([f[2] == "Gene Expression" for f in fl])
        if keep.any():
            M, ids, nm = M[keep], [i for i, k in zip(ids, keep) if k], [n for n, k in zip(nm, keep) if k]
    return M, bl, ids, nm


try:
    base = f"{GEO}/{GSM[:-3]}nnn/{GSM}/suppl/"
    files = P["author_files"].split(";")
    unz = lambda f, b: gzip.decompress(b) if b[:2] == b"" else b  # noqa: E731  (some ".gz" deposits are plain text)
    if P["author_kind"] == "h5":
        A, a_bcs, a_ids, a_nm = read_10x_h5(http(base + files[0]))
    else:
        A, a_bcs, a_ids, a_nm = read_mtx(*(unz(f, http(base + f)) for f in files))
    A = sp.csc_matrix(A).astype(np.float64)
except Exception as e:  # noqa: BLE001
    res["error"] = f"author fetch: {type(e).__name__}: {e}"
    dump()
    sys.exit(0)
a_bc = np.array([norm_bc(x) for x in a_bcs])
a_tot = np.asarray(A.sum(0)).ravel()
res["author"] = {"files": files, "n_barcodes": int(len(a_bc)), "n_genes": int(A.shape[0]),
                 "median_umi": float(np.median(a_tot)) if len(a_tot) else None,
                 "duplicate_barcodes": int(len(a_bc) - len(set(a_bc)))}
in_matrix = set(s_bc) - {None}
a_set = set(a_bc) - {None}
res["author"]["frac_barcodes_in_singlet_matrix"] = len(a_set & in_matrix) / max(1, len(a_set))
if len(a_bc) > 60000 and len(a_bc) > 8 * max(1, s_called.sum()):
    res["excluded_reason"] = "author matrix looks unfiltered (raw barcodes)"
elif res["author"]["frac_barcodes_in_singlet_matrix"] < 0.2:
    res["excluded_reason"] = "author barcodes do not match Singlet's (renamed, multiplexed or different library)"
elif res["author"]["duplicate_barcodes"] > 0.05 * len(a_bc):
    res["excluded_reason"] = "author barcodes repeat (several libraries merged in one matrix)"
if "excluded_reason" in res:
    dump()
    sys.exit(0)

# ------------------------------------------------------------------ cell calls
s_call = set(s_bc[s_called]) - {None}
both = s_call & a_set
res["calls"] = {
    "singlet": len(s_call), "author": len(a_set), "both": len(both),
    "recovery": len(both) / max(1, len(a_set)), "precision": len(both) / max(1, len(s_call)),
    "jaccard": len(both) / max(1, len(s_call | a_set)),
    "author_in_singlet_matrix": len(a_set & in_matrix) / max(1, len(a_set)),
}
s_idx = {b: i for i, b in enumerate(s_bc) if b}
a_idx = {b: i for i, b in enumerate(a_bc) if b}
s_ex_tot = np.asarray(S_ex.sum(0)).ravel()
s_in_tot = np.asarray(S_in.sum(0)).ravel()
U = sorted(s_call | a_set)
T = pd.DataFrame({"barcode": U})
si = T.barcode.map(s_idx)
ai = T.barcode.map(a_idx)
T["singlet_called"] = T.barcode.isin(s_call)
T["author_called"] = T.barcode.isin(a_set)
for name, arr, ix in (("singlet_exon", s_ex_tot, si), ("singlet_intron", s_in_tot, si), ("author", a_tot, ai)):
    v = np.full(len(T), np.nan)
    ok = ix.notna().values
    v[ok] = arr[ix[ok].astype(int).values]
    T[name] = v
T.to_csv(os.path.join(OUT, f"{GSM}.cells.csv.gz"), index=False)
missed = T[T.author_called & ~T.singlet_called]
res["calls"]["missed_author_cells_median_author_umi"] = float(missed.author.median()) if len(missed) else None
res["calls"]["shared_cells_median_author_umi"] = float(T[T.author_called & T.singlet_called].author.median()) if len(both) else None
res["calls"]["missed_author_cells_median_singlet_umi"] = float((missed.singlet_exon + missed.singlet_intron).median()) if len(missed) else None


def rstats(x, y):
    m = (x > 0) & (y > 0)
    if m.sum() < 20:
        return None
    r = x[m] / y[m]
    return {"median": float(np.median(r)), "p10": float(np.quantile(r, 0.1)), "p90": float(np.quantile(r, 0.9)),
            "log_pearson": float(pearsonr(np.log1p(x[m]), np.log1p(y[m]))[0]), "n": int(m.sum())}


sh = T[T.singlet_called & T.author_called]
res["umi_ratio"] = {"exon_over_author": rstats(sh.singlet_exon.values, sh.author.values),
                    "exon_intron_over_author": rstats((sh.singlet_exon + sh.singlet_intron).values, sh.author.values)}
res["singlet_median_umi_shared_cells"] = {"exon": float(sh.singlet_exon.median()) if len(sh) else None,
                                          "exon_intron": float((sh.singlet_exon + sh.singlet_intron).median()) if len(sh) else None}

# ------------------------------------------------------------------ gene level
ens = np.array([bool(re.match(r"^ENS[A-Z]*G\d+", x)) for x in a_ids])
if ens.mean() >= 0.5:
    a_key, s_key, res["gene_match"] = np.array([strip_ver(x) for x in a_ids]), s_genes, "ensembl"
else:
    a_key, s_key, res["gene_match"] = np.array([str(x).upper() for x in a_nm]), s_sym, "symbol"
sk = pd.Series(np.arange(len(s_key)), index=s_key)
sk = sk[~sk.index.duplicated()]
ak = pd.Series(np.arange(len(a_key)), index=a_key)
ak = ak[~ak.index.duplicated()]
common = sorted(set(sk.index) & set(ak.index))
res["genes"] = {"singlet": int(len(sk)), "author": int(len(ak)), "common": len(common)}
if len(sh) >= 20 and len(common) >= 1000:
    rng = np.random.default_rng(0)
    cells = sh.sample(min(2000, len(sh)), random_state=0) if len(sh) > 2000 else sh
    cs = cells.barcode.map(s_idx).astype(int).values
    ca = cells.barcode.map(a_idx).astype(int).values
    rs, ra = sk[common].values, ak[common].values
    Sx = S_ex[:, cs][rs, :].toarray().astype(np.float64)
    Si = S_in[:, cs][rs, :].toarray().astype(np.float64)
    Am = A[:, ca][ra, :].toarray()
    Am = np.rint(Am)  # some deposits store floats; counts are integers
    St = Sx + Si
    pbA = Am.sum(1)
    lp = lambda x, y: float(pearsonr(np.log1p(x), np.log1p(y))[0])  # noqa: E731
    pb_fit = {"exon": lp(Sx.sum(1), pbA), "exon_intron": lp(St.sum(1), pbA)}
    layer = max(pb_fit, key=pb_fit.get)
    res["author_mode_inferred"] = "exon only (Cell Ranger < 7 or include-introns off)" if layer == "exon" else "exon + intron (Cell Ranger >= 7 default)"

    def corr_cols(X, Y):
        out = []
        for k in range(X.shape[1]):
            x, y = np.log1p(X[:, k]), np.log1p(Y[:, k])
            if x.std() > 0 and y.std() > 0:
                out.append(np.corrcoef(x, y)[0, 1])
        return float(np.median(out)) if out else None

    def thin(X, target):
        p = np.clip(target / np.maximum(X.sum(0), 1), 0, 1)
        return rng.binomial(X.astype(np.int64), p[None, :]).astype(np.float64)

    gl = {"layer_matched": layer, "pseudobulk_log_pearson_by_layer": pb_fit, "n_cells": int(len(cells)), "n_genes": len(common)}
    for nm_, S in (("exon", Sx), ("exon_intron", St)):
        tgt = np.minimum(S.sum(0), Am.sum(0))
        St_, At_, At2 = thin(S, tgt), thin(Am, tgt), thin(Am, tgt)
        gl[nm_] = {"cell_corr": corr_cols(S, Am), "cell_corr_depth_matched": corr_cols(St_, At_),
                   "cell_corr_noise_floor": corr_cols(At2, At_),
                   "pseudobulk_spearman": float(spearmanr(S.sum(1), pbA)[0]),
                   "depth_matched_median_umi": float(np.median(tgt))}
    res["gene_level"] = gl
    m = gl[layer]
    res["headline"] = {"cell_corr_depth_matched": m["cell_corr_depth_matched"], "noise_floor": m["cell_corr_noise_floor"],
                       "gap_to_floor": (m["cell_corr_noise_floor"] - m["cell_corr_depth_matched"]) if (m["cell_corr_noise_floor"] is not None and m["cell_corr_depth_matched"] is not None) else None,
                       "pseudobulk_spearman": m["pseudobulk_spearman"],
                       "umi_ratio": (res["umi_ratio"]["exon_over_author" if layer == "exon" else "exon_intron_over_author"] or {}).get("median")}
    lr = np.log2((St.sum(1) + 1) / (pbA + 1))
    top = np.argsort(-np.abs(lr) * np.log1p(St.sum(1) + pbA))[:10]
    sym_of = dict(zip(s_key, s_sym))
    res["top_discordant_genes"] = [{"gene": common[i], "symbol": str(sym_of.get(common[i], "")), "singlet": float(St.sum(1)[i]),
                                    "author": float(pbA[i]), "log2": float(lr[i])} for i in top]
dump()
print(json.dumps({k: res.get(k) for k in ("calls", "umi_ratio", "headline", "reads_used_over_available", "author_mode_inferred")}, indent=1, default=str))
