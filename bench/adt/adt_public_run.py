# SPDX-License-Identifier: MIT
"""Public ADT benchmark harness (results contract 1.0.0, capability ``adt``): every tool on the same antibody FASTQs.

    python -P bench/adt/adt_public_run.py --dataset pbmc1k_cite_v3 --tool singlet --data $WORK/data/adt_bench \
        --out $WORK/bench-public/artifacts/adt [--threads 8]

``--tool prepare`` downloads the dataset (10x FASTQ tar, Cell Ranger 3.0 outputs) and writes the truth and shared inputs;
every other tool reads them. Each tool runs as its own process under ``/usr/bin/time -v`` (the harness re-invokes itself with
``--inner``), so wall time and peak RSS are measured the same way for all. Outputs in ``--out`` (small, per cell):

* ``adt_<dataset>_<tool>.parquet``: ``barcode``, ``feature``, ``count`` (non-zero entries, Cell Ranger's filtered cells,
  features named as in Cell Ranger's reference; ``cellranger`` is the truth);
* ``adt_<dataset>_rna.parquet``: ``barcode``, ``gene``, ``count``: the RNA (Cell Ranger's matrix) of the genes the panel's
  antibodies map to, the same RNA for every tool's protein-RNA concordance;
* ``adt_<dataset>_panel.tsv``: ``feature``, ``sequence``, ``gene``;
* ``adt_<dataset>_cells.tsv``: ``barcode``, ``rna_total`` (the scored cells: Cell Ranger's filtered barcodes);
* ``resources_adt_<dataset>_<tool>.json``: ``wall_s``, ``max_rss_gb``, ``threads``, command line, tool version.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TENX = "https://cf.10xgenomics.com/samples"
DATASETS = {
    "pbmc1k_cite_v3": {
        "tar": f"{TENX}/cell-exp/3.0.0/pbmc_1k_protein_v3/pbmc_1k_protein_v3_fastqs.tar",
        "h5": f"{TENX}/cell-exp/3.0.0/pbmc_1k_protein_v3/pbmc_1k_protein_v3_filtered_feature_bc_matrix.h5",
        "fastq_glob": "*antibody*_R1_*.fastq.gz", "chem": "10xv3", "umi": 12,
    },
    "pbmc10k_cite_v3": {
        "tar": "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/3.0.0/pbmc_10k_protein_v3/"
               "pbmc_10k_protein_v3_fastqs.tar",
        "h5": f"{TENX}/cell-exp/3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5",
        "fastq_glob": "*antibody*_R1_*.fastq.gz", "chem": "10xv3", "umi": 12,
    },
    "pbmc5p_v1": {
        "tar": f"{TENX}/cell-vdj/3.0.0/vdj_v1_hs_pbmc2_5gex_protein/vdj_v1_hs_pbmc2_5gex_protein_fastqs.tar",
        "h5": f"{TENX}/cell-vdj/3.0.0/vdj_v1_hs_pbmc2_5gex_protein/"
              "vdj_v1_hs_pbmc2_5gex_protein_filtered_feature_bc_matrix.h5",
        "fastq_glob": "*antibody*_R1_*.fastq.gz", "chem": "10xv2", "umi": 10,
    },
}
WL_V3 = ("3M-february-2018.txt.gz", "translation_3M-february-2018.txt.gz")
WL_V2 = ("737K-august-2016.txt.gz", None)


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def fetch(url, dest):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        sh(["curl", "-sSfL", "--retry", "8", "--retry-delay", "15", "-C", "-", "-o", dest + ".part", url])
        os.replace(dest + ".part", dest)
    return dest


def ddir(a):
    return os.path.join(a.data, a.dataset)


def fastqs(a):
    d = os.path.join(ddir(a), "fastqs")
    cfg = DATASETS[a.dataset]
    r1 = sorted(glob.glob(os.path.join(d, "**", cfg["fastq_glob"]), recursive=True))
    return r1, [x.replace("_R1_", "_R2_") for x in r1]


def wl(a, name):
    from singlet.adt.chemistry import WL_BASE

    return fetch(WL_BASE + name, os.path.join(a.data, "whitelists", name))


# ----------------------------------------------------------------------------------------------------------- prepare
def prepare(a):
    import h5py
    from scipy import sparse

    cfg = DATASETS[a.dataset]
    d = ddir(a)
    tar = fetch(cfg["tar"], os.path.join(d, os.path.basename(cfg["tar"])))
    fq = os.path.join(d, "fastqs")
    if not os.path.isdir(fq):
        with tarfile.open(tar) as t:
            members = [m for m in t.getmembers() if "antibod" in m.name.lower() and m.name.endswith(".fastq.gz")]
            t.extractall(fq, members=members)
    h5 = fetch(cfg["h5"], os.path.join(d, os.path.basename(cfg["h5"])))
    with h5py.File(h5, "r") as f:
        g = f["matrix"]
        M = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
        ft = {k: [x.decode() for x in g["features"][k][:]] for k in g["features"].keys()
              if g["features"][k].dtype.kind in "SO" and len(g["features"][k]) == M.shape[0]}
        bcs = [x.decode() for x in g["barcodes"][:]]
    ft = pd.DataFrame(ft)
    ab = np.flatnonzero(ft.feature_type.values == "Antibody Capture")
    gx = np.flatnonzero(ft.feature_type.values == "Gene Expression")
    os.makedirs(a.out, exist_ok=True)
    panel = pd.DataFrame({"feature": ft.name.values[ab], "sequence": ft.sequence.values[ab],
                          "pattern": ft.pattern.values[ab] if "pattern" in ft else ""})
    panel["gene"] = [antibody_gene(n) for n in panel.feature]
    panel.to_csv(os.path.join(a.out, f"adt_{a.dataset}_panel.tsv"), sep="\t", index=False)
    write_long(M[ab], list(panel.feature), bcs, os.path.join(a.out, f"adt_{a.dataset}_cellranger.parquet"))
    genes = [g for g in panel.gene.dropna().unique() if g]
    sym = ft.name.values[gx]
    rows = [gx[i] for i, s in enumerate(sym) if s in set(genes)]
    write_long(M[rows], list(ft.name.values[rows]), bcs, os.path.join(a.out, f"adt_{a.dataset}_rna.parquet"),
               cols=("barcode", "gene", "count"))
    rna_total = np.asarray(M[gx].sum(axis=0)).ravel().astype(np.int64)
    pd.DataFrame({"barcode": bcs, "rna_total": rna_total}).to_csv(
        os.path.join(a.out, f"adt_{a.dataset}_cells.tsv"), sep="\t", index=False)
    with open(os.path.join(d, "barcodes.txt"), "w") as fh:
        fh.write("\n".join(bcs) + "\n")
    json.dump({"cells": len(bcs), "antibodies": len(ab), "cellranger_umis": int(M[ab].sum())},
              open(os.path.join(d, "prepare.json"), "w"))


# antibody name -> gene symbol, for protein-RNA concordance (10x reference names such as "CD8a_TotalSeqB")
GENE = {"CD3": "CD3E", "CD4": "CD4", "CD8a": "CD8A", "CD8": "CD8A", "CD14": "CD14", "CD15": "FUT4", "CD16": "FCGR3A",
        "CD56": "NCAM1", "CD19": "CD19", "CD25": "IL2RA", "CD45RA": None, "CD45RO": None, "PD-1": "PDCD1",
        "TIGIT": "TIGIT", "CD127": "IL7R", "CD20": "MS4A1", "CD11c": "ITGAX", "CD11b": "ITGAM", "CD197": "CCR7",
        "CCR7": "CCR7", "CD62L": "SELL", "HLA-DR": "HLA-DRA", "CD27": "CD27", "CD28": "CD28", "CD69": "CD69",
        "CD38": "CD38", "CD34": "CD34", "CD1c": "CD1C", "CD123": "IL3RA", "CD161": "KLRB1", "CD2": "CD2", "CD7": "CD7",
        "CD5": "CD5", "CD33": "CD33", "CD36": "CD36", "CD44": "CD44", "CD86": "CD86", "CD10": "MME", "CD103": "ITGAE"}


def antibody_gene(name):
    base = re.split(r"[_ .(]", name)[0]
    if base.lower().startswith("igg") or "control" in name.lower():
        return ""
    g = GENE.get(base, GENE.get(base.upper(), ""))
    return g or ""


def write_long(M, rows, bcs, path, cols=("barcode", "feature", "count")):
    """Non-zero entries of a features x cells matrix as a long table (``cols``: cell, row name, count)."""
    C = M.tocoo()
    pd.DataFrame({cols[0]: np.asarray(bcs)[C.col], cols[1]: np.asarray(rows)[C.row],
                  cols[2]: C.data.astype(np.int64)}).to_parquet(path, index=False)


# ----------------------------------------------------------------------------------------------------------- tools
def tool_singlet(a):
    from singlet.adt import count as C
    from singlet.adt import panel as P
    from singlet.adt import seq as S

    panel = pd.read_csv(os.path.join(a.out, f"adt_{a.dataset}_panel.tsv"), sep="\t", dtype=str, keep_default_na=False)
    bcs = [x.strip() for x in open(os.path.join(ddir(a), "barcodes.txt")) if x.strip()]
    r1, r2 = fastqs(a)
    cfg = DATASETS[a.dataset]
    wlf, trf = WL_V3 if cfg["chem"] == "10xv3" else WL_V2
    head = []
    for i, p in enumerate(C.iter_pairs(r1, r2)):
        head.append(p)
        if i >= 300_000:
            break
    offs = P.detect_offset([b for _, b in head], list(panel.sequence))
    off = max(offs, key=offs.get)
    gcodes = np.sort(S.encode_strings([b.split("-")[0] for b in bcs], 16)[0])
    top = S.encode_strings([x[:16].decode() for x, _ in head if b"N" not in x[:16]], 16)[0]
    table = None
    if trf:
        table, _ = C.best_translation(top, gcodes, wl(a, trf))
    wlc = table[0] if table is not None else np.sort(S.encode_strings(C.load_barcode_list(wl(a, wlf)), 16)[0])
    lc = C.count_library(r1, r2, cb_len=16, umi_len=cfg["umi"], offset=off, panel_sequences=list(panel.sequence),
                         whitelist_codes=wlc, gex_barcodes=bcs, translation=table,
                         pairs_factory=lambda: C.iter_pairs(r1, r2))
    save_matrix(a, lc.matrix, list(panel.feature), bcs)


def save_matrix(a, M, features, bcs):
    M = M.tocsc()
    M.eliminate_zeros()
    write_long(M, features, bcs, os.path.join(a.out, f"adt_{a.dataset}_{a.tool}.parquet"))


def tool_kite(a):
    import anndata
    from scipy import sparse

    panel = pd.read_csv(os.path.join(a.out, f"adt_{a.dataset}_panel.tsv"), sep="\t", dtype=str, keep_default_na=False)
    bcs = [x.strip() for x in open(os.path.join(ddir(a), "barcodes.txt")) if x.strip()]
    d = os.path.join(a.scratch, "kite")
    os.makedirs(d, exist_ok=True)
    ft = os.path.join(d, "features.tsv")
    panel[["sequence", "feature"]].to_csv(ft, sep="\t", header=False, index=False)
    sh(["kb", "ref", "-i", f"{d}/idx", "-g", f"{d}/t2g.txt", "-f1", f"{d}/fa.fa", "--workflow", "kite", ft])
    r1, r2 = fastqs(a)
    cfg = DATASETS[a.dataset]
    tech = "10xv3" if cfg["chem"] == "10xv3" else "10xv2"
    wf = "kite:10xFB" if cfg["chem"] == "10xv3" else "kite"
    sh(["kb", "count", "-i", f"{d}/idx", "-g", f"{d}/t2g.txt", "-x", tech, "--workflow", wf, "-t", str(a.threads),
        "-o", f"{d}/out", "--h5ad"] + [x for p in zip(r1, r2) for x in p])
    ad = anndata.read_h5ad(glob.glob(f"{d}/out/counts_unfiltered/*.h5ad")[0])
    idx = {b: i for i, b in enumerate(ad.obs_names)}
    X = sparse.csr_matrix(ad.X)
    names = list(ad.var_names)
    rows = [idx.get(b.split("-")[0]) for b in bcs]
    sub = sparse.vstack([X[i] if i is not None else sparse.csr_matrix((1, X.shape[1])) for i in rows]).T.tocsc()
    save_matrix(a, sub, names, bcs)


def tool_citeseqcount(a):
    """CITE-seq-Count 1.4.5 (Stoeckius lab), the counter of the original CITE-seq and hashing papers."""
    from scipy.io import mmread
    from singlet.adt import count as C
    from singlet.adt import seq as S

    panel = pd.read_csv(os.path.join(a.out, f"adt_{a.dataset}_panel.tsv"), sep="\t", dtype=str, keep_default_na=False)
    bcs = [x.strip() for x in open(os.path.join(ddir(a), "barcodes.txt")) if x.strip()]
    cfg = DATASETS[a.dataset]
    d = os.path.join(a.scratch, "csc")
    os.makedirs(d, exist_ok=True)
    tags = os.path.join(d, "tags.csv")
    panel[["sequence", "feature"]].to_csv(tags, header=False, index=False)
    # CITE-seq-Count matches the read's barcode against a whitelist: for 3' v3 TotalSeq-B that is the feature-side barcode
    # of each Cell Ranger cell (inverse of 10x's translation), mapped back afterwards
    gcodes = S.encode_strings([b.split("-")[0] for b in bcs], 16)[0]
    back = {}
    if cfg["chem"] == "10xv3":
        fb, gex = C.load_translation(wl(a, WL_V3[1]))
        o = np.argsort(gex)
        i = S.lookup(gex[o], gcodes)
        fside = np.where(i >= 0, fb[o][np.maximum(i, 0)], gcodes)
        wl_b = S.decode(fside, 16)
        back = dict(zip(wl_b, [b.split("-")[0] for b in bcs]))
    else:
        wl_b = [b.split("-")[0] for b in bcs]
        back = {b: b for b in wl_b}
    wlf = os.path.join(d, "wl.txt")
    open(wlf, "w").write("\n".join(wl_b) + "\n")
    r1, r2 = fastqs(a)
    cat1, cat2 = os.path.join(d, "R1.fastq.gz"), os.path.join(d, "R2.fastq.gz")
    for src, dst in ((r1, cat1), (r2, cat2)):
        with open(dst, "wb") as fo:
            for x in src:
                shutil.copyfileobj(open(x, "rb"), fo)
    trim = 10 if any(str(p).startswith(("5P", "^NNNNNNNNNN")) for p in panel.pattern) else 0
    cmd = ["CITE-seq-Count", "-R1", cat1, "-R2", cat2, "-t", tags, "-cbf", "1", "-cbl", "16", "-umif", "17",
           "-umil", str(16 + cfg["umi"]), "-cells", str(len(bcs)), "-wl", wlf, "-o", f"{d}/out", "-T", str(a.threads)]
    if trim:
        cmd += ["--start-trim", str(trim)]
    sh(cmd)
    mtx = mmread(f"{d}/out/umi_count/matrix.mtx.gz").tocsc()
    feats = [x.split("\t")[0] for x in pd.read_csv(f"{d}/out/umi_count/features.tsv.gz", header=None)[0]]
    cells = [x.strip() for x in pd.read_csv(f"{d}/out/umi_count/barcodes.tsv.gz", header=None)[0]]
    col = {back.get(c, c): j for j, c in enumerate(cells)}
    names = [re.sub(r"-[ACGTN]+$", "", f) for f in feats]
    from scipy import sparse

    sub = sparse.hstack([mtx[:, col[b.split("-")[0]]] if b.split("-")[0] in col else sparse.csc_matrix((len(feats), 1))
                         for b in bcs]).tocsc()
    keep = [i for i, n in enumerate(names) if n != "unmapped"]
    save_matrix(a, sub[keep], [names[i] for i in keep], bcs)


def tool_alevinfry(a):
    """salmon 1.10 ``--features`` index + ``salmon alevin --sketch`` + alevin-fry (cr-like UMI resolution); 3' v3 feature
    barcodes are listed feature-side in the permit list and translated to GEX barcodes afterwards."""
    from scipy import sparse
    from scipy.io import mmread
    from singlet.adt import count as C
    from singlet.adt import seq as S

    b = os.environ.get("AFRY_BIN", "")
    salmon, fry = os.path.join(b, "salmon"), os.path.join(b, "alevin-fry")
    panel = pd.read_csv(os.path.join(a.out, f"adt_{a.dataset}_panel.tsv"), sep="\t", dtype=str, keep_default_na=False)
    bcs = [x.strip() for x in open(os.path.join(ddir(a), "barcodes.txt")) if x.strip()]
    cfg = DATASETS[a.dataset]
    d = os.path.join(a.scratch, "afry")
    os.makedirs(d, exist_ok=True)
    ft = os.path.join(d, "features.tsv")
    panel[["feature", "sequence"]].to_csv(ft, sep="\t", header=False, index=False)
    t2g = os.path.join(d, "t2g.tsv")
    pd.DataFrame({"a": panel.feature, "b": panel.feature}).to_csv(t2g, sep="\t", header=False, index=False)
    sh([salmon, "index", "-t", ft, "-i", f"{d}/idx", "--features", "-k7"])
    r1, r2 = fastqs(a)
    start = 11 if any(str(p).startswith(("5P", "^NNNNNNNNNN")) for p in panel.pattern) else 1
    L = len(panel.sequence.iloc[0])
    sh([salmon, "alevin", "-l", "ISR", "-i", f"{d}/idx", "-1"] + r1 + ["-2"] + r2 +
       ["--read-geometry", f"2[{start}-{start + L - 1}]", "--bc-geometry", "1[1-16]",
        "--umi-geometry", f"1[17-{16 + cfg['umi']}]", "-o", f"{d}/map", "-p", str(a.threads), "--sketch"])
    if cfg["chem"] == "10xv3":
        fb, gex = C.load_translation(wl(a, WL_V3[1]))
        plist = S.decode(fb, 16)
    else:
        fb = gex = None
        plist = C.load_barcode_list(wl(a, WL_V2[0]))
    pl = os.path.join(d, "permit.txt")
    open(pl, "w").write("\n".join(plist) + "\n")
    sh([fry, "generate-permit-list", "-d", "fw", "-i", f"{d}/map", "-o", f"{d}/quant", "--unfiltered-pl", pl,
        "--min-reads", "1"])
    sh([fry, "collate", "-i", f"{d}/quant", "-r", f"{d}/map", "-t", str(a.threads)])
    sh([fry, "quant", "-i", f"{d}/quant", "-o", f"{d}/res", "-t", str(a.threads), "-m", t2g, "-r", "cr-like"])
    X = sparse.csr_matrix(mmread(f"{d}/res/alevin/quants_mat.mtx"))
    rows = [x.strip() for x in open(f"{d}/res/alevin/quants_mat_rows.txt")]
    names = [x.strip() for x in open(f"{d}/res/alevin/quants_mat_cols.txt")]
    rc = S.encode_strings(rows, 16)[0]
    if fb is not None:
        rc = C.translate(rc, (fb, gex))[0]
    idx = {bc: i for i, bc in enumerate(S.decode(rc, 16))}
    pick = [idx.get(bc.split("-")[0]) for bc in bcs]
    sub = sparse.vstack([X[i] if i is not None else sparse.csr_matrix((1, X.shape[1])) for i in pick]).T.tocsc()
    save_matrix(a, sub, names, bcs)


TOOLS = {"singlet": tool_singlet, "kite": tool_kite, "citeseqcount": tool_citeseqcount, "alevinfry": tool_alevinfry}
VERSIONS = {"kite": ["kb", "--version"], "citeseqcount": ["CITE-seq-Count", "--version"],
            "alevinfry": [os.path.join(os.environ.get("AFRY_BIN", ""), "alevin-fry"), "--version"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--tool", required=True, choices=["prepare"] + sorted(TOOLS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/tmp")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--inner", action="store_true")
    a = ap.parse_args()
    if a.tool == "prepare":
        return prepare(a)
    if a.inner:
        return TOOLS[a.tool](a)
    # outer: time the inner run as its own process
    cmd = [sys.executable, "-P", os.path.abspath(__file__), "--inner"] + sys.argv[1:]
    t0 = time.time()
    r = subprocess.run(["/usr/bin/time", "-v"] + cmd, capture_output=True, text=True)
    wall = time.time() - t0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", r.stderr)
    ver = ""
    if a.tool in VERSIONS:
        try:
            ver = sh(VERSIONS[a.tool]).stdout.strip().splitlines()[-1]
        except Exception:
            ver = "unknown"
    rec = {"tool": a.tool, "dataset": a.dataset, "wall_s": round(wall, 1),
           "max_rss_gb": round(int(m.group(1)) / 1024 ** 2, 3) if m else None, "threads": a.threads,
           "returncode": r.returncode, "version": ver, "command": " ".join(cmd),
           "singlet_commit": os.environ.get("SINGLET_COMMIT", ""),
           "stderr_tail": r.stderr[-1500:] if r.returncode else ""}
    with open(os.path.join(a.out, f"resources_adt_{a.dataset}_{a.tool}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "stderr_tail"}))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
