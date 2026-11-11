# SPDX-License-Identifier: MIT
"""Molecule/read-level test of whether ambient RNA differs from native RNA, with a molecule-level truth: cross-species reads in pure cells of a human+mouse library.
  python soup_molecules.py RUN_DIR BAM FASTA GTF OUT_PREFIX

Cells: called_v2 barcodes whose filtered-matrix UMIs are >= 95% one species (pure human / pure mouse).
Ambient set A(X in Y): reads of species X genes in pure-Y cells. Native set N(X in X): reads of species X genes in pure-X cells. Same gene catalogue on both sides, so
differences are about the molecule, not about the species. Reads: primary, NH == 1, with GX (gene) and RE in {E, I}. Native reads are subsampled 1/16 by a hash of the read; every
ambient read is kept. Features (per read): aligned length, soft clip left/right, mismatches nM, spliced, exonic(RE), antisense (read vs gene strand), distance of the nearest alignment end to the gene 3' end
(bp, log1p) and as a fraction of gene length, A-richness (transcript-sense A fraction) of the 20 bp beyond each alignment end, reads in the molecule (CB, UB, gene) over ALL reads of called cells (dup), mean base quality.
Gene-adjusted: each feature minus the median of the native reads of the same gene. Depth strata: pure cells in quartiles of total UMI within species. AUC = P(ambient > native)."""
import gzip, sys, json, hashlib
import numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as sp
import pysam
run, bam, fasta, gtf, outp = sys.argv[1:6]
bare = lambda b: b.split("-")[0]
# --- cells and their species purity from the filtered matrix
M = sio.mmread(gzip.open(f"{run}/filtered_feature_bc_matrix/matrix.mtx.gz", "rb")).tocsr()
feats = [l.split("\t")[0] for l in gzip.open(f"{run}/filtered_feature_bc_matrix/features.tsv.gz", "rt")]
bcs = [bare(l.strip()) for l in gzip.open(f"{run}/filtered_feature_bc_matrix/barcodes.tsv.gz", "rt")]
mouse = np.array([f.startswith("ENSMUSG") for f in feats])
um = np.asarray(M[mouse].sum(0)).ravel(); uh = np.asarray(M[~mouse].sum(0)).ravel(); tot = um + uh
fm = um / np.maximum(tot, 1)
cell_class = {}; cell_depth = {}
for i, b in enumerate(bcs):
    if tot[i] >= 500 and fm[i] < 0.05: cell_class[b] = "human"
    elif tot[i] >= 500 and fm[i] > 0.95: cell_class[b] = "mouse"
    cell_depth[b] = int(tot[i])
print("pure cells", pd.Series(cell_class).value_counts().to_dict(), flush=True)
# depth quartiles within species
dq = {}
for sp_ in ("human", "mouse"):
    cs = [b for b, c in cell_class.items() if c == sp_]; d = np.array([cell_depth[b] for b in cs]); q = np.quantile(d, [.25, .5, .75])
    for b, x in zip(cs, d): dq[b] = int(np.searchsorted(q, x))
# --- gene models: strand and extent
gene = {}
with open(gtf) as f:
    for ln in f:
        if ln.startswith("#"): continue
        t = ln.split("\t")
        if t[2] != "gene": continue
        gid = t[8].split('gene_id "')[1].split('"')[0].split(".")[0]
        gene[gid] = (t[0], int(t[3]), int(t[4]), t[6])
fa = pysam.FastaFile(fasta)
comp = {"A": "T", "T": "A"}
def a_frac(chrom, start, end, strand):  # transcript-sense A fraction of [start,end) (0-based)
    start = max(0, start); end = min(fa.get_reference_length(chrom), end)
    if end - start < 10: return np.nan
    s = fa.fetch(chrom, start, end).upper()
    return (s.count("A") if strand == "+" else s.count("T")) / len(s)
# --- one pass
B = pysam.AlignmentFile(bam, "rb")
keys_all = []; rows = []
n = 0
for r in B.fetch(until_eof=True):
    n += 1
    if r.is_secondary or r.is_unmapped: continue
    try:
        cb = r.get_tag("CB")
    except KeyError:
        continue
    cb = bare(cb)
    if cb not in cell_class and cb not in cell_depth: continue
    if r.get_tag("NH") != 1: continue
    try:
        gx = r.get_tag("GX"); re_ = r.get_tag("RE"); ub = r.get_tag("UB")
    except KeyError:
        continue
    if "," in gx or re_ not in ("E", "I"): continue
    keys_all.append((cb, ub, gx))
    cls = cell_class.get(cb)
    if cls is None: continue
    gsp = "mouse" if gx.startswith("ENSMUSG") else "human"
    amb = gsp != cls
    if not amb:
        h = int(hashlib.blake2b((r.query_name).encode(), digest_size=4).hexdigest(), 16)
        if h % 16: continue
    g = gene.get(gx.split(".")[0]);
    if g is None: continue
    cig = r.cigartuples; sl = cig[0][1] if cig[0][0] == 4 else 0; sr = cig[-1][1] if cig[-1][0] == 4 else 0
    st, en = r.reference_start, r.reference_end
    e3 = g[2] if g[3] == "+" else g[1]
    d3 = min(abs(st - e3), abs(en - e3)); glen = g[2] - g[1] + 1
    rows.append((cb, gx, cls, int(amb), dq[cb], ub, r.query_alignment_length, sl, sr, r.get_tag("nM"), int(any(c[0] == 3 for c in cig)), int(re_ == "E"),
                 int(r.is_reverse != (g[3] == "-")), np.log1p(d3), d3 / glen, a_frac(r.reference_name, en, en + 20, g[3]), a_frac(r.reference_name, st - 20, st, g[3]),
                 float(np.mean(r.query_alignment_qualities)) if r.query_alignment_qualities is not None else np.nan, r.reference_name, st, en))
    if n % 5_000_000 == 0: print("reads", n, "kept rows", len(rows), flush=True)
cols = ["cb", "gene", "cell_species", "ambient", "depth_q", "ub", "alen", "clip_l", "clip_r", "nM", "spliced", "exonic", "antisense", "log_d3", "frac_d3", "A_after_end", "A_before_start", "mean_q", "chrom", "st", "en"]
D = pd.DataFrame(rows, columns=cols)
K = pd.Series(keys_all).value_counts()
dup = K.to_dict()
D["reads_in_molecule"] = [dup.get((c, u, g), 1) for c, u, g in zip(D.cb, D.ub, D.gene)]
D.drop(columns=["chrom", "st", "en", "ub"]).to_parquet(outp + ".reads.parquet") if False else None
D.drop(columns=["chrom", "st", "en", "ub"]).to_csv(outp + ".reads.tsv.gz", sep="\t", index=False)
print("rows", len(D), "ambient", int(D.ambient.sum()), "native sample", int((1 - D.ambient).sum()), flush=True)
