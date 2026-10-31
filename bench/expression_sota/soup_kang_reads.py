# SPDX-License-Identifier: MIT
"""Read-level features (same definitions as soup_molecules.py, without the A-richness windows) for all reads of a random sample of genotype-confident Kang cells.
  python soup_kang_reads.py BAM GTF TRUTH_TSV.gz N_CELLS OUT.tsv.gz
Cells: from the ambient truth artifact (rho_truth, weight = informative-read mass), weight >= 100, random N_CELLS (seed 1). Reads: primary, NH == 1, GX, RE in {E, I}."""
import gzip, sys, hashlib
import numpy as np, pandas as pd, pysam
bam, gtf, truth, n, out = sys.argv[1:6]; n = int(n)
bare = lambda b: b.split("-")[0]
T = pd.read_csv(truth, sep="\t"); T = T[T.weight >= 100]; T["b"] = T.barcode.map(bare)
cells = set(T.sample(min(n, len(T)), random_state=1).b)
gene = {}
for ln in open(gtf):
    if ln.startswith("#"): continue
    t = ln.split("\t")
    if t[2] == "gene": gene[t[8].split('gene_id "')[1].split('"')[0].split(".")[0]] = (t[3], t[4], t[6])
B = pysam.AlignmentFile(bam, "rb"); rows = []; keys = []
for r in B.fetch(until_eof=True):
    if r.is_secondary or r.is_unmapped: continue
    try: cb = bare(r.get_tag("CB"))
    except KeyError: continue
    if cb not in cells: continue
    try:
        if r.get_tag("NH") != 1: continue
        gx = r.get_tag("GX"); re_ = r.get_tag("RE"); ub = r.get_tag("UB")
    except KeyError: continue
    if "," in gx or re_ not in ("E", "I"): continue
    g = gene.get(gx.split(".")[0])
    if g is None: continue
    keys.append((cb, ub, gx))
    cig = r.cigartuples; sl = cig[0][1] if cig[0][0] == 4 else 0; sr = cig[-1][1] if cig[-1][0] == 4 else 0
    st, en = r.reference_start, r.reference_end; s0, e0 = int(g[0]), int(g[1]); e3 = e0 if g[2] == "+" else s0
    d3 = min(abs(st - e3), abs(en - e3))
    rows.append((cb, gx, ub, r.query_alignment_length, sl, sr, r.get_tag("nM"), int(any(c[0] == 3 for c in cig)), int(r.is_reverse != (g[2] == "-")), np.log1p(d3), d3 / (e0 - s0 + 1)))
D = pd.DataFrame(rows, columns=["cb", "gene", "ub", "alen", "clip_l", "clip_r", "nM", "spliced", "antisense", "log_d3", "frac_d3"])
dup = pd.Series(keys).value_counts().to_dict()
D["reads_in_molecule"] = [dup.get((c, u, g), 1) for c, u, g in zip(D.cb, D.ub, D.gene)]
D.drop(columns="ub").to_csv(out, sep="\t", index=False)
print("cells", len(cells), "reads", len(D), flush=True)
