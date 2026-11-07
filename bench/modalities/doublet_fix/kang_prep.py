"""Kang pool (genotype_pool benchmark): Singlet's called cells -> raw binary CSC for dbl_run.
  OUT_PREFIX      exon matrix (310,797 exon intervals) + .f2g exon->gene map (what export.h passes to detect_doublets)
  OUT_PREFIX_in   intron matrix (272,191 intervals) + .f2g into the same gene index (second source in export.h)
  OUT_PREFIXg     gene-level filtered_feature_bc_matrix of the same called cells (what Scrublet was run on)
  OUT_PREFIXi     exon + intron counts folded to genes (should equal the filtered matrix)
Usage: kang_prep.py SINGLET_DIR OUT_PREFIX      (SINGLET_DIR has exon_counts.mtx.gz, exon_counts_barcodes/features.tsv.gz, cell_calls.tsv)"""
import gzip, sys, os, glob, numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
sd, pre = sys.argv[1:3]
os.makedirs(os.path.dirname(pre), exist_ok=True)
cc = pd.read_csv(f"{sd}/cell_calls.tsv", sep="\t")
called = set(cc.loc[cc.is_cell.astype(str).str.upper() == "TRUE", "barcode"])

def write(E, bcs, p, f2g=None):
    E = E.tocsc(); E.data = np.minimum(E.data, 65535).astype(np.uint16)
    E.indices = E.indices.astype(np.int32); E.indptr = E.indptr.astype(np.int32)
    np.array([E.shape[0], E.shape[1], E.nnz], dtype=np.uint32).tofile(p + ".hdr")
    E.indptr.tofile(p + ".indptr"); E.indices.tofile(p + ".indices"); E.data.tofile(p + ".data")
    if f2g is not None: np.asarray(f2g, dtype=np.uint32).tofile(p + ".f2g")
    pd.Series(bcs).to_csv(p + ".barcodes.tsv", index=False, header=False)
    print(p, E.shape, "nnz", E.nnz)

def rd(name):
    with gzip.open(f"{sd}/{name}_counts.mtx.gz", "rt") as f: M = scipy.io.mmread(f).tocsc()
    b = pd.read_csv(f"{sd}/{name}_counts_barcodes.tsv.gz", sep="\t", header=None)[0].astype(str).values
    ft = pd.read_csv(f"{sd}/{name}_counts_features.tsv.gz", sep="\t", header=None)[0].astype(str).values
    return M, b, pd.Series([x.split("_")[0] for x in ft])

E, bc, eg = rd("exon")
keep = np.array([i for i, b in enumerate(bc) if b in called])
have_i = os.path.exists(f"{sd}/intron_counts.mtx.gz")
if have_i:
    I, ibc, ig = rd("intron"); assert (ibc == bc).all()
    allg = pd.factorize(pd.concat([eg, ig]))[0].astype(np.uint32)
    ge, gi = allg[:len(eg)], allg[len(eg):]
else:
    ge = pd.factorize(eg)[0].astype(np.uint32)
write(E[:, keep], bc[keep], pre, ge)
if have_i:
    write(I[:, keep], bc[keep], pre + "_in", gi)
    ng = int(allg.max()) + 1
    Me = sp.csr_matrix((np.ones(len(ge)), (ge, np.arange(len(ge)))), shape=(ng, len(ge)))
    Mi = sp.csr_matrix((np.ones(len(gi)), (gi, np.arange(len(gi)))), shape=(ng, len(gi)))
    write((Me @ E[:, keep] + Mi @ I[:, keep]).tocsc(), bc[keep], pre + "i")

mdir = f"{sd}/filtered_feature_bc_matrix"
op = lambda p: gzip.open(p, "rt") if p.endswith(".gz") else open(p)
with op(glob.glob(mdir + "/matrix.mtx*")[0]) as f: X = scipy.io.mmread(f).tocsc()
gb = np.array([l.strip() for l in op(glob.glob(mdir + "/barcodes.tsv*")[0])])
called_short = {c.split("-")[0] for c in called}
gk = np.array([i for i, b in enumerate(gb) if b.split("-")[0] in called_short])
write(X[:, gk], np.array([b.split("-")[0] for b in gb[gk]]), pre + "g")
