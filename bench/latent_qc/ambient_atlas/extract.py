"""Per-sample ambient / called-cell summaries from the fetched gene-level npz files (+ binary dumps for the C++ harness).

usage: extract.py OUT_REC_DIR DUMP_DIR CHUNK NCHUNK NPZ_ROOT [NPZ_ROOT ...]   (run inside rt, python -P)

Per sample writes OUT_REC_DIR/<GSM>.npz with gene-level (exon+intron summed) count vectors over the sample's own genes:
  called   sum over called cells           pool1  ambient pool, rule 1 (uncalled, UMI < median(uncalled) and < 10% of the called median)
  pool2    uncalled with <= 100 UMIs (production rule)    pool3  stricter (uncalled, UMI < 25th pct of uncalled and < 5% of called median)
  poolA / poolB   random halves (by barcode) of pool1      called_i / pool1_i   intronic parts
and a JSON ``stat`` with pool sizes, UMIs, intronic fractions, protocol, strand check. Skipped samples go to skipped.tsv.
If DUMP_DIR != '-', exon / intron binary CSC + called columns are dumped for the harness.
"""
import glob
import json
import os
import sys
import zlib

import numpy as np
import scipy.sparse as sp

rec_dir, dump_dir, chunk, nchunk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
roots = sys.argv[5:]
os.makedirs(rec_dir, exist_ok=True)
files = {}
for r in roots:
    for f in glob.glob(f"{r}/*/GSM*.npz"):
        files.setdefault(os.path.basename(f)[:-4], f)
gsms = sorted(files)[chunk::nchunk]
skipped = []


def load(z, kind, G, B):
    ip = z[f"{kind}_indptr"]
    if len(ip) != B + 1:
        return sp.csc_matrix((G, B), dtype=np.float64)
    return sp.csc_matrix((z[f"{kind}_data"].astype(np.float64), z[f"{kind}_indices"], ip), shape=(G, B))


for gsm in gsms:
    out = f"{rec_dir}/{gsm}.npz"
    if os.path.exists(out) and (dump_dir == "-" or os.path.exists(f"{dump_dir}/{gsm}/dims.txt")):
        continue
    try:
        z = np.load(files[gsm], allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        genes, bcs, called = z["genes"], z["barcodes"], z["called"].astype(bool)
        G, B = len(genes), len(bcs)
        if G == 0 or B == 0:
            skipped.append((gsm, "empty matrix")); continue
        ps = meta.get("pileup_stats", {}) or {}
        sm = meta.get("summary", {}) or {}
        ws, eh = ps.get("wrong_strand", 0), ps.get("exon_hits", 0)
        wrong = ws / max(ws + eh, 1)
        if wrong > 0.5:
            skipped.append((gsm, f"wrong strand {wrong:.2f}")); continue
        if called.sum() < 20:
            skipped.append((gsm, f"only {int(called.sum())} called cells")); continue
        E, I = load(z, "exon", G, B), load(z, "intron", G, B)
        T = (E + I).tocsc()
        tot = np.asarray(T.sum(0)).ravel()
        itot = np.asarray(I.sum(0)).ravel()
        unc = ~called
        med_c = float(np.median(tot[called]))
        if unc.sum() < 50:
            skipped.append((gsm, f"{int(unc.sum())} uncalled barcodes")); continue
        med_u = float(np.median(tot[unc]))
        p1 = unc & (tot < med_u) & (tot < 0.10 * med_c)
        p2 = unc & (tot <= 100)
        p3 = unc & (tot < np.percentile(tot[unc], 25)) & (tot < 0.05 * med_c)
        rng = np.random.default_rng(zlib.crc32(gsm.encode()))
        half = rng.random(B) < 0.5

        def vec(m, M=T):
            return np.asarray(M @ m.astype(np.float64)).ravel()

        a1, a2, a3 = vec(p1), vec(p2), vec(p3)
        aA, aB = vec(p1 & half), vec(p1 & ~half)
        ccall = vec(called)
        cI = vec(called, I)
        aI = vec(p1, I)
        stat = dict(gsm=gsm, gse=meta.get("gse"), n_genes=G, n_barcodes=B, n_called=int(called.sum()),
                    med_called_umi=med_c, med_uncalled_umi=med_u,
                    pool1_n=int(p1.sum()), pool1_umi=float(a1.sum()), pool2_n=int(p2.sum()), pool2_umi=float(a2.sum()),
                    pool3_n=int(p3.sum()), pool3_umi=float(a3.sum()), poolA_umi=float(aA.sum()), poolB_umi=float(aB.sum()),
                    called_umi=float(ccall.sum()), intron_frac_called=float(cI.sum() / max(ccall.sum(), 1)),
                    intron_frac_pool=float(aI.sum() / max(a1.sum(), 1)),
                    protocol=sm.get("protocol_name"), species=sm.get("species"), ref=sm.get("reference_build"),
                    wrong_strand_frac=wrong, strand_flipped=ps.get("strand_auto_flipped"),
                    source=(meta.get("gsm_meta") or {}).get("sample_source"),
                    chars=((meta.get("gsm_meta") or {}).get("sample_characteristics") or "")[:300])
        np.savez_compressed(out, genes=genes, symbols=z["symbols"], chroms=z["chroms"], called=ccall.astype(np.float32),
                            pool1=a1.astype(np.float32), pool2=a2.astype(np.float32), pool3=a3.astype(np.float32),
                            poolA=aA.astype(np.float32), poolB=aB.astype(np.float32), called_i=cI.astype(np.float32),
                            pool1_i=aI.astype(np.float32), stat=np.array(json.dumps(stat)))
        if dump_dir != "-":
            d = f"{dump_dir}/{gsm}"
            os.makedirs(d, exist_ok=True)
            for tag, M in (("exon", E), ("intron", I)):
                M = M.tocsc(); M.sort_indices()
                M.indptr.astype(np.int64).tofile(f"{d}/{tag}.indptr")
                M.indices.astype(np.int32).tofile(f"{d}/{tag}.indices")
                M.data.astype(np.uint32).tofile(f"{d}/{tag}.data")
            ci = np.flatnonzero(called).astype(np.uint32)
            ci.tofile(f"{d}/cells.idx")
            with open(f"{d}/dims.txt", "w") as f:
                f.write(f"{G} {B} {E.nnz} {I.nnz} {len(ci)} {int(max(10, 0.10 * med_c))}\n")
        print(gsm, "ok", f"called={int(called.sum())} pool1={int(p1.sum())}/{int(a1.sum())}", flush=True)
    except Exception as e:  # noqa: BLE001
        skipped.append((gsm, f"{type(e).__name__}: {e}"))
        print(gsm, "FAILED", e, flush=True)
with open(f"{rec_dir}/skipped_{chunk}.tsv", "w") as f:
    for g, r in skipped:
        f.write(f"{g}\t{r}\n")
