#!/usr/bin/env python
"""w4_corpus: split of nnz (exon and intron matrices) between called cells (cell_calls.tsv) and the other columns ("empties").

    python -P tools/w4_corpus_empties.py  ->  results/w4_corpus_empties.tsv
Payload-bit split is NOT computed here (needs per-column code lengths from a codec); nnz share is the proxy, bits per nnz are
higher for empties (see Track D), so the nnz share understates the empties' share of payload bits.
"""
import os, sys, csv, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import e4_lib as L
CORP = os.path.join(BC, "corpus_w4_corpus"); OUT = os.path.join(BC, "results", "w4_corpus_empties.tsv")

def main():
    pz = L.load_pz()
    ids = sys.argv[1:] or sorted(d for d in os.listdir(CORP) if not d.startswith("_") and d != "x_stub" and os.path.isdir(os.path.join(CORP, d)))
    f = open(OUT, "w", newline=""); w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["id", "protocol", "member", "n_cols", "n_called", "nnz", "nnz_called", "umi_total", "umi_called", "frac_cols_called", "frac_nnz_called", "frac_umi_called", "nnz_per_called_col", "nnz_per_empty_col"])
    for sid in ids:
        sj = json.load(open(os.path.join(CORP, sid, "summary.json")))
        calls = {ln.split("\t")[0] for ln in open(os.path.join(CORP, sid, "cell_calls.tsv")).read().split("\n")[1:] if ln}
        for m in ("exon", "intron"):
            p = os.path.join(CORP, sid, f"{m}_counts.1pz")
            M = pz.read_1pz(p)
            ip = M["indptr"].astype(np.int64); n = M["n"]
            called = np.fromiter((c in calls for c in M["colnames"]), bool, count=n)
            cn = np.diff(ip)
            cs = np.add.reduceat(M["data"].astype(np.uint64), ip[:-1][cn > 0]) if M["nnz"] else np.zeros(0, np.uint64)
            umi = np.zeros(n, np.uint64); umi[cn > 0] = cs
            nnz = int(cn.sum()); nc = int(cn[called].sum())
            w.writerow([sid, sj.get("protocol_name"), m, n, int(called.sum()), nnz, nc, int(umi.sum()), int(umi[called].sum()),
                        round(called.mean(), 5), round(nc / max(nnz, 1), 5), round(float(umi[called].sum()) / max(float(umi.sum()), 1), 5),
                        round(nc / max(called.sum(), 1), 2), round((nnz - nc) / max((~called).sum(), 1), 2)])
        f.flush(); print(sid, flush=True)
    f.close()

if __name__ == "__main__":
    main()
