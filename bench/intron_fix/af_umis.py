"""Per-UMI USA labels from an alevin-fry RAD file (alevin-fry view on stdin), for cells only.

    alevin-fry view --rad map.rad | python af_umis.py T2G CELLS_BARCODES OUT.tsv.gz

Targets: ENST... (spliced transcript, gene from t2g) or <gene>-G (spliceu unspliced gene body). Per (CB, UMI): genes hit
by each read; the UMI goes to the gene hit by the most reads (cr-like: ties -> M, not counted). Label for that gene over
its reads: spliced targets only -> S, unspliced only -> U, both -> A. CB is the raw barcode (not corrected).
"""
import gzip
import sys
from collections import defaultdict

t2g = {}
for ln in open(sys.argv[1]):
    p = ln.rstrip("\n").split("\t")
    t2g[p[0]] = p[1]
cells = {l.strip().split("-")[0] for l in open(sys.argv[2])}
out = gzip.open(sys.argv[3], "wt", compresslevel=1)

umis = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (cb, umi) -> gene -> [n_reads, status bits]
cur_id, cur_key, cur_genes = None, None, {}


def flush():
    if cur_key is None:
        return
    d = umis[cur_key]
    for g, b in cur_genes.items():
        d[g][0] += 1
        d[g][1] |= b


for ln in sys.stdin:
    if not ln.startswith("ID:"):
        continue
    p = ln.rstrip("\n").split("\t")
    rid = p[0]
    if rid != cur_id:
        flush()
        cur_id, cur_genes = rid, {}
        cb = p[3][3:]
        cur_key = (cb, p[4][4:]) if cb in cells else None
    if cur_key is None:
        continue
    t = p[6]
    if t.endswith("-G"):
        g, bit = t[:-2], 2
    else:
        g, bit = t2g.get(t, t), 1
    cur_genes[g] = cur_genes.get(g, 0) | bit
flush()

for (cb, umi), d in umis.items():
    best = max(v[0] for v in d.values())
    top = [g for g, v in d.items() if v[0] == best]
    if len(top) > 1:
        out.write(f"{cb}\t{umi}\t-\tM\n")
        continue
    g = top[0]
    b = d[g][1]
    out.write(f"{cb}\t{umi}\t{g}\t{'S' if b == 1 else 'U' if b == 2 else 'A'}\n")
out.close()
