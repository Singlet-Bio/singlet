"""w3_coexp stage 1 prep: gene-level presence of called cells from exon .1pz files (numpy only).
Run:  python -P tools/w3_coexp_prep.py            (all 7 corpus + 9 corpus_train files)
Output: results/w3_coexp_cache/<id>.npz   (corpus files are untrusted input: only parsed, never executed)
"""
import sys, os, time, csv
import numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, '../../python')

def read_tsv(p):
    with open(p, newline='') as f:
        r = csv.reader(f, delimiter='\t')
        hdr = next(r)
        return hdr, [row for row in r]

def one(path_dir, cid, study):
    t0 = time.time()
    r = read_1pz(os.path.join(path_dir, 'exon_counts.1pz'))
    m, n = r['m'], r['n']
    rn = r['rownames']
    ids, syms = [], []
    for s in rn:
        a = s.split('_', 2)
        ids.append(a[0]); syms.append(a[1])
    ids = np.array(ids); syms = np.array(syms)
    uid, first, inv = np.unique(ids, return_index=True, return_inverse=True)
    # keep gene order of first appearance (rows are grouped by gene)
    order = np.argsort(first)
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    ginv = rank[inv].astype(np.int32)                 # interval row -> gene index (by first appearance)
    gids = uid[order]; gsym = syms[first[order]]
    ng = len(gids)
    nint_per_gene = np.bincount(ginv, minlength=ng).astype(np.int32)
    hdr, rows = read_tsv(os.path.join(path_dir, 'cell_calls.tsv'))
    ic = hdr.index('is_cell'); bc = hdr.index('barcode')
    called = {row[bc] for row in rows if row[ic].upper() == 'TRUE'}
    cn = r['colnames']
    cmask = np.fromiter((c in called for c in cn), dtype=bool, count=n)
    indptr = r['indptr'].astype(np.int64); indices = r['indices']; data = r['data'].astype(np.int64)
    out = {}
    for tag, mask in (('c', cmask), ('e', ~cmask)):
        cols = np.nonzero(mask)[0]
        if tag == 'e':   # empties: keep only a random 5000 with >= 20 nnz for control use later (cheap)
            lens = indptr[cols + 1] - indptr[cols]
            cols = cols[lens >= 5]
            rng = np.random.default_rng(1); 
            if len(cols) > 5000: cols = np.sort(rng.choice(cols, 5000, replace=False))
        gi_ptr = [0]; gi_idx = []; gi_cnt = []; gi_nint = []; dint = []; ntot = []
        for c in cols:
            s, e = indptr[c], indptr[c + 1]
            g = ginv[indices[s:e]]
            v = data[s:e]
            # rows ascending within column and grouped by gene -> g is non-decreasing
            ug, st = np.unique(g, return_index=True)
            cnt = np.add.reduceat(v, st)
            nin = np.diff(np.append(st, len(g)))
            gi_idx.append(ug.astype(np.int32)); gi_cnt.append(cnt.astype(np.int32)); gi_nint.append(nin.astype(np.int16))
            gi_ptr.append(gi_ptr[-1] + len(ug)); dint.append(e - s); ntot.append(int(v.sum()))
        out[tag] = dict(ptr=np.array(gi_ptr, dtype=np.int64),
                        idx=np.concatenate(gi_idx) if gi_idx else np.zeros(0, np.int32),
                        cnt=np.concatenate(gi_cnt) if gi_cnt else np.zeros(0, np.int32),
                        nint=np.concatenate(gi_nint) if gi_nint else np.zeros(0, np.int16),
                        d=np.array(dint, dtype=np.int32), N=np.array(ntot, dtype=np.int64))
    np.savez_compressed(os.path.join('results/w3_coexp_cache', cid + '.npz'),
        gids=gids, gsym=gsym, nint_per_gene=nint_per_gene, study=np.array(study), m=m, n=n,
        c_ptr=out['c']['ptr'], c_idx=out['c']['idx'], c_cnt=out['c']['cnt'], c_nint=out['c']['nint'], c_d=out['c']['d'], c_N=out['c']['N'],
        e_ptr=out['e']['ptr'], e_idx=out['e']['idx'], e_cnt=out['e']['cnt'], e_nint=out['e']['nint'], e_d=out['e']['d'], e_N=out['e']['N'])
    nc = len(out['c']['d'])
    print(f"{cid}: genes={ng} intervals={m} called={nc} nnz_int_called={int(out['c']['d'].sum())} nnz_gene_called={len(out['c']['idx'])} "
          f"medd={int(np.median(out['c']['d']))} empties_kept={len(out['e']['d'])}  {time.time()-t0:.1f}s", flush=True)

if __name__ == '__main__':
    want = sys.argv[1:]
    rows = []
    for tsv, base in (('corpus.tsv', 'corpus'), ('corpus_train.tsv', 'corpus_train')):
        hdr, rr = read_tsv(tsv)
        for x in rr:
            rows.append((x[0], x[1], base))
    for cid, gse, base in rows:
        if want and cid not in want: continue
        one(os.path.join(base, cid), cid, gse)
