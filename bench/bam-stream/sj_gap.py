"""Where does the stream's extra junction count sit? (5' serial-vs-sorted gap, ws-w3b update h)

Usage: python -I sj_gap.py <decoded A sj_counts prefix> <decoded B prefix> <A out dir> <B out dir>
Prints the junctions with the largest A-B count difference, how the difference splits by
per-cell count level (1, 2, 3+ in B) and whether the junction passes sj_features' filter.
"""
import gzip, sys
import numpy as np, pandas as pd, scipy.io


def load(p):
    m = scipy.io.mmread(p + ".mtx").tocsr().astype(np.int64)
    rows = pd.read_csv(p + ".rows.tsv", header=None, dtype=str, keep_default_na=False, skip_blank_lines=False)[0].fillna("").values
    cols = pd.read_csv(p + ".cols.tsv", header=None, dtype=str, keep_default_na=False, skip_blank_lines=False)[0].fillna("").values
    keep = rows != ""
    return m[keep], rows[keep], cols


ma, ra, ca = load(sys.argv[1])
mb, rb, cb = load(sys.argv[2])
rows = pd.Index(ra).intersection(pd.Index(rb))
cols = pd.Index(ca).intersection(pd.Index(cb))
A = ma[pd.Index(ra).get_indexer(rows)][:, pd.Index(ca).get_indexer(cols)]
B = mb[pd.Index(rb).get_indexer(rows)][:, pd.Index(cb).get_indexer(cols)]
D = (A - B).tocoo()
print("total A", A.sum(), "B", B.sum(), "diff", D.sum(), "entries", (D.data != 0).sum())
print("diff sign: A>B", (D.data > 0).sum(), "A<B", (D.data < 0).sum())
bvals = np.asarray(B[D.row, D.col]).ravel()
for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 10**9)):
    sel = (bvals >= lo) & (bvals <= hi)
    print(f"B count {lo}-{hi if hi < 10**9 else 'inf'}: entries {sel.sum()} diff {D.data[sel].sum()}")
per_j = pd.Series(np.asarray((A - B).sum(1)).ravel(), index=rows)
tot_b = pd.Series(np.asarray(B.sum(1)).ravel(), index=rows)
top = per_j.sort_values(ascending=False).head(25)
feat = {}
for side, out in (("A", sys.argv[3]), ("B", sys.argv[4])):
    try:
        f = pd.read_csv(f"{out}/sj_features.tsv.gz", sep="\t")
        feat[side] = f
    except Exception as e:
        print("sj_features", side, e)
if "B" in feat:
    print("sj_features columns:", list(feat["B"].columns)[:12])
print(pd.DataFrame({"diff": top, "B_total": tot_b[top.index], "rel": top / tot_b[top.index].clip(lower=1)}).to_string())
print("junctions with any diff:", int((per_j != 0).sum()), "of", len(per_j))
print("share of diff in top 25:", float(top.sum() / max(per_j.sum(), 1)))
cell_diff = pd.Series(np.asarray((A - B).sum(0)).ravel(), index=cols)
print("cells with diff:", int((cell_diff != 0).sum()), "top cell diffs:", cell_diff.sort_values(ascending=False).head(5).to_dict())
