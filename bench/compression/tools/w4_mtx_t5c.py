import os, sys, numpy as np, math
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); CORP = os.path.join(BC, "corpus_w4_mtx")
def H(c):
    c = np.asarray(c, float); c = c[c > 0]; return float(-(c * np.log2(c / c.sum())).sum())
for d in sys.argv[1:]:
    p = os.path.join(CORP, d)
    raw = open(os.path.join(p, "ambient_profile.tsv"), "rb").read().decode()
    L = raw.split("\n")[1:-1]
    names = [l.split("\t")[0] for l in L]; strs = [l.split("\t")[1] for l in L]
    v = np.array([float(x) for x in strs]); N = len(v); u = v.min()
    k = np.rint((v / u - 1) / 2).astype(np.int64); T = int(k.sum())
    # regenerate text from integer counts alone
    pred = [("%.6g" % ((int(x) + 0.5) / (T + 0.5 * N))) for x in k.tolist()]
    ok = sum(1 for a, b in zip(pred, strs) if a == b)
    regen = "feature\tambient_fraction\n" + "".join(f"{n}\t{s}\n" for n, s in zip(names, pred))
    print(f"[{d}] N={N} T(implied)={T}; strings regenerated from integer counts k with (k+0.5)/(T+0.5N), %.6g: {ok}/{N} equal; whole file byte-identical={regen==raw}; file bytes {len(raw)}")
    # find the T from the file which best reproduces strings
    ex = read_1pz(os.path.join(p, "exon_counts.1pz"))
    assert ex["rownames"] == names
    cs = set(l.split("\t")[0] for l in open(os.path.join(p, "cell_calls.tsv")).read().split("\n")[1:-1])
    iscell = np.array([c in cs for c in ex["colnames"]])
    ip = ex["indptr"].astype(np.int64); cn = np.repeat(np.arange(ex["n"]), np.diff(ip)); ix = ex["indices"].astype(np.int64); dv = ex["data"].astype(np.int64)
    sel = (~iscell)[cn]; c = np.bincount(ix[sel], weights=dv[sel], minlength=ex["m"]).astype(np.int64)
    r = k - c
    print(f"   non-cell column sums of exon_counts.1pz: {int(c.sum())} counts; implied {T}; residual r=k-c: nonzero rows {(r!=0).sum()}, negative {(r<0).sum()}, sum {int(r.sum())}, max {int(r.max())}")
    # costs (bytes): order-0 entropy of k alone; of residual r; positions of nonzero residual
    kk = np.bincount(k); rr = np.bincount(np.clip(r, 0, None))
    print(f"   cost estimates (order-0, bytes): k alone {H(kk)/8:.0f}; residual r (all rows, order-0) {H(rr)/8:.0f}; T and N header ~8")
    # colset check: which columns does the 1pz keep vs count
    print(f"   1pz columns {ex['n']}, cell columns {int(iscell.sum())}, non-cell columns {int((~iscell).sum())}")
