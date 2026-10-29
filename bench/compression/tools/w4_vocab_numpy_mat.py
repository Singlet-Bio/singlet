#!/usr/bin/env python
"""w4_vocab item 3 (Python side): cost of turning a materialised name blob into the objects a Python client needs,
and of the lazy alternative.   python -P tools/w4_vocab_numpy_mat.py REF_DIR [build]
REF_DIR has <build>.exon.nul written by tools/w4_vocab_dump.py (the eager-materialisation output of kernels/w4_vocab.h is byte identical)."""
import sys, os, time, random
import numpy as np
import pandas as pd
ref = sys.argv[1]; build = sys.argv[2] if len(sys.argv) > 2 else "GRCh38-2024-A"
raw = open(os.path.join(ref, build + ".exon.nul"), "rb").read()

def best(f, reps=5):
    b = 1e9
    for _ in range(reps):
        t = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t)
    return b * 1e3, r
res = []
arr = np.frombuffer(raw, np.uint8)
t, nul = best(lambda: np.flatnonzero(arr == 0)); res.append(("offsets index: flatnonzero(blob==0)", t))
n = len(nul)
off = np.concatenate([[0], nul[:-1] + 1]).astype(np.int64)
t, lst = best(lambda: raw.decode("ascii").split("\0")[:-1]); res.append(("list[str] via blob.decode().split(NUL)", t))
t, oarr = best(lambda: np.array(lst, dtype=object)); res.append(("np.array(list[str], dtype=object)", t))
t, idx = best(lambda: pd.Index(lst)); res.append(("pandas.Index(list[str])  (AnnData var_names)", t))
t, _ = best(lambda: pd.Index(lst).get_loc(lst[12345])); res.append(("pandas.Index(...) + first get_loc (hash build)", t))
def sarr():
    w = int((np.diff(np.concatenate([off, [len(raw)]])).max()))
    a = np.zeros((n, w), np.uint8)
    ln = np.diff(np.concatenate([off, [len(raw)]])) - 1
    rows = np.repeat(np.arange(n), ln); cols = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln)
    a[rows, cols] = arr[np.repeat(off, ln) + cols]
    return a.view("S%d" % w).ravel()
t, S = best(sarr, 3); res.append(("numpy fixed-width S%d array (scatter)" % S.dtype.itemsize, t))
# lazy lookups from blob + offsets (what a thin client holds): python slice + decode
q = [random.randrange(n) for _ in range(200000)]
def lazy():
    o = off.tolist(); e = (nul).tolist()
    s = 0
    for i in q: s += len(raw[o[i]:e[i]].decode())
    return s
t, _ = best(lazy, 3); res.append(("lazy lookup us/name (bytes slice+decode, offsets as python lists)", t * 1e3 / len(q)))
def lazy_np():
    s = 0
    for i in q: s += len(raw[off[i]:nul[i]].decode())
    return s
t, _ = best(lazy_np, 3); res.append(("lazy lookup us/name (offsets as numpy arrays)", t * 1e3 / len(q)))
mv = memoryview(raw); o = off.tolist(); e = nul.tolist()
def lazy_mv():
    s = 0
    for i in q: s += len(str(mv[o[i]:e[i]], "ascii"))
    return s
t, _ = best(lazy_mv, 3); res.append(("lazy lookup us/name (memoryview slice + str())", t * 1e3 / len(q)))
print("build", build, "names", n, "blob", len(raw))
for k, v in res: print("%-70s %10.3f %s" % (k, v, "us" if "lookup" in k else "ms"))
