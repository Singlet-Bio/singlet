# SPDX-License-Identifier: MIT
# D-152 diagnostic (run on Clipper in debug-cpu; see the PR body of ve/non10x-routes for what it measured).
import sys, collections
W = "/mnt/projects/debruinz_project/debruinz/singlet/jobs/validation-engine/non10x/whitelists/"
wl = [set(l.strip() for l in open(W + "bd_cls%d.txt" % i) if l.strip()) for i in (1, 2, 3)]
def hv(s):
    return {s[:i] + c + s[i+1:] for i in range(len(s)) for c in "ACGT"} | {s}
wl1 = [set().union(*[hv(b) for b in w]) for w in wl]
L1, L2 = "ACTGGCCTGCGA", "GGTAGCGGTGACA"
def ham(a, b): return sum(x != y for x, y in zip(a, b))
def fnd(r, m, lo, hi, mm=1):
    best = None
    for p in range(lo, hi + 1):
        if p + len(m) <= len(r):
            h = ham(r[p:p+len(m)], m)
            if h <= mm and (best is None or h < best[1]): best = (p, h)
    return best
N = 0; fixed_exact = 0; fixed_1mm = 0; anch = 0; anch_gain = collections.Counter(); l1off = collections.Counter(); l2off = collections.Counter()
with open(sys.argv[1]) as f:
    for i, ln in enumerate(f):
        if i % 4 != 1: continue
        r = ln.strip(); N += 1
        if N > 300000: break
        segs = (r[0:9], r[21:30], r[43:52])
        ok_e = all(s in w for s, w in zip(segs, wl)); ok_1 = all(s in w for s, w in zip(segs, wl1))
        fixed_exact += ok_e; fixed_1mm += ok_1
        a = fnd(r, L1, 5, 13); b = fnd(r, L2, 24, 36)
        l1off[a[0] if a else None] += 1; l2off[b[0] if b else None] += 1
        if a and b:
            p1, p2 = a[0], b[0]
            s1, s2, s3 = r[p1-9:p1] if p1 >= 9 else None, r[p1+12:p1+21], r[p2+13:p2+22]
            if s1 and len(s1) == 9 and s1 in wl1[0] and s2 in wl1[1] and s3 in wl1[2]: anch += 1
print("reads", N, "fixed positions all-exact %.3f 1MM %.3f | linker-anchored all-valid (1MM) %.3f" % (fixed_exact / N, fixed_1mm / N, anch / N))
print("L1 offsets", l1off.most_common(6)); print("L2 offsets", l2off.most_common(6))
