# SPDX-License-Identifier: MIT
# D-152 diagnostic (run on Clipper in debug-cpu; see the PR body of ve/non10x-routes for what it measured).
import sys, collections, re
# usage: inspect_reads.py R1 R2 N
r1, r2, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
def reads(p, n):
    out = []
    with open(p) as f:
        for i, ln in enumerate(f):
            if i % 4 == 1:
                out.append(ln.strip())
                if len(out) >= n: break
    return out
A, B = reads(r1, n), reads(r2, n)
print("lens R1", collections.Counter(map(len, A)).most_common(4), "R2", collections.Counter(map(len, B)).most_common(4))
for i in range(8): print("R1", A[i]); print("R2", B[i]); print()
W1 = "GAGTGATTGCTTGTGACGCCTT"
def hamm(a, b): return sum(x != y for x, y in zip(a, b))
def find(read, motif, mm=2):
    best = None
    for p in range(len(read) - len(motif) + 1):
        h = hamm(read[p:p+len(motif)], motif)
        if h <= mm and (best is None or h < best[1]): best = (p, h)
    return best
for name, S in (("R1", A), ("R2", B)):
    c = collections.Counter()
    for r in S[:100000]:
        f = find(r, W1)
        c[f[0] if f else None] += 1
    print(name, "W1 offset", c.most_common(8))
for name, S in (("R1", A), ("R2", B)):
    comp = [collections.Counter() for _ in range(max(map(len, S[:20000])))]
    for r in S[:20000]:
        for j, ch in enumerate(r): comp[j][ch] += 1
    print(name, "base comp per position (first 60): " + " ".join("".join(k for k, _ in x.most_common(1)) + "%02d" % (100 * x.most_common(1)[0][1] // max(1, sum(x.values()))) for x in comp[:60]))
# BD linkers
L1, L2 = "ACTGGCCTGCGA", "GGTAGCGGTGACA"
for name, S in (("R1", A), ("R2", B)):
    c1, c2 = collections.Counter(), collections.Counter()
    for r in S[:100000]:
        f = find(r, L1, 1); g = find(r, L2, 1)
        c1[f[0] if f else None] += 1; c2[g[0] if g else None] += 1
    print(name, "BD L1 offs", c1.most_common(5), "L2 offs", c2.most_common(5))
# polyT / polyA tails
for name, S in (("R1", A), ("R2", B)):
    tails = collections.Counter()
    for r in S[:100000]:
        tails["T8_end" if r.endswith("T"*8) else "A8_end" if r.endswith("A"*8) else "other"] += 1
    print(name, tails)
