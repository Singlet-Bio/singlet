# SPDX-License-Identifier: MIT
# D-152 diagnostic: reads with an internal poly(A) run (the insert runs through the poly(A) tail into the barcode). usage: indrop_polya.py FASTQ [N]
import sys, re, collections
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
pat = re.compile("A{10,}"); tot = 0; withrun = 0; term = 0; pre = collections.Counter(); after = collections.Counter(); W1 = "AAGGCGTC"
with open(sys.argv[1]) as f:
    for i, ln in enumerate(f):
        if i % 4 != 1: continue
        s = ln.strip(); tot += 1
        m = pat.search(s)
        if m:
            withrun += 1
            if m.end() >= len(s) - 1: term += 1
            else: after[min((len(s) - m.end()) // 5 * 5, 30)] += 1
            pre[min(m.start() // 5 * 5, 50)] += 1
        if tot >= n: break
print("reads", tot, "with an A-run of 10 or more: %d (%.1f%%); run at the read end: %d; run followed by more sequence: %d" % (withrun, 100 * withrun / tot, term, withrun - term))
print("bases before the run (5-nt bins)", sorted(pre.items()))
print("bases after the run (5-nt bins)", sorted(after.items()))
