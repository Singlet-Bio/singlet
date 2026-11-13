# SPDX-License-Identifier: MIT
# D-152 diagnostic: what are the reads STAR leaves unmapped in the inDrop islet library? usage: indrop_unmapped.py star_Unmapped.out.mate1 TOTAL_READS
import sys, collections
W1RC = "AAGGCGTCACAAGCAATCACTC"; ADA = "AGATCGGAAGAGC"
def has(read, motif, mm=2):
    L = len(motif)
    for p in range(len(read) - L + 1):
        if sum(x != y for x, y in zip(read[p:p+L], motif)) <= mm: return p
    return -1
tot = int(sys.argv[2]); c = collections.Counter(); n = 0; ins = collections.Counter()
with open(sys.argv[1]) as f:
    for i, ln in enumerate(f):
        if i % 4 == 1: s = ln.strip()
        elif i % 4 == 3:
            q = ln.strip(); n += 1
            fa = s.count("A") / len(s); lowq = sum(ch in "#,:" for ch in q) / len(q)
            wp = has(s, W1RC); ap = has(s, ADA, 1)
            k = []
            if fa >= 0.6: k.append("polyA_rich")
            if wp >= 0: k.append("W1rc")
            if ap >= 0: k.append("adapter")
            if lowq >= 0.5: k.append("lowq_half")
            if s.count("N") >= 3: k.append("Ns")
            c[("+".join(k) or "none")] += 1
            if wp >= 0: ins[min(wp // 5 * 5, 50)] += 1
print("unmapped reads", n, "of", tot, "(%.1f%%)" % (100 * n / tot))
for k, v in c.most_common(12): print("%-32s %7d  %5.1f%% of unmapped  %5.1f%% of all reads" % (k, v, 100 * v / n, 100 * v / tot))
print("insert length before W1rc (5-nt bins)", sorted(ins.items()))
