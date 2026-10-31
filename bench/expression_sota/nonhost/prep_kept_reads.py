"""Pair the reads Singlet's non-host screen keeps (nonhost_reads.fq = cDNA mate, STAR-unmapped, classified viral/bacterial/fungal)
with their barcode mates (STAR Unmapped.out.mate2, CB16+UMI12) by read name.
usage: python -I prep_kept_reads.py nonhost_reads.fq Unmapped.out.mate2 OUT_PREFIX  -> OUT_PREFIX.cdna.fq, OUT_PREFIX.bc.fq (same order), OUT_PREFIX.json"""
import sys, json
kept_f, m2_f, out = sys.argv[1:4]
kept = {}
order = []
with open(kept_f) as f:
    while True:
        h = f.readline()
        if not h:
            break
        s, p, q = f.readline(), f.readline(), f.readline()
        rid = h[1:].split()[0]
        kept[rid] = (s, q)
        order.append(rid)
bc = {}
with open(m2_f) as f:
    while True:
        h = f.readline()
        if not h:
            break
        s, p, q = f.readline(), f.readline(), f.readline()
        rid = h[1:].split()[0]
        if rid in kept:
            bc[rid] = (s, q)
n = 0
with open(out + '.cdna.fq', 'w') as a, open(out + '.bc.fq', 'w') as b:
    for rid in order:
        if rid not in bc:
            continue
        s, q = kept[rid]
        a.write(f'@{rid}\n{s}+\n{q}')
        s2, q2 = bc[rid]
        b.write(f'@{rid}\n{s2}+\n{q2}')
        n += 1
json.dump({'kept_reads': len(order), 'paired_with_barcode_mate': n}, open(out + '.json', 'w'))
print(len(order), n)
