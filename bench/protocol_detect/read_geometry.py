#!/usr/bin/env python3
"""Read geometry of one library head: modal read lengths, the fixed regions of each read and the reuse of the leading k-mers.

A fixed region is a run of at least 6 positions where one base holds at least 90% of the reads (a linker or adapter). Barcode segments show
as variable positions whose k-mers repeat across reads (distinct/total well below 1), UMIs as variable positions that do not repeat.
usage: read_geometry.py ID R1.fastq R2.fastq   -> one TSV line: id, r1_len, r2_len, r1_fixed, r2_fixed, r1_kmer_reuse, r2_top100_15mer_share
"""
import collections
import sys


def reads(path, n=20000):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                out.append(line.strip())
                if len(out) >= n:
                    break
    return out


def fixed_regions(seqs, width=100):
    if not seqs:
        return ""
    mode = collections.Counter(len(s) for s in seqs).most_common(1)[0][0]
    keep = [s for s in seqs if len(s) >= min(mode, width)]
    w = min(mode, width)
    cons = []
    for p in range(w):
        c = collections.Counter(s[p] for s in keep)
        b, k = c.most_common(1)[0]
        cons.append(b if k / len(keep) >= 0.90 else ".")
    s = "".join(cons)
    runs, i = [], 0
    while i < len(s):
        if s[i] != ".":
            j = i
            while j < len(s) and s[j] != ".":
                j += 1
            if j - i >= 6:
                runs.append(f"{i}-{j}:{s[i:j]}")
            i = j
        else:
            i += 1
    return ",".join(runs) if runs else "none"


def kmer_reuse(seqs, k=12, off=0):
    ks = [s[off:off + k] for s in seqs if len(s) >= off + k and "N" not in s[off:off + k]]
    if not ks:
        return -1
    c = collections.Counter(ks)
    return round(sum(v for v in c.values() if v >= 2) / len(ks), 3)


def top100_share(seqs, k=15, off=0):
    ks = [s[off:off + k] for s in seqs if len(s) >= off + k]
    if not ks:
        return -1
    c = collections.Counter(ks)
    return round(sum(v for _, v in c.most_common(100)) / len(ks), 3)


def main(id_, p1, p2):
    a, b = reads(p1), reads(p2)
    ml = lambda x: collections.Counter(len(s) for s in x).most_common(1)[0][0] if x else 0
    print("\t".join(map(str, [id_, ml(a), ml(b), fixed_regions(a), fixed_regions(b), kmer_reuse(a), kmer_reuse(b), top100_share(b)])))


if __name__ == "__main__":
    main(*sys.argv[1:4])
