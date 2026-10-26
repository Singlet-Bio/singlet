"""Where does Singlet's default download path change the cDNA read?

    python lossy_diagnose.py REF.fastq DEFAULT.fastq LOSSLESS.fastq [SAMPLE]

Three decodes of the same run: NCBI's fasterq-dump (REF), `singlet download` defaults then `singlet decode` (DEFAULT), and
`singlet download --no-trim --quality full` (LOSSLESS). The .1fq is not in archive order (reads are sorted by barcode), so the files are
compared as multisets of reads, not in lockstep:
  1. read order: does lockstep comparison match? (it does not when the .1fq is reordered)
  2. LOSSLESS vs REF: multiset of (sequence, quality) equal?
  3. DEFAULT vs REF: how many DEFAULT sequences are absent from REF, and for a sample of them (SAMPLE, default 200000) the ref read
     they came from (found through the first 20 bases): 3' trim (DEFAULT is a prefix of the ref read), 5' trim, substitution, other;
     the trimmed length and the base composition of the removed tail; the share of ref reads that end in a poly(A) run.
  4. qualities: alphabet and counts of each file.
Needs numpy. Reads that are identical in sequence (duplicates) are counted with multiplicity.
"""
import collections
import sys
import zlib

import numpy as np


def stream(path):
    with open(path, "rb") as fh:
        while True:
            h = fh.readline()
            if not h:
                return
            s = fh.readline().rstrip(b"\n")
            fh.readline()
            q = fh.readline().rstrip(b"\n")
            yield s, q


def h64(b):
    return (zlib.crc32(b) << 32) | zlib.adler32(b)


def hashes(path, with_q):
    hs = []
    alpha = collections.Counter()
    first = []
    for i, (s, q) in enumerate(stream(path)):
        x = h64(s)
        if with_q:
            x = (x * 1000003 + h64(q)) & 0xFFFFFFFFFFFFFFFF
        hs.append(x)
        if i < 5:
            first.append(s)
        if i < 200000:
            alpha.update(q)
    return np.array(hs, dtype=np.uint64), alpha, first


def main():
    ref, dfl, lsl = sys.argv[1:4]
    nsamp = int(sys.argv[4]) if len(sys.argv) > 4 else 200000
    rs, ra, rfirst = hashes(ref, False)
    ds, da, dfirst = hashes(dfl, False)
    ls, la, lfirst = hashes(lsl, False)
    print(f"reads: ref {len(rs):,} default {len(ds):,} lossless {len(ls):,}")
    print("lockstep: first 5 reads equal ref vs default:", sum(a == b for a, b in zip(rfirst, dfirst)), "of 5; ref vs lossless:", sum(a == b for a, b in zip(rfirst, lfirst)), "of 5")
    rs_sorted = np.sort(rs)
    print("lossless sequences multiset equal to ref:", bool(np.array_equal(np.sort(ls), rs_sorted)))
    ru, rc = np.unique(rs_sorted, return_counts=True)
    du, dc = np.unique(np.sort(ds), return_counts=True)
    idx = np.searchsorted(ru, du)
    idx[idx >= len(ru)] = 0
    present = ru[idx] == du
    absent_reads = int(dc[~present].sum())
    print(f"default reads whose sequence is absent from ref: {absent_reads:,} ({absent_reads / len(ds):.3%}); distinct absent sequences {int((~present).sum()):,}")
    print("quality alphabets (first 200k reads): ref", dict(ra), "| default", dict(da), "| lossless", dict(la))
    # classify a sample of absent default reads against the ref read they came from
    absent_set = set(int(x) for x in du[~present])
    sample = {}
    for s, _ in stream(dfl):
        if h64(s) in absent_set:
            sample.setdefault(s[:20], []).append(s)
            if sum(len(v) for v in sample.values()) >= nsamp:
                break
    c = collections.Counter()
    tail = collections.Counter()
    trim = collections.Counter()
    polya_ref = 0
    n_ref = 0
    for s, _ in stream(ref):
        n_ref += 1
        if s.endswith(b"AAAAAAAA"):
            polya_ref += 1
        for d in sample.get(s[:20], ()):
            if d == s:
                continue
            if s.startswith(d):
                c["3prime_trim"] += 1
                t = s[len(d):]
                trim[len(t)] += 1
                tail["A" if set(t) <= {65, 78} else "mixed"] += 1  # bytes: A=65 N=78
            elif len(d) == len(s):
                c["same_length_substitution"] += 1
            else:
                c["other"] += 1
    tot = sum(c.values())
    print(f"classified {tot:,} sampled absent default reads by their 20-base prefix: {dict(c)}")
    print("3' trim: removed tail all A/N vs mixed:", dict(tail), "; removed length top:", trim.most_common(8))
    print(f"ref reads ending in >= 8 A: {polya_ref / max(n_ref, 1):.2%}")


if __name__ == "__main__":
    main()
