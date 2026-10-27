"""M2: distinct-sequence fraction of real cDNA reads, within and across samples (docs/gpu-engine/exploration.md, M2).

    python bench/gpu/distinct_fraction.py A_R2.fastq.gz [B_R2.fastq.gz ...] [--len 90] [--json out.json]
    singlet 1fq decode s.1fq | python bench/gpu/distinct_fraction.py - --len 90

Input: the cDNA read only (10x R2), FASTQ, plain, gzip or stdin. Reads are trimmed to --len bases (the length STAR sees
after clipping; pass the sample's real clip settings, or the answer will be optimistic for 5' data with TSO clipping).
Reads with N are counted apart (STAR splits them; they are not cacheable as one key in this sketch).

Reports per sample:
  d          distinct / reads (the model's d; 3' PBMC 1k v3 = 0.47 from SeqCache.h)
  d@depth    the same on the first 1, 5, 20, 50% of the reads (file order ~ flowcell order, a fair subsample)
  lcp>=k     share of distinct sequences whose sorted neighbour shares >= k leading bases (k = 32, 50, 70): an upper
             proxy for exact seed reuse in prefix-sorted wavefronts (exploration.md 4.5)
and across all samples given: the share of each sample's distinct sequences seen in any other sample (cross-sample
dedup within one batch) and the union's distinct count.

Keys are 64-bit BLAKE2b digests of the 2-bit sequence: at 1e9 keys the expected number of colliding pairs is ~0.03,
negligible for a measurement (the engine itself compares full sequences). Memory: 8 B per read (numpy).
"""

import argparse
import gzip
import hashlib
import json
import sys

import numpy as np

ENC = bytes.maketrans(b"ACGT", b"\x00\x01\x02\x03")


def reads(path, length):
    fh = sys.stdin.buffer if path == "-" else (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb"))
    with fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                s = line.rstrip()[:length]
                if len(s) == length:
                    yield s


def digest(seq):
    return int.from_bytes(hashlib.blake2b(seq.translate(ENC), digest_size=8).digest(), "little")


def lcp_share(seqs, ks=(32, 50, 70)):
    seqs.sort()
    out = {}
    for k in ks:
        hit = np.zeros(len(seqs), dtype=bool)
        for i in range(1, len(seqs)):
            if seqs[i - 1][:k] == seqs[i][:k]:
                hit[i - 1] = hit[i] = True
        out[f"lcp>={k}"] = float(hit.mean()) if len(seqs) else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fastq", nargs="+")
    ap.add_argument("--len", type=int, default=90)
    ap.add_argument("--lcp-sample", type=int, default=2_000_000, help="distinct sequences kept for the LCP proxy")
    ap.add_argument("--json")
    a = ap.parse_args()

    res, uniq = {}, {}
    for path in a.fastq:
        keys, n_n, sample = [], 0, []
        for s in reads(path, a.len):
            if b"N" in s:
                n_n += 1
                continue
            keys.append(digest(s))
            if len(sample) < a.lcp_sample and (len(keys) & 15) == 0:
                sample.append(s)        # every 16th read: a cheap, order-preserving subsample for the LCP proxy
        k = np.asarray(keys, dtype=np.uint64)
        u = np.unique(k)
        r = {"reads": int(k.size), "reads_with_N": n_n, "d": float(u.size / max(k.size, 1))}
        for frac in (0.01, 0.05, 0.20, 0.50):
            m = int(k.size * frac)
            r[f"d@{int(frac * 100)}%"] = float(np.unique(k[:m]).size / max(m, 1))
        r.update(lcp_share(sorted(set(sample))))
        res[path], uniq[path] = r, u
        print(json.dumps({path: r}), flush=True)

    if len(uniq) > 1:
        allk = np.concatenate(list(uniq.values()))
        vals, counts = np.unique(allk, return_counts=True)
        shared = vals[counts > 1]
        cross = {p: float(np.isin(u, shared).mean()) for p, u in uniq.items()}
        res["_cross"] = {"union_distinct": int(vals.size), "sum_distinct": int(allk.size),
                         "batch_dedup_gain": float(allk.size / vals.size), "share_seen_elsewhere": cross}
        print(json.dumps({"_cross": res["_cross"]}), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
