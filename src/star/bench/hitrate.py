"""How often does a 10x cDNA read repeat an earlier read's sequence exactly?

    python hitrate.py OUT.json CLIP5P R2_1.fastq.gz [R2_2.fastq.gz ...]

Reads the cDNA reads in input order, clips CLIP5P bases from the 5' end, hashes the rest
(length included) and reports, for the whole file and for a random 25% subsample:
  infinite      share of reads whose sequence occurred earlier anywhere (perfect cache)
  chunk_<N>     share within chunks of N consecutive reads (grouping inside a chunk only)
  window_<G>    share whose previous occurrence is at most G reads back (cache of recent reads)
  live_max      most sequences that are 'live' at once (seen, and seen again later): the
                memory an oracle cache needs to catch every repeat
"""
import json
import subprocess
import sys

import numpy as np


def hashes(files, clip):
    out = []
    for f in files:
        p = subprocess.Popen(["bash", "-c", f"zcat {f} | awk 'NR%4==2'"], stdout=subprocess.PIPE, bufsize=1 << 24)
        for line in p.stdout:
            out.append(hash(line[clip:].rstrip()))
        p.wait()
    return np.array(out, dtype=np.int64)


def stats(h):
    n = len(h)
    idx = np.arange(n, dtype=np.int64)
    order = np.lexsort((idx, h))
    hs = h[order]
    same = np.empty(n, dtype=bool); same[0] = False; same[1:] = hs[1:] == hs[:-1]
    prev = np.full(n, -1, dtype=np.int64)
    prev[order[same]] = order[np.where(same)[0] - 1]
    res = {"reads": int(n), "infinite": float(same.mean())}
    gap = idx - prev
    rep = prev >= 0
    for g in (75_000, 300_000, 1_000_000, 4_000_000, 16_000_000):
        res[f"window_{g}"] = float((rep & (gap <= g)).mean())
    for c in (75_000, 300_000, 1_200_000):
        chunk = idx // c
        res[f"chunk_{c}"] = float((rep & (chunk[np.maximum(prev, 0)] == chunk)).mean())
    # live set: a sequence is live from its first to its last occurrence
    first = np.ones(n, dtype=bool); first[1:] = hs[1:] != hs[:-1]
    last = np.ones(n, dtype=bool); last[:-1] = hs[1:] != hs[:-1]
    starts = order[first & ~last]; ends = order[last & ~first]
    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, starts, 1); np.add.at(delta, ends, -1)
    res["live_max"] = int(np.cumsum(delta).max())
    res["distinct"] = int(first.sum())
    res["distinct_repeated"] = int((first & ~last).sum())
    return res


def main():
    out, clip, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    h = hashes(files, clip)
    res = {"all": stats(h)}
    rng = np.random.default_rng(1)
    res["sub25"] = stats(h[rng.random(len(h)) < 0.25])
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
