"""Hit rate of bounded sequence caches (entries = distinct cDNA sequences held).

    python policy.py OUT.json CLIP5P R2.fastq.gz...

fifo     insert on miss, evict oldest insert
second   fifo, but a hit on an entry older than half the capacity re-inserts it (second chance)
lru      least recently used
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from hitrate import hashes  # noqa: E402


def fifo(h, cap, second):
    d = {}; q = collections.deque(); hits = 0; t = 0
    half = cap // 2
    for x in h:
        t += 1
        ins = d.get(x)
        if ins is not None:
            hits += 1
            if second and t - ins > half:
                d[x] = t; q.append((x, t))
            continue
        d[x] = t; q.append((x, t))
        while len(d) > cap:
            y, ty = q.popleft()
            if d.get(y) == ty:
                del d[y]
    return hits / len(h)


def lru(h, cap):
    d = collections.OrderedDict(); hits = 0
    for x in h:
        if x in d:
            hits += 1; d.move_to_end(x)
        else:
            d[x] = None
            if len(d) > cap:
                d.popitem(last=False)
    return hits / len(h)


def main():
    out, clip, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    h = hashes(files, clip)
    rng = np.random.default_rng(1)
    res = {}
    for name, arr in (("all", h), ("sub25", h[rng.random(len(h)) < 0.25])):
        lst = arr.tolist(); r = {"reads": len(lst)}
        for cap in (500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000):
            r[f"fifo_{cap}"] = fifo(lst, cap, False)
            r[f"second_{cap}"] = fifo(lst, cap, True)
            if cap <= 2_000_000:
                r[f"lru_{cap}"] = lru(lst, cap)
            print(name, cap, {k: round(v, 4) for k, v in r.items() if k.endswith(f"_{cap}")}, flush=True)
        res[name] = r
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
