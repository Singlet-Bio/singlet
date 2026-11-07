"""Compare private/snp_ad|snp_dp.mtx.gz of the 8-bit (a) and 16-bit (b) engines: every entry whose a-value is below 255 must
be identical in b; entries at 255 (clamped in a) may be larger in b. Usage: snp_below_clamp.py <dir_a> <dir_b>"""
import gzip
import sys


def load(p):
    d = {}
    with gzip.open(p, "rt") as fh:
        for ln in fh:
            if ln.startswith("%"):
                continue
            t = ln.split()
            if len(t) == 3 and "." not in t[0]:
                d[(int(t[0]), int(t[1]))] = int(float(t[2]))
    return d


a_dir, b_dir = sys.argv[1:3]
for name in ("snp_ad", "snp_dp"):
    a, b = load(f"{a_dir}/private/{name}.mtx.gz"), load(f"{b_dir}/private/{name}.mtx.gz")
    below = [k for k, v in a.items() if v < 255]
    bad = [k for k in below if b.get(k) != a[k]]
    clamped = [k for k, v in a.items() if v >= 255]
    grew = sum(1 for k in clamped if b.get(k, 0) > a[k])
    extra = [k for k in b if k not in a]
    print(f"{name}: entries a={len(a)} b={len(b)}; below 255 in a: {len(below)}, differing in b: {len(bad)}; "
          f"at 255 in a: {len(clamped)}, larger in b: {grew}; only in b: {len(extra)}; max b = {max(b.values())}")
