#!/usr/bin/env python3
"""Replay protocol.h detect_protocol's per-candidate scoring on the first N read pairs of a FASTQ pair (diagnostic, counts only).

  protocol_probe.py <R1.fastq.gz> <R2.fastq.gz> <whitelist dir> <out.json> [N=5000]

For each candidate (10x-3p-v3, 10x-3p-v2, splitseq) it reports the whitelist hit rate (exact, and Hamming-1 when the detector would
fall back to it), UMI entropy, R2 poly(A) fraction, the R1 geometry bonus, the poly(T) right after BC+UMI, and the score and confidence
detect_protocol computes from them (whitelist-only branch). It also reports R1/R2 length modes and base composition of R1 by position.
"""
import gzip, itertools, json, math, os, sys
from collections import Counter

CANDS = {  # tag: (protocol_id, r1_len, bc_off, bc_len, umi_off, umi_len, whitelist)
    "10x-3p-v3": (1, 28, 0, 16, 16, 12, "3M-february-2018.txt"),
    "10x-3p-v2": (2, 26, 0, 16, 16, 10, "737K-august-2016.txt"),
    "splitseq": (11, 66, 10, 8, 0, 10, "splitseq_bc_rd3.txt"),
}


def reads(p, n):
    with gzip.open(p, "rt") as f:
        for rec in itertools.islice(zip(*[f] * 4), n):
            yield rec[1].strip()


def ham1(s, wl):
    if s in wl: return True
    for i, c in enumerate(s):
        for b in "ACGT":
            if b != c and (s[:i] + b + s[i + 1:]) in wl: return True
    return False


def entropy(seqs, off, ln):
    tot = 0.0
    for p in range(ln):
        c = Counter(s[off + p] for s in seqs if len(s) > off + p and s[off + p] in "ACGT")
        v = sum(c.values())
        if v < 10: continue
        tot += -sum(x / v * math.log2(x / v) for x in c.values())
    return tot / ln


def main(r1p, r2p, wld, out, n=5000):
    n = int(n)
    R1 = list(reads(r1p, n)); R2 = list(reads(r2p, n))
    r1len = Counter(map(len, R1)).most_common(1)[0][0]; r2len = Counter(map(len, R2)).most_common(1)[0][0]
    polya = sum(1 for s in R2 if len(s) >= 20 and s[-15:].count("A") >= 10) / len(R2)
    res = {"n": len(R1), "r1_len_mode": r1len, "r2_len_mode": r2len, "r2_polya_frac": polya, "candidates": {}}
    comp = []
    for p in range(min(r1len, 101)):
        c = Counter(s[p] for s in R1 if len(s) > p); v = sum(c.values())
        comp.append({b: round(c[b] / v, 3) for b in "ACGTN"})
    res["r1_base_composition"] = comp
    for tag, (pid, el, bo, bl, uo, ul, wf) in CANDS.items():
        wl = set()
        path = os.path.join(wld, wf)
        if os.path.exists(path):
            with open(path) as f:
                wl = {l.split()[0][:bl] for l in f if l.strip()}
        exact = sum(1 for s in R1 if len(s) >= bo + bl and s[bo:bo + bl] in wl) / len(R1) if wl else 0.0
        use_exact = len(wl) >= 100000
        h1 = sum(1 for s in R1 if len(s) >= bo + bl and ham1(s[bo:bo + bl], wl)) / len(R1) if wl and (not use_exact or exact < 0.3) else None
        rate = exact if use_exact and exact >= 0.3 else (h1 if h1 is not None else exact)
        H = entropy(R1, uo, ul)
        diff = r1len - el
        geo = 0.15 if diff == 0 else 0.10 if -2 <= diff <= 2 else 0.05 if 0 < diff <= 10 else -0.20 if diff < -2 else 0.0
        if uo + ul > r1len: geo -= 0.10 * (uo + ul - r1len) / ul
        pts = bo + bl + ul if bo == 0 else max(bo + bl, uo + ul)
        ipt = sum(1 for s in R1 if len(s) >= pts + 6 and (s[pts:pts + 12].count("T") >= 6 or s[pts:pts + 12].count("A") >= 6)) / len(R1)
        score = 0.5 * rate + 0.2 * min(H / 2, 1) + 0.1 * polya + 0.2 * (rate > 0.5) + geo if wl else None
        conf = None if score is None else "HIGH" if score >= 0.85 else "MEDIUM" if score >= 0.6 else "LOW" if score >= 0.4 else "NONE"
        res["candidates"][tag] = {"whitelist_found": bool(wl), "whitelist_size": len(wl), "rate_exact": exact, "rate_hamming1": h1,
                                  "rate_used": rate, "umi_entropy": H, "geometry_bonus": geo, "polyT_after_bc_umi": ipt,
                                  "score": score, "confidence": conf}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "r1_base_composition"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
