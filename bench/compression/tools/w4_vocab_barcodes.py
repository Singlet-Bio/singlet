#!/usr/bin/env python
"""w4_vocab item 4: barcode whitelist packs, per-sample saving of whitelist-index coding, break-even.

    python -P tools/w4_vocab_barcodes.py     -> results/w4_vocab_barcodes_*.tsv and a text dump on stdout
Whitelists are the files shipped in python/singlet/data/whitelists (read only).
"""
import lzma, math, os, sys, collections, glob
import numpy as np
import zstandard

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WL = os.path.join(ROOT, "..", "..", "python", "singlet", "data", "whitelists")
Z19 = zstandard.ZstdCompressor(level=19)
rng = np.random.default_rng(12345)
B2 = {"A": 0, "C": 1, "G": 2, "T": 3}
OUT = os.path.join(ROOT, "results")


def code2(s):
    v = 0
    for ch in s:
        v = v * 4 + B2[ch]
    return v


def leb(a):
    out = bytearray()
    for v in a.tolist():
        while v >= 128:
            out.append((v & 127) | 128); v >>= 7
        out.append(v)
    return bytes(out)


def rice_bytes(sorted_vals):
    """best Rice code (parameter k) for the gaps of a sorted strictly increasing sequence -> bytes."""
    g = np.diff(np.concatenate([[-1], sorted_vals])).astype(np.int64) - 1
    best = None
    for k in range(0, 40):
        bits = int(((g >> k) + 1 + k).sum())
        if best is None or bits < best[0]:
            best = (bits, k)
    return (best[0] + 7) // 8 + 1


def log2_binom(U, n):
    # log2 C(U, n) with lgamma
    return (math.lgamma(U + 1) - math.lgamma(n + 1) - math.lgamma(U - n + 1)) / math.log(2)


def main():
    lines = []
    P = lambda *a: (lines.append(" ".join(str(x) for x in a)), print(*a, flush=True))
    # ---- whitelist pack sizes
    P("# whitelist pack sizes (bytes); pack = sorted 2-bit codes, gaps as LEB128, zstd-19 / xz-9e")
    P("%-22s %9s %3s %11s %11s %11s %11s %11s %9s" % ("list", "n", "len", "text", "2bit", "gaps+zstd19", "gaps+xz", "gaps_rice", "idx_bits"))
    WLS = {}
    rows = []
    for f in sorted(glob.glob(os.path.join(WL, "*.txt"))):
        name = os.path.basename(f)[:-4]
        w = [l.strip() for l in open(f) if l.strip()]
        k = len(w[0])
        codes = np.array(sorted(code2(x) for x in w), np.uint64)
        WLS[name] = (w, codes, k)
        gaps = np.diff(np.concatenate([[0], codes.astype(np.int64)])).astype(np.uint64)
        gb = leb(gaps)
        z = len(Z19.compress(gb)); x = len(lzma.compress(gb, preset=9 | lzma.PRESET_EXTREME))
        ri = rice_bytes(codes.astype(np.int64))
        txt = sum(len(s) + 1 for s in w)
        tb = (len(w) * 2 * k + 7) // 8
        bits = math.ceil(math.log2(len(w)))
        P("%-22s %9d %3d %11d %11d %11d %11d %11d %9d" % (name, len(w), k, txt, tb, z, x, ri, bits))
        rows.append((name, len(w), k, txt, tb, z, x, ri, bits))
    with open(os.path.join(OUT, "w4_vocab_barcodes_whitelists.tsv"), "w") as o:
        o.write("list\tn\tlen\ttext_bytes\ttwobit_bytes\tgaps_leb_zstd19\tgaps_leb_xz9e\tgaps_rice\tidx_bits\n")
        for r in rows: o.write("\t".join(map(str, r)) + "\n")

    # ---- per sample cost of n barcodes (random subset of the whitelist, arbitrary column order)
    P("\n# per-sample cost of n barcodes, bytes (random n-subset of the list; 'file order' = arbitrary column order as today)")
    hdr = ["list", "n", "2bit_packed", "wl_index(file order)", "sorted_gaps_rice+order_perm", "sorted_gaps_rice(set only)",
           "wl_rank_gaps_rice(set only)", "bound_set_no_wl", "bound_set_wl", "saving_idx_vs_sortedperm", "saving_idx_vs_2bit", "saving_set_wl_vs_nowl",
           "pack_zstd19", "breakeven_samples_idx_vs_sortedperm", "breakeven_samples_idx_vs_2bit", "breakeven_samples_set"]
    res = []
    for name in ("737K-august-2016", "3M-february-2018", "gex_737K-arc-v1"):
        w, codes, k = WLS[name]
        W = len(w)
        pack = [r for r in rows if r[0] == name][0][5]
        bits = math.ceil(math.log2(W))
        U = 4 ** k
        for n in (500, 1000, 10000, 100000):
            if n > W: continue
            reps = []
            for _ in range(5):
                sel = rng.choice(W, n, replace=False)
                kmers = np.sort(codes[sel].astype(np.int64))
                ranks = np.sort(sel)
                reps.append((rice_bytes(kmers), rice_bytes(ranks)))
            rk = float(np.mean([r[0] for r in reps])); rr = float(np.mean([r[1] for r in reps]))
            twobit = n * 2 * k / 8
            idx = n * bits / 8
            perm = n * math.ceil(math.log2(n)) / 8
            b_nowl = log2_binom(U, n) / 8
            b_wl = log2_binom(W, n) / 8
            s_perm = rk + perm - idx          # saving of whitelist index over (sorted gaps + order permutation)
            s_2b = twobit - idx
            s_set = rk - rr                   # canonical order: set coded against the whitelist vs without
            be = lambda s: (pack / s) if s > 0 else float("inf")
            res.append((name, n, twobit, idx, rk + perm, rk, rr, b_nowl, b_wl, s_perm, s_2b, s_set, pack, be(s_perm), be(s_2b), be(s_set)))
    with open(os.path.join(OUT, "w4_vocab_barcodes_persample.tsv"), "w") as o:
        o.write("\t".join(hdr) + "\n")
        for r in res: o.write("\t".join(("%.1f" % v) if isinstance(v, float) else str(v) for v in r) + "\n")
    P("\t".join(hdr))
    for r in res: P("\t".join(("%.1f" % v) if isinstance(v, float) else str(v) for v in r))

    # ---- non-10x component structure
    P("\n# combinatorial protocols: component lists shipped (n, length) and bits per barcode as component indices")
    comb = [("splitseq (rd1,rd2,rd3)", ["splitseq_bc_rd1", "splitseq_bc_rd2", "splitseq_bc_rd3"]),
            ("BD Rhapsody (cls1-3)", ["bd_cls1", "bd_cls2", "bd_cls3"]),
            ("inDrop (bc1,bc2)", ["indrop_bc1", "indrop_bc2"]),
            ("Microwell-seq (3 x bc1)", ["microwell_bc1_bare"] * 3),
            ("Quartz-Seq2", ["quartzseq2_bc"]),
            ("sci-RNA-seq3 RT", ["scirna3_rt_bc"])]
    for lab, parts in comb:
        ns = [len(WLS[p][0]) for p in parts]
        ls = [WLS[p][2] for p in parts]
        bits = sum(math.log2(n) for n in ns)
        fix = sum(math.ceil(math.log2(n)) for n in ns)
        sz = sum(len(Z19.compress("\n".join(WLS[p][0]).encode())) for p in dict.fromkeys(parts))
        P("%-26s components=%s len=%s  entropy %.1f bits (fixed %d bits = %.2f B) vs 2-bit packed %.2f B; component lists zstd-19: %d B" %
          (lab, ns, ls, bits, fix, fix / 8, sum(ls) * 2 / 8, sz))
    open(os.path.join(OUT, "w4_vocab_barcodes.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
