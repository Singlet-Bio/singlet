"""Kernel versus the Python model on real pairs: replay the windows the kernel built through paired_model.map_windows (the join,
stitching, mate rules and selection) with the real genome, and compare with what the kernel returned.

The model has no suffix array, so seeding and window assignment are taken from the kernel (its window dump, win.bin). That
leaves the stage this phase changes under test: any difference is in the paired-end rules or their port, not in seeding. The
end-to-end check against STAR is compare_star_sam.py.

usage: python -P replay_model.py GENOME_DIR r1.fq r2.fq win.bin kernel.tsv [--mode spliced|atac] [--n 20000]
(needs numpy and numba; the genome is memory-mapped, so a human genome costs no RAM)
"""
import struct
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from paired_model import (Genome, Params, build_read, map_windows, WC_gStart, WC_gEnd)  # noqa: E402


def load_star_genome(gdir):
    par = {}
    for line in open(f"{gdir}/genomeParameters.txt"):
        t = line.split()
        if len(t) >= 2 and t[0] != "###":
            par[t[0]] = t[1]
    G = Genome.__new__(Genome)
    G.gChrBinNbits = int(par["genomeChrBinNbits"])
    G.names = [l.split()[0] for l in open(f"{gdir}/chrName.txt")]
    G.chrStart = [int(x) for x in open(f"{gdir}/chrStart.txt").read().split()]
    G.chrLength = [int(x) for x in open(f"{gdir}/chrLength.txt").read().split()]
    G.G = np.asarray(np.memmap(f"{gdir}/Genome", dtype=np.int8, mode="r"))
    G.nGenome = G.G.shape[0]
    nb = G.chrStart[len(G.names)] // (1 << G.gChrBinNbits) + 1
    G.chrBin = np.zeros(nb, dtype=np.int64)
    ichr = 1
    for ii in range(nb):
        if ichr < len(G.chrStart) and ii * (1 << G.gChrBinNbits) >= G.chrStart[ichr]:
            ichr += 1
        G.chrBin[ii] = ichr - 1
    return G


def fastq_seqs(path, n):
    out = []
    with open(path) as f:
        while len(out) < n:
            h = f.readline()
            if not h:
                break
            out.append(f.readline().strip().upper())
            f.readline()
            f.readline()
    return out


def read_dump(path):
    """per read: list of (Str, Chr, WA rows, WlastAnchor)"""
    out = []
    with open(path, "rb") as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            n = struct.unpack("<Q", h)[0]
            w = np.frombuffer(f.read(8 * n), dtype="<u8") if n else np.zeros(0, dtype="<u8")
            wins = []
            if n:
                nW, p = int(w[0]), 1
                for _ in range(nW):
                    s, c, nwa, last = int(w[p]), int(w[p + 1]), int(w[p + 2]), int(w[p + 3])
                    p += 4
                    rows = [[int(x) for x in w[p + 7 * k:p + 7 * k + 7]] for k in range(nwa)]
                    p += 7 * nwa
                    wins.append((s, c, rows, last))
            out.append(None if n == 0 else wins)  # None: the kernel did not finish this pair (host lane)
    return out


def main(gdir, r1, r2, dump, ktsv, mode="spliced", n=20000):
    G = load_star_genome(gdir)
    P = Params.atac() if mode == "atac" else Params.spliced()
    G.prepare(P)
    m1s, m2s = fastq_seqs(r1, n), fastq_seqs(r2, n)
    wins = read_dump(dump)
    kern = {}
    with open(ktsv) as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            kern[int(c[0])] = c
    same = diff = skipped = 0
    shown = 0
    for i in range(min(len(m1s), len(m2s), len(wins))):
        k = kern.get(i)
        if k is None or int(k[1]) != 0 or wins[i] is None:
            skipped += 1
            continue
        rd = build_read(m1s[i], m2s[i])
        WC = [[s, c, 0, 0] for (s, c, _r, _l) in wins[i]]
        res = map_windows(G, P, rd, WC, [r for (_s, _c, r, _l) in wins[i]], [l for (_s, _c, _r, l) in wins[i]])
        # the kernel's TSV columns: idx status unmap nW nTr iBest Chr Str score nMM nMatch gLength iFrag nExons exons canonSJ
        km = (int(k[2]), int(k[3]), int(k[4]), int(k[5]))
        mm = (res.unmapType, res.nW, len(res.transcripts), res.primary)
        ok = km == mm
        if ok and res.primary >= 0:
            t = res.transcripts[res.primary]
            exs = ";".join(":".join(str(int(v)) for v in t.exons[e][:4]) for e in range(t.nExons))
            sj = ",".join(str(int(v)) for v in t.canonSJ[:t.nExons - 1])
            ok = (k[6:16] == [str(t.Chr), str(t.Str), str(t.maxScore), str(t.nMM), str(t.nMatch), str(t.gLength), str(t.iFrag),
                              str(t.nExons), exs, sj])
        if ok:
            same += 1
        else:
            diff += 1
            if shown < 10:
                shown += 1
                print(f"#{i} kernel {k[2:]}  model {mm} "
                      f"{res.transcripts[res.primary].summary() if res.primary >= 0 else ''}")
    print(f"replayed {same + diff} pairs: {same} identical, {diff} different; {skipped} skipped (host lane or missing)")
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    mode = a[a.index("--mode") + 1] if "--mode" in a else "spliced"
    n = int(a[a.index("--n") + 1]) if "--n" in a else 20000
    sys.exit(main(a[0], a[1], a[2], a[3], a[4], mode, n))
