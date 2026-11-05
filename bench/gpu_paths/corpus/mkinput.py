#!/usr/bin/env python3
"""Cut a small FASTQ pair from data already on Clipper, optionally with seeded SYNTHETIC low qualities.

Usage:
  mkinput.py --r1 F1[,F2..] --r2 G1[,G2..] --n N --out1 R1.fastq --out2 R2.fastq
             [--r1cut L] [--r2cut L] [--synth SHARE --seed S] [--stats stats.json]
  mkinput.py --sra-pair FILE_A FILE_B --wl WHITELIST --n N --out1 ... --out2 ... [--r1cut 28] [--synth ...]
      (fastq-dump --split-files output: the file whose first 16 bases are on the whitelist is the barcode read)

N read pairs are taken from the start of the concatenated inputs (plain or gzip). The R1 read is cut to --r1cut bases.
The cDNA read is cut to --r2cut bases. With --synth SHARE, a seeded pseudorandom share of the non-N cDNA bases whose
quality is the Illumina 4-bin value Q11 (',') is lowered to Q5 ('&'), so low quality is no longer tied to N bases and the
GPU pileup needs the per-base mask. For reads over 128 bases a fixed rule adds a low base (non-N) at 0-based position 128
in the reads with index mod 1000 = 0 and at the last position in the reads with index mod 1000 = 1, so the third mask
word carries bits in a known set of reads. The stats file counts what was changed. The output is SYNTHETIC, not real data.
"""
import argparse
import json
import subprocess
import sys

import numpy as np


def stream(files):
    p = subprocess.Popen(["zcat", "-f"] + files, stdout=subprocess.PIPE, bufsize=1 << 22)
    return p, p.stdout


def take(f, nlines):
    out = []
    for _ in range(nlines):
        l = f.readline()
        if not l:
            break
        out.append(l)
    return out


def whitelist_rate(path, wl):
    ok = n = 0
    with open(path) as f:
        for i, l in enumerate(f):
            if i % 4 == 1:
                n += 1
                ok += l[:16] in wl
                if n >= 5000:
                    break
    return ok / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1"); ap.add_argument("--r2")
    ap.add_argument("--sra-pair", nargs=2); ap.add_argument("--wl")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out1", required=True); ap.add_argument("--out2", required=True)
    ap.add_argument("--r1cut", type=int, default=0); ap.add_argument("--r2cut", type=int, default=0)
    ap.add_argument("--synth", type=float, default=0.0); ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--stats")
    a = ap.parse_args()
    if a.sra_pair:
        wl = set(l.strip() for l in open(a.wl))
        fa, fb = whitelist_rate(a.sra_pair[0], wl), whitelist_rate(a.sra_pair[1], wl)
        print("barcode read test: file A %.3f, file B %.3f" % (fa, fb), flush=True)
        bc, cd = (a.sra_pair[0], a.sra_pair[1]) if fa > fb else (a.sra_pair[1], a.sra_pair[0])
        r1f, r2f = [bc], [cd]
    else:
        r1f, r2f = a.r1.split(","), a.r2.split(",")
    p1, s1 = stream(r1f)
    p2, s2 = stream(r2f)
    rng = np.random.default_rng(a.seed)
    CH = 250000
    st = dict(reads=0, synthetic=a.synth > 0, seed=a.seed, share=a.synth, r1cut=a.r1cut, r2cut=a.r2cut, cdna_len=0,
              bases_q11_nonN=0, lowered_random=0, lowered_forced_extra=0, reads_with_lowered=0,
              reads_with_lowered_ge128=0, lowered_at_pos128=0, reads_lowq_nonN=0, reads_lowq_nonN_ge128=0)
    done = 0
    with open(a.out1, "wb") as w1, open(a.out2, "wb") as w2:
        while done < a.n:
            k = min(CH, a.n - done)
            l1, l2 = take(s1, 4 * k), take(s2, 4 * k)
            n = min(len(l1), len(l2)) // 4
            if n == 0:
                break
            out = []
            for i in range(n):
                s, q = l1[4*i+1].rstrip(b"\n"), l1[4*i+3].rstrip(b"\n")
                if a.r1cut:
                    s, q = s[:a.r1cut], q[:a.r1cut]
                out += [l1[4*i].split(b" ")[0].rstrip(b"\n") + b"\n", s + b"\n", b"+\n", q + b"\n"]
            w1.write(b"".join(out))
            seqs = [l2[4*i+1].rstrip(b"\n") for i in range(n)]
            quals = [l2[4*i+3].rstrip(b"\n") for i in range(n)]
            if a.r2cut:
                seqs = [x[:a.r2cut] for x in seqs]; quals = [x[:a.r2cut] for x in quals]
            L = len(seqs[0])
            if a.synth > 0:
                if any(len(x) != L for x in seqs):
                    sys.exit("synthetic lowering needs reads of one length")
                seq = np.frombuffer(b"".join(seqs), dtype=np.uint8).reshape(n, L)
                q = np.frombuffer(b"".join(quals), dtype=np.uint8).reshape(n, L).copy()
                isN = seq == ord("N")
                q11 = (q == ord(",")) & ~isN
                pick = q11 & (rng.random((n, L)) < a.synth)
                forced = np.zeros_like(pick)
                if L > 128:
                    idx = st["reads"] + np.arange(n)
                    r0, r1 = (idx % 1000 == 0), (idx % 1000 == 1)
                    forced[r0, 128] = ~isN[r0, 128]
                    forced[r1, L - 1] = ~isN[r1, L - 1]
                before = q.copy()
                q[pick | forced] = ord("&")
                ch = q != before
                low = ((q.astype(np.int16) - 33) < 10) & ~isN
                st["bases_q11_nonN"] += int(q11.sum()); st["lowered_random"] += int(pick.sum())
                st["lowered_forced_extra"] += int((forced & ~pick).sum())
                st["reads_with_lowered"] += int(ch.any(1).sum())
                if L > 128:
                    st["reads_with_lowered_ge128"] += int(ch[:, 128:].any(1).sum())
                    st["lowered_at_pos128"] += int(ch[:, 128].sum())
                    st["reads_lowq_nonN_ge128"] += int(low[:, 128:].any(1).sum())
                st["reads_lowq_nonN"] += int(low.any(1).sum())
                quals = [q[i].tobytes() for i in range(n)]
            st["cdna_len"] = L
            out = []
            for i in range(n):
                out += [l2[4*i].split(b" ")[0].rstrip(b"\n") + b"\n", seqs[i] + b"\n", b"+\n", quals[i] + b"\n"]
            w2.write(b"".join(out))
            st["reads"] += n
            done += n
            if n < k:
                break
    for p in (p1, p2):
        p.terminate()
    if a.stats:
        json.dump(st, open(a.stats, "w"), indent=1)
    print(json.dumps(st))


if __name__ == "__main__":
    main()
