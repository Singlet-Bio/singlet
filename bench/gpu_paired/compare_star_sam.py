"""Compare the kernel's primary alignments (paired_microbench --out kernel.tsv) with a STAR run on the same pairs: the phase 1d
kill criterion, >= 99.9% identical primary alignments on unique reads.

STAR reference run (same genome directory, same first N pairs, same parameters as the kernel's --mode):
  head -n $((4*N)) R1.fastq > r1.fq ; head -n $((4*N)) R2.fastq > r2.fq        (plain FASTQ, as the harness reads it)
  STAR --genomeDir G --readFilesIn r1.fq r2.fq --outSAMtype SAM --outSAMunmapped Within --outSAMattributes NH HI AS nM \\
       --outFileNamePrefix ref_ [--alignIntronMax 1 --alignMatesGapMax 2000]      (the last two for --mode atac)
usage: python compare_star_sam.py kernel.tsv ref_Aligned.out.sam GENOME_DIR [--show 20]

A pair is "identical" when both mates have the same chromosome, strand and reference blocks (start, end, split at junctions; a
deletion or insertion does not split a block). The comparison is over pairs STAR reports as unique (NH:i:1 on both records); the
kernel's own unique/multi status and its overflows (status != 0, the host lane) are reported next to it, not hidden in it.
"""
import re
import sys
from collections import Counter

CIG = re.compile(r"(\d+)([MIDNSHP=X])")


def sam_blocks(pos1, cigar):
    blocks, s, e = [], pos1 - 1, pos1 - 1
    for n, op in CIG.findall(cigar):
        n = int(n)
        if op in "M=XD":
            e += n
        elif op == "N":
            blocks.append((s, e))
            s = e = e + n
    blocks.append((s, e))
    return blocks


def kernel_blocks(exons, canon):
    """exons: [(R, G, L, mate)], canon: canonSJ between consecutive exons. Returns {mate: [(start, end), ...]}"""
    out = {}
    for i, (r, g, ln, m) in enumerate(exons):
        prev_sj = canon[i - 1] if i > 0 else None
        if i == 0 or prev_sj == -3:
            out.setdefault(m, []).append([g, g + ln])
        elif prev_sj >= 0:
            out[m].append([g, g + ln])
        else:  # contiguous, deletion (-1) or insertion (-2): one block
            out[m][-1][1] = g + ln
    return {m: [tuple(b) for b in v] for m, v in out.items()}


def main(kernel_tsv, sam, gdir, show=20):
    names = [l.split()[0] for l in open(f"{gdir}/chrName.txt")]
    ref = {}  # read index -> (NH, {mate: (chr, rev, blocks)}) for mapped pairs; None for unmapped
    cur = []
    with open(sam) as f:
        for line in f:
            if line.startswith("@"):
                continue
            c = line.rstrip("\n").split("\t")
            cur.append(c)
    # STAR writes both records of a pair together (and both mates of an unmapped pair); read names are 1..N in input order
    order, i = [], 0
    while i < len(cur):
        name = cur[i][0]
        j = i
        while j < len(cur) and cur[j][0] == name:
            j += 1
        order.append(cur[i:j])
        i = j
    for idx, recs in enumerate(order):
        flag0 = int(recs[0][1])
        if flag0 & 4 and (len(recs) == 1 or int(recs[1][1]) & 4):
            ref[idx] = None
            continue
        nh = int(next((t[5:] for t in recs[0][11:] if t.startswith("NH:i:")), "1"))
        mates = {}
        for c in recs:
            fl = int(c[1])
            if fl & 4:
                continue
            mate = 0 if fl & 0x40 else 1
            mates[mate] = (c[2], bool(fl & 0x10), sam_blocks(int(c[3]), c[5]))
        ref[idx] = (nh, mates)
    cnt = Counter()
    diffs = []
    with open(kernel_tsv) as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            idx, status = int(c[0]), int(c[1])
            r = ref.get(idx)
            if status != 0:
                cnt["kernel overflow / host lane"] += 1
                continue
            unmap = int(c[2])
            nTr = int(c[4])
            if r is None:
                cnt["STAR unmapped; kernel " + ("unmapped" if unmap != -1 else "mapped")] += 1
                continue
            nh, mates = r
            if nh != 1:
                cnt["STAR multi (NH>1), not compared; kernel " + ("unmapped" if unmap != -1 else f"nTr={nTr}")] += 1
                continue
            cnt["STAR unique"] += 1
            if unmap != -1:
                cnt["  STAR unique, kernel unmapped"] += 1
                diffs.append((idx, "kernel unmapped", None, mates))
                continue
            chr_, strand, score, nMM, nMatch, gLen, iFrag, nEx, exs, sjs = c[6:16]
            exons = [tuple(int(v) for v in e.split(":")) for e in exs.split(";")]
            canon = [int(v) for v in sjs.split(",")] if sjs else []
            kb = kernel_blocks(exons, canon)
            rev1 = int(strand) == 1
            kern = {m: (names[int(chr_)], (rev1 if m == 0 else not rev1), kb[m]) for m in kb}
            if kern == mates:
                cnt["  identical"] += 1
            else:
                cnt["  STAR unique, different alignment"] += 1
                diffs.append((idx, "different", kern, mates))
    n_unique = cnt["STAR unique"]
    for k in sorted(cnt):
        print(f"{cnt[k]:>10}  {k}")
    if n_unique:
        ident = cnt["  identical"]
        print(f"\nidentical primary alignments on STAR-unique pairs: {ident}/{n_unique} = {100.0 * ident / n_unique:.4f}%  (criterion >= 99.9%)")
    for idx, why, kern, mates in diffs[:show]:
        print(f"\n#{idx} {why}\n  kernel: {kern}\n  STAR:   {mates}")


if __name__ == "__main__":
    a = sys.argv[1:]
    show = int(a[a.index("--show") + 1]) if "--show" in a else 20
    main(a[0], a[1], a[2], show)
