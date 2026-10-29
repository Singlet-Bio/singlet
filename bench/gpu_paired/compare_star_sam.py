"""Compare the kernel's alignments (paired_microbench --out kernel.tsv) with a STAR run on the same pairs. "Identical" means the
whole SAM-visible alignment of both mates, not just where they start:

  per alignment and per mate: FLAG (strand, mate-strand, first/second, proper pair 0x2, mate-unmapped 0x8), chromosome, POS,
  the full CIGAR (M, I, D, N and the S soft clips, which include the bases removed by --clip*), and PNEXT (the mate's position);
  per read: the number of alignments (NH), the set of alignments (the tie set of a multimapper) and which one is primary.

The kernel's transcripts are turned into SAM fields by star_sam.py, a transcription of ReadAlign_outputTranscriptSAM.cpp.

STAR reference run (same genome directory, the same first N pairs, the same parameters as the kernel's --mode and --clip* options):
  head -n $((4*N)) R1.fastq > r1.fq ; head -n $((4*N)) R2.fastq > r2.fq        (plain FASTQ, as the harness reads it)
  STAR --genomeDir G --readFilesIn r1.fq r2.fq --outSAMtype SAM --outSAMunmapped Within --outSAMattributes NH HI AS nM \\
       --outSAMmultNmax -1 --outFileNamePrefix ref_  [--alignIntronMax 1 --alignMatesGapMax 2000]  [--clip3pAdapterSeq A B ...]
usage: python compare_star_sam.py kernel.tsv ref_Aligned.out.sam GENOME_DIR [--show 20] [--protrude N --concordant]
  (--protrude and --concordant repeat --alignEndsProtrude N ConcordantPair, which changes FLAG 0x2)

Reported, for the pairs the kernel finished (status 0; overflows and mates clipped to nothing go to the host lane and are counted
separately, they are STAR's own result by construction):
  - the criterion: identical alignments among pairs STAR reports as unique (NH:i:1). A kernel-unmapped or multimapping result for a
    STAR-unique pair counts against it;
  - the same over all STAR-mapped pairs, with the differences split: NH differs, tie set differs, only the primary differs
    (a tie-break, which follows window order, i.e. suffix-array order), a field differs (shown).
"""
import re
import sys
from collections import Counter

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from star_sam import sam_records  # noqa: E402


def parse_sam(path):
    """-> list of reads (consecutive records of one read name): None if unmapped, else (NH, {HI: [record, ...]})"""
    groups, cur, name = [], [], None
    with open(path) as f:
        for line in f:
            if line.startswith("@"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[0] != name:
                if cur:
                    groups.append(cur)
                cur, name = [], c[0]
            cur.append(c)
    if cur:
        groups.append(cur)
    out = []
    for recs in groups:
        mapped = [c for c in recs if not int(c[1]) & 4]
        if not mapped:
            out.append(None)
            continue
        by = {}
        nh = 1
        for c in mapped:
            attrs = {t[:2]: t[5:] for t in c[11:] if len(t) > 5 and t[2] == ":"}
            nh = int(attrs.get("NH", 1))
            hi = int(attrs.get("HI", 1))
            by.setdefault(hi, []).append(dict(flag=int(c[1]), chr=c[2], pos=int(c[3]), cigar=c[5], pnext=int(c[7])))
        out.append((nh, by))
    return out


def rec_key(r, chr_name):
    mate = 0 if r["flag"] & 0x40 else 1
    return (mate, r["flag"] & ~0x100, chr_name, r["pos"], r["cigar"], r.get("pnext", 0) if r["flag"] & 0x1 and not r["flag"] & 0x8 else 0)


def star_alignments(entry):
    """-> list of (primary, key) with key = sorted tuple of the mates' record keys"""
    nh, by = entry
    out = []
    for hi, recs in sorted(by.items()):
        primary = all(not r["flag"] & 0x100 for r in recs)
        out.append((primary, tuple(sorted(rec_key(r, r["chr"]) for r in recs))))
    return nh, out


def parse_transcript(s):
    f = s.split(",")
    chr_, strand, score, nMM, nMatch, gLen, iFrag, nEx = (int(v) for v in f[:8])
    exons = [tuple(int(v) for v in e.split(":")) for e in f[8].split("+")]
    canon = [int(v) for v in f[9].split("/")] if len(f) > 9 and f[9] != "" else []
    return dict(chr=chr_, Str=strand, score=score, exons=exons, canon=canon)


def kernel_alignments(c, names, chrStart, protrude, concordant):
    lo = (int(c[7]), int(c[8]))
    c5 = (int(c[9]), int(c[11]))
    c3 = (int(c[10]), int(c[12]))
    rl = (lo[0] - c5[0] - c3[0], lo[1] - c5[1] - c3[1])
    iBest = int(c[5])
    out = []
    for k, ts in enumerate(c[6].split("|")):
        t = parse_transcript(ts)
        recs = sam_records(t["exons"], t["canon"], t["Str"], chrStart[t["chr"]], rl, lo, c5, c3, rl[0] + rl[1] + 1,
                           primary=(k == iBest), protrude=protrude, concordant_pair=concordant)
        keys = tuple(sorted((r["mate"], r["flag"] & ~0x100, names[t["chr"]], r["pos"], r["cigar"], r.get("pnext", 0)) for r in recs))
        out.append((k == iBest, keys))
    return len(out), out


def main(kernel_tsv, sam, gdir, show=20, protrude=0, concordant=False):
    names = [l.split()[0] for l in open(f"{gdir}/chrName.txt")]
    chrStart = [int(x) for x in open(f"{gdir}/chrStart.txt").read().split()]
    ref = parse_sam(sam)
    cnt = Counter()
    diffs = []
    with open(kernel_tsv) as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            idx, status, unmap = int(c[0]), int(c[1]), int(c[2])
            r = ref[idx] if idx < len(ref) else None
            if status != 0:
                cnt["host lane (kernel status %d)" % status] += 1
                continue
            k_mapped = unmap == -1
            if r is None:
                cnt["STAR unmapped, kernel " + ("mapped" if k_mapped else "unmapped")] += 1
                if k_mapped:
                    diffs.append((idx, "STAR unmapped, kernel mapped", None, None))
                continue
            nh, s_al = star_alignments(r)
            unique = nh == 1
            cnt["STAR mapped"] += 1
            cnt["STAR unique"] += unique
            if not k_mapped:
                cnt["differs: kernel unmapped, STAR mapped"] += 1
                cnt["  of which STAR unique"] += unique
                diffs.append((idx, "kernel unmapped, STAR mapped (NH=%d)" % nh, None, s_al))
                continue
            knh, k_al = kernel_alignments(c, names, chrStart, protrude, concordant)
            same_set = sorted(a[1] for a in k_al) == sorted(a[1] for a in s_al)
            k_prim = [a[1] for a in k_al if a[0]]
            s_prim = [a[1] for a in s_al if a[0]]
            if knh != nh:
                why = "differs: NH (kernel %d, STAR %d)" % (knh, nh)
            elif not same_set:
                why = "differs: alignment set (fields)" if nh == 1 else "differs: tie set"
            elif k_prim != s_prim:
                why = "differs: primary only (tie-break)"
            else:
                why = "identical"
            cnt[why if why == "identical" else why.split(" (")[0]] += 1
            if why != "identical":
                diffs.append((idx, why, k_al, s_al))
            if unique:
                cnt["STAR unique, identical"] += why == "identical"
    for k in sorted(cnt):
        print(f"{cnt[k]:>10}  {k}")
    nu = cnt["STAR unique"]
    if nu:
        u_ident = cnt["STAR unique, identical"]
        print(f"\nidentical alignments on STAR-unique pairs: {u_ident}/{nu} = {100.0 * u_ident / nu:.4f}%  (criterion >= 99.9%)")
    nm = cnt["STAR mapped"]
    if nm:
        print(f"identical alignments on all STAR-mapped pairs: {cnt['identical']}/{nm} = {100.0 * cnt['identical'] / nm:.4f}%")
    for idx, why, k, s in diffs[:show]:
        print(f"\n#{idx} {why}\n  kernel: {k}\n  STAR:   {s}")


if __name__ == "__main__":
    a = sys.argv[1:]
    show = int(a[a.index("--show") + 1]) if "--show" in a else 20
    prot = int(a[a.index("--protrude") + 1]) if "--protrude" in a else 0
    main(a[0], a[1], a[2], show, prot, "--concordant" in a)
