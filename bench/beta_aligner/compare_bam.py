#!/usr/bin/env python3
"""Per-read identity of two STAR BAMs, joined on the read sequence (as sequenced, i.e. reverse-complemented back when
the alignment is on the minus strand).  The pipeline renames reads (barcode + offset), so names cannot be the key.

usage: compare_bam.py bench.bam pipeline.bam [samtools]
For every distinct read sequence present in both BAMs, compares the SET of alignments (strand, chrom, pos, CIGAR, NH, AS).
Identical sequences align identically in a deterministic aligner, so one set per sequence is exact.  Sequences present
in only one BAM (unmapped in the other, or dropped by the encoder) are counted separately.  Prints a JSON summary.
"""
import json, subprocess, sys

RC = str.maketrans("ACGTN", "TGCAN")

def load(bam, st):
    d = {}
    p = subprocess.Popen([st, "view", "-F", "0x900", bam], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for ln in p.stdout:
        f = ln.split("\t")
        flag = int(f[1])
        seq = f[9]
        if flag & 0x10: seq = seq.translate(RC)[::-1]
        tags = {t[:2]: t[5:] for t in f[11:]}
        d.setdefault(seq, set()).add((flag & 0x10, f[2], int(f[3]), f[5], tags.get("NH", ""), tags.get("AS", "")))
    p.wait()
    return d

st = sys.argv[3] if len(sys.argv) > 3 else "samtools"
a, b = load(sys.argv[1], st), load(sys.argv[2], st)
both = set(a) & set(b)
same = [k for k in both if a[k] == b[k]]
diff = [k for k in both if a[k] != b[k]]
print(json.dumps({"distinct_seqs_bench": len(a), "distinct_seqs_pipeline": len(b), "in_both": len(both),
                  "only_bench": len(set(a) - set(b)), "only_pipeline": len(set(b) - set(a)),
                  "identical_alignment_sets": len(same), "different_sets": len(diff),
                  "frac_identical_of_both": len(same) / len(both) if both else None,
                  "examples": [{"seq": k, "bench": sorted(a[k])[:3], "pipeline": sorted(b[k])[:3]} for k in diff[:5]]}, indent=1))
