#!/usr/bin/env python3
"""Concordance of a transcriptome-first alignment with STAR's genomic record.
Usage: tx_concordance.py TX.tsv TX.bam GENOME.bam SAMPLE OUT.json
TX.bam: STAR on the transcriptome (end to end, at most 1 mismatch, no introns). GENOME.bam: STAR on the genome, unmapped reads within.
Reads claimed by the transcriptome pass: aligned, at most 1 mismatch, all alignments in one gene. Each alignment is projected to the genome
through the exon list of its transcript and compared with the primary genomic record.
"""
import bisect
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict

txtsv, txbam, gbam, sample, outjson = sys.argv[1:6]
TX = {}
exons_by_chr = defaultdict(list)
genespan = defaultdict(dict)
for l in open(txtsv):
    t, chrom, strand, gene, bt, ex = l.rstrip("\n").split("\t")
    e = [tuple(map(int, x.split("-"))) for x in ex.split(",")]
    TX[t] = (chrom, strand, gene, bt, e, sum(b - a + 1 for a, b in e))
    exons_by_chr[chrom].extend(e)
    g = genespan[chrom].get(gene)
    lo, hi = e[0][0], e[-1][1]
    genespan[chrom][gene] = (min(lo, g[0]), max(hi, g[1])) if g else (lo, hi)


def merge(iv):
    iv = sorted(list(x) for x in iv)
    m = []
    for a, b in iv:
        if m and a <= m[-1][1] + 1:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    return [x[0] for x in m], [x[1] for x in m]


EXM = {c: merge(v) for c, v in exons_by_chr.items()}
GSM = {c: merge(list(g.values())) for c, g in genespan.items()}


def overlaps(idx, c, a, b):
    if c not in idx:
        return False
    st, en = idx[c]
    i = bisect.bisect_right(st, b) - 1
    return i >= 0 and en[i] >= a


CIG = re.compile(r"(\d+)([MIDNSHP=X])")


def cigar_span(cig):
    return sum(int(n) for n, o in CIG.findall(cig) if o in "MDN=X")


def project(tx, flag, pos, rl):
    chrom, strand, gene, bt, ex, L = TX[tx]
    p = pos - 1
    lo, hi = (p, p + rl) if strand == "+" else (L - p - rl, L - p)
    merged, off = [], 0
    for a, b in ex:
        n = b - a + 1
        s, e = max(lo, off), min(hi, off + n)
        if s < e:
            g0, g1 = a + s - off, a + e - off - 1
            if merged and g0 == merged[-1][1] + 1:
                merged[-1][1] = g1
            else:
                merged.append([g0, g1])
        off += n
    cig, prev = "", None
    for a, b in merged:
        if prev is not None:
            cig += "%dN" % (a - prev - 1)
        cig += "%dM" % (b - a + 1)
        prev = b
    rev = (strand == "-") != bool(flag & 16)
    return (chrom, merged[0][0], "-" if rev else "+", cig)


def tag(f, name, default=None):
    for x in f[11:]:
        if x.startswith(name + ":"):
            return x[5:]
    return default


# transcriptome pass
txr = defaultdict(list)
p = subprocess.Popen(["samtools", "view", "-@", "4", "-F", "4", txbam], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
for l in p.stdout:
    f = l.rstrip("\n").split("\t")
    nm = int(tag(f, "nM", 0))
    if nm > 1:
        continue
    txr[f[0]].append((f[2], int(f[1]), int(f[3]), len(f[9]), nm, int(tag(f, "AS", 0))))
p.wait()

cls_n, cls_claim, cls_full = Counter(), Counter(), Counter()
comp, why, unclaimed, claim_kind, asdelta, nh_hist = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
total = 0


def lane_class(r):
    if r is None:
        return "unmapped"
    if r["nh"] > 1:
        return "heavy_multi"
    if "S" in r["cigar"]:
        return "heavy_clip"
    if r["cigar"].count("N") > 1:
        return "other_multisplice"
    return "light"


def locus(r):
    return (r["chrom"], r["pos"], "-" if r["flag"] & 16 else "+", r["cigar"])


def handle(name, recs):
    global total
    total += 1
    prim = next((r for r in recs if not r["flag"] & 256 and not r["flag"] & 4), None)
    c = lane_class(prim)
    cls_n[c] += 1
    t = txr.get(name)
    if prim is not None:
        nh_hist[min(prim["nh"], 11)] += 1
    if not t:
        if prim is not None:
            a = prim["pos"]
            b = a + cigar_span(prim["cigar"]) - 1
            k = "exonic_other" if overlaps(EXM, prim["chrom"], a, b) else ("intronic" if overlaps(GSM, prim["chrom"], a, b) else "intergenic")
            unclaimed[c + "|" + k] += 1
        else:
            unclaimed[c + "|none"] += 1
        return
    genes = {TX[x[0]][2] for x in t}
    if len(genes) > 1:
        claim_kind["multi_gene"] += 1
        unclaimed[c + "|tx_multigene"] += 1
        return
    claim_kind["one_gene"] += 1
    cls_claim[c] += 1
    projs = {project(x[0], x[1], x[2], x[3]) for x in t}
    best = min(t, key=lambda x: (-x[5], x[0]))
    bproj = project(best[0], best[1], best[2], best[3])
    bt = TX[best[0]][3]
    if prim is None:
        why["genome_unmapped"] += 1
        return
    gp = locus(prim)
    loc = gp in projs
    locpos = any(q[:3] == gp[:3] for q in projs)
    nh1 = prim["nh"] == 1
    asd = prim["as"] - best[5]
    asdelta[max(-6, min(6, asd))] += 1
    comp["claimed"] += 1
    comp["locus_cigar_equal"] += loc
    comp["chrom_pos_strand_equal"] += locpos
    comp["nh_equal_1"] += nh1
    comp["as_equal"] += (asd == 0)
    comp["nm_equal"] += (prim["nm"] == best[4])
    comp["isoform_projections_differ"] += (len(projs) > 1)
    if loc and nh1 and asd == 0:
        cls_full[c] += 1
        why["match"] += 1
    elif not nh1:
        alts = [locus(r) for r in recs if r is not prim and not r["flag"] & 4]
        why["genome_multimapper" + ("_contains_projection" if loc or any(a in projs for a in alts) else "")] += 1
    elif "S" in prim["cigar"]:
        why["genome_softclip"] += 1
    elif not locpos and (prim["chrom"] == "chrM" or bproj[0] == "chrM"):
        why["numt_locus_differs"] += 1
    elif not locpos and "pseudogene" in bt:
        why["pseudogene_locus_differs"] += 1
    elif not locpos:
        why["other_locus_differs"] += 1
    elif not loc:
        why["same_pos_cigar_differs_" + ("genome_unspliced" if "N" not in prim["cigar"] and "N" in bproj[3] else "other")] += 1
    else:
        why["score_differs_only"] += 1


p = subprocess.Popen(["samtools", "view", "-@", "4", gbam], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
cur, recs = None, []
for l in p.stdout:
    f = l.rstrip("\n").split("\t")
    r = {"flag": int(f[1]), "chrom": f[2], "pos": int(f[3]), "cigar": f[5], "nh": int(tag(f, "NH", 1)),
         "as": int(tag(f, "AS", 0)), "nm": int(tag(f, "nM", 0))}
    if cur is not None and f[0] != cur:
        handle(cur, recs)
        recs = []
    cur = f[0]
    recs.append(r)
if cur is not None:
    handle(cur, recs)
p.wait()

res = {"sample": sample, "reads": total, "lane_class": dict(cls_n), "claimed_by_class": dict(cls_claim), "fullmatch_by_class": dict(cls_full),
       "claim_kind": dict(claim_kind), "claimed_components": dict(comp), "disagreement": dict(why), "unclaimed_fate": dict(unclaimed),
       "as_delta_genome_minus_tx": {str(k): v for k, v in sorted(asdelta.items())}, "genome_nh_hist": dict(nh_hist)}
json.dump(res, open(outjson, "w"), indent=1)
print(json.dumps(res)[:1800])
