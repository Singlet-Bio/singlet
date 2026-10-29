"""Which UMI grouping should R9 use? (ws-w3b update i)

Reads `samtools view` text of STAR's sorted BAM (tags CB, UR, UB, GX, NH) on stdin, for chr1 and the IG/TR loci, and
counts molecules at V(D)J segments and at chr1 panel SNP sites four ways, on the same reads:
  ub        distinct STAR UB per (cell, feature)                  = what prod/campaign counted
  ur        distinct raw UR per (cell, feature)                   = no correction
  feat      R9 directional on UR within (cell, feature)           = perf/bam-stream 253c2c6
  gene      R9 directional on UR within (cell, STAR gene GX), using every read of that gene in the cell;
            molecules at a feature = distinct corrected UMIs there; reads without GX fall back to `feat`
  gene_all  1MM_All within (cell, GX) (connected components, no count rule), STAR's --soloUMIdedup default
For UMI pairs merged by `feat` but kept apart by `gene` (and the reverse) it records the gene-wide read counts of both
UMIs: an error copy carries few reads compared with its parent; two templates carry similar numbers.

Usage: samtools view BAM chr1 <loci> | python -I umi_group_probe.py <genes.gtf> <snp panel vcf.gz> <out.json>
"""
import gzip, json, sys
from collections import defaultdict

gtf, vcf, out = sys.argv[1:4]
VDJ_TYPES = {f"{a}_{b}_gene" for a in ("IG", "TR") for b in "VDJC"}

vdj = []  # (chrom, start0, end0, name), GTF order
with open(gtf) as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.split("\t", 9)
        if p[2] != "gene": continue
        a = p[8]
        gt = a.split('gene_type "', 1)[1].split('"', 1)[0] if 'gene_type "' in a else ""
        if gt in VDJ_TYPES:
            nm = a.split('gene_name "', 1)[1].split('"', 1)[0] if 'gene_name "' in a else a.split('gene_id "', 1)[1].split('"', 1)[0]
            vdj.append((p[0], int(p[3]) - 1, int(p[4]), nm))
vdj_by_chr = defaultdict(list)
for i, (c, s, e, n) in enumerate(vdj): vdj_by_chr[c].append((s, e, i))

snps = []
with gzip.open(vcf, "rt") as f:
    for line in f:
        if line[0] == "#": continue
        if not line.startswith("chr1\t"):
            if snps: break
            continue
        snps.append(int(line.split("\t", 2)[1]) - 1)   # 0-based
snps.sort()
import bisect

def blocks(pos0, cigar):
    out, n = [], ""
    for ch in cigar:
        if ch.isdigit(): n += ch; continue
        L = int(n); n = ""
        if ch in "M=X": out.append((pos0, pos0 + L)); pos0 += L
        elif ch in "DN": pos0 += L
    return out

# gene_reads[(cb, gx)][ur] = reads (every unique read of the gene in that cell: the gene-wide evidence)
gene_reads = defaultdict(lambda: defaultdict(int))
# evidence[kind][(cb, feature)] = list of (ur, ub, gx)
evidence = {"vdj": defaultdict(list), "snp": defaultdict(list)}
n_reads = n_used = 0
tags_seen = defaultdict(int)
for line in sys.stdin:
    p = line.rstrip("\n").split("\t")
    n_reads += 1
    flag = int(p[1])
    if flag & 0x904: continue
    t = {}
    for x in p[11:]:
        t[x[:2]] = x[5:]
    if t.get("NH", "1") != "1": continue
    cb, ur = t.get("CB", "-"), t.get("UR", "")
    if cb == "-" or not ur: continue
    ub, gx = t.get("UB", ""), t.get("GX", "-")
    tags_seen["ub_present" if ub and ub != "-" else "ub_missing"] += 1
    tags_seen["gx_present" if gx != "-" else "gx_missing"] += 1
    if ub and ub != "-" and ub != ur: tags_seen["ub_ne_ur"] += 1
    n_used += 1
    if gx != "-": gene_reads[(cb, gx)][ur] += 1
    bl = blocks(int(p[3]) - 1, p[5])
    if not bl: continue
    chrom = p[2]
    if chrom in vdj_by_chr:
        s0, e0 = bl[0][0], bl[-1][1]
        hit = [i for (s, e, i) in vdj_by_chr[chrom] if s < e0 and e > s0]
        if hit: evidence["vdj"][(cb, min(hit))].append((ur, ub, gx))
    if chrom == "chr1":
        for (s, e) in bl:
            k = bisect.bisect_left(snps, s)
            while k < len(snps) and snps[k] < e:
                evidence["snp"][(cb, snps[k])].append((ur, ub, gx)); k += 1

BASES = "ACGTN"

def roots(counts, rule):
    """counts: {umi: reads}. rule 'dir' = R9 directional, 'all' = 1MM_All. Returns {umi: root}."""
    order = sorted(counts, key=lambda u: (-counts[u], u))
    rank = {u: r for r, u in enumerate(order)}
    par = {u: u for u in order}
    def find(u):
        while par[u] != u:
            par[u] = par[par[u]]; u = par[u]
        return u
    for j in order:
        cj = counts[j]
        for pos in range(len(j)):
            for b in BASES:
                if b == j[pos]: continue
                i = j[:pos] + b + j[pos + 1:]
                if i not in counts: continue
                if rule == "dir" and not (rank[i] < rank[j] and counts[i] >= 2 * cj - 1): continue
                ri, rj = find(i), find(j)
                if ri != rj:
                    if rank[ri] < rank[rj]: par[rj] = ri
                    else: par[ri] = rj
    return {u: find(u) for u in order}

gene_root_cache = {}
def gene_roots(key, rule):
    k = (key, rule)
    if k not in gene_root_cache: gene_root_cache[k] = roots(gene_reads[key], rule)
    return gene_root_cache[k]

res = {"n_reads_in": n_reads, "n_reads_used": n_used, "tags": dict(tags_seen), "n_vdj_segments": len(vdj), "n_snp_sites_chr1": len(snps)}
for kind, ev in evidence.items():
    tot = defaultdict(int)
    pairs = {"feat_merge_gene_apart": [], "gene_merge_feat_apart": []}
    n_ev_reads = n_ev_gx = 0
    for (cb, feat), rows in ev.items():
        fc = defaultdict(int)
        for ur, ub, gx in rows: fc[ur] += 1
        n_ev_reads += len(rows)
        tot["ur"] += len(fc)
        tot["ub"] += len({ub if ub and ub != "-" else "raw:" + ur for ur, ub, gx in rows})
        fr = roots(fc, "dir")
        tot["feat"] += len(set(fr.values()))
        gx_of = {}
        for ur, ub, gx in rows:
            if gx != "-": gx_of.setdefault(ur, gx)
        n_ev_gx += sum(1 for ur, ub, gx in rows if gx != "-")
        for rule, name in (("dir", "gene"), ("all", "gene_all")):
            mol = set()
            for ur in fc:
                gx = gx_of.get(ur)
                mol.add(("g", gx, gene_roots((cb, gx), rule)[ur]) if gx else ("f", fr[ur]))
            tot[name] += len(mol)
        # disagreement evidence (directional only), UMIs that have a gene
        g = {ur: gene_roots((cb, gx_of[ur]), "dir")[ur] for ur in fc if ur in gx_of}
        for u in g:
            for v in g:
                if u >= v or gx_of[u] != gx_of[v]: continue
                same_f, same_g = fr[u] == fr[v], g[u] == g[v]
                if same_f == same_g: continue
                gw = gene_reads[(cb, gx_of[u])]
                hi, lo = (u, v) if (gw[u], v) > (gw[v], u) else (v, u)
                rec = [gw[hi], gw[lo], fc[hi], fc[lo]]
                (pairs["feat_merge_gene_apart"] if same_f else pairs["gene_merge_feat_apart"]).append(rec)
    summ = {"groups": len(ev), "evidence_reads": n_ev_reads, "evidence_reads_with_gx": n_ev_gx, "molecules": dict(tot)}
    for nm, lst in pairs.items():
        lst.sort()
        n = len(lst)
        if n:
            ratios = sorted(r[1] / r[0] for r in lst)
            summ[nm] = {
                "n": n,
                "gene_wide_ratio_low_over_high_quantiles": [round(ratios[int(q * (n - 1))], 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
                "share_ratio_ge_0.5": round(sum(r >= 0.5 for r in ratios) / n, 4),
                "share_low_gene_reads_ge_3": round(sum(r[1] >= 3 for r in lst) / n, 4),
                "examples_gene_hi_lo_feat_hi_lo": lst[:: max(1, n // 12)][:12],
            }
        else:
            summ[nm] = {"n": 0}
    res[kind] = summ
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
