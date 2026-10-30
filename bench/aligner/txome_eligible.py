"""How many reads could a transcriptome-first step align exactly as STAR does?

    python txome_eligible.py <possorted.bam> <genes.gtf> <out.json> [stride]

Classifies every stride-th primary alignment of a STAR BAM against the GTF's transcripts:

  tx_unspliced   one block (M/S only), inside one exon of at least one transcript
  tx_spliced     several blocks; some transcript has every block in consecutive exons and every N gap equal
                 to that transcript's intron (annotated junctions only)
  multimapper    NH > 1
  indel          CIGAR with I or D (a gapless projection cannot reproduce it)
  chrM           mitochondrial (kept on STAR: heteroplasmy calls)
  other          intronic, intergenic, novel junctions, exon-boundary overhangs

tx_* are split by whether the compatible transcripts belong to one gene and whether the read also overlaps
another gene's span (where STAR's label could depend on the other gene). A projection step that emits the same
genomic alignment would leave junctions, mtDNA and velocity labels unchanged for those reads.
"""

import bisect
import collections
import json
import re
import sys

import pysam

bam_path, gtf_path, out_path = sys.argv[1:4]
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 10

# --- GTF -------------------------------------------------------------------------------------------------
tx_exons = collections.defaultdict(list)          # tx -> [(start0, end0_exclusive)]
tx_info = {}                                      # tx -> (chrom, gene)
gene_span = {}                                    # gene -> [chrom, start, end]
attr_re = re.compile(r'(gene_id|transcript_id) "([^"]+)"')
with open(gtf_path) as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 9:
            continue
        a = dict(attr_re.findall(p[8]))
        s, e = int(p[3]) - 1, int(p[4])
        if p[2] == "gene":
            gene_span[a["gene_id"]] = [p[0], s, e]
        elif p[2] == "exon":
            tx_exons[a["transcript_id"]].append((s, e))
            tx_info[a["transcript_id"]] = (p[0], a["gene_id"])

BIN = 1 << 16
tx_bins = collections.defaultdict(list)           # (chrom, bin) -> [tx]
for tx, ex in tx_exons.items():
    ex.sort()
    c = tx_info[tx][0]
    for b in range(ex[0][0] // BIN, (ex[-1][1] - 1) // BIN + 1):
        tx_bins[(c, b)].append(tx)
gene_bins = collections.defaultdict(list)
for g, (c, s, e) in gene_span.items():
    for b in range(s // BIN, (e - 1) // BIN + 1):
        gene_bins[(c, b)].append(g)
tx_starts = {tx: [s for s, _ in ex] for tx, ex in tx_exons.items()}


def exon_index(tx, s, e):
    """Index of the exon of tx containing [s, e), or -1."""
    ex = tx_exons[tx]
    i = bisect.bisect_right(tx_starts[tx], s) - 1
    return i if i >= 0 and ex[i][1] >= e else -1


def compatible(tx, blocks):
    idx = [exon_index(tx, s, e) for s, e in blocks]
    if min(idx) < 0:
        return False
    ex = tx_exons[tx]
    for k in range(len(blocks) - 1):        # consecutive exons, gap exactly the intron
        i, j = idx[k], idx[k + 1]
        if j != i + 1 or blocks[k][1] != ex[i][1] or blocks[k + 1][0] != ex[j][0]:
            return False
    return True


# --- BAM -------------------------------------------------------------------------------------------------
n = collections.Counter()
seen = 0
with pysam.AlignmentFile(bam_path) as bam:
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        seen += 1
        if seen % stride:
            continue
        n["total"] += 1
        c = r.reference_name
        if r.has_tag("NH") and r.get_tag("NH") > 1:
            n["multimapper"] += 1
            continue
        if c in ("chrM", "MT"):
            n["chrM"] += 1
            continue
        if any(op in (1, 2) for op, _ in r.cigartuples):
            n["indel"] += 1
            continue
        blocks = r.get_blocks()
        merged = []                           # get_blocks splits at nothing but N here (no I/D left)
        for s, e in blocks:
            if merged and s == merged[-1][1]:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        lo, hi = merged[0][0], merged[-1][1]
        cands = {t for b in range(lo // BIN, (hi - 1) // BIN + 1) for t in tx_bins.get((c, b), ())}
        comp = [t for t in cands if compatible(t, merged)]
        kind = "tx_spliced" if len(merged) > 1 else "tx_unspliced"
        if not comp:
            n["other_spliced" if len(merged) > 1 else "other"] += 1
            continue
        genes = {tx_info[t][1] for t in comp}
        over = {g for b in range(lo // BIN, (hi - 1) // BIN + 1) for g in gene_bins.get((c, b), ())
                if gene_span[g][1] < hi and gene_span[g][2] > lo}
        if len(genes) > 1:
            n[kind + "_multigene"] += 1
        elif over - genes:
            n[kind + "_overlap"] += 1
        else:
            n[kind + "_clean"] += 1

tot = n["total"]
res = {"bam": bam_path, "stride": stride, "primary_seen": seen, "counts": dict(n),
       "share": {k: round(v / tot, 4) for k, v in sorted(n.items()) if k != "total"}}
res["share"]["tx_clean_all"] = round((n["tx_unspliced_clean"] + n["tx_spliced_clean"]) / tot, 4)
res["share"]["tx_any_all"] = round(sum(v for k, v in n.items() if k.startswith("tx_")) / tot, 4)
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res, indent=1))
