"""Orientation-free strand probe of a coordinate-sorted, indexed BAM (the engine's probe in Python).

    python -I strand_probe_bam.py <genes.gtf> <out.tsv> <name=bam> [<name=bam> ...]

For each BAM: the 4 nuclear chromosomes with the most mapped reads, 16 windows along each,
up to 320 strand-informative reads per window (MAPQ >= 20, primary). A read is "sense" if its
aligned blocks touch exons of '+' genes only when forward / '-' genes only when reversed (the
10x 3' orientation), "antisense" if the opposite; reads touching both strands are skipped.

Also writes, per BAM, how concentrated the reads are on genes in each orientation: the share of
informative reads that fall on the 50 genes with the most reads. A library read in the wrong
orientation looks like a normal transcriptome on the antisense side (a few genes carry many
reads); genuine antisense noise is diffuse.
"""
import bisect
import collections
import re
import subprocess
import sys

gtf, out = sys.argv[1], sys.argv[2]
bams = [a.split("=", 1) for a in sys.argv[3:]]
MITO = {"chrM", "MT", "chrMT", "M"}

# exons per chrom: sorted starts, ends, strand, gene
ex = collections.defaultdict(list)
with open(gtf) as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.split("\t", 9)
        if len(p) < 9 or p[2] != "exon" or p[6] not in "+-":
            continue
        m = re.search(r'gene_id "([^"]+)"', p[8])
        ex[p[0]].append((int(p[3]) - 1, int(p[4]), p[6], m.group(1) if m else "?"))
idx = {}
for c, v in ex.items():
    v.sort()
    starts = [e[0] for e in v]
    # running max of ends for the left scan bound
    maxlen = max(e[1] - e[0] for e in v)
    idx[c] = (starts, v, maxlen)


def overlaps(c, s, e):
    if c not in idx:
        return []
    starts, v, maxlen = idx[c]
    i = bisect.bisect_left(starts, s - maxlen)
    j = bisect.bisect_right(starts, e)
    return [x for x in v[i:j] if x[0] < e and x[1] > s]


CIG = re.compile(r"(\d+)([MIDNSHP=X])")


def probe(bam):
    st = subprocess.run(["samtools", "idxstats", bam], capture_output=True, text=True, check=True).stdout
    chroms = []
    for ln in st.splitlines():
        c, ln_, mapped, _ = ln.split("\t")
        if c in MITO or c == "*" or c not in idx or int(mapped) == 0:
            continue
        chroms.append((int(mapped), c, int(ln_)))
    chroms = sorted(chroms, reverse=True)[:4]
    sense = anti = 0
    genes = {"sense": collections.Counter(), "anti": collections.Counter()}
    for _, c, L in chroms:
        for w in range(16):
            beg, end = L * w // 16, L * (w + 1) // 16
            proc = subprocess.Popen(["samtools", "view", "-q", "20", "-F", "0x904", bam, f"{c}:{beg + 1}-{end}"],
                                    stdout=subprocess.PIPE, text=True)
            got = scanned = 0
            for ln in proc.stdout:
                p = ln.split("\t", 6)
                pos = int(p[3]) - 1
                if pos < beg:
                    continue
                scanned += 1
                rev = int(p[1]) & 16 != 0
                r, s_ok, a_ok, gs = pos, False, False, set()
                for n, op in CIG.findall(p[5]):
                    n = int(n)
                    if op in "M=X":
                        for x in overlaps(c, r, r + n):
                            if rev == (x[2] == "-"):
                                s_ok = True
                            else:
                                a_ok = True
                            gs.add(x[3])
                        r += n
                    elif op in "DN":
                        r += n
                if s_ok != a_ok:
                    k = "sense" if s_ok else "anti"
                    if s_ok:
                        sense += 1
                    else:
                        anti += 1
                    for g in gs:
                        genes[k][g] += 1
                    got += 1
                if got >= 320 or scanned >= 16000:
                    break
            proc.kill()
            proc.wait()
    n = sense + anti

    def top50(cn):
        t = sum(cn.values())
        return sum(v for _, v in cn.most_common(50)) / t if t else float("nan")

    return {"informative": n, "sense": sense, "antisense": anti,
            "antisense_frac": anti / n if n else float("nan"),
            "top50_share_sense": top50(genes["sense"]), "top50_share_anti": top50(genes["anti"]),
            "chroms": ",".join(c for _, c, _ in chroms)}


cols = ["name", "informative", "sense", "antisense", "antisense_frac", "top50_share_sense",
        "top50_share_anti", "chroms"]
with open(out, "w") as f:
    f.write("\t".join(cols) + "\n")
    for name, bam in bams:
        try:
            r = probe(bam)
        except Exception as e:  # noqa: BLE001
            print(name, "failed:", e, file=sys.stderr)
            continue
        f.write("\t".join(str(r.get(k, name)) if k != "name" else name for k in cols) + "\n")
        f.flush()
        print(name, r, flush=True)
