"""Where do the reads of mouse mt-Co2/mt-Atp6/mt-Co3/mt-Nd3 align, and how does Singlet see them?

    python mt_probe.py BAM GTF SAMTOOLS
"""
import collections
import re
import subprocess
import sys

BAM, GTF, SAMT = sys.argv[1:4]
want = {"mt-Co1", "mt-Co2", "mt-Atp8", "mt-Atp6", "mt-Co3", "mt-Nd3", "mt-Nd4", "mt-Cytb"}
genes = {}
spans = []  # all genes (chrom, start, end, name, strand) for annotating other loci
for ln in open(GTF):
    if ln.startswith("#"):
        continue
    p = ln.split("\t")
    if p[2] != "gene":
        continue
    nm = re.search(r'gene_name "([^"]+)"', p[8])
    nm = nm.group(1) if nm else re.search(r'gene_id "([^"]+)"', p[8]).group(1)
    spans.append((p[0], int(p[3]), int(p[4]), nm, p[6]))
    if nm in want:
        genes[nm] = (p[0], int(p[3]), int(p[4]), p[6])
print("genes", genes)
by_chrom = collections.defaultdict(list)
for s in spans:
    by_chrom[s[0]].append(s)


def genes_at(chrom, pos):
    return [s[3] for s in by_chrom.get(chrom, []) if s[1] <= pos <= s[2]]


qsets = {}
for nm in ["mt-Co2", "mt-Atp6", "mt-Co3", "mt-Nd3", "mt-Nd4", "mt-Cytb"]:
    c, s, e, strand = genes[nm]
    out = subprocess.run([SAMT, "view", BAM, f"{c}:{s}-{e}"], capture_output=True, text=True).stdout.splitlines()
    nh = collections.Counter()
    prim = sec = cb_sec = 0
    qn = set()
    mapq = collections.Counter()
    for l in out:
        f = l.split("\t")
        flag = int(f[1])
        tags = dict((t[:2], t[5:]) for t in f[11:])
        nh[int(tags.get("NH", 1))] += 1
        mapq[f[4]] += 1
        if flag & 256:
            sec += 1
            cb_sec += "CB" in tags
        else:
            prim += 1
        if int(tags.get("NH", 1)) > 1:
            qn.add(f[0])
    print(f"== {nm} {c}:{s}-{e}{strand}: alignments {len(out)} primary {prim} secondary {sec} (with CB {cb_sec}); NH {dict(nh.most_common(5))}; MAPQ {dict(mapq.most_common(4))}")
    qsets[nm] = set(list(qn)[:3000])

allq = {q: nm for nm, qs in qsets.items() for q in qs}
stats = {nm: [collections.Counter(), collections.Counter(), collections.Counter()] for nm in qsets}
p = subprocess.Popen([SAMT, "view", BAM], stdout=subprocess.PIPE, text=True)
for l in p.stdout:
    q = l[: l.index(chr(9))]
    nm = allq.get(q)
    if nm is None:
        continue
    f = l.rstrip(chr(10)).split(chr(9))
    flag = int(f[1])
    hascb = any(t.startswith("CB:") for t in f[11:])
    kind = "secondary" if flag & 256 else "primary"
    other, where, cbs = stats[nm]
    if f[2] not in ("chrM", "MT"):
        gs = genes_at(f[2], int(f[3]))
        other[(f[2], int(f[3]) // 100000 * 100000, ",".join(gs[:3]) or "intergenic")] += 1
        where[kind + "@other"] += 1
    else:
        where[kind + "@chrM"] += 1
    cbs[(kind, hascb)] += 1
p.wait()
for nm, (other, where, cbs) in stats.items():
    print("==", nm, "multimapped reads", len(qsets[nm]), "alignments", dict(where), "CB tag (kind, has):", dict(cbs))
    print("   other loci:", other.most_common(6))
