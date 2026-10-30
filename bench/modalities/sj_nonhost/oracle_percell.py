"""Per-cell SARS-CoV-2 UMI counts from a minimap2 PAF of STAR-unmapped cDNA reads vs NC_045512.2.
usage: python oracle_percell.py PAF BARCODE_MATE_FASTQ CELLS_TSV OUT_TSV
A read counts if aligned >= 60 nt at >= 90% identity; molecules = distinct (cell, UMI)."""
import sys, collections
paf, m2, cells_f, out = sys.argv[1:5]
hit = set()
for l in open(paf):
    f = l.split('\t')
    al = int(f[10]); mt = int(f[9])
    if al >= 60 and mt / al >= 0.9:
        hit.add(f[0])
cells = [l.split('\t')[0].strip() for l in open(cells_f) if l.strip()]
cmap = {c: c for c in cells}
for c in cells:
    for i in range(len(c)):
        for b in 'ACGT':
            if b != c[i]:
                cmap.setdefault(c[:i] + b + c[i + 1:], c)
reads = collections.Counter(); umis = collections.defaultdict(set)
with open(m2) as f:
    while True:
        h = f.readline()
        if not h:
            break
        s = f.readline().strip(); f.readline(); f.readline()
        if h[1:].split()[0] not in hit:
            continue
        cb = cmap.get(s[:16])
        if cb is None:
            continue
        reads[cb] += 1; umis[cb].add(s[16:28])
with open(out, 'w') as o:
    o.write('barcode\treads\tumis\n')
    for cb, n in reads.items():
        o.write(f'{cb}\t{n}\t{len(umis[cb])}\n')
print(f'reads aligned: {len(hit)}; cells with hits: {len(reads)}; total UMIs: {sum(len(v) for v in umis.values())}')
