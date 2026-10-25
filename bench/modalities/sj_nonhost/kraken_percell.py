"""Per-cell and per-species summaries of Kraken2 output on STAR-unmapped reads.
usage: python kraken_percell.py KRAKEN_DIR BARCODE_MATE_FASTQ CELLS_TSV KTAXONOMY_TSV [CB_LEN UMI_LEN]
Reads KRAKEN_DIR/out_conf*.txt; writes KRAKEN_DIR/percell_conf*.tsv (cell, species, reads, umis) and species_conf*.tsv.
The barcode mate is the STAR *_Unmapped.out.mate2 (read 1 of a 10x 3' library, CB then UMI), in the same order as mate1.
"""
import sys, glob, os, collections

kd, m2, cells_f, ktax = sys.argv[1:5]
cb_len = int(sys.argv[5]) if len(sys.argv) > 5 else 16
umi_len = int(sys.argv[6]) if len(sys.argv) > 6 else 12

parent, rank, name = {}, {}, {}
for l in open(ktax):
    f = [x.strip() for x in l.rstrip('\n').split('|')]
    t = int(f[0]); parent[t] = int(f[1]); rank[t] = f[2]; name[t] = f[4]


def species(t):
    x = t
    while x in parent and x != 1:
        if rank[x] == 'S':
            return x
        x = parent[x]
    return t


cells = [l.split('\t')[0].strip() for l in open(cells_f) if l.strip()]
cmap = {}
for c in cells:
    cmap.setdefault(c, c)
for c in cells:
    for i in range(len(c)):
        for b in 'ACGT':
            if b != c[i]:
                n = c[:i] + b + c[i + 1:]
                if n not in cmap:
                    cmap[n] = c
                elif cmap[n] != c:
                    cmap[n] = None  # ambiguous
for conf in sorted(glob.glob(f'{kd}/out_conf*.txt')):
    tag = os.path.basename(conf)[4:-4]
    r2sp = {}
    n_total = n_class = 0
    with open(conf) as f:
        for l in f:
            n_total += 1
            if l[0] != 'C':
                continue
            p = l.split('\t')
            r2sp[p[1]] = species(int(p[2]))
            n_class += 1
    sp_reads = collections.Counter(r2sp.values())
    with open(f'{kd}/species_{tag}.tsv', 'w') as o:
        o.write('taxid\tname\treads\n')
        for t, n in sp_reads.most_common():
            o.write(f'{t}\t{name.get(t, "?")}\t{n}\n')
    reads = collections.Counter()
    umis = collections.defaultdict(set)
    nbar = 0
    with open(m2) as f:
        while True:
            h = f.readline()
            if not h:
                break
            s = f.readline().strip(); f.readline(); f.readline()
            rid = h[1:].split()[0]
            sp = r2sp.get(rid)
            if sp is None:
                continue
            cb = cmap.get(s[:cb_len])
            if cb is None:
                continue
            nbar += 1
            reads[(cb, sp)] += 1
            umis[(cb, sp)].add(s[cb_len:cb_len + umi_len])
    with open(f'{kd}/percell_{tag}.tsv', 'w') as o:
        o.write('barcode\ttaxid\tname\treads\tumis\n')
        for (cb, sp), n in reads.items():
            o.write(f'{cb}\t{sp}\t{name.get(sp, "?")}\t{n}\t{len(umis[(cb, sp)])}\n')
    print(f'{tag}: reads={n_total} classified={n_class} ({100 * n_class / max(1, n_total):.3f}%) in_called_cells={nbar} '
          f'top={[(name.get(t, "?"), n) for t, n in sp_reads.most_common(5)]}')
