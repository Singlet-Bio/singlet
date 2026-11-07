"""Per-species and per-cell summaries of Kraken2 output on STAR-unmapped reads.

    python kraken_percell.py KRAKEN_OUT KRAKEN_REPORT BARCODE_MATE_FASTQ CELLS_TXT OUT_DIR [CB_LEN UMI_LEN]

KRAKEN_OUT / KRAKEN_REPORT   kraken2 --output / --report files for the cDNA mate (STAR *_Unmapped.out.mate1)
BARCODE_MATE_FASTQ           the barcode mate (STAR *_Unmapped.out.mate2: cell barcode then UMI), same read order
CELLS_TXT                    called-cell barcodes, one per line (the set every tool is scored on)
Writes OUT_DIR/species.tsv (taxid, name, domain, reads) and OUT_DIR/percell.tsv.gz (barcode, taxid, name, domain, reads,
umis). Reads classified below species rank are rolled up to their species. Reads assigned above species rank (genus,
"cellular organisms", root, ...) are not species-level calls and are dropped, as are host reads (Homo sapiens, taxid 9606).
The species and domain of every taxon come from the Kraken2 report itself (indentation = depth), so no taxonomy file is
needed. A cell barcode matches a called cell exactly or with one mismatch (ambiguous one-mismatch matches are dropped);
a molecule is a distinct (cell, UMI, species).
"""
import collections
import gzip
import sys


def load_report(path):
    """taxid -> (species taxid, species name, domain name) from a Kraken2 report."""
    stack = {}
    out = {}
    names = {}
    with open(path) as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            rank, taxid, raw = f[3], int(f[4]), f[5]
            depth = (len(raw) - len(raw.lstrip(" "))) // 2
            name = raw.strip()
            stack[depth] = (taxid, rank, name)
            for d in [k for k in stack if k > depth]:
                del stack[d]
            sp = None
            dom = ""
            for d in range(depth + 1):
                t = stack.get(d)
                if not t:
                    continue
                if t[1] == "S":
                    sp = t
                if t[1] in ("D", "K", "R1") and not dom:
                    dom = t[2]
            # reads assigned above species rank (genus, family, "cellular organisms", root, ...) are not species-level calls
            out[taxid] = (sp[0], sp[2], dom) if sp is not None else (None, name, dom)
            names[taxid] = name
    return out


def main():
    ko, rep, m2, cells_f, od = sys.argv[1:6]
    cb_len = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    umi_len = int(sys.argv[7]) if len(sys.argv) > 7 else 12
    tax = load_report(rep)
    cells = [x.split("\t")[0].strip() for x in open(cells_f) if x.strip()]
    cmap = {c: c for c in cells}
    for c in cells:
        for i in range(len(c)):
            for b in "ACGT":
                if b != c[i]:
                    n = c[:i] + b + c[i + 1:]
                    if n not in cmap:
                        cmap[n] = c
                    elif cmap[n] != c:
                        cmap[n] = None
    r2sp = {}
    n_total = n_class = 0
    import gzip as _gz
    with (_gz.open(ko, "rt") if ko.endswith(".gz") else open(ko)) as fh:
        for ln in fh:
            n_total += 1
            if ln[0] != "C":
                continue
            p = ln.split("\t")
            t = int(p[2])
            sp = tax.get(t)
            if sp is None or sp[0] is None or sp[0] == 9606:
                continue
            r2sp[p[1]] = sp
            n_class += 1
    sp_reads = collections.Counter(v[0] for v in r2sp.values())
    info = {v[0]: v for v in r2sp.values()}
    with open(f"{od}/species.tsv", "w") as o:
        o.write("taxid\tname\tdomain\treads\n")
        for t, n in sp_reads.most_common():
            o.write(f"{t}\t{info[t][1]}\t{info[t][2]}\t{n}\n")
    reads = collections.Counter()
    umis = collections.defaultdict(set)
    with open(m2) as fh:
        while True:
            h = fh.readline()
            if not h:
                break
            s = fh.readline().strip()
            fh.readline()
            fh.readline()
            sp = r2sp.get(h[1:].split()[0])
            if sp is None:
                continue
            cb = cmap.get(s[:cb_len])
            if cb is None:
                continue
            reads[(cb, sp[0])] += 1
            umis[(cb, sp[0])].add(s[cb_len:cb_len + umi_len])
    with gzip.open(f"{od}/percell.tsv.gz", "wt") as o:
        o.write("barcode\ttaxid\tname\tdomain\treads\tumis\n")
        for (cb, t), n in reads.items():
            o.write(f"{cb}\t{t}\t{info[t][1]}\t{info[t][2]}\t{n}\t{len(umis[(cb, t)])}\n")
    print(f"reads={n_total} classified_non_host={n_class} species={len(sp_reads)} cell_species_pairs={len(reads)} "
          f"top={[(info[t][1], n) for t, n in sp_reads.most_common(5)]}")


if __name__ == "__main__":
    main()
