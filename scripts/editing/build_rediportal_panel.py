#!/usr/bin/env python3
"""Build the D-101 RNA editing panel from a REDIportal TABLE1 file.

Writes <out>.tsv.gz (CHROM POS REF ALT STRAND ACCESSION TYPE, 1-based, REF/ALT on the genome + strand: A>G for a + strand
site, T>C for a - strand site, sorted in genome .fai order) and <out>.json, the manifest the engine copies into provenance
(source URL, REDIportal version, sha256 of the source and of the panel, counts by filter, license note).

Filters, in order:
  1. single-base A>G on + or T>C on - (REDIportal's own convention; anything else is reported and dropped)
  2. optional liftover (mm10 -> GRCm39 for mouse) with a UCSC chain; sites that do not map one to one are dropped
  3. the reference base at the position in the target genome fasta must equal REF
  4. sites at positions of the default SNP panel (1000G AF >= 5% / Ensembl common) are dropped: their editing level is
     confounded with the donor genotype, and the SNP pileup already reports those positions

Usage:
  build_rediportal_panel.py --table TABLE1_hg38_v3.txt.gz --fasta genome.fa --snp-panel snps.vcf.gz \
      --version "REDIportal v3" --source-url URL --out rna_editing_GRCh38 [--chain mm10ToMm39.over.chain.gz]
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def read_fai(fasta):
    order, info = [], {}
    with open(fasta + ".fai") as f:
        for line in f:
            name, length, offset, lb, lw = line.split("\t")[:5]
            order.append(name)
            info[name] = (int(length), int(offset), int(lb), int(lw))
    return order, info


class Fasta:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.order, self.info = read_fai(path)

    def base(self, chrom, pos1):
        length, offset, lb, lw = self.info[chrom]
        if pos1 < 1 or pos1 > length:
            return "N"
        i = pos1 - 1
        self.f.seek(offset + (i // lb) * lw + i % lb)
        return self.f.read(1).decode().upper()


def contig_mapper(names):
    s = set(names)

    def m(c):
        if c in s:
            return c
        if c.startswith("chr") and c[3:] in s:
            return c[3:]
        if c == "chrM" and "MT" in s:
            return "MT"
        if ("chr" + c) in s:
            return "chr" + c
        return None
    return m


def load_snp_positions(path, cmap):
    pos = set()
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            a = line.split("\t", 2)
            c = cmap(a[0])
            if c is not None:
                pos.add((c, int(a[1])))
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--snp-panel", default="")
    ap.add_argument("--chain", default="")
    ap.add_argument("--version", required=True)
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--assembly-source", required=True, help="assembly of the table (hg38, mm10)")
    ap.add_argument("--assembly-target", required=True, help="genome the panel is for (GRCh38, GRCm39)")
    ap.add_argument("--out", required=True, help="output prefix (writes .tsv.gz and .json)")
    a = ap.parse_args()

    fa = Fasta(a.fasta)
    cmap = contig_mapper(fa.order)
    lo = None
    if a.chain:
        from pyliftover import LiftOver
        lo = LiftOver(a.chain)
    snp = load_snp_positions(a.snp_panel, cmap) if a.snp_panel else set()
    print(f"[panel] {len(snp)} SNP panel positions", file=sys.stderr)

    n = dict(rows=0, kept=0, not_a_to_i=0, contig_absent=0, liftover_failed=0, ref_mismatch=0, snp_panel_overlap=0)
    by_type = {}
    sites = []
    with gzip.open(a.table, "rt") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        col = {h: i for i, h in enumerate(hdr)}
        ci, pi, ri, ei, si = col["Region"], col["Position"], col["Ref"], col["Ed"], col["Strand"]
        ai, ti = col.get("Accession"), col.get("type")
        for line in f:
            v = line.rstrip("\n").split("\t")
            n["rows"] += 1
            ref, ed, strand = v[ri], v[ei], v[si]
            if not ((strand == "+" and ref == "A" and ed == "G") or (strand == "-" and ref == "T" and ed == "C")):
                n["not_a_to_i"] += 1
                continue
            chrom, pos = v[ci], int(v[pi])
            if lo is not None:
                r = lo.convert_coordinate(chrom, pos - 1, strand)
                if not r or len(r) != 1:
                    n["liftover_failed"] += 1
                    continue
                chrom, pos0, nstrand = r[0][0], r[0][1], r[0][2]
                pos = pos0 + 1
                if nstrand != strand:   # the site's genomic orientation flipped: complement REF/ALT and strand
                    strand = nstrand
                    ref, ed = ("A", "G") if strand == "+" else ("T", "C")
            c = cmap(chrom)
            if c is None:
                n["contig_absent"] += 1
                continue
            if fa.base(c, pos) != ref:
                n["ref_mismatch"] += 1
                continue
            if (c, pos) in snp:
                n["snp_panel_overlap"] += 1
                continue
            t = v[ti] if ti is not None else "."
            by_type[t] = by_type.get(t, 0) + 1
            sites.append((c, pos, ref, ed, strand, v[ai] if ai is not None else ".", t))
            n["kept"] += 1
            if n["rows"] % 2000000 == 0:
                print(f"[panel] {n}", file=sys.stderr)

    rank = {c: i for i, c in enumerate(fa.order)}
    sites.sort(key=lambda s: (rank[s[0]], s[1]))
    dedup, last = [], None
    for s in sites:
        if (s[0], s[1]) == last:
            continue
        dedup.append(s)
        last = (s[0], s[1])
    n["duplicate_position"] = len(sites) - len(dedup)
    n["kept"] = len(dedup)
    tsv = a.out + ".tsv.gz"
    with gzip.open(tsv + ".tmp", "wt", compresslevel=6) as g:
        g.write(f"#chrom\tpos\tref\talt\tstrand\taccession\ttype\t{a.version}; {a.assembly_target}\n")
        for s in dedup:
            g.write("\t".join(map(str, s)) + "\n")
    os.replace(tsv + ".tmp", tsv)
    man = {
        "panel": os.path.basename(tsv),
        "source": "REDIportal (Picardi lab, University of Bari / CNR-IBIOM), A-to-I RNA editing sites",
        "version": a.version,
        "source_url": a.source_url,
        "source_file": os.path.basename(a.table),
        "source_sha256": sha256(a.table),
        "source_bytes": os.path.getsize(a.table),
        "assembly_source": a.assembly_source,
        "assembly_target": a.assembly_target,
        "liftover_chain": os.path.basename(a.chain) if a.chain else None,
        "snp_panel_excluded": os.path.basename(a.snp_panel) if a.snp_panel else None,
        "counts": n,
        "sites_by_type": by_type,
        "panel_sha256": sha256(tsv),
        "license": ("REDIportal states no license or terms of use (download, help and contact pages checked 2026-11-07). "
                    "The panel is used as an internal reference and is not redistributed; per-site editing outputs are "
                    "owner-only. Cite Picardi et al., Nucleic Acids Res 2025;53:D233 and Mansi et al., Nucleic Acids Res 2021."),
        "built": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": "scripts/editing/build_rediportal_panel.py",
    }
    with open(a.out + ".json", "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
