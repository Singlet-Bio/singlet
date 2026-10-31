# SPDX-License-Identifier: MIT
"""Add immunoglobulin V genes that the genome annotation lacks (GENCODE / RefSeq on GRCh38: IGHV1-8, IGHV4-30-4 ...) from the
AIRR Community reference sets (OGRDB, published under CC0 1.0), as a reference-version addition.

    python -P add_airrc.py IN.tsv.gz OUT.tsv.gz IGH.fasta IGK.fasta IGL.fasta --version "IGH_VDJ r10, ..." [--first-cys]

Per gene absent from IN by name: its first allele in the set (``*01`` style before inferred ``*i`` alleles), the ungapped V
region (FR1 .. Cys104 and the germline 3' end; no leader), frame 0, Cys104 = the last in-frame TGT/TGC in the last 45 nt
(``reference._v_anchor``), or with ``--first-cys`` the first of two adjacent Cys codons (Cell Ranger >= 5 convention, as in
``vdj_human_imgtcys.tsv.gz``). Genes with a stop codon before Cys104 or without an anchor are skipped (pseudogenes).
gene_id = ``AIRR-C:<allele>``, contig = ``AIRR-C``.
"""
from __future__ import annotations

import argparse
import gzip
import re

from singlet.vdj import reference as R


def fasta(path):
    name, seq, out = None, [], []
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith(">"):
            if name:
                out.append((name, "".join(seq).upper()))
            name, seq = ln[1:].split()[0].split("|")[0], []
        elif name:
            seq.append(ln)
    if name:
        out.append((name, "".join(seq).upper()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("fastas", nargs="+")
    ap.add_argument("--version", required=True)
    ap.add_argument("--first-cys", action="store_true")
    a = ap.parse_args()
    segs = R.load(a.inp)
    with gzip.open(a.inp, "rt") as f:
        first = f.readline()
    prov = first[1:].strip() if first.startswith("#") else ""
    have = {s.name for s in segs}
    best = {}
    for p in a.fastas:
        for allele, seq in fasta(p):
            gene = allele.split("*")[0]
            if not re.match(r"^IG[HKL]V\d", gene) or gene in have or set(seq) - set("ACGT"):
                continue
            key = (allele.split("*")[1].startswith("i") if "*" in allele else True, allele)
            if gene not in best or key < best[gene][0]:
                best[gene] = (key, allele, seq)
    added, skipped = [], []
    for gene, (_, allele, seq) in sorted(best.items()):
        anc = R._v_anchor(seq, 0)
        if anc < 0 or "*" in R.translate(seq[: anc + 3], 0):
            skipped.append(allele)
            continue
        if a.first_cys:
            anc = R._first_cys(seq, anc)
        segs.append(R.Segment(gene, f"AIRR-C:{allele}", R.chain_of(gene), "V", seq, anc, 0, "AIRR-C", 0))
        added.append(allele)
    segs.sort(key=lambda s: (s.chain, "VDJC".index(s.region), s.name))
    R.write(segs, a.out, (prov + "; " if prov else "") + f"+ {len(added)} IG V genes from AIRR-C ({a.version}; OGRDB, CC0 1.0): "
            + ",".join(added))
    print("added", len(added), added)
    print("skipped (no anchor / stop)", skipped)


if __name__ == "__main__":
    main()
