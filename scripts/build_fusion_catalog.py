#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the fusion catalogue read by `singlet --chimeric --fusion-catalog` (include/singlet/pileup/chimeric_table.h).

Open sources only (never COSMIC):
  * Mitelman Database of Chromosome Aberrations and Gene Fusions in Cancer (CC BY 4.0), bulk file mitelman_db.zip: the molecular gene
    fusions of MCGENE.TXT.DATA (MolClin == "M", Gene "A::B"; a chain "A::B::C" gives A--B and B--C). The order in Mitelman names is not
    the 5'/3' order, so these pairs are written as unordered (ordered = 0).
  * CTAT human fusion annotation library (FusionAnnotator/CTAT_HumanFusionLib, fusion_lib.*.dat.gz): lines "A--B <TAB> [tags] <TAB> {json}",
    5' gene first. It carries the ChimerDB modules (ChimerSeq, ChimerPub, ChimerKB, chimerdb_*), Mitelman, and recurrent tumour, cell line and
    normal-tissue fusions (TCGA_StarF2019, CCLE_StarF2019, GTEx_recurrent_StarF2019, Babiceanu_Normal, ...). Every tag is kept as a source so a
    caller can tell cancer fusions from normal-tissue ones, except:
      - tags containing "cosmic" (any case) are dropped, as is a line whose only tags are dropped;
      - HGNC_GENEFAM (9.8 M gene family pairs, not fusion evidence) is dropped; DGD_PARALOGS is kept (a paralog pair is a common false positive).

Output (gzip, sorted, deterministic): `#` provenance lines, then `gene_a<TAB>gene_b<TAB>sources<TAB>ordered`, sources comma separated and sorted.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import zipfile

DROP_TAGS = {"HGNC_GENEFAM"}
PAIR = re.compile(r"^[^\s]+--[^\s]+$")


def keep_tag(tag):
    return "cosmic" not in tag.lower() and tag not in DROP_TAGS and tag != "."


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_ctat(path):
    """-> {(a, b): set(tags)} from fusion_lib.*.dat(.gz)."""
    out = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 2 or not PAIR.match(f[0]):
                continue
            a, b = f[0].split("--", 1)
            if not a or not b or a == b:
                continue
            try:
                tags = json.loads(f[1])
            except ValueError:
                continue
            tags = {t for t in tags if keep_tag(t)}
            if tags:
                out.setdefault((a, b), set()).update(tags)
    return out


def read_mitelman(path):
    """-> set of unordered (a, b) gene pairs from mitelman_db.zip (or the extracted directory) MCGENE.TXT.DATA."""
    if os.path.isdir(path):
        fh = open(os.path.join(path, "MCGENE.TXT.DATA"), "rt", encoding="utf-8", errors="replace")
    else:
        z = zipfile.ZipFile(path)
        fh = __import__("io").TextIOWrapper(z.open("MCGENE.TXT.DATA"), encoding="utf-8", errors="replace")
    pairs = set()
    with fh:
        head = fh.readline().rstrip("\n").split("\t")
        ig, im = head.index("Gene"), head.index("MolClin")
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) <= ig or f[im] != "M" or "::" not in f[ig]:
                continue
            genes = [g.strip() for g in f[ig].split("::") if g.strip()]
            for a, b in zip(genes, genes[1:]):
                if a != b:
                    pairs.add((a, b))
    return pairs


def build(ctat_path, mitelman_path):
    ordered = read_ctat(ctat_path) if ctat_path else {}
    rows = {}
    for (a, b), tags in ordered.items():
        rows[(a, b)] = [set(tags), 1]
    if mitelman_path:
        for a, b in read_mitelman(mitelman_path):
            if (a, b) in rows:
                rows[(a, b)][0].add("Mitelman")
            elif (b, a) in rows:
                rows[(b, a)][0].add("Mitelman")
            else:
                rows[(a, b)] = [{"Mitelman"}, 0]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ctat", help="CTAT fusion_lib.*.dat.gz")
    ap.add_argument("--mitelman", help="mitelman_db.zip or the extracted directory")
    ap.add_argument("--out", required=True, help="fusion_catalog.tsv.gz")
    a = ap.parse_args(argv)
    rows = build(a.ctat, a.mitelman)
    prov = ["# singlet fusion catalogue: Mitelman (CC BY 4.0), ChimerDB and CTAT sources via the CTAT human fusion library; COSMIC excluded"]
    for name, p in (("ctat", a.ctat), ("mitelman", a.mitelman)):
        if p and os.path.isfile(p):
            prov.append("# %s %s sha256 %s" % (name, os.path.basename(p), sha256(p)))
    prov.append("# pairs %d" % len(rows))
    with gzip.GzipFile(a.out, "wb", compresslevel=6, mtime=0) as gz:
        for ln in prov:
            gz.write((ln + "\n").encode())
        gz.write(b"gene_a\tgene_b\tsources\tordered\n")
        for (x, y) in sorted(rows):
            tags, o = rows[(x, y)]
            gz.write(("%s\t%s\t%s\t%d\n" % (x, y, ",".join(sorted(tags)), o)).encode())
    print("wrote %s: %d pairs" % (a.out, len(rows)), file=sys.stderr)


if __name__ == "__main__":
    main()
