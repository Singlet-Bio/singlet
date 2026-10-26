# SPDX-License-Identifier: MIT
"""Guide table for Replogle 2022 K562 essential from the sgRNA ids and the genome.

    python -P prep_replogle.py DATA_DIR GENOME.fa.gz GTF.gz [--gemgroup 41]

The published h5ad names each cell's guide pair ``sgID_AB = A|B`` with ids ``GENE_<strand>_<coord>.23-<transcript>``
(hCRISPRi-v2 style: strand and hg19 coordinate of a 23-nt protospacer + PAM). Table S1 has no sequences, so the
protospacers are cut from hg19 here (the library was designed on hg19; GRCh38 coordinates match no read). Which offset the coordinate marks is decided from the reads: every candidate
convention is scored by the share of observed protospacer windows (20 nt before the two scaffold anchors) it explains.

Writes ``DATA_DIR/guides.tsv`` (id, sequence, target_gene, target_gene_id, kind, position A/B) and
``DATA_DIR/reference_gg<N>.csv`` (barcode, sgID_AB, gene) for the gem group.
"""
import argparse
import gzip
import re
from collections import Counter

import numpy as np
import pandas as pd

_C = str.maketrans("ACGTN", "TGCAN")


def rc(s):
    return s.translate(_C)[::-1]


def read_chroms(path, wanted):
    out, cur, buf = {}, None, []
    with gzip.open(path, "rt") as f:
        for ln in f:
            if ln[0] == ">":
                if cur:
                    out[cur] = "".join(buf)
                n = ln[1:].split()[0]
                cur = n if n in wanted else None
                buf = []
            elif cur:
                buf.append(ln.strip().upper())
    if cur:
        out[cur] = "".join(buf)
    return out


def gene_chroms(gtf):
    m = {}
    with gzip.open(gtf, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.split("\t", 9)
            if p[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^".]+)', p[8]).group(1)
            gn = re.search(r'gene_name "([^"]+)"', p[8])
            m[gid] = p[0]
            if gn:
                m.setdefault(gn.group(1), p[0])
    return m


def windows(fq, n=400000):
    """Observed 20-nt windows before the A (GTTTCAGAGC) and B (GTTTAAGAGC) scaffold anchors."""
    obs = {"A": Counter(), "B": Counter()}
    with gzip.open(fq, "rt") as f:
        for i, ln in enumerate(f):
            if i % 4 != 1:
                continue
            for pos, anc in (("A", "GTTTCAGAGC"), ("B", "GTTTAAGAGC")):
                j = ln.find(anc)
                if j >= 20:
                    obs[pos][ln[j - 20:j]] += 1
            if i > 4 * n:
                break
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("genome")
    ap.add_argument("gtf")
    ap.add_argument("--gemgroup", type=int, default=41)
    a = ap.parse_args()
    import anndata as ad

    adata = ad.read_h5ad(f"{a.data}/K562_essential_raw_singlecell_01.h5ad", backed="r")
    obs = adata.obs
    ids = sorted(set(obs["sgID_AB"].astype(str)))
    gene_of = dict(zip(obs["sgID_AB"].astype(str), obs["gene"].astype(str)))
    gid_of = dict(zip(obs["sgID_AB"].astype(str), obs["gene_id"].astype(str)))
    rows = []
    for ab in ids:
        parts = ab.split("|")
        for pos, sg in zip("AB", parts):
            m = re.match(r"(.+)_([+-])_(\d+)\.(\d+)-(.+)$", sg)
            rows.append({"id": sg, "pos": pos, "gene": gene_of[ab], "gene_id": gid_of[ab],
                         "strand": m.group(2) if m else "", "coord": int(m.group(3)) if m else -1,
                         "len": int(m.group(4)) if m else 0, "pair": ab})
    g = pd.DataFrame(rows).drop_duplicates(["id", "pos"])
    gc = gene_chroms(a.gtf)
    g["chrom"] = [gc.get(x, gc.get(y, "")) for x, y in zip(g.gene_id, g.gene)]
    chroms = read_chroms(a.genome, set(g["chrom"]) - {""})
    import glob

    fq = sorted(glob.glob(f"{a.data}/fastqs/*_R2_*.fastq.gz"))[0]
    seen = windows(fq)
    # conventions: protospacer (L nt, 5'->3' on the guide strand) at an offset from coord, for each strand. hCRISPRi-v2
    # protospacers are often 19 nt behind a U6 +1 G, so 19-nt genomic matches are scored against the window's last 19
    seen19 = {k: {w[1:] for w in v} for k, v in seen.items()}

    def cut(chrom_seq, r, off, L):
        # measured on the reads (bench/crispr/locate_replogle.py): a "-" id reads its protospacer on the + strand of
        # hg19 at coord - 21 (19 nt), a "+" id on the - strand at coord + 3; the offset is searched per strand here
        w = chrom_seq[r.coord + off: r.coord + off + L]
        return w if r.strand == "-" else rc(w)

    best = {}
    sample = g[(g.coord > 0) & (g.chrom != "")].sample(min(3000, len(g)), random_state=0)
    for strand in "+-":
        sub = sample[sample.strand == strand]
        for L in (19, 20):
            pool = seen19 if L == 19 else seen
            for off in range(-40, 41):
                hit = 0
                for r in sub.itertuples():
                    cs = chroms.get(r.chrom)
                    if cs is not None:
                        hit += cut(cs, r, off, L) in pool[r.pos]
                if strand not in best or hit > best[strand][2]:
                    best[strand] = (L, off, hit, len(sub))
    for strand, (L, off, hit, n) in best.items():
        print(f"id strand {strand}: {L} nt at offset {off}: {hit} / {n} sampled guides seen in the reads")
    seqs = []
    for r in g.itertuples():
        cs = chroms.get(r.chrom)
        if cs is None or r.coord < 0 or r.strand not in best:
            seqs.append("")
            continue
        L, off = best[r.strand][0], best[r.strand][1]
        p_ = cut(cs, r, off, L)
        if L == 19:  # the read window: the observed 5' base (G for U6 transcripts, often prepended) + 19 nt
            c = Counter(w[0] for w in seen[r.pos] if w[1:] == p_)
            p_ = (c.most_common(1)[0][0] if c else "G") + p_
        seqs.append(p_)
    g["sequence"] = seqs
    g["seen_reads"] = [seen[p].get(x, 0) for p, x in zip(g.pos, g.sequence)]
    ok = g[g.sequence.str.len() == 20].drop_duplicates("sequence")
    # the most frequent windows that no guide explains (e.g. non-targeting guides, which have no coordinate)
    known = set(ok.sequence)
    unexplained = [(w, c) for pos in "AB" for w, c in seen[pos].most_common(3000) if w not in known]
    nt = g[g.sequence.str.len() != 20]
    print(f"{len(g)} guides, {len(ok)} with sequence, {len(nt)} without (non-targeting: {(g.gene == 'non-targeting').sum()})")
    print("top unexplained windows:", unexplained[:10])
    tab = pd.DataFrame({"id": ok["id"], "sequence": ok["sequence"], "target_gene": ok["gene"],
                        "target_gene_id": ok["gene_id"],
                        "kind": np.where(ok["gene"] == "non-targeting", "non-targeting", "targeting"), "position": ok["pos"]})
    # non-targeting guides have no coordinate: abundant windows no guide explains are counted under their sequence
    # ("unidentified"; most are the library's non-targeting guides), so their cells are not left unassigned
    extra = {}
    for pos in "AB":
        for w, c in seen[pos].most_common(5000):
            if c >= 20 and "N" not in w and w not in known and w not in extra:
                extra[w] = pos
    tab = pd.concat([tab, pd.DataFrame({"id": [f"window_{p}_{w}" for w, p in extra.items()], "sequence": list(extra),
                                        "target_gene": "unidentified", "target_gene_id": "", "kind": "unidentified",
                                        "position": list(extra.values())})], ignore_index=True)
    print(f"{len(extra)} unidentified abundant windows added")
    tab.to_csv(f"{a.data}/guides.tsv", sep="\t", index=False)
    sel = obs["gem_group"].astype(int) == a.gemgroup
    ref = pd.DataFrame({"barcode": [i.split("-")[0] + "-1" for i in obs.index[sel]],
                        "sgID_AB": obs.loc[sel, "sgID_AB"].astype(str).to_numpy(),
                        "gene": obs.loc[sel, "gene"].astype(str).to_numpy()})
    ref.to_csv(f"{a.data}/reference_gg{a.gemgroup}.csv", index=False)
    print("guides seen in reads:", float((ok.seen_reads > 0).mean()), "cells in gem group", int(sel.sum()))


if __name__ == "__main__":
    main()
