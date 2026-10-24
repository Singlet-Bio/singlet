#!/usr/bin/env python
"""w4_mtx test 3: STARsolo features.tsv vs feature_vocab.json (set, names, order) and vs interval names in exon_counts.1pz."""
import os, sys, json, re
import numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); CORP = os.path.join(BC, "corpus_w4_mtx")
P = "star_Solo.out__Gene__"
for d in sys.argv[1:]:
    gse = d.split(".")[0]
    feat = [l.split("\t") for l in open(os.path.join(CORP, d, P + "raw__features.tsv")).read().split("\n")[:-1]]
    voc = json.load(open(os.path.join(CORP, gse, "feature_vocab.json")))["genes"]
    vid = [g["gene_id"] for g in voc]; vnm = {g["gene_id"]: g["gene_name"] for g in voc}
    fid = [f[0] for f in feat]
    print(f"[{d}] features {len(feat)} vocab {len(voc)}; same id set={set(fid)==set(vid)}; vocab sorted by id={vid==sorted(vid)}; features sorted by id={fid==sorted(fid)}")
    print("   names equal per id:", all(f[1] == vnm[f[0]] for f in feat), "; type column values:", sorted(set(f[2] for f in feat)), "; 3 cols:", all(len(f)==3 for f in feat))
    # name==id fallback for unnamed genes?
    print("   n genes with name==id:", sum(1 for f in feat if f[1]==f[0]), " (vocab:", sum(1 for g in voc if g['gene_name']==g['gene_id']), ")")
    # order: compare to position order derived from 1pz exon row names ENSG_SYM_chr:s-e
    r = read_1pz(os.path.join(CORP, d, "exon_counts.1pz"))
    rn = r["rownames"]
    pos = {}
    for n in rn:
        m = re.match(r"^(.*)_([^_:]+):(\d+)-(\d+)$", n)
        if not m: continue
        gid = n.split("_")[0]; chrom = m.group(2); s = int(m.group(3)); e = int(m.group(4))
        if gid not in pos: pos[gid] = [chrom, s, e]
        else:
            pos[gid][1] = min(pos[gid][1], s); pos[gid][2] = max(pos[gid][2], e)
    have = [g for g in fid if g in pos]
    print(f"   genes with exon intervals in 1pz: {len(have)} / {len(fid)}; example rowname {rn[0]}")
    # chromosome order of appearance in features order
    order_chr = []
    for g in have:
        c = pos[g][0]
        if not order_chr or order_chr[-1] != c: order_chr.append(c)
    print("   chrom runs in features order (first 30):", order_chr[:30], "n runs", len(order_chr))
    # within chromosome, is start increasing in features order?
    inc = 0; tot = 0; prev = None
    for g in have:
        c, s, e = pos[g]
        if prev is not None and prev[0] == c:
            tot += 1; inc += s >= prev[1]
        prev = (c, s)
    print(f"   consecutive same-chrom pairs with non-decreasing start: {inc}/{tot}")
    # try sort by (chrom runs rank, start, end) and compare exactly
    crank = {c: i for i, c in enumerate(order_chr)} if len(set(order_chr)) == len(order_chr) else None
    if crank:
        srt = sorted(have, key=lambda g: (crank[pos[g][0]], pos[g][1], pos[g][2]))
        eq = sum(1 for a, b in zip(srt, have) if a == b)
        print(f"   sort by (chrom order, start, end) reproduces features order at {eq}/{len(have)} positions; exact={srt==have}")
