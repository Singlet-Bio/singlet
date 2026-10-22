#!/usr/bin/env python
"""w4_vocab: (1) invariance of row names across studies, (2) structural derivation tests, per reference build.

    python -P tools/w4_vocab_analyze.py            # reads corpus_w4_vocab/ (see w4_vocab_fetch.py), writes results/w4_vocab_invariance.txt,
                                                   # results/w4_vocab_derivation.txt and results/w4_vocab_derivation.tsv
"""
import collections, csv, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4_vocab_lib as L

OUT = os.path.join(L.ROOT, "results")


def rep_for_build(index):
    """First (gse, gsm) per build in index order, preferring the one with the most studies."""
    by = collections.OrderedDict()
    for r in index:
        by.setdefault(r["build"], []).append(r)
    return by


def derive(build, exon, intron, vocab, tag_of_gene=None):
    """All derivation tests for one build.  exon/intron: row lists; vocab: parsed feature_vocab.json."""
    res = collections.OrderedDict()
    res["build"] = build
    res["exon_rows"] = len(exon)
    res["intron_rows"] = len(intron)
    genes = [(g["gene_id"], g["gene_name"]) for g in vocab["genes"]]
    res["vocab_genes"] = len(genes)
    # barnyard: split by tag
    tags = []
    if "___" in exon[0]:
        tags = sorted({re.search(r"_([A-Za-z0-9.]+___)", r).group(1) for r in exon[:1] + exon[-1:]})
    res["species_tags"] = ",".join(tags) if tags else ""
    notes = []

    def run(exon_rows, intron_rows, genes_sub, tag):
        r = collections.OrderedDict()
        gid, sym, chrom, st, en = L.parse_rows(exon_rows, "exon", genes_sub, tag)
        grp = L.group_genes(gid, sym, chrom, st, en)
        r["genes_in_exon_rows"] = len(grp)
        r["gene_order_equals_vocab"] = int([g[0] for g in grp] == [g for g, _ in genes_sub])
        r["symbol_equals_vocab_name"] = sum(1 for g, (vg, vn) in zip(grp, genes_sub) if g[1] == vn)
        r["vocab_name_equals_symbol_or_id_if_empty"] = sum(1 for g, (vg, vn) in zip(grp, genes_sub) if (g[1] if g[1] else g[0]) == vn)
        # symbol kinds
        r["sym_empty"] = sum(1 for g in grp if g[1] == "")
        r["sym_is_id"] = sum(1 for g in grp if g[1] == g[0])
        r["sym_explicit"] = len(grp) - r["sym_empty"] - r["sym_is_id"]
        r["sym_has_underscore"] = sum(1 for g in grp if "_" in g[1])
        # ids
        pref = {re.match(r"^([A-Za-z]+)", g[0]).group(1) for g in grp}
        digs = {len(g[0]) - len(re.match(r"^([A-Za-z]+)", g[0]).group(1)) for g in grp}
        nums = np.array([int(re.sub(r"^[A-Za-z]+", "", g[0])) for g in grp], np.int64)
        r["id_prefixes"] = "|".join(sorted(pref))
        r["id_digit_widths"] = "|".join(map(str, sorted(digs)))
        r["ids_sorted_lex"] = int([g[0] for g in grp] == sorted(g[0] for g in grp))
        r["ids_strictly_increasing_numeric"] = int(bool((np.diff(nums) > 0).all()))
        r["id_delta_mean"] = float(np.diff(nums).mean())
        # exons
        neg = ov = touch = zero = 0
        nex = np.array([len(g[3]) for g in grp])
        r["exons_per_gene_mean"] = float(nex.mean()); r["exons_per_gene_max"] = int(nex.max()); r["single_exon_genes"] = int((nex == 1).sum())
        for g in grp:
            ee = g[3]
            for i, (a, b) in enumerate(ee):
                if b < a: neg += 1
                if b == a: zero += 1
                if i:
                    pb = ee[i - 1][1]
                    if a < pb: ov += 1
                    elif a == pb: touch += 1
        r["exon_end_lt_start"] = neg; r["exon_zero_length"] = zero; r["exon_overlaps_prev"] = ov; r["exon_touches_prev"] = touch
        # chromosomes
        chs = [g[2] for g in grp]
        r["chromosomes"] = len(set(chs))
        r["chrom_names_with_underscore"] = len({c for c in chs if "_" in c})
        r["chrom_prefix_chr"] = int(any(c.startswith("chr") for c in set(chs)))
        r["gene_same_chrom_as_prev_frac"] = sum(1 for i in range(1, len(chs)) if chs[i] == chs[i - 1]) / len(chs)
        # genomic order?  (is the gene list sorted by (chrom, start)?)
        gs = [(g[2], g[3][0][0]) for g in grp]
        r["gene_pairs_increasing_pos_same_chrom_frac"] = sum(1 for i in range(1, len(gs)) if gs[i][0] == gs[i - 1][0] and gs[i][1] > gs[i - 1][1]) / len(gs)
        # introns: derived == actual ?
        mid = "intron_"
        der = []
        for g in grp:
            ee = g[3]
            for i in range(len(ee) - 1):
                der.append("%s_%s_%s%s%s:%d-%d" % (g[0], g[1], mid, tag, g[2], ee[i][1], ee[i + 1][0]))
        r["intron_rows_derived"] = len(der)
        r["intron_rows_actual"] = len(intron_rows)
        r["intron_equals_gaps_exact"] = int(der == intron_rows)
        if der != intron_rows:
            sa, sd = set(intron_rows), set(der)
            r["intron_only_in_actual"] = len(sa - sd); r["intron_only_in_derived"] = len(sd - sa)
            r["intron_order_differs"] = int(len(sa) == len(sd) and sa == sd)
        else:
            r["intron_only_in_actual"] = 0; r["intron_only_in_derived"] = 0; r["intron_order_differs"] = 0
        r["intron_rows_eq_exon_rows_minus_genes"] = int(len(intron_rows) == len(exon_rows) - len(grp))
        return r, grp

    if not tags:
        r, _ = run(exon, intron, genes, "")
        res.update(r)
    else:
        # barnyard: partition exon / intron / genes by tag; every part is analysed as a standalone species build
        parts = []
        for t in tags:
            ex = [x for x in exon if ("_" + t) in x]
            it = [x for x in intron if ("_" + t) in x]
            res["part_%s_exon_rows" % t] = len(ex)
            res["part_%s_intron_rows" % t] = len(it)
            parts.append((t, ex, it))
        res["parts_contiguous_in_order"] = int(exon == sum((p[1] for p in parts), []) and intron == sum((p[2] for p in parts), []))
        # genes belong to parts by id prefix
        for t, ex, it in parts:
            gsub = [(g, n) for g, n in genes if (g.startswith("ENSMUSG") if "m39" in t else g.startswith("ENSG"))]
            r, _ = run(ex, it, gsub, t)
            for k, v in r.items():
                res["%s:%s" % (t, k)] = v
    return res


def main():
    index = list(csv.DictReader(open(os.path.join(L.DATA, "index.tsv")), delimiter="\t"))
    by = rep_for_build(index)
    # ---- invariance
    lines = ["# Row-name invariance across studies (hash = sha256 over NUL-terminated names)\n"]
    for build, rs in by.items():
        for kind in ("exon", "intron"):
            sub = [r for r in rs if r["kind"] == kind]
            hs = collections.OrderedDict()
            for r in sub:
                hs.setdefault(r["rows_sha256"], []).append("%s/%s(%s,%s)" % (r["gse"], r["gsm"], r["protocol"], r["git_sha"][:7]))
            lines.append("%-24s %-6s rows=%s studies=%d distinct_lists=%d" % (build, kind, sorted({r["m"] for r in sub}), len({r["gse"] for r in sub}), len(hs)))
            for h, who in hs.items():
                lines.append("    %s  %s" % (h[:16], "  ".join(who)))
    # feature_vocab.json invariance over every study fetched (including the light and atlas scans)
    lines.append("\n# feature_vocab.json (gene table) over every study fetched\n")
    tab = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(os.path.join(L.DATA, "*.feature_vocab.json"))):
        gse = os.path.basename(f).split(".")[0]
        raw = open(f, "rb").read()
        d = json.loads(raw)
        h = L.names_sha256([x["gene_id"] + "\t" + x["gene_name"] for x in d["genes"]])[:16]
        sj = glob.glob(os.path.join(L.DATA, gse + ".*.summary.json"))
        git = json.load(open(sj[0])).get("git_sha", "")[:7] if sj else "?"
        tab[d["reference_build"]][h].append("%s(%s)" % (gse, git))
    for b, hs in sorted(tab.items()):
        n = sum(len(v) for v in hs.values())
        lines.append("%-24s studies=%d distinct_gene_tables=%d" % (b, n, len(hs)))
        for h, who in hs.items():
            lines.append("    %s  %d studies: %s" % (h, len(who), " ".join(who[:12]) + (" ..." if len(who) > 12 else "")))
    open(os.path.join(OUT, "w4_vocab_invariance.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- derivation, one representative sample per build
    recs = []
    for build, rs in by.items():
        r0 = rs[0]
        gse, gsm = r0["gse"], r0["gsm"]
        ex, _, _ = L.load_meta(os.path.join(L.DATA, "%s.%s.exon.meta" % (gse, gsm)))
        it, _, _ = L.load_meta(os.path.join(L.DATA, "%s.%s.intron.meta" % (gse, gsm)))
        vocab = L.load_vocab(os.path.join(L.DATA, "%s.feature_vocab.json" % gse))
        res = derive(build, ex, it, vocab)
        res["sample"] = "%s/%s" % (gse, gsm)
        recs.append(res)
    keys = []
    for r in recs:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(OUT, "w4_vocab_derivation.tsv"), "w") as o:
        o.write("metric\t" + "\t".join(r["build"] for r in recs) + "\n")
        for k in keys:
            o.write(k + "\t" + "\t".join(str(r.get(k, "")) for r in recs) + "\n")
    print()
    print(open(os.path.join(OUT, "w4_vocab_derivation.tsv")).read())


if __name__ == "__main__":
    main()
