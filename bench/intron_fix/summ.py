import json, sys
d = json.load(open(sys.argv[1]))
want = sys.argv[2].split(",") if len(sys.argv) > 2 else None
for ds, vs in d.items():
    print("=====", ds)
    for v, e in vs.items():
        if want and v not in want:
            continue
        ms = {m["id"]: m for m in e["metrics"]}
        def f(i):
            m = ms.get(i)
            if not m: return "   -   "
            c = m["competitors"][0][1] if m["competitors"] else float("nan")
            return f"{m['singlet']:.4f}/{c:.4f}{'+' if m['pass']=='pass' else '-'}"
        row = [f(i) for i in ("umi_ratio_distance.unspliced", "gene_corr.unspliced", "pseudobulk_rho.unspliced",
                              "umi_ratio_distance.spliced_ambiguous", "gene_corr.spliced_ambiguous", "pseudobulk_rho.spliced_ambiguous",
                              "gene_corr.total", "pseudobulk_rho.total", "adt_rna_spearman.total")]
        npass = sum(m["pass"] == "pass" for m in e["metrics"])
        by = e.get("barnyard", {})
        bys = f" by U={by.get('singlet_U',0)*100:.3f} E={by.get('singlet_E',0)*100:.3f} EU={by.get('singlet_EU',0)*100:.3f}" if by else ""
        print(f"{v:10s} {npass:2d}/{len(e['metrics'])} U={e['totals']['U']:.0f} " + " ".join(row) + bys)
