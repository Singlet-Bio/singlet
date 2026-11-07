"""Print scores.json compactly (developer aid)."""
import json
import sys

d = json.load(open(sys.argv[1]))
for ds, r in d.items():
    print("==", ds, r.get("timing"))
    if "barnyard" in r:
        b = r["barnyard"]
        print({k: v for k, v in b.items() if k not in ("tools", "top_wrong_genes")})
        for t, v in b["tools"].items():
            print(" ", t, {k: round(x, 5) if isinstance(x, float) else x for k, x in v.items() if k != "ci"}, [round(c, 5) for c in v["ci"]])
        for t, v in b["top_wrong_genes"].items():
            print(" ", t, "h-in-m", v["human_genes_in_mouse_cells"][:6], "m-in-h", v["mouse_genes_in_human_cells"][:6])
    c = r["chrY"]
    print(" chrY", c["n_cells"], c["n_chrY_genes"])
    for t, v in c["tools"].items():
        print("  ", t, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items() if k != "ci_per_million"})
    for t, v in c["per_gene_umi"].items():
        print("  ", t, dict(sorted(v.items(), key=lambda kv: -kv[1])[:8]))
    if "cite" in r:
        ct = r["cite"]
        print(" cite pairs", ct["pairs"])
        for t, v in ct["tools"].items():
            print("  ", t, round(v["mean_spearman"], 4), [round(x, 4) for x in v["ci"]], {k: round(x, 3) for k, x in v["per_pair"].items()})
        print(json.dumps(ct["paired_diff_mean_ci"]))
