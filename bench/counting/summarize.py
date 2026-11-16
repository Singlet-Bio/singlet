"""Tabulate sweep results: python3 summarize.py <OUT> <library> [<library>...]  (rows: metric ids; columns: rule sets)."""
import json, os, sys, glob
out, libs = sys.argv[1], sys.argv[2:]
tags = sorted(os.path.basename(p)[1:] for p in glob.glob(f"{out}/r*") if os.path.isdir(p))
def load(t, cap, lib):
    p = f"{out}/r{t}/results/{cap}.json"
    if not os.path.exists(p): return {}
    d = json.load(open(p)); ms = d.get("metrics") or [m for ds in d.get("datasets", []) for m in (ds.get("metrics", []) if isinstance(ds, dict) else [])]
    return {m["id"]: m for m in ms if m.get("dataset") == lib}
for lib in libs:
    print(f"## {lib}\nrow\t" + "\t".join(tags) + "\tcomparator")
    for cap in ("gex-counts", "spliced-unspliced"):
        D = {t: load(t, cap, lib) for t in tags}; ids = sorted(set().union(*[set(d) for d in D.values()]))
        for i in ids:
            cells, comp = [], "-"
            for t in tags:
                m = D[t].get(i)
                if m is None: cells.append("-"); continue
                if m.get("pass") == "na": cells.append(f"{m['singlet']['value']:.4g}"); continue
                cells.append(f"{m['singlet']['value']:.4g} ({m.get('pass')})")
                c = [x["value"] for x in m.get("competitors", []) if x.get("value") is not None]
                if c: comp = f"{(max(c) if m.get('direction') == 'higher' else min(c)):.4g}"
            print(f"{cap[:3]} {i}\t" + "\t".join(cells) + f"\t{comp}")
