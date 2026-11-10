"""Compare two singlet run dirs: which files differ, and for donor VCFs whether records, order, AD/DP or GT differ."""
import sys, os, filecmp, collections
a, b = sys.argv[1:3]
def files(root):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs: out.append(os.path.relpath(os.path.join(d, f), root))
    return sorted(out)
fa, fb = set(files(a)), set(files(b))
print("only_a", sorted(fa - fb)[:10], "only_b", sorted(fb - fa)[:10])
skip = ("log", "time", "txt", "Log", "out")
diff = [f for f in sorted(fa & fb) if not filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False)]
print("n_common", len(fa & fb), "n_differ", len(diff))
for f in diff: print("DIFF", f, os.path.getsize(os.path.join(a, f)), os.path.getsize(os.path.join(b, f)))
def vcf(p):
    recs = []
    for l in open(p):
        if l[0] == "#": continue
        x = l.rstrip("\n").split("\t"); recs.append((x[0], x[1], x[2], x[9]))
    return recs
for f in diff:
    if f.endswith(".vcf"):
        ra, rb = vcf(os.path.join(a, f)), vcf(os.path.join(b, f))
        da, db = {r[2]: r[3] for r in ra}, {r[2]: r[3] for r in rb}
        print(f, "n", len(ra), len(rb), "same_order_ids", [r[2] for r in ra] == [r[2] for r in rb],
              "sites_only_a", len(set(da) - set(db)), "only_b", len(set(db) - set(da)))
        c = collections.Counter(); n_ad = 0
        for k in set(da) & set(db):
            ga, gb = da[k].split(":"), db[k].split(":")
            c["gt_diff" if ga[0] != gb[0] else "gt_same"] += 1
            if ga[2] != gb[2]: n_ad += 1
        print("  ", dict(c), "AD differs at", n_ad)
        # which donor index maps to which: compare total depth
        print("  total DP a/b", sum(int(r[3].split(":")[3]) for r in ra), sum(int(r[3].split(":")[3]) for r in rb))
