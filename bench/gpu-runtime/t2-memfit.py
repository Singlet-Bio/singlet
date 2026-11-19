#!/usr/bin/env python3
# hostcut round 3: per-item host peak (gpu_pool.cpp's definition: max over time of the sum of RssAnon + RssShmem, GiB, of the item's process group, from
# memsamp.tsv) against the item's reads (pairs of samples.tsv, M), least squares peak = a + b x reads(M), and the comment's margin rule.
# usage: t2-memfit.py <rt-node results dir> [csv out]    (pool0/, pool1/ with samples.tsv and memsamp.tsv)
# env EXCLUDE=name,name: samples left out of the fit (the table still lists them)
import sys, os, collections, math
d = sys.argv[1]
items = []   # (pool, item, sample, reads M, peak GiB)
for p in sorted(x for x in os.listdir(d) if x.startswith('pool') and os.path.isdir(os.path.join(d, x))):
    meta = {}
    sp = os.path.join(d, p, 'samples.tsv')
    if not os.path.exists(sp): continue
    for ln in open(sp).read().splitlines()[1:]:
        f = ln.split('\t')
        if len(f) < 15: continue
        if f[11] != '0': continue   # exit code
        meta[int(f[0])] = (os.path.basename(f[2]).replace('.1fq', ''), float(f[13]) / 1e6)
    tot = collections.defaultdict(float)
    for ln in open(os.path.join(d, p, 'memsamp.tsv')):
        f = ln.rstrip('\n').split('\t')
        if len(f) < 7: continue
        it = int(f[1])
        if it < 0: continue
        tot[(f[0], it)] += (int(f[4]) + int(f[6])) / 1048576.0
    peak = {}
    for (t, it), v in tot.items(): peak[it] = max(peak.get(it, 0), v)
    for it, (s, m) in meta.items():
        if it in peak: items.append((p, it, s, m, peak[it]))
EX = set(x for x in os.environ.get('EXCLUDE', '').split(',') if x)
all_items = items
items = [i for i in items if i[2] not in EX]
n = len(items)
xs = [i[3] for i in items]; ys = [i[4] for i in items]
mx = sum(xs) / n; my = sum(ys) / n
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
a = my - b * mx
res = [y - (a + b * x) for x, y in zip(xs, ys)]
rmse = math.sqrt(sum(r * r for r in res) / n)
w = max(range(n), key=lambda k: res[k])
print("items %d, samples %d: peak = %.2f + %.4f x reads(M); rmse %.1f GiB; worst under-prediction %.1f GiB (%s, %.1f M reads, measured %.1f)" %
      (n, len(set(i[2] for i in items)), a, b, rmse, res[w], items[w][2], items[w][3], items[w][4]))
margin = math.ceil(res[w] / 5.0) * 5
print("margin rule (worst under-prediction rounded up to 5): margin %d, genome off 16.4: a_new = %.1f, b = %.4f" % (margin, a + margin - 16.4, b))
bys = collections.defaultdict(list)
for i in all_items: bys[i[2]].append(i)
for s, l in sorted(bys.items(), key=lambda kv: -kv[1][0][3]):
    pk = [i[4] for i in l]
    print("  %-16s n=%2d reads %8.1f M peak mean %6.1f max %6.1f  est(old 61.8+0.2235x) %6.1f" % (s, len(l), l[0][3], sum(pk) / len(pk), max(pk), 61.8 + 0.2235 * l[0][3]))
if len(sys.argv) > 2:
    with open(sys.argv[2], 'w') as f:
        for i in items: f.write("%s\t%d\t%s\t%.3f\t%.2f\n" % i)
