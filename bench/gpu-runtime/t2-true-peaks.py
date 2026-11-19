#!/usr/bin/env python3
"""pooladmit loss B: true per-item host peak against the pool's figures, from a one-worker run of t2-pooladmit-true.sbatch.
usage: t2-true-peaks.py <results dir> [profiles.json out] [table.tsv out]
Per item (one in flight at a time): reads (M pairs, samples.tsv), the pool's estimate (61.8 + 0.2235 x), the RSS-sum peak (gpu_pool.cpp's definition: max over time of the sum
of RssAnon + RssShmem of the item's processes, minus nothing; memsamp.tsv), and the TRUE peak = max of the job cgroup's anon+shmem (cg1hz.tsv, 1 s) while the item runs minus the
idle baseline just before the first item starts (median of the 5 s before; holder, server, pool). True level at the "[engine_gpu] level" line, at the "[allsite] async: waited" line
(markers.tsv, first second the line showed in the item's log) and at the item's end. Then the fit true = a + b x reads (the t2-memfit.py procedure) and the RSS-sum / true ratio by depth.
profiles.json: per sample class, the 1 s true level (GiB) against seconds since the item's start, with the level / waited / end offsets (input of t2-admit-replay.py --true)."""
import sys, os, re, json, math, collections
import numpy as np
D = sys.argv[1]; G = 1073741824.0
rows = [l.rstrip('\n').split('\t') for l in open(os.path.join(D, 'cg1hz.tsv')) if not l.startswith('#')]
A = np.array([[float(x) for x in r[:9]] for r in rows]); t = A[:, 0]
anonshm = (A[:, 3] + A[:, 5]) / G
mk = collections.defaultdict(dict)
if os.path.exists(os.path.join(D, 'markers.tsv')):
    for ln in open(os.path.join(D, 'markers.tsv')):
        f = ln.split()
        if len(f) == 3: mk[int(f[1])][f[2]] = float(f[0])
items = {}
for ln in open(os.path.join(D, 'items.tsv')):
    i, s = ln.split(); items[int(i)] = s
starts, ends = {}, {}
for ln in open(os.path.join(D, 'pool0/pool.log'), errors='replace'):
    m = re.match(r'(\d+) \[gpu-pool [^]]*\] (start|end) item (\d+) ', ln)
    if m: (starts if m.group(2) == 'start' else ends)[int(m.group(3))] = float(m.group(1))
meta = {}
for ln in open(os.path.join(D, 'pool0/samples.tsv')).read().splitlines()[1:]:
    f = ln.split('\t')
    if len(f) >= 15: meta[int(f[0])] = dict(pairs=float(f[13]) / 1e6, exit=f[11], s=float(f[4]), e=float(f[5]), wall=float(f[6]), maxrss=float(f[10]))
tot = collections.defaultdict(float)
for ln in open(os.path.join(D, 'pool0/memsamp.tsv')):
    f = ln.rstrip('\n').split('\t')
    if len(f) < 7 or int(f[1]) < 0: continue
    tot[(f[0], int(f[1]))] += (int(f[4]) + int(f[6])) / 1048576.0
rss = {}
for (ts, i), v in tot.items(): rss[i] = max(rss.get(i, 0), v)
first = min(starts.values())
base = float(np.median(anonshm[(t >= first - 6) & (t <= first - 1)]))
print('idle baseline before the first item (median anon+shmem of the 5 s before the start): %.2f GiB; sampler rows %d' % (base, len(t)))
out = []; prof = {}
for i in sorted(items):
    if i not in starts: continue
    s0 = starts[i]; e0 = ends.get(i, t[-1]); m_ = mk.get(i, {})
    sel = (t >= s0) & (t <= e0 + 1)
    L = anonshm[sel] - base
    at = lambda x: float(np.interp(x, t, anonshm)) - base
    true = float(L.max())
    pk_t = float(t[sel][np.argmax(L)] - s0)
    tt = t[sel]
    lv = at(m_['level']) if 'level' in m_ else None
    grow_l = float(L[tt >= m_['level']].max() - lv) if lv is not None else None
    grow_w = float(L[tt >= m_['waited']].max() - at(m_['waited'])) if 'waited' in m_ else None
    wt = at(m_['waited']) if 'waited' in m_ else None
    ex = at(e0)
    r = dict(item=i, sample=items[i], reads=meta[i]['pairs'], exit=meta[i]['exit'], est=61.8 + 0.2235 * meta[i]['pairs'], rss=rss.get(i, float('nan')), true=true, peak_at=pk_t,
             lvl_t=(m_['level'] - s0) if 'level' in m_ else None, lvl=lv, wt_t=(m_['waited'] - s0) if 'waited' in m_ else None, waited=wt, end_t=e0 - s0, exitlvl=ex, grow_after_level=grow_l, grow_after_waited=grow_w, wall=meta[i]['wall'])
    out.append(r)
    prof[items[i]] = dict(peak=true, t=[float(x) for x in (t[sel] - s0)], L=[float(x) for x in L], level=r['lvl_t'], waited=r['wt_t'], end=r['end_t'])
f = lambda x, fmt='%.1f': '-' if x is None else fmt % x
print('%-3s %-15s %9s %6s %7s %7s %6s %7s | %-18s %-18s %-14s | %6s %6s' % ('it', 'sample', 'reads(M)', 'est', 'RSSsum', 'TRUE', 'ratio', 'peak@s', 'level (t s, GiB)', 'waited (t s, GiB)', 'exit (t s,GiB)', 'g.lvl', 'g.wait'))
for r in out:
    print('%-3d %-15s %9.1f %6.1f %7.1f %7.1f %6.2f %7.0f | %-18s %-18s %-14s | %6s %6s%s' % (r['item'], r['sample'], r['reads'], r['est'], r['rss'], r['true'], r['rss'] / r['true'], r['peak_at'],
          '%s, %s' % (f(r['lvl_t'], '%.0f'), f(r['lvl'])), '%s, %s' % (f(r['wt_t'], '%.0f'), f(r['waited'])), '%.0f, %.1f' % (r['end_t'], r['exitlvl']), f(r['grow_after_level'], '%.0f'), f(r['grow_after_waited'], '%.0f'), '' if r['exit'] == '0' else '  EXIT ' + r['exit']))
ok = [r for r in out if r['exit'] == '0']
def fit(key, label):
    xs = np.array([r['reads'] for r in ok]); ys = np.array([r[key] for r in ok])
    b = ((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum(); a = ys.mean() - b * xs.mean()
    res = ys - (a + b * xs); w = int(np.argmax(res))
    print('fit %-7s n=%d: = %.2f + %.4f x reads(M); rmse %.1f GiB; worst under-prediction %.1f GiB (%s, %.1f M reads, measured %.1f); margin rule (5 up) %d' % (
        label, len(ok), a, b, math.sqrt((res ** 2).mean()), res[w], ok[w]['sample'], ok[w]['reads'], ok[w][key], math.ceil(res[w] / 5) * 5))
fit('true', 'TRUE'); fit('rss', 'RSS-sum')
# the pool's constants against the true peak
over = [(r['sample'], r['true'] - r['est']) for r in ok]
print('true peak minus the pool estimate (61.8 + 0.2235 x): ' + ', '.join('%s %+.0f' % o for o in over))
print('max over items of true / estimate: %.2f (%s)' % max((r['true'] / r['est'], r['sample']) for r in ok))
if len(sys.argv) > 2: json.dump(prof, open(sys.argv[2], 'w'))
if len(sys.argv) > 3:
    with open(sys.argv[3], 'w') as fo:
        ks = list(out[0].keys()); fo.write('\t'.join(ks) + '\n')
        for r in out: fo.write('\t'.join('' if r[k] is None else str(r[k]) for k in ks) + '\n')
