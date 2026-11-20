#!/usr/bin/env python3
"""pooladmit loss A: what the gate counted against what the job's cgroup had.
usage: t2-pooladmit-cg.py <results dir of t2-pooladmit-cg.sbatch: cg1hz.tsv, pool0/pool.log, pool0/memsamp.tsv>
For every "memory admission: item N waits" line of the pool: its `free + cache share` figure next to the cgroup headroom (memory.max - memory.current) at that second.
When the pool logged no wait (the gate admitted every start at once, nothing is logged for an admission), the gate's inequality is evaluated at every start from the
1 Hz series: freeEff = MemFree + 0.25 (Cached - Shmem) [gateTry], own = sum over the items in flight of max(0, charge - (RssAnon+RssShmem of the item - the pool's RssAnon)),
against the same with the headroom of avail_gb() (limit - current + 0.5 (file - shmem)), the figure the pool already prints in its stagger line."""
import sys, os, re
import numpy as np
D = sys.argv[1]; G = 1073741824.0
rows = [l.rstrip('\n').split('\t') for l in open(os.path.join(D, 'cg1hz.tsv')) if not l.startswith('#')]
A = np.array([[float(x) for x in r[:9]] for r in rows])
t, mx, cur, an, fi, sh, mf, ca, shm = A.T
ev = sorted(set(r[9] for r in rows))
def at(x, arr): return float(np.interp(x, t, arr))
est = lambda pairs: 61.8 + 0.2235 * pairs / 1e6
print('cgroup memory.max %.1f GiB; samples %d (%.0f s); memory.events (max oom oom_kill) values seen: %s' % (mx[0] / G, len(t), t[-1] - t[0], ev))
print('peaks (GiB): memory.current %.1f, anon+shmem %.1f, anon %.1f, file %.1f; minimum headroom max-current %.1f GiB at +%.0f s; node MemFree min %.1f GB, Cached max %.1f GB' % (
    cur.max() / G, (an + sh).max() / G, an.max() / G, fi.max() / G, (mx - cur).min() / G, t[np.argmin(mx - cur)] - t[0], mf.min() / 1048576, ca.max() / 1048576))
# pool log
starts = {}; ends = {}; drops = {}; waits = []; admit_lines = []
for ln in open(os.path.join(D, 'pool0/pool.log'), errors='replace'):
    m = re.match(r'(\d+) \[gpu-pool [^]]*\] (.*)', ln)
    if not m: continue
    ts = float(m.group(1)); s = m.group(2)
    mm = re.match(r'start item (\d+) ', s)
    if mm: starts[int(mm.group(1))] = ts
    mm = re.match(r'end item (\d+) ', s)
    if mm: ends[int(mm.group(1))] = ts
    mm = re.match(r'phase drop: item (\d+) charge (\d+) -> (\d+)', s)
    if mm: drops.setdefault(int(mm.group(1)), []).append((ts, float(mm.group(3))))
    if 'memory admission: item' in s and 'waits' in s: waits.append((ts, s))
    if s.startswith('admission:'): admit_lines.append((ts, s))
print('"memory admission: item N waits" lines: %d' % len(waits))
for ts, s in waits:
    mm = re.search(r'free \+ cache share (\d+) GB', s)
    hr = (at(ts, mx) - at(ts, cur)) / G
    hre = (at(ts, mx) - at(ts, cur) + 0.5 * max(0, at(ts, fi) - at(ts, sh))) / G
    print('  +%.0f s: %s | cgroup headroom max-current %.0f GiB, avail_gb rule %.0f GiB' % (ts - t[0], s[:160], hr, hre))
# memsamp per item
ms = {}; shared = None
for ln in open(os.path.join(D, 'pool0/memsamp.tsv')):
    f = ln.rstrip('\n').split('\t')
    if len(f) < 7: continue
    it = int(f[1])
    if it == -1 and shared is None: shared = int(f[4]) / 1048576.0
    if it < 0: continue
    ms.setdefault(it, {}); ms[it][f[0]] = ms[it].get(f[0], 0) + (int(f[4]) + int(f[6])) / 1048576.0
held = {i: (np.array(sorted(float(k) for k in d)), np.array([d[k] for k in sorted(d, key=float)])) for i, d in ms.items()}
pairs = 300223349
print('pool RssAnon (sharedGb) %.1f GB; estimate per item %.0f GB; per-item RSS-sum peak (pool definition): %s' % (shared, est(pairs), ', '.join('item %d %.1f' % (i, v[1].max()) for i, v in sorted(held.items()))))
def own(tt, excl=None):
    r = 0
    for i, ts in starts.items():
        if i == excl or ts > tt or ends.get(i, 1e18) <= tt: continue
        c = est(pairs)
        for td, nc in drops.get(i, []):
            if td <= tt: c = nc
        h = float(np.interp(tt, *held[i])) if i in held else 0
        r += max(0, c - max(0, h - shared))
    return r
print('gate at every start (1 Hz series; estimate 129; reserve 40):')
print('  item +s  MemFree  Cached-Shm  freeEff | own(others)  proj=freeEff-own-est | cg max-current  cg avail_gb-rule  | would a cgroup-aware gate admit: min(freeEff, avail_gb) - own - est')
for i in sorted(starts):
    ts = starts[i] - 1   # the second before the start: the gate saw this state
    fe = (at(ts, mf) + 0.25 * max(0, at(ts, ca) - at(ts, shm))) / 1048576.0
    o = own(ts, i)
    pj = fe - o - est(pairs)
    hr = (at(ts, mx) - at(ts, cur)) / G
    hre = (at(ts, mx) - at(ts, cur) + 0.5 * max(0, at(ts, fi) - at(ts, sh))) / G
    alone = not any(starts[k] <= ts < ends.get(k, 1e18) for k in starts if k != i)
    print('  %d  +%3.0f  %7.1f  %9.1f  %7.1f | %6.1f  %7.1f%s | %7.1f  %7.1f | %7.1f' % (i, starts[i] - t[0], at(ts, mf) / 1048576, max(0, at(ts, ca) - at(ts, shm)) / 1048576, fe, o, pj, ' (alone)' if alone else '', hr, hre, min(fe, hre) - o - est(pairs)))
# what the cgroup had at the peak
k = int(np.argmax(cur))
print('at the peak of memory.current (+%.0f s): current %.1f GiB = anon %.1f + file %.1f + (shmem %.1f inside both); MemFree %.1f GB' % (t[k] - t[0], cur[k] / G, an[k] / G, fi[k] / G, sh[k] / G, mf[k] / 1048576))
tot_in_flight_est = 3 * est(pairs)
print('sum of the three estimates %.0f (pool GB = GiB) against memory.max %.0f GiB; true anon+shmem peak %.1f GiB' % (tot_in_flight_est, mx[0] / G, (an + sh).max() / G))
