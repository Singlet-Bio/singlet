#!/usr/bin/env python3
"""pooladmit (throughput track 2): replay of the gpu-pool's memory-admission gate over the logged series of a mix run.

usage: t2-admit-replay.py <rt-mix results dir> [--true true_profiles.json] [--csv out.tsv] [--maxwait 1800]

Needs in the directory: pool0/ pool1/ (pool.log, memsamp.tsv), mem2.tsv (10 s: epoch MemFree-GB pressure sys compact), mon.tsv (10 s: epoch anon+shmem-GB
memory.current-GB ...), items.tsv, samples.all, results.txt (the "Mem:" line of free -g), node_start.txt.

What it does. For every "memory admission: item N waits" line of both pools it takes the head item's wait window [line, "admitted"/"starts" line] and
re-evaluates, once a second inside that window, the inequality of gpu_pool.cpp gateTry
    proj = freeEff - ownRem - othRem - est - othWait >= reserve (40 GB)     (or nothing else in flight on the node)
together with the two other conditions that make the head wait: a free worker slot, and the 40 s stagger after the start of an item of 100 GB or more.
 * freeEff = MemFree + share * (Cached - Shmem). MemFree comes from mem2.tsv (10 s, linear interpolation). Cached - Shmem is NOT in any sample of these runs:
   it is modelled as MemTotal - MemFree - (cgroup anon+shmem) - K, K = the median over the logged wait lines of what is left when the logged
   "free + cache share" figure is solved for Cached - Shmem (share 0.25). The residuals of that model at the logged lines are printed.
 * ownRem / othRem = sum over the items in flight of max(0, charge - held): charge = estimate (61.8 + 0.2235 x M pairs; = the logged estimates), lowered at the
   times of the "phase drop" lines to the logged new charge; held = RssAnon+RssShmem of the item's processes (memsamp.tsv, 2 s) minus the pool's own RssAnon (what
   the pool subtracts as sharedGb).
 * variants: cache1 = cache share 1.0 instead of 0.25 (all of Cached - Shmem counted free); true = every item charged its TRUE figure (--true: the solo,
   one-second cgroup profile of its sample class, anon+shmem minus idle baseline): the head's estimate becomes the true peak of its class, an item in flight
   is charged (maximum of its true profile from now on) - (its true level now), at the point of its solo profile that its own phases map to (start, level
   line = first phase drop, end; piecewise linear); both = both changes.
It does NOT simulate knock-on effects: the in-flight sets, the free memory, the other pool and the items that started ahead of the head are those of the
real run; only the head's wait is re-timed. A counterfactual wait end earlier than the real one would have changed everything after it.
"""
import sys, os, re, json, bisect, argparse, statistics, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('--true')
ap.add_argument('--csv')
ap.add_argument('--maxwait', type=float, default=1800)
ap.add_argument('--reserve', type=float, default=40)
ap.add_argument('--stagger', type=float, default=40)
ap.add_argument('--tail', type=float, default=80)
ap.add_argument('--workers', type=int, default=3)
a = ap.parse_args()
D = a.dir
TRUE = json.load(open(a.true)) if a.true else None

def rd(p): return open(os.path.join(D, p), errors='replace').read().splitlines()

# ---- items.tsv: pool j sample
sample_of = {}
for ln in rd('items.tsv'):
    f = ln.split('\t'); sample_of[(int(f[0]), int(f[1]))] = f[2]
pairs_of = {}
for ln in rd('samples.all')[1:]:
    f = ln.split('\t')
    if len(f) < 15 or not f[0].isdigit(): continue
    s = os.path.basename(f[2]).replace('.1fq', '')
    pairs_of[s] = max(pairs_of.get(s, 0), float(f[13]) / 1e6)
def est_of(s): return 61.8 + 0.2235 * pairs_of[s]

mem2 = np.array([[float(x) for x in ln.split('\t')[:2]] for ln in rd('mem2.tsv') if ln.strip()])
mon = np.array([[float(x) for x in ln.split('\t')[:3]] for ln in rd('mon.tsv') if ln.strip()])
total = 0
for ln in rd('results.txt'):
    if ln.startswith('Mem:'): total = float(ln.split()[1]); break
def memfree(t): return float(np.interp(t, mem2[:, 0], mem2[:, 1]))
def anonshm(t): return float(np.interp(t, mon[:, 0], mon[:, 1]))
def cgcur(t): return float(np.interp(t, mon[:, 0], mon[:, 2]))

# ---- pool logs
class Item: pass
pools = {}
waits = []
for p in (0, 1):
    items = {}
    drops = collections.defaultdict(list)
    for ln in rd('pool%d/pool.log' % p):
        m = re.match(r'(\d+) \[gpu-pool [^]]*\] (.*)', ln)
        if not m: continue
        t = float(m.group(1)); s = m.group(2)
        mm = re.match(r'start item (\d+) ', s)
        if mm: it = items.setdefault(int(mm.group(1)), Item()); it.start = t; continue
        mm = re.match(r'end item (\d+) ', s)
        if mm: items.setdefault(int(mm.group(1)), Item()).end = t; continue
        mm = re.match(r'phase drop: item (\d+) charge (\d+) -> (\d+) GB \(holds (\d+) GB\)', s)
        if mm: drops[int(mm.group(1))].append((t, float(mm.group(3)))); continue
        mm = re.match(r'memory admission: item (\d+) waits \(estimate (\d+) GB; free \+ cache share (\d+) GB, own items still to allocate (\d+) GB, other pools (\d+) GB, reserve (\d+) GB; projected (-?\d+) GB\)', s)
        if mm:
            w = dict(pool=p, item=int(mm.group(1)), t=t, est=float(mm.group(2)), free=float(mm.group(3)), own=float(mm.group(4)), oth=float(mm.group(5)),
                     reserve=float(mm.group(6)), proj=float(mm.group(7)), how='?')
            w['othwait'] = w['free'] - w['own'] - w['oth'] - w['est'] - w['proj']
            waits.append(w); continue
        mm = re.match(r'memory admission: item (\d+) admitted after (\d+) s', s)
        if mm:
            for w in waits:
                if w['pool'] == p and w['item'] == int(mm.group(1)) and w['how'] == '?': w['end'] = t; w['how'] = 'admitted'; w['waited'] = float(mm.group(2))
            continue
        mm = re.match(r'memory admission: item (\d+) starts after (\d+) s', s)
        if mm:
            for w in waits:
                if w['pool'] == p and w['item'] == int(mm.group(1)) and w['how'] == '?': w['how'] = 'bounded'
            continue
    # memsamp: (ts, item) -> GB
    held = collections.defaultdict(lambda: ([], []))
    tot = collections.defaultdict(float); shared = None
    for ln in open(os.path.join(D, 'pool%d/memsamp.tsv' % p)):
        f = ln.rstrip('\n').split('\t')
        if len(f) < 7: continue
        it = int(f[1])
        if it == -1 and shared is None: shared = int(f[4]) / 1048576.0
        if it < 0: continue
        tot[(f[0], it)] += (int(f[4]) + int(f[6])) / 1048576.0
    for (ts, it), v in sorted(tot.items(), key=lambda kv: float(kv[0][0])):
        held[it][0].append(float(ts)); held[it][1].append(v)
    pools[p] = dict(items=items, drops=drops, held=held, shared=shared or 17.6)
    for it in items.values():
        if not hasattr(it, 'end'): it.end = 1e18
for w in waits:
    if w['how'] == 'bounded': w['end'] = w['t'] + a.maxwait
    w['sample'] = sample_of[(w['pool'], w['item'])]
    w.setdefault('end', w['t'])

def heldgb(p, i, t):
    ts, v = pools[p]['held'][i]
    if not ts: return 0.0
    return float(np.interp(t, ts, v))

def charge(p, i, t):
    it = pools[p]['items'][i]; c = est_of(sample_of[(p, i)])
    for (td, nc) in pools[p]['drops'].get(i, []):
        if td <= t: c = nc
    return c

# ---- true profiles
def warp(p, i, t):
    """time in the solo profile of item i's class that corresponds to wall time t of the mix run"""
    it = pools[p]['items'][i]; s = sample_of[(p, i)]; pr = TRUE[s]
    xs = [it.start]; ys = [0.0]
    dr = pools[p]['drops'].get(i, [])
    if dr and pr.get('level') is not None: xs.append(dr[0][0]); ys.append(pr['level'])
    if pr.get('end') is not None: xs.append(it.end if it.end < 1e17 else it.start + pr['end']); ys.append(pr['end'])
    xs2, ys2 = [xs[0]], [ys[0]]
    for x, y in zip(xs[1:], ys[1:]):
        if x > xs2[-1] and y > ys2[-1]: xs2.append(x); ys2.append(y)
    return float(np.interp(t, xs2, ys2))
def true_rem(p, i, t):
    s = sample_of[(p, i)]; pr = TRUE[s]
    ts = np.array(pr['t']); L = np.array(pr['L'])
    u = warp(p, i, t)
    cur = float(np.interp(u, ts, L))
    fut = float(max(L[ts >= u].max() if (ts >= u).any() else cur, cur))
    return max(0.0, fut - cur)

EXCL = None   # (pool, item) of the head being replayed: it is not in flight while it waits
def inflight(p, t): return [i for i, it in pools[p]['items'].items() if hasattr(it, 'start') and it.start <= t < it.end and (p, i) != EXCL]
def own_rem(p, t, true):
    r = 0.0
    for i in inflight(p, t):
        if true: r += true_rem(p, i, t)
        else: r += max(0.0, charge(p, i, t) - max(0.0, heldgb(p, i, t) - pools[p]['shared']))
    return r

# ---- cache model
Ks = []
for w in waits:
    cs = (w['free'] - memfree(w['t'])) / 0.25
    w['cache_logged'] = cs
    Ks.append(total - memfree(w['t']) - anonshm(w['t']) - cs)
K = statistics.median(Ks)
def cache_sub(t): return max(0.0, total - memfree(t) - anonshm(t) - K)
def free_eff(t, share): return memfree(t) + share * cache_sub(t)

res = []
for w in waits:
    cs_model = cache_sub(w['t'])
    w['cache_model'] = cs_model
    w['free_model'] = free_eff(w['t'], 0.25)

def replay(w, share, true):
    global EXCL
    EXCL = (w['pool'], w['item'])
    p = w['pool']; q = 1 - p
    est = TRUE[w['sample']]['peak'] if true else w['est']
    t = w['t']
    tmax = w['t'] + max(a.maxwait, w['end'] - w['t']) + 60
    while t <= tmax:
        inf_all = len(inflight(p, t)) + len(inflight(q, t))
        ok = False
        # slot and stagger
        slot = len(inflight(p, t)) < a.workers
        started = [(it.start, sample_of[(p, i)]) for i, it in pools[p]['items'].items() if hasattr(it, 'start') and it.start <= t and (p, i) != EXCL]
        stag = False
        if started and len(inflight(p, t)) > 0:
            last = max(started)
            stag = est_of(last[1]) >= 100 and t - last[0] < a.stagger
        if slot and not stag:
            if inf_all == 0: ok = True
            else:
                proj = free_eff(t, share) - own_rem(p, t, true) - own_rem(q, t, true) - est - w['othwait']
                ok = proj >= a.reserve
        if ok: return t - w['t']
        if t - w['t'] >= a.maxwait: return a.maxwait
        t += 1.0
    return None

hdr = ['pool', 'item', 'sample', 'wall_t0', 'est', 'free', 'own', 'oth', 'reserve', 'proj', 'MemFree', 'cgAnonShm', 'cgCur', 'cache_logged', 'cache_model', 'free_model',
       'actual_wait', 'how', 'base', 'cache1', 'true', 'both']
rows = []
t0 = min(w['t'] for w in waits) if waits else 0
for w in sorted(waits, key=lambda x: x['t']):
    actual = w['end'] - w['t'] if w['how'] != 'bounded' else a.maxwait
    row = dict(pool=w['pool'], item=w['item'], sample=w['sample'], t=w['t'] - 0, est=w['est'], free=w['free'], own=w['own'], oth=w['oth'], reserve=w['reserve'], proj=w['proj'],
               MemFree=memfree(w['t']), cgAnonShm=anonshm(w['t']), cgCur=cgcur(w['t']), cache_logged=w['cache_logged'], cache_model=w['cache_model'], free_model=w['free_model'],
               actual=actual, how=w['how'])
    row['base'] = replay(w, 0.25, False)
    row['cache1'] = replay(w, 1.0, False)
    if TRUE:
        row['true'] = replay(w, 0.25, True)
        row['both'] = replay(w, 1.0, True)
    # the gate at the first moment (the logged line) under each variant, from the logged figures (cache variant uses the modelled Cached - Shmem)
    c1 = w['proj'] + 0.75 * w['cache_logged']
    row['proj_cache1_first'] = c1
    if TRUE:
        ptrue = w['proj'] + w['est'] - TRUE[w['sample']]['peak']
        row['proj_true_first_est_only'] = ptrue
    rows.append(row)

print('run %s: %d waits (%s); MemTotal %.0f GB; cache model K = %.1f GB (median; spread %.1f..%.1f over the lines)' % (
    D, len(rows), ', '.join('pool%d %d' % (p, sum(1 for r in rows if r['pool'] == p)) for p in (0, 1)), total, K, min(Ks), max(Ks)))
resid = [w['free_model'] - w['free'] for w in waits]
print('model residual of free + cache share at the logged lines (model - logged), GB: mean %.1f, abs max %.1f' % (statistics.mean(resid), max(abs(x) for x in resid)))
cols = ['pool', 'item', 'sample', 'actual', 'base', 'cache1'] + (['true', 'both'] if TRUE else [])
print('%-4s %-5s %-14s %5s %6s %6s %6s %6s %5s %5s %5s | %7s %6s | %6s %7s %7s | %s' % ('pool', 'item', 'sample', 'est', 'free', 'own', 'oth', 'proj', 'MemF', 'cgAS', 'cgCur', 'cacheLg', 'cacheM', 'actual', 'base', 'cache1', 'true  both' if TRUE else ''))
for r in rows:
    f = lambda x: '%7s' % ('-' if x is None else '%.0f' % x)
    print('%-4d %-5d %-14s %5.0f %6.0f %6.0f %6.0f %6.0f %5.0f %5.0f %5.0f | %7.0f %6.0f | %6.0f %s %s %s' % (
        r['pool'], r['item'], r['sample'], r['est'], r['free'], r['own'], r['oth'], r['proj'], r['MemFree'], r['cgAnonShm'], r['cgCur'], r['cache_logged'], r['cache_model'],
        r['actual'], f(r['base']), f(r['cache1']), (f(r.get('true')) + f(r.get('both'))) if TRUE else ''))

# ---- (i) the page cache while items waited: node total (modelled / from the logged figure), the job's part (memory.current - anon+shmem = file cache without
# shmem, plus a little kernel memory), the rest (not the job's: it was there at the start, node_start.txt, and was not reclaimed)
print('page cache at the logged wait lines (GiB; Cached - Shmem from the logged "free + cache share" and MemFree, 0.25 share): node total / job cgroup (memory.current - anon - shmem) / not the job')
def q(x): return '%.0f..%.0f (median %.0f)' % (min(x), max(x), statistics.median(x))
nodec = [r['cache_logged'] for r in rows]; jobc = [r['cgCur'] - r['cgAnonShm'] for r in rows]; nonj = [n - j for n, j in zip(nodec, jobc)]
print('  node %s; job %s; not the job %s' % (q(nodec), q(jobc), q(nonj)))
print('  node cache as a share of MemTotal %.0f..%.0f %%; cgroup anon+shmem %s; MemFree %s' % (100 * min(nodec) / total, 100 * max(nodec) / total, q([r['cgAnonShm'] for r in rows]), q([r['MemFree'] for r in rows])))
print('  node start: ' + ' '.join(rd('node_start.txt')[0].split()[3:]) )
def tot_(k): return sum((r[k] if r[k] is not None else 0) for r in rows)
print('TOTAL actual wait %.0f s; replayed baseline %.0f s; cache1 %.0f s (saves %.0f vs replayed baseline)' % (tot_('actual'), tot_('base'), tot_('cache1'), tot_('base') - tot_('cache1')))
if TRUE: print('       true %.0f s (saves %.0f); both %.0f s (saves %.0f)' % (tot_('true'), tot_('base') - tot_('true'), tot_('both'), tot_('base') - tot_('both')))
if a.csv:
    with open(a.csv, 'w') as f:
        ks = list(rows[0].keys())
        f.write('\t'.join(ks) + '\n')
        for r in rows: f.write('\t'.join('' if r[k] is None else str(r[k]) for k in ks) + '\n')
