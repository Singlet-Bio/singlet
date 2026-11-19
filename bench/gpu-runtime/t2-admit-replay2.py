#!/usr/bin/env python3
"""peakmodel (throughput track 2): replay of a limit-aware memory-admission rule over the logged series of a mix run, next to the replay of today's gate (t2-admit-replay.py),
and the safety test of the rule's "what the items in flight can still allocate" term on the same series.

usage: t2-admit-replay2.py <rt-mix results dir> --pred predictor.json [--margin 7] [--baseline 60.0] [--limit 700] [--reserve 40] [--maxwait 1800] [--csv out.tsv] [--which B|A]

The directory needs: pool0/ pool1/ (pool.log, memsamp.tsv), pool0.marks / pool1.marks (per item "<n>.log <epoch of the [engine_gpu] level line> [ | <epoch of the [allsite] async: waited line> ["),
mem2.tsv (10 s: epoch MemFree-GB ...), mon.tsv (10 s: epoch cgroup-anon+shmem-GiB cgroup-memory.current-GiB ...), items.tsv, samples.all, results.txt, node_start.txt.

The rule (the one the orchestrator prescribes):
   free  = min( node rule as today = MemFree + share x (Cached - Shmem) ,  job rule = memory.max - memory.current + 0.5 x (file - shmem) )
   that  = max( 0 , sum over the items in flight of F_i  -  (cgroup anon+shmem now - idle baseline) )
   admit when  free - that - F0(new item) >= reserve (40), or when nothing else runs.
   F_i, the most the item can still hold (GiB above the idle baseline; predictor.json): F0 = P + m before its "[engine_gpu] level" line, F1 = min(F0, 0.9 P + 7.1 + m) from that line,
   F2 = min(F1, 0.5 P + 2.6 + m) from its "[allsite] async: waited" line (rules fitted by t2-peak-fit.py).
Limits of the replay (as t2-admit-replay.py): only the head's wait is re-timed; the in-flight sets, MemFree, the cgroup series and the other pool are those of the real run; the 10 s series are
interpolated; Cached - Shmem is modelled (not sampled); file - shmem of the cgroup is taken as memory.current - anon - shmem; a replayed wait that passes the real start is censored at the real
start (the series after it contains the head itself). Nothing here says what a run with the rule would have done.

Safety test (-a): at every second t of the run, with I(t) the items in flight, S(t) = baseline + sum F_i(t) and M(t) = the largest cgroup anon+shmem from t until the next item starts (so the
set only shrinks); a violation is S(t) < M(t). The test also says whether solo peaks add up in the mix."""
import sys, os, re, json, bisect, argparse, statistics, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('dir'); ap.add_argument('--pred', required=True)
ap.add_argument('--margin', type=float, default=7.0); ap.add_argument('--baseline', type=float, default=60.0)
ap.add_argument('--limit', type=float, default=700.0); ap.add_argument('--reserve', type=float, default=40.0)
ap.add_argument('--maxwait', type=float, default=1800.0); ap.add_argument('--stagger', type=float, default=40.0)
ap.add_argument('--workers', type=int, default=3); ap.add_argument('--csv'); ap.add_argument('--which', default='B')
ap.add_argument('--tail', type=float, default=80.0)
a = ap.parse_args()
D = a.dir
PR = json.load(open(a.pred)); PRED = PR['P_' + a.which]; R1 = PR['rules']['level']; R2 = PR['rules']['waited']
def rd(p): return open(os.path.join(D, p), errors='replace').read().splitlines()
sample_of = {}
for ln in rd('items.tsv'):
    f = ln.split('\t'); sample_of[(int(f[0]), int(f[1]))] = f[2]
pairs_of = {}
for ln in rd('samples.all')[1:]:
    f = ln.split('\t')
    if len(f) < 15 or not f[0].isdigit(): continue
    s = os.path.basename(f[2]).replace('.1fq', ''); pairs_of[s] = max(pairs_of.get(s, 0), float(f[13]) / 1e6)
def est_of(s): return 61.8 + 0.2235 * pairs_of[s]
mem2 = np.array([[float(x) for x in ln.split('\t')[:2]] for ln in rd('mem2.tsv') if ln.strip()])
mon = np.array([[float(x) for x in ln.split('\t')[:3]] for ln in rd('mon.tsv') if ln.strip()])
total = 0
for ln in rd('results.txt'):
    if ln.startswith('Mem:'): total = float(ln.split()[1]); break
def memfree(t): return float(np.interp(t, mem2[:, 0], mem2[:, 1]))
def anonshm(t): return float(np.interp(t, mon[:, 0], mon[:, 1]))
def cgcur(t): return float(np.interp(t, mon[:, 0], mon[:, 2]))

class Item: pass
pools = {}; waits = []
for p in (0, 1):
    items = {}; drops = collections.defaultdict(list)
    for ln in rd('pool%d/pool.log' % p):
        m = re.match(r'(\d+) \[gpu-pool [^]]*\] (.*)', ln)
        if not m: continue
        t = float(m.group(1)); s = m.group(2)
        mm = re.match(r'start item (\d+) ', s)
        if mm: items.setdefault(int(mm.group(1)), Item()).start = t; continue
        mm = re.match(r'end item (\d+) ', s)
        if mm: items.setdefault(int(mm.group(1)), Item()).end = t; continue
        mm = re.match(r'phase drop: item (\d+) charge (\d+) -> (\d+) GB \(holds (\d+) GB\)', s)
        if mm: drops[int(mm.group(1))].append((t, float(mm.group(3)))); continue
        mm = re.match(r'memory admission: item (\d+) waits \(estimate (\d+) GB; free \+ cache share (\d+) GB, own items still to allocate (\d+) GB, other pools (\d+) GB, reserve (\d+) GB; projected (-?\d+) GB\)', s)
        if mm:
            w = dict(pool=p, item=int(mm.group(1)), t=t, est=float(mm.group(2)), free=float(mm.group(3)), own=float(mm.group(4)), oth=float(mm.group(5)), reserve=float(mm.group(6)), proj=float(mm.group(7)), how='?')
            w['othwait'] = w['free'] - w['own'] - w['oth'] - w['est'] - w['proj']
            waits.append(w); continue
        mm = re.match(r'memory admission: item (\d+) admitted after (\d+) s', s)
        if mm:
            for w in waits:
                if w['pool'] == p and w['item'] == int(mm.group(1)) and w['how'] == '?': w['end'] = t; w['how'] = 'admitted'
            continue
        mm = re.match(r'memory admission: item (\d+) starts after (\d+) s', s)
        if mm:
            for w in waits:
                if w['pool'] == p and w['item'] == int(mm.group(1)) and w['how'] == '?': w['how'] = 'bounded'
            continue
    marks = {}
    for ln in rd('pool%d.marks' % p):
        m = re.match(r'(\d+)\.log (\d+) \[ \| (\d+)?', ln)
        if m: marks[int(m.group(1))] = (float(m.group(2)), float(m.group(3)) if m.group(3) else None)
    for i, it in items.items():
        it.level, it.waited = marks.get(i, (None, None))
        if not hasattr(it, 'end'): it.end = 1e18
    held = collections.defaultdict(lambda: ([], [])); tot = collections.defaultdict(float); shared = None
    for ln in open(os.path.join(D, 'pool%d/memsamp.tsv' % p)):
        f = ln.rstrip('\n').split('\t')
        if len(f) < 7: continue
        it = int(f[1])
        if it == -1 and shared is None: shared = int(f[4]) / 1048576.0
        if it < 0: continue
        tot[(f[0], it)] += (int(f[4]) + int(f[6])) / 1048576.0
    for (ts, it), v in sorted(tot.items(), key=lambda kv: float(kv[0][0])): held[it][0].append(float(ts)); held[it][1].append(v)
    pools[p] = dict(items=items, drops=drops, held=held, shared=shared or 17.6)
for w in waits:
    if w['how'] == 'bounded': w['end'] = w['t'] + a.maxwait
    w['sample'] = sample_of[(w['pool'], w['item'])]; w.setdefault('end', w['t'])

m_ = a.margin
def Pof(p, i): return PRED[sample_of[(p, i)]]
def F_item(p, i, t):
    it = pools[p]['items'][i]; P = Pof(p, i)
    F0 = P + m_; F1 = min(F0, R1['u'] * P + R1['v'] + m_); F2 = min(F1, R2['u'] * P + R2['v'] + m_)
    if it.waited is not None and t >= it.waited: return F2
    if it.level is not None and t >= it.level: return F1
    return F0
def inflight(p, t, excl=None): return [i for i, it in pools[p]['items'].items() if hasattr(it, 'start') and it.start <= t < it.end and (p, i) != excl]

def heldgb(p, i, t):
    ts, v = pools[p]['held'][i]
    return float(np.interp(t, ts, v)) if ts else 0.0
def charge(p, i, t):
    c = est_of(sample_of[(p, i)])
    for (td, nc) in pools[p]['drops'].get(i, []):
        if td <= t: c = nc
    return c
def own_rem_old(p, t, excl):
    return sum(max(0.0, charge(p, i, t) - max(0.0, heldgb(p, i, t) - pools[p]['shared'])) for i in inflight(p, t, excl))

# cache model (as t2-admit-replay.py): Cached - Shmem = MemTotal - MemFree - cgroup anon+shmem - K
Ks = []
for w in waits:
    cs = (w['free'] - memfree(w['t'])) / 0.25; w['cache_logged'] = cs
    Ks.append(total - memfree(w['t']) - anonshm(w['t']) - cs)
K = statistics.median(Ks)
def cache_sub(t): return max(0.0, total - memfree(t) - anonshm(t) - K)
def node_free(t, share): return memfree(t) + share * cache_sub(t)
def job_free(t): cur = cgcur(t); return a.limit - cur + 0.5 * max(0.0, cur - anonshm(t))
def that_new(t, excl):
    s = sum(F_item(p, i, t) for p in (0, 1) for i in inflight(p, t, excl))
    return max(0.0, s - (anonshm(t) - a.baseline))

def replay(w, mode, share):
    """mode: old (today's inequality) | new (the limit-aware rule). Returns (seconds waited, censored?)"""
    p = w['pool']; q = 1 - p; excl = (p, w['item'])
    est_new = Pof(p, w['item']) + m_
    t = w['t']; tmax = w['end']
    while t <= tmax + 0.5:
        n_in = len(inflight(p, t, excl)) + len(inflight(q, t, excl))
        slot = len(inflight(p, t, excl)) < a.workers
        started = [(it.start, sample_of[(p, i)]) for i, it in pools[p]['items'].items() if hasattr(it, 'start') and it.start <= t and (p, i) != excl]
        stag = bool(started) and len(inflight(p, t, excl)) > 0 and est_of(max(started)[1]) >= 100 and t - max(started)[0] < a.stagger
        if slot and not stag:
            if n_in == 0: return t - w['t'], False
            if mode == 'old':
                proj = node_free(t, share) - own_rem_old(p, t, excl) - own_rem_old(q, t, excl) - est_of(w['sample']) - w['othwait']
            else:
                proj = min(node_free(t, share), job_free(t)) - that_new(t, excl) - est_new
            if proj >= a.reserve: return t - w['t'], False
        if t - w['t'] >= a.maxwait: return a.maxwait, False
        t += 1.0
    return w['end'] - w['t'], True

rows = []
for w in sorted(waits, key=lambda x: x['t']):
    actual = w['end'] - w['t'] if w['how'] != 'bounded' else a.maxwait
    r = dict(pool=w['pool'], item=w['item'], sample=w['sample'], t=w['t'], actual=actual, how=w['how'], P=Pof(w['pool'], w['item']))
    r['base'], _ = replay(w, 'old', 0.25)
    r['cache1'], _ = replay(w, 'old', 1.0)
    r['new25'], r['c25'] = replay(w, 'new', 0.25)
    r['new100'], r['c100'] = replay(w, 'new', 1.0)
    rows.append(r)
print('run %s: %d waits; MemTotal %.0f GB; limit %.0f GiB, baseline %.1f GiB, margin %.1f, reserve %.0f; cache model K = %.1f GB' % (D, len(rows), total, a.limit, a.baseline, m_, a.reserve, K))
print('%-4s %-5s %-14s %6s | %7s %7s | %7s %7s %7s %7s' % ('pool', 'item', 'sample', 'P', 'actual', 'base', 'cache1', 'new.25', 'new1.0', 'cens'))
for r in rows:
    print('%-4d %-5d %-14s %6.1f | %7.0f %7.0f | %7.0f %7.0f %7.0f %s' % (r['pool'], r['item'], r['sample'], r['P'], r['actual'], r['base'], r['cache1'], r['new25'], r['new100'], ('%s%s' % ('*' if r['c25'] else '', '*' if r['c100'] else ''))))
def tot_(k): return sum(r[k] for r in rows)
print('TOTAL wait: real %.0f s; replayed baseline %.0f s; old gate with cache share 1.0 %.0f s; NEW rule, node term at cache share 0.25: %.0f s; at 1.0: %.0f s' % (tot_('actual'), tot_('base'), tot_('cache1'), tot_('new25'), tot_('new100')))
print('  waits under 10 s: baseline %d, new .25 %d, new 1.0 %d of %d; censored at the real start: new .25 %d, new 1.0 %d' % (
    sum(r['base'] < 10 for r in rows), sum(r['new25'] < 10 for r in rows), sum(r['new100'] < 10 for r in rows), len(rows), sum(r['c25'] for r in rows), sum(r['c100'] for r in rows)))

# ---- safety test: baseline + sum F_i(t) against the later maximum of the cgroup's anon+shmem, until the next start
starts_all = sorted(it.start for p in (0, 1) for it in pools[p]['items'].values() if hasattr(it, 'start'))
t0 = max(starts_all[0], mon[0, 0]); t1 = min(max(it.end for p in (0, 1) for it in pools[p]['items'].values() if it.end < 1e17), mon[-1, 0])
ts = np.arange(t0, t1, 1.0)
cg = np.array([anonshm(x) for x in ts])
nxt = np.searchsorted(np.array(starts_all), ts, side='right')   # index of the next start after t
viol = []; nchk = 0; worst = (0, None)
for k, x in enumerate(ts):
    I = [(p, i) for p in (0, 1) for i in inflight(p, x)]
    if not I: continue
    S = a.baseline + sum(F_item(p, i, x) for p, i in I)
    tn = starts_all[nxt[k]] if nxt[k] < len(starts_all) else t1
    Mx = cg[k:int(min(len(ts), max(k + 1, tn - t0)))].max() if tn > x else cg[k]
    nchk += 1
    if Mx > S + 1e-9: viol.append((x - t0, Mx - S, S, Mx, [sample_of[c] for c in I]))
print('SAFETY (%s): %d seconds checked, %d with baseline + sum F_i below the later maximum (until the next start)' % (D, nchk, len(viol)))
if viol:
    sizes = np.array([v[1] for v in viol]); print('  size of the violations (GiB): max %.1f, median %.1f; seconds %d' % (sizes.max(), np.median(sizes), len(sizes)))
    # group consecutive seconds into episodes
    eps = []; cur = [viol[0]]
    for v in viol[1:]:
        if v[0] - cur[-1][0] <= 1.5: cur.append(v)
        else: eps.append(cur); cur = [v]
    eps.append(cur)
    print('  %d episodes; the 15 largest (start s, length s, max size GiB, S, M, items in flight):' % len(eps))
    for e in sorted(eps, key=lambda e: -max(v[1] for v in e))[:15]:
        b = max(e, key=lambda v: v[1]); print('   +%6.0f %4d s  %6.1f  S %6.1f M %6.1f  %s' % (e[0][0], len(e), b[1], b[2], b[3], ' '.join(b[4])))
# do solo peaks add up? at the second of the cgroup's maximum: the items in flight, baseline + their sum of F0 and of the predictor P
kmax = int(np.argmax(cg)); xm = ts[kmax]
Im = [(p, i) for p in (0, 1) for i in inflight(p, xm)]
print('PEAK of the run: cgroup anon+shmem %.1f GiB at +%.0f s with %d items in flight: %s; baseline + sum P = %.1f, baseline + sum F (phases) = %.1f' % (
    cg[kmax], xm - t0, len(Im), ' '.join('%s(%s)' % (sample_of[c], 'F0' if F_item(*c, xm) == Pof(*c) + m_ else 'F1/2') for c in Im), a.baseline + sum(Pof(*c) for c in Im), a.baseline + sum(F_item(*c, xm) for c in Im)))
if a.csv:
    with open(a.csv, 'w') as f:
        ks = list(rows[0].keys()); f.write('\t'.join(ks) + '\n')
        for r in rows: f.write('\t'.join(str(r[k]) for k in ks) + '\n')
