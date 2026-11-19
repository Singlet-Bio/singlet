#!/usr/bin/env python3
"""peakmodel (throughput track 2): offline tests of the limit-aware admission rule on the single-job recordings, and the prediction of what today's gate lets through on a 365 GiB lane.

usage: t2-peak-gate-tests.py <profiles.json> <predictor.json> <dir 818938> <dir 818937> <dir 818940> [--margin 12] [--which a|b|c|abc] [--fe0 450] [--limit 365] [--json triples.json]

(a) Safety on the recordings 818938 (one item at a time), 818937 and 818940 (three d300_3p, 200 and 150 GiB jobs): at every second, with the items in flight I(t),
    S(t) = idle baseline + sum F_i(t) against M(t) = the largest cgroup anon+shmem from t until the next item starts. Violations are listed with their size.
(b) The rule at the real starts of 818937 / 818940 (free, that, P(new), reserve) and the counterfactual start times when the rule is applied to the same queue, with the cgroup
    series rebuilt from the solo profile of d300_3p (baseline + sum of solo levels; MemFree, memory.current - anon - shmem taken from the recording).
(c) A PREDICTION from solo profiles, not a measurement: the triples of the 18 classes that today's gate (gpu_pool.cpp gateTry: free = MemFree + 0.25 (Cached - Shmem); admit when
    free - own items still to allocate - estimate >= 40; estimate 61.8 + 0.2235 reads(M); charge x1.0 in the front half, x0.4 after the level line, min(that, 80) after the waited line;
    own = max(0, charge - held), held = the pool's RSS sum minus its genome; the stagger of 40 s after the start of an item of estimate >= 100) would admit together on a node with
    free + cgroup anon+shmem = FE0 (default 450 GiB: 818937/818940 measured 444-455 at the starts), with three workers, and that exceed the lane's limit (default 365 GiB) by the
    overlay of the solo true profiles (baseline + sum of the three levels at their real offsets)."""
import sys, os, re, json, itertools, math, collections
import numpy as np
G = 1073741824.0
args = [x for x in sys.argv[1:] if not x.startswith('--')]
def opt(name, d):
    return type(d)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else d
M_ = opt('--margin', 12.0); WHICH = opt('--which', 'abc'); FE0 = opt('--fe0', 450.0); LIMIT = opt('--limit', 365.0)
prof = json.load(open(args[0])); C = prof['classes']; BASE = prof['baseline']
PR = json.load(open(args[1])); PRED = PR['P_B']; R1 = PR['rules']['level']; R2 = PR['rules']['waited']
D938, D937, D940 = args[2], args[3], args[4]
def F(P, phase):   # phase 0: before the level line, 1: after it, 2: after the waited line
    F0 = P + M_; F1 = min(F0, R1['u'] * P + R1['v'] + M_); F2 = min(F1, R2['u'] * P + R2['v'] + M_)
    return (F0, F1, F2)[phase]
def phase_of(p, t):
    if p['waited'] is not None and t >= p['waited']: return 2
    if p['level'] is not None and t >= p['level']: return 1
    return 0
def load_cg(d):
    rows = [l.rstrip('\n').split('\t') for l in open(os.path.join(d, 'cg1hz.tsv')) if not l.startswith('#')]
    A = np.array([[float(x) if x not in ('max',) else 1e18 for x in r[:9]] for r in rows])
    return A
def pool_log(d):
    st, en = {}, {}
    for ln in open(os.path.join(d, 'pool0/pool.log'), errors='replace'):
        m = re.match(r'(\d+) \[gpu-pool [^]]*\] (start|end) item (\d+) ', ln)
        if m: (st if m.group(2) == 'start' else en)[int(m.group(3))] = float(m.group(1))
    return st, en
def marks(d):
    mk = collections.defaultdict(dict)
    for ln in open(os.path.join(d, 'markers.tsv')):
        f = ln.split()
        if len(f) == 3 and f[1].isdigit(): mk[int(f[1])][f[2]] = float(f[0])
    return mk

def safety(name, A, items, base):
    """items: list of (class, start, end, level_epoch, waited_epoch)"""
    t = A[:, 0]; cg = (A[:, 3] + A[:, 5]) / G
    starts = sorted(i[1] for i in items)
    viol = []; n = 0
    for k, x in enumerate(t):
        I = [i for i in items if i[1] <= x < i[2]]
        if not I: continue
        S = base + sum(F(PRED[c], 2 if (w and x >= w) else 1 if (l and x >= l) else 0) for c, s, e, l, w in I)
        nxt = [s for s in starts if s > x]
        tn = nxt[0] if nxt else t[-1] + 1
        Mx = cg[(t >= x) & (t < tn)].max(); n += 1
        if Mx > S: viol.append((x - t[0], Mx - S, S, Mx, [c for c, *_ in I]))
    print('(a) %s: %d seconds checked, %d with idle baseline %.1f + sum F_i below the later maximum' % (name, n, len(viol), base))
    if viol:
        v = max(viol, key=lambda z: z[1]); print('    largest: +%.0f s, %.1f GiB (S %.1f, M %.1f, in flight %s)' % (v[0], v[1], v[2], v[3], ' '.join(v[4])))
    return viol

if 'a' in WHICH:
    A = load_cg(D938); st, en = pool_log(D938); mk = marks(D938)
    names = {int(l.split()[0]): l.split()[1] for l in open(os.path.join(D938, 'items.tsv'))}
    items = [(names[i], st[i], en[i], mk[i].get('level'), mk[i].get('waited')) for i in st]
    safety('818938 (18 items one at a time)', A, items, BASE)
    for lab, d in (('818937 (3 x d300_3p, 200 GiB)', D937), ('818940 (3 x d300_3p, 150 GiB)', D940)):
        A = load_cg(d); st, en = pool_log(d); mk = marks(d)
        items = [('d300_3p', st[i], en[i], mk[i].get('level'), mk[i].get('waited')) for i in st]
        safety(lab, A, items, BASE)

# ---------- (b)
if 'b' in WHICH:
    for lab, d in (('818937 (--mem=200G)', D937), ('818940 (--mem=150G)', D940)):
        A = load_cg(d); t = A[:, 0]; st, en = pool_log(d); mk = marks(d)
        s0 = {i: st[i] for i in st}; Pn = PRED['d300_3p']; solo = C['d300_3p']
        lim = A[0, 1] / G
        def state(x, real=True, starts=None):
            k = int(np.argmin(abs(t - x))); r = A[k]
            memfree = r[6] / 1048576; cached = r[7] / 1048576; shm = r[8] / 1048576
            anon_real = (r[3] + r[5]) / G; cur_real = r[2] / G
            if real: return memfree, cached, shm, anon_real, cur_real, (r[4] - r[5]) / G
            # counterfactual series: baseline + solo levels at the counterfactual starts; the file cache part (current - anon - shmem) from the recording
            lev = sum(float(np.interp(x - s, solo['t'], solo['L'], left=0, right=0)) for s in starts)
            anon_cf = BASE + lev
            return memfree + (anon_real - anon_cf), cached, shm, anon_cf, cur_real - anon_real + anon_cf, (r[4] - r[5]) / G
        def evaluate(x, started, real=True, startlist=None):
            memfree, cached, shm, anon, cur, file_shm = state(x, real, startlist)
            node = memfree + 0.25 * max(0.0, cached - shm)
            job = lim - cur + 0.5 * file_shm
            fsum = 0.0
            for i in started:
                p = solo; fsum += F(Pn, phase_of(p, x - (s0[i] if real else startlist[started.index(i)])))
            that = max(0.0, fsum - (anon - BASE))
            free = min(node, job)
            return free, node, job, that, free - that - F(Pn, 0)
        print('(b) %s: memory.max %.0f GiB, F0(d300_3p) = %.1f GiB (P %.1f + m %.1f), reserve 40' % (lab, lim, F(Pn, 0), Pn, M_))
        order = sorted(st, key=lambda i: st[i])
        for k, i in enumerate(order):
            x = st[i] - 1.0
            started = order[:k]
            if not started:
                print('   item %d real start +%.0f s: nothing else runs: admitted' % (i, st[i] - t[0])); continue
            free, node, job, that, proj = evaluate(x, started)
            print('   item %d real start +%.0f s: node %.1f job %.1f -> free %.1f; items still to allocate %.1f; F0 %.1f; margin to the reserve %.1f -> %s' % (
                i, st[i] - t[0], node, job, free, that, F(Pn, 0), proj - 40, 'ADMIT' if proj >= 40 else 'WAIT'))
        # counterfactual
        cf = []
        for k, i in enumerate(order):
            if k == 0: cf.append(st[i]); continue
            x = cf[-1] if k else st[i]
            x = max(x, st[order[0]])
            while x < t[-1]:
                started = list(range(k)); free, node, job, that, proj = evaluate(x, started, False, cf)
                if proj >= 40: break
                # an item ends: when all earlier items ended, it runs alone
                alive = [j for j in range(k) if x < cf[j] + solo['end']]
                if not alive: break
                x += 1.0
            cf.append(x)
        print('   counterfactual starts (rule applied, series rebuilt from the solo profile): %s' % ', '.join('+%.0f s' % (c - t[0]) for c in cf))
        peak = max(BASE + sum(float(np.interp(x - s, solo['t'], solo['L'], left=0, right=0)) for s in cf) for x in np.arange(t[0], t[-1], 1.0))
        print('   counterfactual solo-overlay peak anon+shmem %.1f GiB against memory.max %.0f; the recording: peak %.1f' % (peak, lim, ((A[:, 3] + A[:, 5]) / G).max()))

# ---------- (c)
if 'c' in WHICH:
    names = list(C)
    NT = 4000
    xs = np.arange(0, NT, 1.0)
    def est(n): return 61.8 + 0.2235 * n
    LV = {}; OW = {}; AL = {}
    for s in names:
        p = C[s]; e = est(p['reads'] / 1e6)
        LV[s] = np.interp(xs, p['t'], p['L'], left=0, right=0)
        held = np.maximum(0.0, np.interp(xs, p['held_t'], p['held'], left=0, right=p['held'][-1]))
        lvl = p['level'] if p['level'] is not None else 1e9; wt = p['waited'] if p['waited'] is not None else 1e9
        want = np.where(xs < lvl, e, 0.4 * e); want = np.where(xs >= wt, np.minimum(want, 80.0), want)
        OW[s] = np.maximum(0.0, want - held); AL[s] = (xs < p['end']).astype(float)
    def shifted(arr, off):   # arr evaluated at (x - off); off integer seconds; zero before the start
        off = int(round(off)); out = np.zeros(NT)
        if off < NT: out[off:] = arr[:NT - off]
        return out
    def admit(prev, cand, lo, fe0):
        """prev: list of (class, start). Earliest second >= lo at which today's gate admits cand; 'alone' counts as admitted. Returns (t, alone)"""
        e = est(C[cand]['reads'] / 1e6)
        alive = sum(shifted(AL[s], st) for s, st in prev)
        anon = BASE + sum(shifted(LV[s], st) for s, st in prev)
        own = sum(shifted(OW[s], st) * shifted(AL[s], st) for s, st in prev)
        ok = (alive == 0) | ((fe0 - anon - own - e) >= 40.0)
        ok[: int(lo)] = False
        idx = np.nonzero(ok)[0]
        if len(idx) == 0: return None, False
        return float(idx[0]), bool(alive[idx[0]] == 0)
    hits = {}
    for combo in itertools.combinations_with_replacement(names, 3):
        for perm in set(itertools.permutations(combo)):
            a, b, c = perm
            lo = 40.0 if est(C[a]['reads'] / 1e6) >= 100 else 0.0
            tB, aloneB = admit([(a, 0.0)], b, lo, FE0)
            if tB is None or aloneB: continue
            lo = tB + (40.0 if est(C[b]['reads'] / 1e6) >= 100 else 0.0)
            tC, aloneC = admit([(a, 0.0), (b, tB)], c, lo, FE0)
            if tC is None or aloneC: continue
            tot = BASE + shifted(LV[a], 0) + shifted(LV[b], tB) + shifted(LV[c], tC)
            pk = float(tot.max())
            if pk > LIMIT:
                key = tuple(sorted(combo))
                if key not in hits or pk > hits[key][0]: hits[key] = (pk, perm, tB, tC, float(np.argmax(tot)))
    allc = len(list(itertools.combinations_with_replacement(names, 3)))
    print('(c) PREDICTION (solo profiles, FE0 = %.0f GiB, limit %.0f GiB, baseline %.1f): %d of %d multisets of three classes have an order that the current gate admits together and whose overlaid true peak exceeds the limit' % (FE0, LIMIT, BASE, len(hits), allc))
    sm = lambda k: sum(C[s]['peak'] for s in k)
    for key, (pk, perm, tB, tC, tp) in sorted(hits.items(), key=lambda kv: sm(kv[0]))[:25]:
        print('   %-48s sum of solo peaks %6.1f  overlay peak %6.1f (+%.0f s) over %.1f ; order %s, starts 0 / %.0f / %.0f s' % (' + '.join(key), sm(key), pk, tp, pk - LIMIT, '>'.join(perm), tB, tC))
    # the same admissions seen by the limit-aware rule (job term only: limit - anon+shmem, the cgroup's file cache counted 0; that = sum F_i - sum of the levels now), margin m
    def phs(s, tt):
        p = C[s]; return 2 if (p['waited'] is not None and tt >= p['waited']) else 1 if (p['level'] is not None and tt >= p['level']) else 0
    nadm = 0
    for key, (pk, perm, tB, tC, tp) in hits.items():
        a, b, c = perm; ok = True
        for cand, prev, t in ((b, [(a, 0.0)], tB), (c, [(a, 0.0), (b, tB)], tC)):
            lv = sum(float(np.interp(t - st, C[x]['t'], C[x]['L'], left=0, right=0)) for x, st in prev if 0 <= t - st < C[x]['end'])
            fs = sum(F(PRED[x], phs(x, t - st)) for x, st in prev if 0 <= t - st < C[x]['end'])
            if not prev or all(not (0 <= t - st < C[x]['end']) for x, st in prev): continue
            free = LIMIT - (BASE + lv); that = max(0.0, fs - lv)
            if free - that - F(PRED[cand], 0) < 40: ok = False
        nadm += ok
    print('    of these %d triples the limit-aware rule (m = %g, limit %.0f, node term not binding) would still admit %d at the same times' % (len(hits), M_, LIMIT, nadm))
    if hits:
        k0 = min(hits, key=sm); print('   smallest by sum of solo peaks: %s' % ' + '.join(k0))
        k1 = min(hits, key=lambda k: sum(C[s]['reads'] for s in k)); print('   smallest by reads: %s' % ' + '.join(k1))
    if '--json' in sys.argv: json.dump({'|'.join(k): [v[0], list(v[1]), v[2], v[3], v[4]] for k, v in hits.items()}, open(sys.argv[sys.argv.index('--json') + 1], 'w'))
