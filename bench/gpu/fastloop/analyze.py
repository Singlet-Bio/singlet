#!/usr/bin/env python3
"""Effects of stacked GPU ideas from one fastloop item (bench/gpu/fastloop/README.md). Pure Python, no numpy.

Reads every point of a tp_gc results directory (point.json + meta.json). An idea is on in a point when its name is one of the '+'-joined
atoms of the point's config. Per run it takes the metrics below, drops runs that fell back to the CPU and, with --drop-first, the first
run of each one-at-a-time point (warm-up: page cache, a shared-index holder that has to load). For each metric it fits ordinary least
squares on the +/-1 coded design with one term per block (and, with --2fi, every two-way interaction):

    y = b0 + sum_blocks + sum_i beta_i x_i [+ sum_ij gamma_ij x_i x_j]

The effect of idea i is 2 beta_i (switching it on), reported in the metric's unit and relative to the fitted all-off level. The 95 %
interval comes from a bootstrap that resamples runs within each design cell and block, so the design matrix stays fixed.

Metrics (lower is better): held = seconds a run holds device locks (prefill + solo + pileup + non-host, the GPU's cost per sample),
core = host core-seconds per million read pairs, wall = run seconds, and per stage: st.<step> (the engine's step timers: index, reads, e10,
clip, dedup, laneA, laneM, laneB, lanes, hostlane, records, readRecord, e2e), lk.<lock> (seconds each device lock is held: prefill, solo,
pileup, nonhost, handoff), sg.<stage> (wall seconds of the run's stages from its own log: scan, refs, STAR, stream, handoff, nonhost,
export, demux).

  analyze.py --res DIR --ideas a,b,c [--drop-first 1] [--metric held] [--theta 0.03] [--2fi] [--registry ideas.tsv] [--tsv out.tsv]
"""
import argparse
import glob
import json
import os
import random
import re

STEPS = ['index', 'reads', 'e10', 'clip', 'dedup', 'laneA', 'laneM', 'laneB', 'lanes', 'hostlane', 'records', 'readRecord', 'e2e']
LOCKS = ['prefill', 'solo', 'pileup', 'nonhost', 'handoff']
STAGES = ['scan', 'refs', 'STAR', 'stream', 'handoff', 'nonhost', 'export', 'demux']
MAIN = ['held', 'gheld', 'gsteps', 'core', 'wall']


def metric(r, m):
    if m == 'held':
        v = r.get('log_held_s')
        if v is None:
            v = sum((x or {}).get('held') or 0 for x in (r.get('locks') or {}).values())
        return v
    if m == 'gheld':   # device locks without the non-host screen's (its index load is file I/O, noisy on small inputs)
        L = r.get('locks') or {}
        return sum((L.get(k) or {}).get('held') or 0 for k in ('prefill', 'solo', 'pileup', 'handoff')) if L else None
    if m == 'gsteps':   # the engine's own step timers, summed (device work of the prefill, no locks or waits)
        s = r.get('steps') or {}
        return sum(s.values()) if s else None
    if m == 'core':
        return r['core_s'] / r['pairs'] * 1e6 if r.get('core_s') and r.get('pairs') else None
    if m == 'wall':
        return r.get('wall')
    if m.startswith('st.'):
        s = r.get('steps') or {}
        return s.get(m[3:]) if s else None
    if m.startswith('lk.'):
        return ((r.get('locks') or {}).get(m[3:]) or {}).get('held', 0.0 if r.get('locks') else None)
    if m.startswith('sg.'):
        return (r.get('stages') or {}).get(m[3:])
    return None


def fell_back(r):
    return bool(r.get('engine_failed')) or not r.get('level5', True) or not r.get('gpu_pileup_ok', True)


def fallback_copies(d):
    """Copies that tp_gc's fallback.py marks FALLBACK (any stage: prefill, pileup, the non-host screen), from <point>/fallback.txt."""
    out = set()
    try:
        for l in open(os.path.join(d, 'fallback.txt')):
            f = l.split('	')
            if len(f) > 1 and f[1] == 'FALLBACK':
                out.add(f[0])
    except OSError:
        pass
    return out


def inv(A):
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) < 1e-9:
            return None
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
    return [row[n:] for row in M]


def load(res, ideas, drop_first):
    rows, excl = [], {'fallback': 0, 'failed': 0, 'warmup': 0, 'nopoint': 0}
    load.tally = []   # (x, fell back or failed) of every run, for the fallback rate per idea
    for mf in sorted(glob.glob(os.path.join(res, '*', 'meta.json'))):
        d = os.path.dirname(mf)
        try:
            meta = json.load(open(mf))
            pj = json.load(open(os.path.join(d, 'point.json')))
        except (OSError, ValueError):
            excl['nopoint'] += 1
            continue
        if os.path.basename(d).startswith('alone-'):
            continue
        atoms = set((meta.get('cfg') or 'base').split('+'))
        x = [1 if i in atoms else -1 for i in ideas]
        mb = re.search(r'_b(\d+)$', meta.get('tag', ''))
        block = int(mb.group(1)) if mb else 1
        runs = sorted(pj.get('runs', []), key=lambda r: r.get('start', 0))
        fb = fallback_copies(d)
        for j, r in enumerate(runs):
            bad = r.get('rc', 1) != 0 or not r.get('pairs') or fell_back(r) or r.get('copy') in fb
            load.tally.append((x, bad))
            if r.get('rc', 1) != 0 or not r.get('pairs'):
                excl['failed'] += 1
                continue
            if fell_back(r) or r.get('copy') in fb:
                excl['fallback'] += 1
                continue
            if drop_first and int(meta.get('k', 1)) == 1 and j == 0:
                excl['warmup'] += 1
                continue
            rows.append({'x': x, 'block': block, 'cell': tuple(x), 'run': r, 'point': os.path.basename(d)})
    return rows, excl


def design(rows, k, twofi):
    blocks = sorted({r['block'] for r in rows})
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)] if twofi else []
    X = []
    for r in rows:
        v = [1.0] + [1.0 if r['block'] == b else 0.0 for b in blocks[1:]] + [float(t) for t in r['x']]
        v += [float(r['x'][a] * r['x'][b]) for a, b in pairs]
        X.append(v)
    return X, len(blocks) - 1, pairs


def solver(X):
    p = len(X[0])
    XtX = [[sum(row[i] * row[j] for row in X) for j in range(p)] for i in range(p)]
    I = inv(XtX)
    if I is None:
        return None
    return [[sum(I[i][j] * X[n][j] for j in range(p)) for n in range(len(X))] for i in range(p)]   # (X'X)^-1 X'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--ideas', required=True)
    ap.add_argument('--drop-first', type=int, default=0)
    ap.add_argument('--metric', default='held')
    ap.add_argument('--theta', type=float, default=0.03)
    ap.add_argument('--guard', type=float, default=0.03, help='core-s/M ceiling of a device-time win (the campaign rule: +3%%)')
    ap.add_argument('--2fi', dest='twofi', action='store_true')
    ap.add_argument('--registry', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ideas.tsv'))
    ap.add_argument('--reps', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tsv', default=None)
    ap.add_argument('--trim', type=float, default=1.4, help='leave out runs above TRIM x their cell median of the primary metric (0: off)')
    o = ap.parse_args()
    ideas = [x for x in o.ideas.split(',') if x]
    k = len(ideas)
    stage_of, metric_of = {}, {}
    if os.path.exists(o.registry):
        for l in open(o.registry):
            f = l.rstrip('\n').split('\t')
            if len(f) >= 4 and not l.startswith('#'):
                stage_of[f[0]] = f[3]
                if len(f) >= 6 and f[5] not in ('', '-'):
                    metric_of[f[0]] = f[5]   # the idea's own decision metric (a host idea: core), else --metric
    rows, excl = load(o.res, ideas, o.drop_first)
    if o.trim > 0:
        med = {}
        for r in rows:
            v = metric(r['run'], o.metric)
            if v is not None:
                med.setdefault(r['cell'], []).append(v)
        med = {c: sorted(v)[(len(v) - 1) // 2] for c, v in med.items()}   # lower median: robust with two runs
        keep = []
        for r in rows:
            v = metric(r['run'], o.metric)
            if v is not None and med.get(r['cell']) and v > o.trim * med[r['cell']]:
                excl['outlier'] = excl.get('outlier', 0) + 1
                print('outlier left out: %s %s %s = %.2f (cell median %.2f)' % (r['point'], r['run'].get('copy'), o.metric, v, med[r['cell']]))
                continue
            keep.append(r)
        rows = keep
    gone = [i for j, i in enumerate(ideas) if len({r['x'][j] for r in rows}) < 2]
    if gone:
        print('no runs left at one level of: %s (left out of the fit)' % ', '.join(gone))
        idx = [j for j, i in enumerate(ideas) if i not in gone]
        ideas = [ideas[j] for j in idx]
        k = len(ideas)
        for r in rows:
            r['x'] = [r['x'][j] for j in idx]
            r['cell'] = tuple(r['x'])
        load.tally = [([x[j] for j in idx], b) for x, b in load.tally]
    print('# fastloop effects: %s' % o.res)
    print('runs used %d; left out: %s' % (len(rows), ', '.join('%s %d' % kv for kv in excl.items() if kv[1])))
    if not rows:
        return
    cells = {}
    for i, r in enumerate(rows):
        cells.setdefault((r['cell'], r['block']), []).append(i)
    print('cells %d, blocks %d, runs per cell and block %s' % (len({c for c, _ in cells}), len({b for _, b in cells}),
          sorted({len(v) for v in cells.values()})))
    metrics = MAIN + ['st.' + s for s in STEPS] + ['lk.' + s for s in LOCKS] + ['sg.' + s for s in STAGES]
    rng = random.Random(o.seed)
    # one bootstrap index map per replicate, shared by every metric (resample within cell and block)
    groups = list(cells.values())
    reps = []
    for _ in range(o.reps):
        idx = [0] * len(rows)
        for g in groups:
            for pos in g:
                idx[pos] = g[rng.randrange(len(g))]
        reps.append(idx)
    out = {}
    for m in metrics:
        sel = [i for i, r in enumerate(rows) if metric(r['run'], m) is not None]
        if len(sel) < len(rows) * 0.9 or len(sel) < 4:
            continue
        sub = [rows[i] for i in sel]
        y = [float(metric(r['run'], m)) for r in sub]
        if max(y) - min(y) < 1e-9 and max(y) == 0:
            continue
        X, nb, pairs = design(sub, k, o.twofi)
        S = solver(X)
        if S is None:
            continue
        pos = {i: n for n, i in enumerate(sel)}

        def fit(yv):
            b = [sum(S[i][n] * yv[n] for n in range(len(yv))) for i in range(len(S))]
            main = b[1 + nb:1 + nb + k]
            base = b[0] + sum(b[1:1 + nb]) / (nb + 1) - sum(main) + sum(b[1 + nb + k:])   # all off, blocks averaged
            return b, main, base
        b, main, base = fit(y)
        bs = []
        for idx in reps:
            yv = [y[pos[idx[i]]] if idx[i] in pos else y[pos[i]] for i in sel]
            bs.append(fit(yv))
        res = []
        for j in range(k):
            eff = 2 * main[j]
            rel = eff / base if base else float('nan')
            de = sorted(2 * t[1][j] for t in bs)
            dr = sorted((2 * t[1][j] / t[2]) if t[2] else float('nan') for t in bs)
            lo, hi = int(0.025 * len(de)), int(0.975 * len(de)) - 1
            res.append(dict(eff=eff, lo=de[lo], hi=de[hi], rel=rel, rlo=dr[lo], rhi=dr[hi], base=base))
        twofi = []
        for q, (a, c) in enumerate(pairs):
            g = 2 * b[1 + nb + k + q]
            dg = sorted(2 * t[0][1 + nb + k + q] for t in bs)
            twofi.append((ideas[a], ideas[c], g, dg[int(0.025 * len(dg))], dg[int(0.975 * len(dg)) - 1]))
        out[m] = dict(ideas=res, twofi=twofi, n=len(sub))

    def cell(m, j, rel=True):
        if m not in out:
            return 'n/a'
        e = out[m]['ideas'][j]
        if rel:
            return '%+.1f%% [%+.1f, %+.1f]' % (100 * e['rel'], 100 * e['rlo'], 100 * e['rhi'])
        return '%+.2f s [%+.2f, %+.2f]' % (e['eff'], e['lo'], e['hi'])

    def decide(m, j):
        if m not in out:
            return 'n/a'
        e = out[m]['ideas'][j]
        d = 'WIN' if e['rhi'] < -o.theta else 'RULED-OUT' if e['rlo'] > -o.theta else 'OPEN'
        # the campaign's guard: a device-time win may not cost more than GUARD of host core-s/M (its CI wholly above it: GUARD)
        if d == 'WIN' and m != 'core' and 'core' in out and out['core']['ideas'][j]['rlo'] > o.guard:
            d = 'GUARD (core-s/M)'
        return '%s on %s' % (d, m)
    print()
    print('Effect of switching each idea on, relative to the fitted all-off level (95 %% bootstrap CI). Decision on each idea\'s own metric '
          '(ideas.tsv column 6, else %s) at theta %.0f%% (WIN: CI below -theta; RULED-OUT: CI above -theta; OPEN: neither, add a block; '
          'GUARD: a win whose core-s/M CI lies above +%.0f%%).' % (o.metric, 100 * o.theta, 100 * o.guard))
    print()
    print('| idea | stage | held | gheld | gsteps | core-s/M | wall | decision |')
    print('|---|---|---|---|---|---|---|---|')
    for j, i in enumerate(ideas):
        print('| %s | %s | %s | %s | %s | %s | %s | %s |' % (i, stage_of.get(i, ''), cell('held', j), cell('gheld', j), cell('gsteps', j), cell('core', j),
                                                       cell('wall', j), decide(metric_of.get(i, o.metric), j)))
    print()
    print('Where each idea acts: stage metrics whose 95 % CI excludes zero, largest first (seconds per run, then relative).')
    print()
    for j, i in enumerate(ideas):
        hits = []
        for m in out:
            if m in MAIN:
                continue
            e = out[m]['ideas'][j]
            if e['lo'] > 0 or e['hi'] < 0:
                hits.append((abs(e['eff']), m, e))
        hits.sort(reverse=True)
        txt = '; '.join('%s %+.2f s (%+.0f%%) [%+.2f, %+.2f]' % (m, e['eff'], 100 * e['rel'], e['lo'], e['hi']) for _, m, e in hits[:8])
        print('- **%s**: %s' % (i, txt or 'no stage moved beyond its CI'))
    if any(b for _, b in load.tally):
        print()
        print('Runs that failed or fell back to the CPU (left out of the timing above), with each idea on and off:')
        print()
        for j, i in enumerate(ideas):
            on = [b for x, b in load.tally if x[j] > 0]
            off = [b for x, b in load.tally if x[j] < 0]
            print('- %s: on %d of %d, off %d of %d' % (i, sum(on), len(on), sum(off), len(off)))
    if o.twofi and 'held' in out:
        print()
        print('Two-way interactions on held and core-s/M (2 gamma: the extra change when both are on; 95 % CI).')
        print()
        print('| pair | held s | core-s/M |')
        print('|---|---|---|')
        for q, (a, c, g, lo, hi) in enumerate(out['held']['twofi']):
            cc = out.get('core', {}).get('twofi', [None] * (q + 1))[q]
            print('| %s:%s | %+.2f [%+.2f, %+.2f] | %s |' % (a, c, g, lo, hi, '%+.3f [%+.3f, %+.3f]' % cc[2:] if cc else 'n/a'))
    if o.tsv:
        with open(o.tsv, 'w') as f:
            f.write('idea\tmetric\teffect\tlo\thi\trel\trel_lo\trel_hi\tbase\tn\n')
            for m in out:
                for j, i in enumerate(ideas):
                    e = out[m]['ideas'][j]
                    f.write('%s\t%s\t%.5g\t%.5g\t%.5g\t%.5g\t%.5g\t%.5g\t%.5g\t%d\n' % (i, m, e['eff'], e['lo'], e['hi'], e['rel'], e['rlo'],
                                                                                    e['rhi'], e['base'], out[m]['n']))


if __name__ == '__main__':
    main()
