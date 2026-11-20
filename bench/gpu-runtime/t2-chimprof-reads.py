#!/usr/bin/env python3
"""t2-chimprof-reads.py <prefix> [--mate1 <chim_dedup.mate1>] [--out <file>] [--sample-n N]

Analysis of the per-read records of SINGLET_CHIM_PROF=<prefix> (src/star/ChimProf.h): where the cycles of a read go, the cost
distribution over reads, the class of the costliest reads, the chimeric-detection counters. Counts, shares and seconds only: no read
sequence is printed. With --mate1 the sequence of the selected reads (costliest 1 %, 0.1 %, and a random sample) is read from the
pass input to classify them (low complexity classes of docs/gpu-engine/runtime/T2-chimpass.md 3.3); the sequences stay in memory.
"""
import sys, glob, math, random, json
import numpy as np

REC = np.dtype([('id', '<u8'), ('cTotal', '<u8')] +
               [(n, '<u4') for n in ['cLoad', 'cSeed', 'cWinCreate', 'cWinFlank', 'cWinAssign', 'cStitch', 'cMult', 'cBar', 'cChim', 'cChimStitch', 'cChimOut', 'cOut',
                                     'Lread', 'nP', 'nA', 'nWinMade', 'nW', 'nTr']] +
               [('maxScore', '<i4'), ('mapMarker', '<i4')] +
               [(n, '<u4') for n in ['chimVisited', 'chimScored', 'chimCand', 'chimCheckOk', 'chimStitched', 'chimSurvive', 'chimLines', 'flags']])
assert REC.itemsize == 128

FLAGS = [('chim_called', 0), ('chim_trigger', 1), ('chim_pairloop', 2), ('chim_record', 3), ('limit seedMultimapNmax (a seed dropped)', 4),
         ('limit seedPerWindowNmax', 5), ('limit alignWindowsPerReadNmax', 6), ('limit alignTranscriptsPerReadNmax', 7),
         ('limit alignTranscriptsPerWindowNmax', 8), ('seed above winAnchorMultimapNmax (not an anchor)', 9), ('limit seedPerReadNmax', 10),
         ('marker too many anchors per window', 11), ('chimN > chimMultimapNmax', 12)]


def args():
    a = sys.argv[1:]
    d = {'prefix': a[0], 'mate1': None, 'out': None, 'n': 200000}
    i = 1
    while i < len(a):
        if a[i] == '--mate1': d['mate1'] = a[i + 1]; i += 2
        elif a[i] == '--out': d['out'] = a[i + 1]; i += 2
        elif a[i] == '--sample-n': d['n'] = int(a[i + 1]); i += 2
        else: raise SystemExit('bad argument ' + a[i])
    return d


def load(prefix):
    meta = {}
    for l in open(prefix + '.meta'):
        k, v = l.split()
        meta[k] = float(v)
    assert int(meta['rec_bytes']) == 128, 'record size'
    parts = [np.fromfile(f, dtype=REC) for f in sorted(glob.glob(prefix + '.reads.*'))]
    return meta, np.concatenate(parts)


def classify(seq):
    """T2-chimpass 3.3 classes of one read sequence (str); returns a dict of booleans."""
    n = len(seq)
    c = {b: seq.count(b) for b in 'ACGTN'}
    top = max('ACGT', key=lambda b: c[b])
    mono = n > 0 and c[top] >= 0.9 * n
    tand = False
    if not mono and n > 12:
        for p in range(2, 7):
            same = sum(1 for i in range(p, n) if seq[i] == seq[i - p])
            if same >= 0.9 * (n - p):
                tand = True; break
    ent = 0.0
    if n >= 12:
        for b in 'ACGT':
            if c[b] > 0:
                f = c[b] / n; ent -= f * math.log2(f)
    rc5 = 'CCCATATAAGAAA'; f5 = 'TTTCTTATATGGG'; core3 = 'TATCAACGCAGAGTAC'; core3rc = 'GTACTCTGCGTTGATA'
    return {'polyA': mono and top == 'A', 'polyT': mono and top == 'T', 'polyG': mono and top == 'G', 'polyC': mono and top == 'C',
            'tandem2-6': tand, 'entropy<1.5': n >= 12 and ent < 1.5, 'entropy<1.0': n >= 12 and ent < 1.0, 'hasN': c['N'] > 0,
            'TSO5rc': rc5 in seq, 'TSO5fwd': f5 in seq, 'TSO3core': core3 in seq or core3rc in seq}


def fetch(mate1, ids):
    """sequences (str) of the given read ids from the pass input (4-line FASTQ, name '@<id>')"""
    want = set(int(i) for i in ids); got = {}
    with open(mate1, 'r') as f:
        while True:
            h = f.readline()
            if not h: break
            s = f.readline().rstrip('\n'); f.readline(); f.readline()
            try: i = int(h[1:].split()[0])
            except ValueError: continue
            if i in want: got[i] = s
    return got


def main():
    A = args(); out = []
    P = out.append
    meta, r = load(A['prefix'])
    hz = meta['tsc_hz']; n = len(r)
    tot = r['cTotal'].astype(np.float64)
    T = tot.sum()
    P('# t2-chimprof per-read analysis of %s' % A['prefix'])
    P('records %d (meta says %d); threads %d; mapping wall %.1f s; TSC %.3f GHz; process utime %.1f s stime %.1f s' % (n, meta['records'], meta['threads'], meta['wall_s'], hz / 1e9, meta['utime_s'], meta['stime_s']))
    P('sum of per-read oneRead cycles: %.1f CPU-s (%.1f us per read); mapping wall x threads = %.1f' % (T / hz, T / hz / n * 1e6, meta['wall_s'] * meta['threads']))
    P('')
    P('## 1. where the cycles of oneRead go (all reads)')
    ph = ['cLoad', 'cSeed', 'cWinCreate', 'cWinFlank', 'cWinAssign', 'cStitch', 'cMult', 'cBar', 'cChim', 'cOut']
    acc = 0.0
    for p in ph:
        s = r[p].astype(np.float64).sum(); acc += s
        P('%-12s %9.1f CPU-s %6.2f %%   %7.2f us/read' % (p, s / hz, 100 * s / T, s / hz / n * 1e6))
    P('%-12s %9.1f CPU-s %6.2f %%   (oneRead total minus the phases above: hand-off, qualHist, mappedFilter, transformGenome, loop overhead)' % ('other', (T - acc) / hz, 100 * (T - acc) / T))
    P('inside cChim: stitching %.1f CPU-s (%.1f %% of cChim), junction output %.1f CPU-s (%.1f %%), rest (pair loop, segment checks, allocation) %.1f CPU-s' % (
        r['cChimStitch'].astype(np.float64).sum() / hz, 100 * r['cChimStitch'].astype(np.float64).sum() / max(1, r['cChim'].astype(np.float64).sum()),
        r['cChimOut'].astype(np.float64).sum() / hz, 100 * r['cChimOut'].astype(np.float64).sum() / max(1, r['cChim'].astype(np.float64).sum()),
        (r['cChim'].astype(np.float64).sum() - r['cChimStitch'].astype(np.float64).sum() - r['cChimOut'].astype(np.float64).sum()) / hz))
    P('')
    P('## 2. cost distribution (oneRead cycles per read)')
    order = np.argsort(-tot)
    cs = np.cumsum(tot[order])
    for frac in (0.001, 0.01, 0.1, 0.5):
        k = max(1, int(round(frac * n)))
        P('costliest %5.1f %% of reads (%d reads): %.2f %% of the CPU; their mean %.0f us, smallest of them %.0f us' % (100 * frac, k, 100 * cs[k - 1] / T, tot[order[:k]].mean() / hz * 1e6, tot[order[k - 1]] / hz * 1e6))
    q = np.percentile(tot / hz * 1e6, [10, 50, 90, 99, 99.9, 100])
    P('per-read us: p10 %.1f median %.1f p90 %.1f p99 %.1f p99.9 %.1f max %.1f; mean %.1f' % (*q, T / hz / n * 1e6))
    P('')

    def describe(sel, name):
        m = int(sel.sum())
        P('### %s (%d reads, %.2f %% of the CPU)' % (name, m, 100 * tot[sel].sum() / T))
        if m == 0: return
        x = r[sel]
        for f in ['Lread', 'nP', 'nA', 'nWinMade', 'nW', 'nTr']:
            v = x[f].astype(np.float64)
            P('  %-9s mean %9.2f  median %7.0f  p90 %7.0f  max %8.0f' % (f, v.mean(), np.median(v), np.percentile(v, 90), v.max()))
        P('  maxScore/Lread: mean %.3f; maxScore == 0: %.1f %%' % ((x['maxScore'] / np.maximum(1, x['Lread'])).mean(), 100 * (x['maxScore'] == 0).mean()))
        P('  nTr>1 (multimapping): %.1f %%   nTr==0: %.1f %%   nTr>outFilterMultimapNmax(10): %.1f %%' % (100 * (x['nTr'] > 1).mean(), 100 * (x['nTr'] == 0).mean(), 100 * (x['nTr'] > 10).mean()))
        for nm, b in FLAGS:
            P('  flag %-52s %7.2f %%' % (nm, 100 * ((x['flags'] >> b) & 1).mean()))
        for f in ['chimVisited', 'chimScored', 'chimCand', 'chimCheckOk', 'chimStitched', 'chimSurvive', 'chimLines']:
            P('  %-12s sum %12d  mean %9.2f  max %9d' % (f, x[f].astype(np.uint64).sum(), x[f].mean(), x[f].max()))
        for p in ph:
            P('  share of the group CPU: %-10s %6.2f %%' % (p, 100 * x[p].astype(np.float64).sum() / tot[sel].sum()))

    allm = np.ones(n, bool)
    describe(allm, 'all reads')
    topk = {}
    for frac, name in ((0.1, 'costliest 10 %'), (0.01, 'costliest 1 %'), (0.001, 'costliest 0.1 %')):
        k = max(1, int(round(frac * n))); m = np.zeros(n, bool); m[order[:k]] = True; topk[name] = m
        describe(m, name)
    P('')
    P('## 3. chimeric detection (counters of every read; trigger = trBest->maxScore <= Lread - chimNonchimScoreDropMin(20), see T2-chimprof.md)')
    called = ((r['flags'] >> 0) & 1) == 1; trig = ((r['flags'] >> 1) & 1) == 1; loop = ((r['flags'] >> 2) & 1) == 1; rec = ((r['flags'] >> 3) & 1) == 1
    P('reads mapped by the pass %d' % n)
    P('reads that entered chimericDetection() past its early returns: %d (%.2f %%)' % (called.sum(), 100 * called.mean()))
    P('reads passing the trigger (chimericDetectionMult called): %d (%.2f %%)' % (trig.sum(), 100 * trig.mean()))
    P('reads that reached the pair loop (some segment passed segmentCheck): %d (%.2f %%)' % (loop.sum(), 100 * loop.mean()))
    P('reads with at least one junction line: %d (%.2f %%); lines %d' % (rec.sum(), 100 * rec.mean(), r['chimLines'].astype(np.uint64).sum()))
    for f in ['chimVisited', 'chimScored', 'chimCand', 'chimCheckOk', 'chimStitched', 'chimSurvive']:
        P('total %-12s %d' % (f, r[f].astype(np.uint64).sum()))
    cc = r['cChim'].astype(np.float64); CC = cc.sum()
    P('chimericDetection() time: %.1f CPU-s = %.2f %% of oneRead; reads with no junction line: %.1f CPU-s = %.2f %% of it' % (CC / hz, 100 * CC / T, cc[~rec].sum() / hz, 100 * cc[~rec].sum() / CC))
    P('  reads passing the trigger without a line: %.1f CPU-s (%.2f %% of chim time); reads not passing the trigger: %.1f CPU-s (%.2f %%)' % (cc[trig & ~rec].sum() / hz, 100 * cc[trig & ~rec].sum() / CC, cc[~trig].sum() / hz, 100 * cc[~trig].sum() / CC))
    P('  reads in the pair loop without a line: %.1f CPU-s (%.2f %%)' % (cc[loop & ~rec].sum() / hz, 100 * cc[loop & ~rec].sum() / CC))
    bar = r['cBar'].astype(np.float64)
    P('  barcode parse (getCBandUMI, once per read before chimericDetection): %.1f CPU-s = %.2f %% of oneRead' % (bar.sum() / hz, 100 * bar.sum() / T))
    P('  chim time by pair-loop size (chimScored per read): ')
    for lo, hi in ((0, 0), (1, 1), (2, 10), (11, 100), (101, 1000), (1001, 10 ** 9)):
        m = (r['chimScored'] >= lo) & (r['chimScored'] <= hi)
        P('    chimScored %d-%s: %d reads, chim %.1f CPU-s (%.2f %% of chim), stitching %.1f CPU-s' % (lo, hi if hi < 10 ** 9 else 'inf', m.sum(), cc[m].sum() / hz, 100 * cc[m].sum() / CC, r['cChimStitch'][m].astype(np.float64).sum() / hz))
    P('')

    P('## 4. outcome classes (all reads): mean us per read and share of the CPU')
    cls = [('nW == 0 (no window with an alignment)', r['nW'] == 0), ('nW == 1', r['nW'] == 1), ('nW 2-5', (r['nW'] >= 2) & (r['nW'] <= 5)),
           ('nW 6-20', (r['nW'] >= 6) & (r['nW'] <= 20)), ('nW > 20', r['nW'] > 20)]
    for nm, m in cls:
        P('  %-40s %9d reads (%5.1f %%)  %8.1f us/read  %6.2f %% of CPU' % (nm, m.sum(), 100 * m.mean(), (tot[m].mean() / hz * 1e6) if m.any() else 0, 100 * tot[m].sum() / T))
    P('  by read length (Lread after clip):')
    for lo, hi in ((0, 89), (90, 99), (100, 129), (130, 150), (151, 10 ** 9)):
        m = (r['Lread'] >= lo) & (r['Lread'] <= hi)
        if m.any(): P('    Lread %d-%s: %9d reads %8.1f us/read %6.2f %% of CPU' % (lo, hi if hi < 10 ** 9 else 'inf', m.sum(), tot[m].mean() / hz * 1e6, 100 * tot[m].sum() / T))
    P('  by nP (stored seeds):')
    for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)):
        m = (r['nP'] >= lo) & (r['nP'] <= hi)
        if m.any(): P('    nP %d-%s: %9d reads %8.1f us/read %6.2f %% of CPU' % (lo, hi if hi < 10 ** 9 else 'inf', m.sum(), tot[m].mean() / hz * 1e6, 100 * tot[m].sum() / T))
    P('')

    if A['mate1']:
        P('## 5. read classes (T2-chimpass 3.3 definitions, on the unclipped sequence of the pass input)')
        random.seed(1)
        samp = random.sample(range(n), min(A['n'], n))
        sel = set(int(i) for i in samp)
        for name, m in topk.items():
            if name != 'costliest 10 %': sel.update(int(i) for i in np.nonzero(m)[0])
        ids = r['id']
        want = set(int(ids[i]) for i in sel)
        seqs = fetch(A['mate1'], want)
        P('sequences found for %d of %d selected reads' % (len(seqs), len(want)))
        keys = None; tab = {}
        for i in sel:
            s = seqs.get(int(ids[i]))
            if s is None: continue
            c = classify(s); keys = list(c.keys()); tab[i] = c
        def rows(name, idx):
            idx = [i for i in idx if i in tab]
            if not idx: return
            P('### %s (%d reads classified)' % (name, len(idx)))
            tt = sum(tot[i] for i in idx)
            for k in keys:
                m = [i for i in idx if tab[i][k]]
                P('  %-12s %7.2f %% of the reads; mean %8.1f us/read; %6.2f %% of the group CPU' % (k, 100 * len(m) / len(idx), (np.mean([tot[i] for i in m]) / hz * 1e6) if m else 0, 100 * sum(tot[i] for i in m) / tt))
            any_lc = [i for i in idx if tab[i]['polyA'] or tab[i]['polyT'] or tab[i]['polyG'] or tab[i]['polyC'] or tab[i]['tandem2-6'] or tab[i]['entropy<1.5']]
            P('  %-12s %7.2f %% of the reads; mean %8.1f us/read; %6.2f %% of the group CPU' % ('any low-complexity (mono, tandem, entropy<1.5)', 100 * len(any_lc) / len(idx), (np.mean([tot[i] for i in any_lc]) / hz * 1e6) if any_lc else 0, 100 * sum(tot[i] for i in any_lc) / tt))
        rows('random sample (baseline)', samp)
        for name, m in topk.items():
            if name != 'costliest 10 %': rows(name, [int(i) for i in np.nonzero(m)[0]])
    txt = '\n'.join(out) + '\n'
    if A['out']: open(A['out'], 'w').write(txt)
    sys.stdout.write(txt)


if __name__ == '__main__':
    main()
