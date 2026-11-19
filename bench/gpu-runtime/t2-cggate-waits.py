#!/usr/bin/env python3
"""cggate: the admission waits, the refusing terms and the model-exceeded lines of one or more gpu-pool reports.

usage: t2-cggate-waits.py <pool report dir> [<pool report dir> ...]      (each has pool.log; mode 1 also memsamp-cg.tsv)

Per pool: items started / ended / failed, the start time of every item (seconds after the pool's first line), the waits
("memory admission: item N waits ..."): how many, the seconds waited ("admitted after X s" and "starts after X s"), which term refused (the
last "refused by the <node|job|node and job> term" of each wait), the forced starts ("the wait is bounded"), the peak-figure changes, the
"model exceeded" lines and the largest excess, the idle baseline lines. With memsamp-cg.tsv: the largest AS - B - (sum of figures) over the run.
"""
import re, sys, os, collections

def pool(dirn):
    p = os.path.join(dirn, 'pool.log')
    L = [l.rstrip('\n') for l in open(p, errors='replace')]
    t0 = None
    starts = {}; ends = {}; fails = 0
    waits = collections.OrderedDict(); admitted = {}; forced = []; exc = []; base = []; figs = 0; noKill = 0
    refused = collections.defaultdict(list)
    for l in L:
        m = re.match(r'(\d+) \[gpu-pool [^]]*\] (.*)', l)
        if not m: continue
        t = float(m.group(1)); s = m.group(2)
        if t0 is None: t0 = t
        mm = re.match(r'start item (\d+) ', s)
        if mm: starts[int(mm.group(1))] = t - t0; continue
        mm = re.match(r'end item (\d+) .*?: (exit \d+|killed[^,]*), wall ([\d.]+)', s)
        if mm:
            ends[int(mm.group(1))] = (t - t0, mm.group(2))
            if mm.group(2) != 'exit 0': fails += 1
            continue
        mm = re.match(r'memory admission: item (\d+) waits \(', s)
        if mm:
            i = int(mm.group(1)); waits[i] = t - t0
            r = re.search(r'refused by the ([a-z ]+) term', s)
            if r: refused[i].append(r.group(1))
            continue
        mm = re.match(r'memory admission: item (\d+) admitted after (\d+) s', s)
        if mm: admitted[int(mm.group(1))] = float(mm.group(2)); continue
        mm = re.match(r'memory admission: item (\d+) starts after (\d+) s', s)
        if mm: forced.append((int(mm.group(1)), float(mm.group(2)))); continue
        mm = re.match(r'memory admission: item (\d+) still waits, now refused by the ([a-z ]+) term', s)
        if mm: refused[int(mm.group(1))].append(mm.group(2)); continue
        if s.startswith('model exceeded'): exc.append((t - t0, s)); continue
        if s.startswith('peak admission: idle baseline'): base.append((t - t0, s)); continue
        if s.startswith('peak figure:'): figs += 1
    return dict(dir=dirn, starts=starts, ends=ends, fails=fails, waits=waits, admitted=admitted, forced=forced, exc=exc, base=base, figs=figs, refused=refused)

for d in sys.argv[1:]:
    r = pool(d)
    tot = sum(r['admitted'].values()) + sum(w for _, w in r['forced'])
    print('== %s' % d)
    print('items started %d, ended %d, not exit 0: %d' % (len(r['starts']), len(r['ends']), r['fails']))
    print('start times (s after the pool\'s first line): ' + ' '.join('%d:%.0f' % (i, t) for i, t in sorted(r['starts'].items())))
    cnt = collections.Counter()
    for i, v in r['refused'].items():
        cnt[v[-1] if v else '?'] += 1
    print('waits %d, %.0f s waited in all (admitted %d, forced %d); last refusing term per wait: %s' % (len(r['waits']), tot, len(r['admitted']), len(r['forced']), dict(cnt) or '-'))
    for i, v in sorted(r['refused'].items()): print('  item %d refused by: %s' % (i, ' -> '.join(v)))
    print('peak-figure change lines: %d; idle baseline lines: %d' % (r['figs'], len(r['base'])))
    for t, s in r['base']: print('  +%.0f s %s' % (t, s))
    print('model exceeded lines: %d' % len(r['exc']))
    for t, s in r['exc'][:20]: print('  +%.0f s %s' % (t, s))
    f = os.path.join(d, 'memsamp-cg.tsv')
    if os.path.exists(f):
        mx = (-1e9, None); n = 0
        for ln in open(f):
            a = ln.split('\t')
            if a[0] == 't' or len(a) < 9: continue
            AS, B, figs, valid = float(a[1]), float(a[2]), float(a[3]), a[8].strip()
            if valid == '1':
                n += 1
                if AS - B - figs > mx[0]: mx = (AS - B - figs, a[0])
        print('memsamp-cg.tsv: %d samples with a valid baseline; largest AS - B - sum of figures %.1f GiB (t %s)' % (n, mx[0] if n else float('nan'), mx[1]))
