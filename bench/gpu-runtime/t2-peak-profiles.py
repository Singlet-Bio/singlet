#!/usr/bin/env python3
"""peakmodel (throughput track 2): the true per-item host profiles of the one-at-a-time run t2-pooladmit-true.sbatch (job 818938), against the CORRECT idle baseline.

usage: t2-peak-profiles.py <results dir of t2-pooladmit-818938-true> <out profiles.json> [--baseline GiB]

Why this script exists. t2-true-peaks.py took the idle baseline as the median cgroup anon+shmem of the 5 s before the first item (22.03 GiB). At that moment the pool's
genome (17.65 GiB RssAnon) was loaded but the non-host server's page-locked index (24.7 GiB RssShmem) was still coming up (the sampler's first rows: 0, 14.5, 22.2, 24.7 GiB).
Between the items, with nothing running, the cgroup's anon+shmem is 42.4 GiB (median of the second after each item's end, items 1-16; pool 17.65 + server 24.7), so every
"TRUE" figure of T2-pooladmit.md section 2 is 20.4 GiB too large. This script measures the baseline from the item ends and writes the profiles above it.

profiles.json, per sample class: peak (max of anon+shmem above the baseline while the item runs), t / L (1 s series, seconds after the start / GiB above the baseline), level / waited / end
(seconds after the start of the "[engine_gpu] level" line, the "[allsite] async: waited" line, the item's end), held_t / held (the pool's own figure: RssAnon+RssShmem summed over the
item's processes, 2 s, minus the pool's genome 17.65 GiB, as gpu_pool.cpp's gateRefresh does), reads (pairs), plus the scan line's cDNA bases and longest read, and the .1fq size if known."""
import sys, os, re, json, collections
import numpy as np
G = 1073741824.0
D = sys.argv[1]; OUT = sys.argv[2]
base_arg = float(sys.argv[sys.argv.index('--baseline') + 1]) if '--baseline' in sys.argv else None
rows = [l.rstrip('\n').split('\t') for l in open(os.path.join(D, 'cg1hz.tsv')) if not l.startswith('#')]
A = np.array([[float(x) for x in r[:9]] for r in rows]); t = A[:, 0]; ab = (A[:, 3] + A[:, 5]) / G
items = {}
for ln in open(os.path.join(D, 'items.tsv')):
    i, s = ln.split(); items[int(i)] = s
starts, ends = {}, {}
for ln in open(os.path.join(D, 'pool0/pool.log'), errors='replace'):
    m = re.match(r'(\d+) \[gpu-pool [^]]*\] (start|end) item (\d+) ', ln)
    if m: (starts if m.group(2) == 'start' else ends)[int(m.group(3))] = float(m.group(1))
mk = collections.defaultdict(dict)
for ln in open(os.path.join(D, 'markers.tsv')):
    f = ln.split()
    if len(f) == 3: mk[int(f[1])][f[2]] = float(f[0])
meta = {}
for ln in open(os.path.join(D, 'pool0/samples.tsv')).read().splitlines()[1:]:
    f = ln.split('\t'); meta[int(f[0])] = dict(pairs=float(f[13]), wall=float(f[6]), exit=f[11])
# idle baseline: the second after the end of each item that is followed by another one
bl = [float(np.interp(ends[i] + 1.0, t, ab)) for i in sorted(ends) if i < max(ends)]
baseline = base_arg if base_arg is not None else float(np.median(bl))
print('idle baseline (median of the level 1 s after items 0-16 end): %.2f GiB (min %.2f, max %.2f)' % (baseline, min(bl), max(bl)))
tot = collections.defaultdict(float)
for ln in open(os.path.join(D, 'pool0/memsamp.tsv')):
    f = ln.rstrip('\n').split('\t')
    if len(f) < 7 or int(f[1]) < 0: continue
    tot[(float(f[0]), int(f[1]))] += (int(f[4]) + int(f[6])) / 1048576.0
prof = {}
for i in sorted(items):
    s0, e0 = starts[i], ends[i]
    sel = (t >= s0) & (t <= e0 + 1)
    L = ab[sel] - baseline
    held = sorted((ts - s0, v - 17.65) for (ts, j), v in tot.items() if j == i and s0 - 2 <= ts <= e0 + 2)
    txt = open(os.path.join(D, 'pool0/%d.log' % i), errors='replace').read() if os.path.exists(os.path.join(D, 'pool0/%d.log' % i)) else ''
    m = re.search(r'Scanned (\d+) blocks: (\d+) reads .*?, (\d+) cDNA bases, longest (\d+)', txt)
    prof[items[i]] = dict(item=i, reads=meta[i]['pairs'], peak=float(L.max()), peak_t=float(t[sel][np.argmax(L)] - s0), t=[float(x) for x in (t[sel] - s0)], L=[float(x) for x in L],
                          level=mk[i]['level'] - s0 if 'level' in mk[i] else None, waited=mk[i]['waited'] - s0 if 'waited' in mk[i] else None, end=e0 - s0,
                          held_t=[a for a, b in held], held=[b for a, b in held], bases=int(m.group(3)) if m else None, lmax=int(m.group(4)) if m else None,
                          fell_back=('FELL BACK' in txt), exit=meta[i]['exit'])
json.dump(dict(baseline=baseline, classes=prof), open(OUT, 'w'))
print('%-15s %9s %7s %6s %6s %7s %7s %7s' % ('sample', 'reads(M)', 'bases(G)', 'peak', 'peak@s', 'level@s', 'waited@s', 'end@s'))
for s, p in prof.items():
    print('%-15s %9.1f %7.1f %6.1f %6.0f %7.0f %7.0f %7.0f %s' % (s, p['reads'] / 1e6, (p['bases'] or 0) / 1e9, p['peak'], p['peak_t'], p['level'] or -1, p['waited'] or -1, p['end'], 'FELL BACK' if p['fell_back'] else ''))
