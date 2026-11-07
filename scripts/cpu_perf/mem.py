#!/usr/bin/env python3
"""Private vs shared memory over a PACK=1 run, from mem.txt (smaps_rollup of every process of the job, every 15 s). usage: mem.py DIR"""
import sys, collections
blocks, cur, t0 = [], None, None
for l in open(sys.argv[1] + '/mem.txt', errors='replace'):
    if l.startswith('@'):
        cur = {}; blocks.append((int(l.split()[1]), cur)); continue
    p = l.split()
    if len(p) < 3 or cur is None: continue
    kv = {k.rstrip(':'): int(v) for k, v in (x.split('=') for x in p[2:] if '=' in x)}
    cur[(p[0], p[1])] = kv
g = lambda kv, k: kv.get(k, 0) / 1048576.0
t0 = blocks[0][0]; step = max(1, len(blocks) // 14)
print('t_s   Rss_GB  Private_GB  Shared_GB  Anon_GB   per-process Private_GB')
peak = collections.defaultdict(float)
for i, (t, c) in enumerate(blocks):
    rss = sum(g(v, 'Rss') for v in c.values()); pr = sum(g(v, 'Private_Clean') + g(v, 'Private_Dirty') for v in c.values())
    sh = sum(g(v, 'Shared_Clean') + g(v, 'Shared_Dirty') for v in c.values()); an = sum(g(v, 'Anonymous') for v in c.values())
    for k, v in c.items(): peak[k] = max(peak[k], g(v, 'Private_Clean') + g(v, 'Private_Dirty'))
    if i % step == 0 or i == len(blocks) - 1:
        per = ' '.join(f"{k[1]}:{g(v,'Private_Clean')+g(v,'Private_Dirty'):.1f}" for k, v in c.items() if g(v, 'Rss') > 0.5)
        print(f'{t-t0:5d} {rss:7.1f} {pr:10.1f} {sh:9.1f} {an:8.1f}   {per}')
print('peak private per process (GB):', {f'{k[1]}[{k[0]}]': round(v, 1) for k, v in peak.items() if v > 0.3})
