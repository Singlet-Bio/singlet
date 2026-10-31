#!/usr/bin/env python3
"""CPU-seconds per thread name (and the busiest threads) from a PACK=1 run's threads.txt (ps -L samples every 10 s). usage: threads.py DIR [M reads]"""
import collections, sys
last = {}
for l in open(sys.argv[1] + '/threads.txt', errors='replace'):
    p = l.split()
    if len(p) == 4 and ':' in p[3]:
        h, m, s = p[3].split(':') if p[3].count(':') == 2 else ('0',) + tuple(p[3].split(':'))
        last[(p[0], p[1])] = (p[2], int(h) * 3600 + int(m) * 60 + int(s))
by = collections.defaultdict(lambda: [0, 0, 0])
for (pid, lwp), (c, t) in last.items():
    by[c][0] += 1; by[c][1] += t; by[c][2] = max(by[c][2], t)
M = float(sys.argv[2]) if len(sys.argv) > 2 else None
tot = sum(v[1] for v in by.values())
print(f'total {tot} cpu-s over {len(last)} threads' + (f' = {tot/M:.1f} core-s/M' if M else ''))
for c, (n, t, mx) in sorted(by.items(), key=lambda x: -x[1][1])[:12]:
    print(f'  {c:<16} threads={n:3d} cpu-s={t:6d} max-thread={mx:5d}' + (f' {t/M:5.2f}/M' if M else ''))
