#!/usr/bin/env python3
"""pooladmit: second-by-second cgroup trace of a t2-cgsamp.sh series around memory.events changes. usage: t2-cg-events.py <cg1hz.tsv> [every_n_s]"""
import sys
rows = [l.rstrip('\n').split('\t') for l in open(sys.argv[1]) if not l.startswith('#')]
G = 1073741824.; n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
t0 = float(rows[0][0]); prev = None
print('  t(s)  current  anon  file  shmem  MemFree  Cached(GB)  max/oom/oom_kill  pressure_full_avg10')
for k, r in enumerate(rows):
    ev = r[9].split()
    if prev is None or ev[1:] != prev[1:] or k % n == 0 or abs(int(ev[0]) - int(prev[0])) > 2000:
        print('%6.0f %8.1f %5.1f %5.1f %6.1f %8.1f %8.0f   %s  %s' % (float(r[0]) - t0, float(r[2]) / G, float(r[3]) / G, float(r[4]) / G, float(r[5]) / G, float(r[6]) / 1048576, float(r[7]) / 1048576, ' '.join(ev), r[10]))
    prev = ev
