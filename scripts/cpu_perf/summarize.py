#!/usr/bin/env python3
"""One row per run directory under jobs/cpu-perf/results: threads, pack, wall, core-s (user+sys), core-s/M reads, peak RSS, stream-path check.
usage: summarize.py RESULTS_DIR   (stage tables: stage_profile.py DIR M for the pack=1 runs)"""
import glob, os, re, sys
def secs(s):
    p = [float(x) for x in s.split(':')]; return sum(v * 60 ** i for i, v in enumerate(reversed(p)))
for d in sorted(glob.glob(sys.argv[1] + '/*')):
    m = re.match(r'(.+)-t(\d+)-p(\d+)-(singlet-[a-z-]+)-', os.path.basename(d))
    if not m: continue
    S, T, P, B = m.groups(); rows = []
    for i in range(1, int(P) + 1):
        try:
            t = open(f'{d}/time_{i}.txt').read(); L = open(f'{d}/log_{i}.txt', errors='replace').read()
            n = int(re.search(r'Number of input reads \|\s*(\d+)', open(f'{d}/final_{i}.txt').read()).group(1))
        except Exception: continue
        g = lambda k: re.search(k + r'.*?:\s*(\S+)\s*$', t, re.M).group(1)
        u, sy = float(g('User time')), float(g('System time')); w = secs(g('Elapsed'))
        ok = all(x in L for x in ('BAM: unsorted STAR stream into the pileup', 'Pileup: streaming from')) and 'Pileup: parallel' not in L and '\nexit 0' in L.replace(' exit 0', '\nexit 0') and 'bin/../whitelists' in L
        rows.append((n / 1e6, w, u + sy, int(g('Maximum resident')) / 1e6, ok))
    if not rows: continue
    k = len(rows); M = rows[0][0]
    print(f'{S:12} T={T:>2} pack={P} reads={M:6.1f}M wall={sum(r[1] for r in rows)/k:7.0f}s core-s={sum(r[2] for r in rows)/k:7.0f} '
          f'core-s/M={sum(r[2] for r in rows)/k/M:5.1f} RSS={max(r[3] for r in rows):5.1f}GB streamOK={all(r[4] for r in rows)} {B}')
