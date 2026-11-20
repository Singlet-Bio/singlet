#!/usr/bin/env python3
"""t2-chimpass-summary.py <results-dir> <label> [pool-log]: phases, CPU seconds and memory of the --chimeric second pass of one run,
from the files t2-chimpass-watch.sh and jobs/t2-chimpass.sbatch wrote (<label>.watch.tsv, .events.tsv, .chim_Log.out, .chim_Log.final.out, .pool.log).
Prints facts only (seconds, kB, counts)."""
import re, sys, datetime, os

d, label = sys.argv[1], sys.argv[2]
p = lambda s: os.path.join(d, label + s)

rows = []
for ln in open(p('.watch.tsv')):
    if ln.startswith('#'):
        continue
    f = ln.rstrip('\n').split('\t')
    if len(f) < 15:
        continue
    try:
        rows.append(dict(t=float(f[0]), state=f[1], pid=f[2], cpu=float(f[3]) if f[3] else None, nth=int(f[4]) if f[4] else None,
                         rss=int(f[5]) if f[5] else None, hwm=int(f[6]) if f[6] else None, anon=int(f[7]) if f[7] else None,
                         jb=int(f[9]) if f[9] else 0, cgcpu=int(f[10]) if f[10] else None, cganon=int(f[11]), cgshm=int(f[12]), cgfile=int(f[13])))
    except ValueError:
        pass
ev = {}
for ln in open(p('.events.tsv')):
    f = ln.split()
    if len(f) == 2:
        ev[f[1]] = float(f[0])
t0 = rows[0]['t']
print(f'== {label}: watch rows {len(rows)}, first {t0:.1f}')
for k, v in sorted(ev.items(), key=lambda x: x[1]):
    print(f'  event {k}: +{v - t0:.1f} s')
run = [r for r in rows if r['state'] == 'run' and r['cpu'] is not None and r['hwm'] is not None and r['rss'] is not None]
if run:
    last = run[-1]
    print(f"  child pid {last['pid']}, threads (max) {max(r['nth'] for r in run)}, STAR child CPU (user+sys, last sample) {last['cpu']:.1f} s, "
          f"samples {len(run)} over {run[-1]['t'] - run[0]['t']:.0f} s; VmHWM max {max(r['hwm'] for r in run) / 1048576:.1f} GiB, VmRSS max {max(r['rss'] for r in run) / 1048576:.1f} GiB")
    # utilisation per 30 s window
    out = []
    i = 0
    while i < len(run):
        j = i
        while j < len(run) and run[j]['t'] - run[i]['t'] < 30:
            j += 1
        if j >= len(run):
            j = len(run) - 1
        if j > i:
            out.append((run[i]['t'] - t0, (run[j]['cpu'] - run[i]['cpu']) / (run[j]['t'] - run[i]['t'])))
        i = j + 1 if j > i else i + 1
    print('  CPU cores in use per ~30 s window (offset s: cores): ' + ', '.join(f'{a:.0f}:{b:.1f}' for a, b in out))
# cgroup memory: floor before the pass, peak during
pre = [r for r in rows if r['t'] <= (ev.get('logout', t0))]
allr = rows
if rows:
    gib = lambda x: x / 1073741824
    base = rows[0]
    print(f"  cgroup anon+shmem at the first sample {gib(base['cganon'] + base['cgshm']):.1f} GiB; peak over the watch {gib(max(r['cganon'] + r['cgshm'] for r in rows)):.1f} GiB "
          f"(increase {gib(max(r['cganon'] + r['cgshm'] for r in rows) - base['cganon'] - base['cgshm']):.1f}); page cache (file) max {gib(max(r['cgfile'] for r in rows)):.1f} GiB")
    print(f"  cgroup CPU over the watch: {(rows[-1]['cgcpu'] - rows[0]['cgcpu']) / 1e6:.1f} s")
    jb = [r for r in rows if r['jb']]
    if jb:
        print(f"  junction file: first seen +{jb[0]['t'] - t0:.0f} s, {jb[-1]['jb'] / 1e9:.2f} GB at the end of the watch")
# STAR's own time stamps
if os.path.exists(p('.chim_Log.out')):
    print('  STAR Log.out stage lines:')
    for ln in open(p('.chim_Log.out'), errors='replace'):
        if re.match(r'^[A-Z][a-z]{2} +\d+ \d\d:\d\d:\d\d ', ln) or ln.startswith('SEQCACHE') or ln.startswith('ALL DONE'):
            print('    ' + ln.rstrip()[:160])
if os.path.exists(p('.chim_Log.final.out')):
    print('  Log.final.out:')
    for ln in open(p('.chim_Log.final.out'), errors='replace'):
        if ln.strip() and '|' in ln:
            print('    ' + ln.rstrip()[:120])
if os.path.exists(p('.chim_Log.progress.out')):
    print('  Log.progress.out (last 4 lines):')
    for ln in open(p('.chim_Log.progress.out'), errors='replace').read().splitlines()[-4:]:
        print('    ' + ln[:140])
