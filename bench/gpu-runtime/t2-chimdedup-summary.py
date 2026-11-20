#!/usr/bin/env python3
"""t2-chimdedup-summary.py <results-dir> : one line per run of a t2-chimdedup job (counts, hashes, seconds, bytes only).

Reads <dir>/results.txt (RUN ... lines), <dir>/<label>.watch.tsv and <label>.events.tsv of t2-chimdedup-watch.sh.
Per run: item wall, stage A seconds and distinct fraction (the 'dedup:' log line), pass wall (epoch of the 'reads into the second pass' line to
the epoch of the 'second pass' line), the junction-line counters, the table md5, the STAR child's CPU seconds (last sample, user+sys), its VmHWM,
the job cgroup anon+shmem at the first sample, at the pass start (chim_Log.out) and its peak, and the parent's own memory line.
"""
import re, sys, os, collections

d = sys.argv[1]
runs = collections.OrderedDict()
for ln in open(os.path.join(d, 'results.txt'), errors='replace'):
    m = re.match(r'RUN (\S+) (.*)', ln.rstrip('\n'))
    if m:
        runs.setdefault(m.group(1), []).append(m.group(2))
GiB = 1 << 30
for L, lines in runs.items():
    o = collections.OrderedDict()
    for t in lines:
        m = re.match(r'samples: (\d+)\t([\d.]+)\t(\d+)', t)
        if m: o['item_wall_s'] = m.group(2); o['pairs'] = m.group(3)
        m = re.match(r'LOG: (\d+) \[singlet\] chimeric: ([0-9]+) reads into the second pass', t)
        if m: o['t_in'] = int(m.group(1)); o['reads_in'] = int(m.group(2))
        m = re.match(r'LOG: (\d+) \[singlet\] chimeric: second pass .*?; (\d+) junction lines, (\d+) kept, (\d+) without barcode, (\d+) unmatched, (\d+) ambiguous, (\d+) non-canonical, (\d+) turned', t)
        if m:
            o['t_out'] = int(m.group(1))
            o['counters'] = 'lines %s kept %s nobc %s unmatched %s ambiguous %s noncanon %s turned %s' % m.groups()[1:]
        m = re.match(r'LOG: (\d+) \[singlet\] chimeric: dedup: (\d+) reads, (\d+) distinct sequences \(([\d.]+) %\), ([\d.]+) s', t)
        if m: o['dedup'] = 'reads %s distinct %s (%s %%) stageA_s %s' % m.groups()[1:]
        m = re.match(r'LOG: \d+ \[singlet\] chimeric: dedup memory: (.*)', t)
        if m: o['parent_mem'] = m.group(1)
        m = re.match(r'table md5 \(decompressed\) (\w+) rows (\d+)', t)
        if m: o['table_md5'] = m.group(1); o['table_lines'] = m.group(2)
        if t.startswith('RAWNORM') or t.startswith('RAWPROOF'): o.setdefault('raw', []).append(t)
    if 't_in' in o and 't_out' in o: o['pass_wall_s'] = o['t_out'] - o['t_in']
    w = os.path.join(d, L + '.watch.tsv'); e = os.path.join(d, L + '.events.tsv')
    ev = {}
    if os.path.exists(e):
        for ln in open(e):
            p = ln.split()
            if len(p) == 2: ev[p[1]] = float(p[0])
    rows = []
    if os.path.exists(w):
        for ln in open(w):
            if ln.startswith('#'): continue
            p = ln.rstrip('\n').split('\t')
            if len(p) < 15: continue
            try: rows.append((float(p[0]), p[1], p[3], p[6], int(p[11] or 0) + int(p[12] or 0)))
            except ValueError: pass
    if rows:
        o['cg_anon_shmem_first_GiB'] = '%.1f' % (rows[0][4] / GiB)
        lo = [r for r in rows if 'logout' in ev and r[0] >= ev['logout']]
        if lo: o['cg_anon_shmem_pass_start_GiB'] = '%.1f' % (lo[0][4] / GiB)
        o['cg_anon_shmem_peak_GiB'] = '%.1f' % (max(r[4] for r in rows) / GiB)
        cpu = [float(r[2]) for r in rows if r[2]]
        if cpu: o['child_cpu_s'] = '%.1f' % max(cpu)
        hw = [int(r[3]) for r in rows if r[3]]
        if hw: o['child_vmhwm_GiB'] = '%.1f' % (max(hw) / 1048576)
        # peak of the first part (stage A) : samples before the log
        if 'dedupfile' in ev and 'logout' in ev:
            sa = [r for r in rows if ev['dedupfile'] <= r[0] < ev['logout']]
            if sa: o['cg_anon_shmem_stageA_peak_GiB'] = '%.1f' % (max(r[4] for r in sa) / GiB)
    print('== ' + L)
    for k, v in o.items():
        if isinstance(v, list):
            for x in v: print('   ' + k + ': ' + x)
        else:
            print('   %s: %s' % (k, v))
