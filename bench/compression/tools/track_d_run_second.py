#!/usr/bin/env python
"""Run bin/track_d_second.exe on every codec dump that has no row yet in results/track_d_second.tsv.
Re-run it whenever new codecs have been dumped (bin/track_d_dump.exe --skip-existing ...).

    python -P tools/track_d_run_second.py [--threads 2] [--methods zstd22,xz,brotli,bz2,cm] [--max-bytes N]
"""
import csv, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from track_d_common import ROOT


def main():
    threads = 2
    methods = 'zstd22,xz,brotli,bz2,cm'
    max_bytes = 10**12
    codecs = None
    a = sys.argv[1:]
    while a:
        k = a.pop(0)
        if k == '--threads': threads = int(a.pop(0))
        elif k == '--methods': methods = a.pop(0)
        elif k == '--max-bytes': max_bytes = int(a.pop(0))
        elif k == '--codecs': codecs = a.pop(0).split(',')
    dump = os.path.join(ROOT, 'results', 'track_d_dump')
    tsv = os.path.join(ROOT, 'results', 'track_d_second.tsv')
    done = {}
    if os.path.exists(tsv):
        for r in csv.DictReader(open(tsv), delimiter='\t'):
            done.setdefault(r['file'], set()).add(r['method'])
    want = set(methods.split(','))
    todo = []
    for fn in sorted(os.listdir(dump)):
        if not fn.endswith('.bin'):
            continue
        p = os.path.join(dump, fn)
        if codecs and fn[:-4].split('__', 2)[2] not in codecs:
            continue
        if os.path.getsize(p) > max_bytes or time.time() - os.path.getmtime(p) < 120:
            continue
        miss = want - done.get(fn, set())
        if miss:
            todo.append((p, sorted(miss)))
    print(f'{len(todo)} dump files need second-pass work')
    # group by identical missing-method sets
    groups = {}
    for p, miss in todo:
        groups.setdefault(','.join(miss), []).append(p)
    exe = os.path.join(ROOT, 'bin', 'track_d_second.exe')
    for ms, files in groups.items():
        subprocess.run([exe, '--methods', ms, '--out', tsv, '--threads', str(threads)] + files, cwd=ROOT, check=False)


if __name__ == '__main__':
    main()
