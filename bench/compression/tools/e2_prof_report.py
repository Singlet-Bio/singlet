"""Aggregate an e2 sampling profile by source line: python -P tools/e2_prof_report.py prof.txt [exe]"""
import sys, subprocess, collections, os
prof = sys.argv[1]
exe = sys.argv[2] if len(sys.argv) > 2 else 'bin/e2_bench_prof.exe'
tc = '/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix/bin'
a2l = 'C:/RBuildTools/4.4/x86_64-w64-mingw32.static.posix/bin/addr2line.exe'
rows = []
for ln in open(prof):
    if ln.startswith('#'): continue
    c, a = ln.split()
    rows.append((int(c), a))
tot = sum(c for c, _ in rows)
out = subprocess.run([a2l, '-e', exe, '-f', '-C', '-s'] + [a for _, a in rows], capture_output=True, text=True).stdout.split('\n')
byline = collections.Counter(); byfunc = collections.Counter()
for i, (c, a) in enumerate(rows):
    fn = out[2 * i].strip(); loc = out[2 * i + 1].strip()
    byline[loc] += c; byfunc[fn[:60]] += c
print('total samples', tot)
print('-- by function'); [print(f'{100.0*c/tot:5.1f}%  {f}') for f, c in byfunc.most_common(12)]
print('-- by line'); [print(f'{100.0*c/tot:5.1f}%  {l}') for l, c in byline.most_common(40)]
