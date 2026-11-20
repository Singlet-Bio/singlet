#!/usr/bin/env python3
"""peakmodel: the engine's own size figures per sample class, from the per-item engine logs of a mix run (rt_mix / t2-rt_mix: engine.c<pool>.<n>.log = the lines of the item's STAR Log.out
that mention engine_gpu / gpu_pileup), into classes_engine.json.

usage: t2-peak-engine-lines.py <mix results dir with items.tsv> <engine_lines.txt> [out.json]
engine_lines.txt is made on the cluster with
  cd <results dir> && grep -a -H -E 'E8 records on the device|engine_gpu: [0-9]+ reads, [0-9]+ distinct|E22 early release|per-read hand-off entries|Solo inputs kept|E17 budget|fallback|E10 level 5|read chunks' engine.c*.log | cut -c1-520
(items.tsv: "<pool> <item> <sample>" per line). Per class (the first item of the class): mode (normal / lean), read chunks (RC), reads, distinct sequences D, records, alignments, ops, seq bytes, region MiB,
planes GiB (E22 'released'), late store GiB and unmapped reads U (E22 'compact store'), Solo keep GiB, E17's estimates."""
import re, json, sys, os, collections
D = sys.argv[1]; LINES = sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else 'classes_engine.json'
items = {}
for ln in open(os.path.join(D, 'items.tsv')):
    f = ln.split(); items['engine.c%s.%s.log' % (f[0], f[1])] = f[2]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for ln in open(LINES, errors='replace'):
    fn, _, txt = ln.partition(':')
    s = items.get(fn)
    if s: by[s][fn].append(txt.strip())
out = {}
for s, files in by.items():
    fn = sorted(files)[0]; d = dict(file=fn)
    for t in files[fn]:
        m = re.match(r'engine_gpu: E17 budget .*?mode (lean|normal).*?reads ([\d.]+) GB, E10 aux ([\d.]+) GB, entries\+masks ([\d.]+) GB, E11 keep ([\d.]+) GB, clip ([\d.]+) GB; estimates: normal ([\d.]+) GB, lean ([\d.]+) GB', t)
        if m: d.update(mode=m.group(1), RD=float(m.group(2)), AUX=float(m.group(3)), ENTQM=float(m.group(4)), KEEP=float(m.group(5)), CLIP=float(m.group(6)), needN=float(m.group(7)), needL=float(m.group(8)))
        m = re.match(r'engine_gpu: P3-MEM-03 reads phase in (\d+) read chunks', t)
        if m: d['rc'] = int(m.group(1))
        m = re.match(r'engine_gpu: (\d+) reads, (\d+) distinct \((\d+) with N\)', t)
        if m: d.update(n=int(m.group(1)), D=int(m.group(2)), Dn=int(m.group(3)))
        m = re.match(r'engine_gpu: E8 records on the device(?: in [^:]*)?: (\d+) records, (\d+) alignments, (\d+) ops, (\d+) seq bytes', t)
        if m: d.update(nRec=int(m.group(1)), nAln=int(m.group(2)), nOps=int(m.group(3)), nSeq=int(m.group(4)))
        m = re.match(r'ENGINE_GPU: E8: per-read hand-off entries and low-quality masks in the shared region \((\d+) MB in use', t)
        if m: d['regionMiB'] = int(m.group(1))
        m = re.match(r'ENGINE_GPU: E22 early release: released ([\d.]+) GiB of host planes .*compact store ([\d.]+) GiB for (\d+) unmapped reads of (\d+)', t)
        if m: d.update(planesGiB=float(m.group(1)), lateGiB=float(m.group(2)), U=int(m.group(3)))
        m = re.match(r'engine_gpu: E11: Solo inputs kept on the device: .*?, ([\d.]+) GB', t)
        if m: d['soloKeepGiB'] = float(m.group(1))
        if 'fallback' in t and 'E8 records not built' in t:
            m = re.search(r'needs about ([\d.]+) GB', t); d['fallback_need'] = float(m.group(1)) if m else True
    out[s] = d
json.dump(out, open(OUT, 'w'), indent=1)
print('%-15s %6s %3s %9s %8s %8s %8s %7s %7s %7s %7s %8s' % ('class', 'mode', 'rc', 'n(M)', 'D(M)', 'nAln(M)', 'nSeq(G)', 'U(M)', 'regMiB', 'planes', 'late', 'soloKeep'))
for s, d in sorted(out.items(), key=lambda kv: kv[1].get('n', 0)):
    print('%-15s %6s %3s %9.1f %8.2f %8.2f %8.2f %7.2f %7s %7s %7s %8s' % (s, d.get('mode', '?'), d.get('rc', '-'), d.get('n', 0) / 1e6, d.get('D', 0) / 1e6, d.get('nAln', 0) / 1e6, d.get('nSeq', 0) / 1e9, d.get('U', 0) / 1e6,
          d.get('regionMiB', '-'), d.get('planesGiB', '-'), d.get('lateGiB', '-'), d.get('soloKeepGiB', '-')))
