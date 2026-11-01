#!/usr/bin/env python3
"""Resolve a samprof run (samples.bin + maps.txt) into per-thread, per-function CPU tables.
usage: resolve.py RUNDIR BINARY ADDR2LINE [--ms 5] [--top 40]   -> writes RUNDIR/profile.md and prints it.
self = innermost inlined function at the sampled pc; own = the real (outermost) function holding it; inclusive = real functions on the
stack, each counted once per sample. cpu_s = samples * ms / 1000."""
import collections, re, struct, subprocess, sys

rd, binary, a2l = sys.argv[1:4]
ms = int(sys.argv[sys.argv.index('--ms') + 1]) if '--ms' in sys.argv else 5
top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 40
import glob
S = 16 + 4 + 4 + 8 * 32
recs = []
for fnm in sorted(glob.glob(rd + '/samples.*.bin')) or [rd + '/samples.bin']:
    pid = fnm.rsplit('samples.', 1)[-1][:-4]
    with open(fnm, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        raw = bytes(8) + fh.read(n * S)
    for i in range(n):
        o = 8 + i * S
        k = struct.unpack_from('<I', raw, o + 16)[0]
        recs.append(('pid' + pid, struct.unpack_from('<32Q', raw, o + 24)[:k]))   # the "thread" column is the process
maps = []
mapf = (sorted(glob.glob(rd + '/maps.*.txt')) or [rd + '/maps.txt'])[0]
for ln in open(mapf):
    p = ln.split(None, 5)
    if len(p) < 6 or 'x' not in p[1]:
        continue
    lo, hi = (int(x, 16) for x in p[0].split('-'))
    maps.append((lo, hi, int(p[2], 16), p[5].strip()))
exe = binary.rsplit('/', 1)[-1]
exe_maps = [m for m in maps if m[3].endswith('/' + exe)]
exe_base = 0
if exe_maps:
    lo = min(m[0] for m in exe_maps)
    off = min(m[2] for m in exe_maps)
    exe_base = lo - off if lo - off > 0x10000000 else 0   # PIE load base; non-PIE addresses are absolute


def module(pc):
    for lo, hi, off, path in maps:
        if lo <= pc < hi:
            return path
    return None


need = sorted({pc for _, pcs in recs for pc in pcs if (module(pc) or '').endswith('/' + exe)})
chain = {}
if need:
    inp = '\n'.join(hex(pc - exe_base - 1) for pc in need) + '\n'     # -1: return address -> call site
    out = subprocess.run([a2l, '-a', '-f', '-i', '-C', '-e', binary], input=inp, capture_output=True, text=True).stdout.split('\n')
    cur = None
    i = 0
    while i < len(out):
        if re.match(r'^0x[0-9a-f]+$', out[i]):
            cur = int(out[i], 16) + exe_base + 1
            chain[cur] = []
            i += 1
        elif cur is not None and out[i] and i + 1 < len(out):
            chain[cur].append(out[i])
            i += 2
        else:
            i += 1


def fn(pc, which):
    c = chain.get(pc)
    if c:
        return c[0] if which == 'self' else c[-1]
    m = module(pc)
    return '[' + (m.rsplit('/', 1)[-1] if m else '?') + ']'


tot = len(recs)
if '--cpu' in sys.argv:   # the process timer under-counts multithreaded CPU (about 1 sample per wall tick); scale to the measured total
    ms = float(sys.argv[sys.argv.index('--cpu') + 1]) * 1000.0 / tot


def sec(c):
    return c * ms / 1000.0


bythread = collections.Counter(r[0] for r in recs)
L = ['# samprof ' + rd + '\n', '%d samples x %d ms = %.0f CPU-s\n' % (tot, ms, sec(tot)), '\n## By thread name\n',
     '| thread | CPU-s | % |', '|---|---|---|']
for t, c in bythread.most_common(12):
    L.append('| %s | %.0f | %.1f |' % (t, sec(c), 100 * c / tot))
selfc, ownc, inc = collections.Counter(), collections.Counter(), collections.Counter()
for name, pcs in recs:
    if not pcs:
        continue
    selfc[fn(pcs[0], 'self')] += 1
    ownc[fn(pcs[0], 'own')] += 1
    for f in {fn(pc, 'own') for pc in pcs}:
        inc[f] += 1
for title, c in (('Self (innermost inlined function)', selfc), ('Own (real function holding the pc)', ownc),
                 ('Inclusive (real functions on the stack)', inc)):
    L += ['\n## ' + title + '\n', '| function | CPU-s | % |', '|---|---|---|']
    for f, k in c.most_common(top):
        L.append('| `%s` | %.0f | %.1f |' % (f[:110], sec(k), 100 * k / tot))
for t in [t for t, _ in bythread.most_common(4)]:
    c = collections.Counter(fn(p[0], 'own') for nm, p in recs if nm == t and p)
    L += ['\n## Own functions on thread `%s` (%.0f CPU-s)\n' % (t, sec(bythread[t])), '| function | CPU-s | %% of thread |', '|---|---|---|']
    for f, k in c.most_common(15):
        L.append('| `%s` | %.0f | %.1f |' % (f[:110], sec(k), 100 * k / bythread[t]))

# ---- phase categories by the real function holding the pc, per process (largest pid = STAR child, first pid = singlet parent) ----
CATS = [
    ('STAR seed search', r'compareSeqToGenome|maxMappableLength|SeqCache|storeAligns|ReadAlign::seed|stitchAlignToTranscript_disabled'),
    ('STAR clip/opal', r'opal|ClipCR4|searchDatabase|Clip'),
    ('STAR stitch/extend', r'stitchPieces|stitchWindowAligns|extendAlign|stitchAlignToTranscript|alignToTranscript|Transcript::|sjAlignSplit|createExtendWindows|assignAlignToWindow|multMapSelect|mappedFilter|scoreGenomic|stitchGapIndel|blocksOverlap'),
    ('STAR annotate/solo/output', r'classifyAlign|hashLookupCB|alignBAM|Solo|outputAlign|ReadAlign::output|samAttr|Transcriptome|bam_|BAM'),
    ('STAR read/parse', r'oneRead|mapChunk|processChunks|ReadAlignChunk|revComplement|readLoad|ReadAlign::'),
    ('singlet pileup/count', r'PileupEngine|IntervalTree|GeneModel|DirectionalUmi|SparseAccumulator|SnpConsensus|UmiFeature|inline_scan|BarcodeIndex|umi_|extract_splice|pileup_|refine_read'),
    ('singlet barcode correction', r'CbCorrector'),
    ('singlet donor demux (VB/auto-K)', r'run_vb|fit_vb|DonorDemux|donor|Donor|vb_|parallel_det_chunks'),
    ('singlet ambient/decontX/doublet', r'decontx|ambient|doublet|Doublet'),
    ('singlet non-host screen', r'nonhost'),
    ('singlet called_v2/export/other', r'calledv2|export_|singlet::'),
    ('libc/libm/libstdc++/other libs (leaf)', r'^\[lib|^\[\?\]'),
]
def cat(f):
    for name, rx in CATS:
        if re.search(rx, f): return name
    return 'other'
byp = collections.defaultdict(list)
for name, pcs in recs:
    byp[name].append(pcs)
pids = sorted(byp, key=lambda k: -len(byp[k]))
L += ['
## Phase categories per process (CPU-s)
']
for pid in pids:
    if len(byp[pid]) * ms / 1000.0 < 5: continue
    cc = collections.Counter(cat(fn(pcs[0], 'own')) for pcs in byp[pid] if pcs)
    L += ['
### %s (%.0f CPU-s)
' % (pid, sec(len(byp[pid]))), '| category | CPU-s | %% of process |', '|---|---|---|']
    for c, k in cc.most_common(): L.append('| %s | %.0f | %.1f |' % (c, sec(k), 100 * k / len(byp[pid])))
    lc = collections.Counter()
    for pcs in byp[pid]:
        if pcs and fn(pcs[0], 'own').startswith('['):
            nxt = next((fn(q, 'own') for q in pcs[1:] if not fn(q, 'own').startswith('[')), '?')
            lc[nxt] += 1
    L += ['
library leaf time by the nearest function above it (one frame up: libc leaves keep no frame pointer)
', '| function | CPU-s |', '|---|---|']
    for f, k in lc.most_common(14): L.append('| `%s` | %.0f |' % (f[:110], sec(k)))
open(rd + '/profile.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
