#!/usr/bin/env python3
"""t2-chimprof-samples.py <prefix> [--exe <binary>] [--out <file>] [--top N] [--groups-only]

Symbolizes and summarizes the wall-clock stack samples of SINGLET_CHIM_SAMPLE=<prefix> (src/star/ChimProf.cpp): <prefix>.samp.<k>
(binary), <prefix>.maps (copy of /proc/self/maps at the end), <prefix>.meta. Symbols come from `nm` (static symbol table of the binary,
dynamic table of the shared libraries), demangled with c++filt. Self = the innermost frame; inclusive = functions anywhere in the stack
(counted once per sample). Groups: each sample goes to the group of the innermost frame that belongs to one (see GROUPS); a sample whose
innermost frames are a lock or futex wait is "locks and waiting". Percent of all samples (all threads, wall clock at SAMP_HZ per thread).
Function names and counts only.
"""
import sys, glob, struct, subprocess, bisect, re, os, collections
import numpy as np


def args():
    a = sys.argv[1:]
    d = {'prefix': a[0], 'exe': None, 'out': None, 'top': 25}
    i = 1
    while i < len(a):
        if a[i] == '--exe': d['exe'] = a[i + 1]; i += 2
        elif a[i] == '--out': d['out'] = a[i + 1]; i += 2
        elif a[i] == '--top': d['top'] = int(a[i + 1]); i += 2
        else: raise SystemExit('bad argument ' + a[i])
    return d


class Syms:
    def __init__(self, path):
        self.addr = []; self.size = []; self.name = []
        for flags in (['-n', '-S', '--defined-only'], ['-D', '-n', '-S', '--defined-only']):
            try:
                o = subprocess.run(['nm'] + flags + [path], capture_output=True, text=True, timeout=300).stdout
            except Exception:
                o = ''
            rows = []
            for l in o.splitlines():
                p = l.split(None, 3)
                if len(p) == 4 and p[2] in 'TtWwiI':
                    try: rows.append((int(p[0], 16), int(p[1], 16), p[3]))
                    except ValueError: pass
            if len(rows) > 20: break
        rows.sort()
        names = [r[2] for r in rows]
        dem = {}
        if names:
            try:
                o = subprocess.run(['c++filt'], input='\n'.join(names) + '\n', capture_output=True, text=True, timeout=300).stdout.split('\n')
                dem = dict(zip(names, o))
            except Exception:
                pass
        self.addr = [r[0] for r in rows]; self.size = [r[1] for r in rows]; self.name = [dem.get(r[2], r[2]) for r in rows]

    def find(self, a):
        i = bisect.bisect_right(self.addr, a) - 1
        if i < 0: return None
        while i > 0 and self.addr[i - 1] == self.addr[i] and self.name[i].startswith('_GLOBAL__'): i -= 1   # aliases at one address: not the static-init stub
        if self.size[i] and a >= self.addr[i] + self.size[i]: return None
        return self.name[i]


def short(n):
    n = re.sub(r' \[clone [^\]]*\]', '', n)
    n = re.sub(r'\(.*$', '', n) if n.count('(') and not n.startswith('std::') else n
    n = re.sub(r'<[^<>]*>', '', n)
    return n[:110]


# innermost-first match: (group, regex on the demangled name)
GROUPS = [
    ('locks and waiting', r'pthread_mutex_lock|pthread_cond_wait|pthread_cond_timedwait|__lll_lock_wait|futex|pthread_join|sem_wait|nanosleep|clock_nanosleep'),
    ('chimeric: stitching (chimericStitching)', r'ChimericAlign::chimericStitching'),
    ('chimeric: junction output (chimericJunctionOutput, formatting)', r'ChimericAlign::chimericJunctionOutput'),
    ('chimeric: ChimericAlign other (check, construct, generateCigar)', r'ChimericAlign::'),
    ('chimeric: pair loop (chimericDetectionMult, segments, score)', r'ChimericDetection::chimericDetectionMult|chimericAlignScore|ChimericSegment::'),
    ('chimeric: ReadAlign::chimericDetection (entry)', r'ReadAlign::chimericDetection'),
    ('Solo / barcode (getCBandUMI, Solo records)', r'SoloReadBarcode|SoloReadFeature|SoloRead::|Solo::|SoloFeature'),
    ('multimapper selection (multMapSelect, mappedFilter)', r'ReadAlign::multMapSelect|ReadAlign::mappedFilter'),
    ('stitching (stitchWindowAligns, stitchAlignToTranscript, extendAlign, scoring)', r'stitchWindowAligns|stitchAlignToTranscript|extendAlign|scoreGenomicLength|stitchGapIndel|sjdbAlignSplit|sjAlignSplit|binarySearch|blocksOverlap|Transcript::|stitchWindowSeeds|ReadAlign::stitchPieces.*stitch'),
    ('seed storage / windows (createExtendWindowsWithAlign, assignAlignToWindow, stitchPieces)', r'createExtendWindowsWithAlign|assignAlignToWindow|ReadAlign::stitchPieces'),
    ('seed search (maxMappableLength2strands, suffix array)', r'maxMappableLength|compareSeqToGenome|ReadAlign::storeAligns|qualitySplit|ReadAlign::mapOneRead|suffixArraySearch|funCompareSuffixes|compareRefEnds'),
    ('output (outputAlignments, SAM/unmapped/SJ output)', r'ReadAlign::output|chunkFstreamCat|outputTranscript|OutSJ::|ReadAlign::outReads|ReadAlign::softclip'),
    ('input parsing (readLoad, chunk read, clip, chunk sort)', r'readLoad|readBarcodeLoad|processChunks|sortChunkByPrefix|ClipMate|clipChunk|ReadAlignChunk::mapChunk|convertNt|convertNucleotides|complementSeqNumbers'),
    ('per-read driver (oneRead self, qualHist)', r'ReadAlign::oneRead'),
]
GRX = [(g, re.compile(r)) for g, r in GROUPS]
WAIT_LEAF = re.compile(r'pthread_mutex_lock|pthread_cond_wait|pthread_cond_timedwait|__lll_lock_wait|futex|pthread_join|sem_wait|nanosleep')


def main():
    A = args(); out = []; P = out.append
    meta = dict(l.split() for l in open(A['prefix'] + '.meta'))
    sb = int(meta['samp_bytes']); depth = int(meta['samp_depth']); hz = float(meta['samp_hz'])
    dt = np.dtype([('rip', '<u8'), ('depth', '<u4'), ('pad', '<u4'), ('pc', '<u8', (depth,))])
    assert dt.itemsize == sb, (dt.itemsize, sb)
    files = sorted(glob.glob(A['prefix'] + '.samp.*'))
    per = [np.fromfile(f, dtype=dt) for f in files]
    S = np.concatenate(per); n = len(S)
    # address map
    maps = []; base = {}
    for l in open(A['prefix'] + '.maps'):
        p = l.split(None, 5)
        if len(p) < 6: continue
        s, e = [int(x, 16) for x in p[0].split('-')]
        off = int(p[2], 16); path = p[5].strip()
        if path.startswith('/'):   # load bias of a file: the lowest (mapping start - file offset) over all its mappings
            b = s - off
            if path not in base or b < base[path]: base[path] = b
        if 'x' in p[1]: maps.append((s, e, off, path))
    symcache = {}
    exe = A['exe']
    mp = sorted(maps)
    starts = [m[0] for m in mp]

    def sym(a):
        i = bisect.bisect_right(starts, a) - 1
        if i < 0 or a >= mp[i][1]: return '[unmapped]'
        path = mp[i][3]
        use = exe if (exe and os.path.basename(path) == os.path.basename(exe)) else path
        if use not in symcache:
            symcache[use] = Syms(use) if os.path.exists(use) else None
        s = symcache[use]
        v = a - base.get(path, 0)
        nm = s.find(v) if s else None
        if nm is None:
            nm = s.find(v - 1) if s else None   # a return address just past a call
        return nm if nm else '[%s unresolved]' % os.path.basename(path)

    cache = {}

    def sy(a):
        if a not in cache: cache[a] = sym(a)
        return cache[a]

    self_c = collections.Counter(); incl_c = collections.Counter(); grp = collections.Counter(); sub = collections.defaultdict(collections.Counter)
    okrip = 0; unres = 0
    for k in range(n):
        d = int(S['depth'][k]); pcs = [int(x) for x in S['pc'][k][:d]]; rip = int(S['rip'][k])
        j = next((i for i in range(min(3, len(pcs))) if pcs[i] == rip), None)
        if j is not None: okrip += 1; fr = pcs[j:]
        else: fr = [rip] + pcs[2:]
        names = [sy(a) for a in fr]
        if names[0].startswith('['): unres += 1
        self_c[short(names[0]) + ((' <- ' + short(names[1])) if names[0].startswith('[') and len(names) > 1 else '')] += 1
        for nme in set(short(x) for x in names): incl_c[nme] += 1
        g = None
        # waiting: the innermost 4 frames hold a wait primitive
        if any(WAIT_LEAF.search(x) for x in names[:4]): g = 'locks and waiting'
        else:
            for x in names:
                for gname, rx in GRX:
                    if rx.search(x): g = gname; break
                if g: break
        if g is None: g = 'other (' + ('libc/other library' if names[0].startswith('[') else 'main binary, no group') + ')'
        grp[g] += 1; sub[g][short(names[0])] += 1
    P('# t2-chimprof stack samples of %s: %d samples of %d threads at %g Hz (wall clock): %.1f thread-seconds' % (A['prefix'], n, len(files), hz, n / hz))
    P('samples whose first backtrace frame equals the interrupted pc: %.1f %%; leaf unresolved (no symbol): %.1f %%' % (100 * okrip / n, 100 * unres / n))
    P('per thread samples: ' + ' '.join(str(len(p)) for p in per))
    P('')
    P('## groups (innermost frame that belongs to a group; percent of all samples; seconds = samples / Hz summed over threads)')
    for g, c in sorted(grp.items(), key=lambda x: -x[1]):
        P('%-90s %8d  %6.2f %%  %9.1f thread-s' % (g, c, 100 * c / n, c / hz))
    P('')
    P('## self time, top %d (innermost frame)' % A['top'])
    for nme, c in self_c.most_common(A['top']): P('%6.2f %%  %s' % (100 * c / n, nme))
    P('')
    P('## inclusive time, top %d (function anywhere in the stack)' % A['top'])
    for nme, c in incl_c.most_common(A['top']): P('%6.2f %%  %s' % (100 * c / n, nme))
    P('')
    P('## innermost functions inside each group (top 6)')
    for g, c in sorted(grp.items(), key=lambda x: -x[1]):
        P('%s (%.2f %%)' % (g, 100 * c / n))
        for nme, cc in sub[g].most_common(6): P('    %6.2f %% of all  %s' % (100 * cc / n, nme))
    txt = '\n'.join(out) + '\n'
    if A['out']: open(A['out'], 'w').write(txt)
    sys.stdout.write(txt)


if __name__ == '__main__':
    main()
