#!/usr/bin/env python3
"""peakmodel: the ledger of the true host peak (GiB above the idle baseline) of the 18 sample classes, from the code's size rules and the engine's logged sizes.

usage: t2-peak-ledger.py <profiles.json> <classes_engine.json> [predictor.json]

Rule (engine path, level 5; valid for the 16 classes that did not fall back):
  peak = planes + region + 0.9 x Solo keep + late store + (a + b n)
  planes     = 24 n + bases / 2                       .1fq decode: off 8 B/read, barcode read 16 B/read, cDNA 2-bit and its quality 2-bit (0.5 B/base)   [logged: E22 "released N GiB of host planes"]
  region     = (28 + 8 nw) n + 16 D + 18 nAln + 4 nOps + nSeq   E2E shared region: per-read entries 24 B + low-quality masks 8 nw B + readRecord 4 B, records 16 B, alignments 16 + 2 B, ops 4 B, seq
               (nw = ceil(longest read / 64): 2 for 91 bp, 3 for 150 bp; aux = (2 + nw) x 8 B per read in the decode)   [logged: "N MB in use"]
  Solo keep  = page-locked host copy of E11's kept inputs (CR/UR 16 B/read + the records' gene annotation and the read -> record map); 0.9 of the logged "Solo inputs kept ... N GB"
  late store = U x (1.25 L + 38) B                    the compact store of the U unmapped reads (cDNA 1 B/base + quality 0.25 B/base + barcode read 16 B + name and offsets)   [logged: E22 "compact store N GiB"]
  a + b n    = 6.06 GiB + 0.0238 GiB per M reads: the parent item process (2.4-3.1), the non-host server's growth (1.1), STAR's private structures kept to its loop (per-read arrays, offsets 16 B x D, the
               index copy where STAR loads one); measured as the remainder, NOT itemised
Fallback classes (level 0; E8 records not built, the hand-off file path): peak = planes + 6.06 + 0.70 GiB per M distinct sequences (STAR's sequence cache 3 x (blob + seq / 4 + 40 D), host records, hand-off summaries)."""
import sys, json
import numpy as np
G = 2.0 ** 30
prof = json.load(open(sys.argv[1]))['classes']; C = json.load(open(sys.argv[2]))
rows = []
print('%-15s %7s | %7s %7s %7s %7s %7s %5s | %7s %7s | %6s' % ('class', 'n (M)', 'planes', 'region', '.9 keep', 'late', 'a+bn', 'U/n', 'sum', 'measured', 'resid'))
res = []
for s, p in prof.items():
    d = C[s]; n = d['n']; b = p['bases']
    P = (24 * n + 0.5 * b) / G
    if 'regionMiB' in d and 'nAln' in d:
        nw = -(-p['lmax'] // 64); R = d['regionMiB'] / 1024.0; Rf = ((28 + 8 * nw) * n + 16 * d['D'] + 18 * d['nAln'] + 4 * d['nOps'] + d['nSeq']) / G
        K = 0.9 * d['soloKeepGiB']; La = d['lateGiB']; O = 6.06 + 0.0238 * n / 1e6
        tot = P + R + K + La + O; r = p['peak'] - tot; res.append(r)
        print('%-15s %7.1f | %7.1f %7.1f %7.1f %7.1f %7.1f %5.2f | %7.1f %7.1f | %6.1f   (region by the rule %.1f, planes logged %.1f)' % (s, n / 1e6, P, R, K, La, O, d['U'] / n, tot, p['peak'], r, Rf, d['planesGiB']))
    else:
        D = d['D']; tot = P + 6.06 + 0.70 * D / 1e6
        print('%-15s %7.1f | fallback (level 0): planes %.1f + 6.06 + 0.70 x D (%.1f M) = %.1f | %7.1f measured | %6.1f' % (s, n / 1e6, P, D / 1e6, tot, p['peak'], p['peak'] - tot))
res = np.array(res)
print('engine-path classes: n=%d, rmse %.1f GiB, residual (measured - sum) from %.1f to %.1f GiB' % (len(res), np.sqrt((res ** 2).mean()), res.min(), res.max()))
