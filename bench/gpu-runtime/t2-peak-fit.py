#!/usr/bin/env python3
"""peakmodel: the predictor of an item's true host peak, from what the pool can know, fitted on the 18 solo profiles (t2-peak-profiles.py), and the phase rules.

usage: t2-peak-fit.py <profiles.json> <sizes.txt: "<sample> <.1fq bytes> ..." per line> [--out predictor.json] [--zone-lo 80] [--zone-hi 160]

Predictors (GiB above the idle baseline; n = reads in M from the .1fq header, bases = cDNA bases in G, S = .1fq bytes in G):
  P_B(n, bases) = a + b n + c bases      (bases: the first block's read lengths before the start, or the "[1fq-ingest] Scanned" line at +3..40 s)
  P_A(n, S)     = a + b n + c S          (header and stat only)
  fitted on the classes outside the zone; inside the zone [zone-lo, zone-hi) M reads P = max(P_x, P_fb), P_fb = planes + 6.06 + 0.49 n (the engine's E8 records can
  fall back to STAR's loop there: no read chunking, the records' device need above the budget; see T2-peakmodel.md).
Phase rules: after the "[engine_gpu] level" line F1 = u1 P + v1, after the "[allsite] async: waited" line F2 = u2 P + v2 (the smallest line of the given slope that covers every class;
the slopes are arguments of the search, printed). Margin m: the worst under-prediction rounded up (in sample), and the leave-one-out one."""
import sys, json, math
import numpy as np
prof = json.load(open(sys.argv[1])); C = prof['classes']
size = {l.split()[0]: int(l.split()[1]) for l in open(sys.argv[2]) if len(l.split()) > 1}
zlo = float(sys.argv[sys.argv.index('--zone-lo') + 1]) if '--zone-lo' in sys.argv else 70.0
zhi = float(sys.argv[sys.argv.index('--zone-hi') + 1]) if '--zone-hi' in sys.argv else 160.0
names = list(C)
def feat_n(s): return C[s]['reads'] / 1e6
def feat_b(s): return C[s]['bases'] / 1e9
def feat_S(s): return size[s] / 1e9
inzone = lambda s: zlo <= feat_n(s) < zhi
main = [s for s in names if not inzone(s)]
zone = [s for s in names if inzone(s)]
C0_ = 6.06; KFB = 0.70 * 0.70   # fallback: peak = planes + C0 + 0.70 GiB per M distinct sequences, D taken as 0.70 n (measured D / n: 0.62 and 0.67 for the two classes that fell back)
def planes(s): return (24 * C[s]['reads'] + 0.5 * C[s]['bases']) / 2 ** 30
def P_fb(s): return planes(s) + C0_ + KFB * feat_n(s)
Z = 0.0
print('classes outside the zone: %d; inside [%g, %g) M reads: %s' % (len(main), zlo, zhi, ', '.join(zone)))
def fit(fx, label):
    X = np.array([[1.0, feat_n(s), fx(s)] for s in main]); y = np.array([C[s]['peak'] for s in main])
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    loo = []
    for k in range(len(main)):
        m = np.ones(len(main), bool); m[k] = False
        bb = np.linalg.lstsq(X[m], y[m], rcond=None)[0]; loo.append(y[k] - X[k] @ bb)
    print('%s: a=%.3f b=%.4f c=%.4f  rmse %.2f  worst under-prediction %.2f (%s) over-prediction %.2f | leave-one-out worst under %.2f (%s)' % (
        label, b[0], b[1], b[2], math.sqrt((r ** 2).mean()), r.max(), main[int(np.argmax(r))], -r.min(), max(loo), main[int(np.argmax(loo))]))
    return b, r.max(), max(loo)
bB, uB, lB = fit(feat_b, 'P_B(n, bases)')
bA, uA, lA = fit(feat_S, 'P_A(n, S)    ')
def pred(s, which):
    b, f = (bB, feat_b) if which == 'B' else (bA, feat_S)
    p = b[0] + b[1] * feat_n(s) + b[2] * f(s)
    return max(p, P_fb(s)) if inzone(s) else p
mB = math.ceil(max(uB, lB)); mA = math.ceil(max(uA, lA))
print('margin: B %d GiB (max of in-sample and leave-one-out worst under-prediction, rounded up), A %d GiB' % (mB, mA))
print('%-15s %8s %8s | %7s %7s %7s | %7s %7s %7s' % ('sample', 'reads(M)', 'measured', 'P_B', 'err_B', 'P_B+m', 'P_A', 'err_A', 'P_A+m'))
worstB = worstA = -1e9
for s in names:
    pb, pa = pred(s, 'B'), pred(s, 'A'); y = C[s]['peak']
    worstB = max(worstB, y - pb); worstA = max(worstA, y - pa)
    print('%-15s %8.1f %8.1f | %7.1f %7.1f %7.1f | %7.1f %7.1f %7.1f%s' % (s, feat_n(s), y, pb, y - pb, pb + mB, pa, y - pa, pa + mA, '  (zone)' if inzone(s) else ''))
print('worst under-prediction over all 18 (measured - P): B %.1f, A %.1f' % (worstB, worstA))
# phase rules: level line and waited line
def after(s, key):
    p = C[s]; tt = np.array(p['t']); L = np.array(p['L'])
    return float(L[tt >= p[key]].max())
res = {}
for key, lab in (('level', 'F1 (after the level line)'), ('waited', 'F2 (after the waited line)')):
    M = {s: after(s, key) for s in names}
    print('\n%s: largest level from that line on, per class, against P_B' % lab)
    for u in ([0.8, 0.9, 1.0] if key == 'level' else [0.3, 0.4, 0.5, 0.6]):
        v = max(M[s] - u * pred(s, 'B') for s in names)
        over = [(M[s] - (u * pred(s, 'B') + v)) for s in names]
        print('  slope %.2f: smallest intercept covering all 18: %.1f GiB; (measured - F) worst %.1f, most over-predicted %.1f' % (u, v, max(over), min(over)))
    res[key] = M
    print('  ' + ', '.join('%s %.1f' % (s, M[s]) for s in names))
rules = {}
for key, u in (('level', 0.9), ('waited', 0.5)):
    rules[key] = dict(u=u, v=max(res[key][s] - u * pred(s, 'B') for s in names))
print('rules written: F0 = P + m; F1 = min(F0, %.2f P + %.1f + m); F2 = min(F1, %.2f P + %.1f + m), m = %d' % (rules['level']['u'], rules['level']['v'], rules['waited']['u'], rules['waited']['v'], mB))
mm = float(sys.argv[sys.argv.index('--m') + 1]) if '--m' in sys.argv else 12.0
print('phase rules with margin m = %g: F0 = P + m; F1 = min(F0, %.2f P + %.1f + m); F2 = min(F1, %.2f P + %.1f + m)' % (mm, rules['level']['u'], rules['level']['v'], rules['waited']['u'], rules['waited']['v']))
print('%-15s %6s | %6s %6s %6s | %6s %6s %6s | %6s %6s %6s' % ('sample', 'P', 'F0', 'peak', 'F0-pk', 'F1', 'M_lvl', 'F1-M', 'F2', 'M_wait', 'F2-M'))
worst = [1e9, 1e9, 1e9]
for s in names:
    P = pred(s, 'B'); F0 = P + mm; F1 = min(F0, rules['level']['u'] * P + rules['level']['v'] + mm); F2 = min(F1, rules['waited']['u'] * P + rules['waited']['v'] + mm)
    d = (F0 - C[s]['peak'], F1 - res['level'][s], F2 - res['waited'][s]); worst = [min(a, b) for a, b in zip(worst, d)]
    print('%-15s %6.1f | %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f' % (s, P, F0, C[s]['peak'], d[0], F1, res['level'][s], d[1], F2, res['waited'][s], d[2]))
print('smallest (rule - measured) over the 18 classes: F0 %.1f, F1 %.1f, F2 %.1f GiB (positive: the rule covers every class)' % tuple(worst))
if '--out' in sys.argv:
    json.dump(dict(rules=rules, baseline=prof['baseline'], B=dict(a=bB[0], b=bB[1], c=bB[2], margin=mB), A=dict(a=bA[0], b=bA[1], c=bA[2], margin=mA), zone=[zlo, zhi], fallback=dict(c0=C0_, k=KFB, dhat=0.70, gib_per_M_distinct=0.70),
                   P_B={s: pred(s, 'B') for s in names}, P_A={s: pred(s, 'A') for s in names}, M_level=res['level'], M_waited=res['waited']), open(sys.argv[sys.argv.index('--out') + 1], 'w'), indent=1)
