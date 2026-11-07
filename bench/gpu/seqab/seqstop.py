"""Sequential stopping for a paired A/B timing comparison (the campaign's tier 4).

The throughput harness (tp_gc.sbatch with SEQAB=A:B) runs arms in blocks A B B A of one point each (k runs per point) and calls this
after every block. It reads each finished point's point.json, takes the per-run metric of every run of arm A (control) and arm B
(candidate), and bootstraps the relative change of B's mean against A's, resampling runs within each point so a point's level stays its
own. Metrics: core = core-seconds per million read pairs of each run, held = the seconds each run held a device lock.

Decision, with theta the smallest improvement worth adopting (0.03 = B 3 % lower than A):
  WIN        the CI's upper end is below -theta: B is better by at least theta
  RULED-OUT  the CI's lower end is above -theta: B cannot be better by theta
  CONTINUE   neither, and both arms have fewer runs than the cap
  INCONCLUSIVE at the cap (treated as a rejection; the measurement is recorded)
Interim looks use a 99 % interval and the look at the cap a 95 % interval, a simple guard against stopping early on noise.
Prints one summary line per metric and a final line "DECISION <word>" for the primary metric.
"""
import argparse
import glob
import json
import os
import random


def runs_of(res, cfg):
    pts = []
    for meta in sorted(glob.glob(os.path.join(res, '*', 'meta.json'))):
        try:
            m = json.load(open(meta))
            pj = json.load(open(os.path.join(os.path.dirname(meta), 'point.json')))
        except (OSError, ValueError):
            continue
        if m.get('cfg') != cfg:
            continue
        vals = {'core': [], 'held': [], 'fallback': 0}
        for r in pj.get('runs', []):
            if r.get('rc', 0) != 0 or not r.get('pairs'):
                continue
            # a run that fell back to the CPU costs about three times the core-seconds of a GPU run; it would set the point's mean, so it
            # is counted (FALLBACK line) and left out of the timing (bench/gpu/fastloop/README.md: fallbacks are a failure, not a time)
            if r.get('engine_failed') or not r.get('level5', True) or not r.get('gpu_pileup_ok', True):
                vals['fallback'] += 1
                continue
            vals['core'].append(r['core_s'] / r['pairs'] * 1e6)
            if r.get('log_held_s') is not None:
                vals['held'].append(r['log_held_s'])
        pts.append((os.path.basename(os.path.dirname(meta)), vals))
    return pts


def mean(x):
    return sum(x) / len(x) if x else float('nan')


def boot(pa, pb, metric, reps, rng):
    out = []
    for _ in range(reps):
        a = [v for _, p in pa for v in (rng.choices(p[metric], k=len(p[metric])) if p[metric] else [])]
        b = [v for _, p in pb for v in (rng.choices(p[metric], k=len(p[metric])) if p[metric] else [])]
        out.append(mean(b) / mean(a) - 1)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--a', required=True, help='control config name (meta.json cfg)')
    ap.add_argument('--b', required=True, help='candidate config name')
    ap.add_argument('--metric', default='core', choices=['core', 'held'])
    ap.add_argument('--theta', type=float, default=0.03)
    ap.add_argument('--cap', type=int, default=40)
    ap.add_argument('--reps', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=1)
    o = ap.parse_args()
    rng = random.Random(o.seed)
    pa, pb = runs_of(o.res, o.a), runs_of(o.res, o.b)
    na = sum(len(p['core']) for _, p in pa)
    nb = sum(len(p['core']) for _, p in pb)
    fa = sum(p['fallback'] for _, p in pa)
    fb = sum(p['fallback'] for _, p in pb)
    if fa or fb:
        print('seqstop FALLBACK runs left out of the timing: %s %d, %s %d' % (o.a, fa, o.b, fb))
    final = min(na + fa, nb + fb) >= o.cap
    lvl = 0.95 if final else 0.99
    decision = 'CONTINUE'
    for metric in ('core', 'held'):
        if not any(p[metric] for _, p in pa) or not any(p[metric] for _, p in pb):
            continue
        d = boot(pa, pb, metric, o.reps, rng)
        lo, hi = d[int((1 - lvl) / 2 * len(d))], d[int((1 + lvl) / 2 * len(d)) - 1]
        est = mean([v for _, p in pb for v in p[metric]]) / mean([v for _, p in pa for v in p[metric]]) - 1
        word = 'WIN' if hi < -o.theta else 'RULED-OUT' if lo > -o.theta else ('INCONCLUSIVE' if final else 'CONTINUE')
        print('seqstop %s: %s vs %s, runs %d/%d, change %+.2f%%, %d%% CI [%+.2f%%, %+.2f%%], theta %.1f%%: %s'
              % (metric, o.b, o.a, nb, na, 100 * est, 100 * lvl, 100 * lo, 100 * hi, 100 * o.theta, word))
        if metric == o.metric:
            decision = word
    print('DECISION', decision)


if __name__ == '__main__':
    main()
