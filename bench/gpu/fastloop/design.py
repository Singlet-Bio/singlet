#!/usr/bin/env python3
"""Stacked screening of GPU ideas: a two-level (fractional) factorial design written as one pilot item (bench/gpu/fastloop/README.md).

Every idea is a switch (an atom of tp_gc.sbatch's composite configs, usually a name in ideas.tsv). k ideas run in a 2^(k-p) design of
resolution IV or better: main effects are clear of each other and of every two-way interaction. Each design cell becomes one tp_gc point
whose config joins the ideas that are on with '+' (all off: base). Blocks repeat the cells, the second in the reverse order of the first,
so drift over the job falls equally on every factor. The item runs the points on one pinned GPU, then analyze.py fits the effects.

  design.py --ideas wshrink,shidx,nhtries1 --tier S1 --bin 99105bd8-multi --name 60-fl-cal-s1 [--blocks 2] [--full] [--sole]
            [--out DIR] [--seed N]
Prints the design (cells, aliases, run count, estimated minutes) and writes DIR/<name>.g1.sh.
"""
import argparse
import itertools
import os
import random
import sys

# tier: sample, point shape (tp_gc tag without the config), atoms every cell carries, first run of a k1 point dropped, minutes per point (H100)
TIERS = {
    'S1': dict(sample='pbmc2m', tag='k1t8s1r3', extra=[], drop_first=1, min_per_point=2.3,
               note='pbmc2m, one run at a time, 3 runs per cell point (the first is warm-up), whole pipeline'),
    'S2': dict(sample='pbmc2m', tag='k4t2s2r2', extra=[], drop_first=0, min_per_point=2.5,
               note='pbmc2m, 4 in flight on 2 device slots, 8 runs per cell point, whole pipeline'),
    'S3': dict(sample='pbmc1k_v3', tag='k1t8s1r2', extra=['maponly'], drop_first=1, min_per_point=4.5,
               note='pbmc1k_v3, one run at a time, 2 runs per cell point (the first is warm-up), ends after STAR and the pileup'),
    'FULL': dict(sample='pbmc1k_v3', tag='k8t7s2r1', extra=[], drop_first=0, min_per_point=5.5,
                 note='the campaign\'s tier-4 shape: pbmc1k_v3, 8 in flight on 2 slots, one round per cell point'),
}

# generators of the generated factors, by the number of factors (letters index the base factors): resolution IV or V, fewest runs
GEN = {
    4: ['ABC'],                                   # 2^(4-1) IV, 8 cells
    5: ['ABCD'],                                  # 2^(5-1) V, 16
    6: ['ABC', 'BCD'],                            # 2^(6-2) IV, 16
    7: ['ABC', 'BCD', 'ACD'],                     # 2^(7-3) IV, 16
    8: ['BCD', 'ACD', 'ABC', 'ABD'],              # 2^(8-4) IV, 16
    9: ['BCDE', 'ACDE', 'ABDE', 'ABCE'],          # 2^(9-4) IV, 32
    10: ['BCDE', 'ACDE', 'ABDE', 'ABCE', 'ABCD'],  # 2^(10-5) IV, 32
}
L = 'ABCDEFGHJK'


def cells(k, full):
    """Rows of +1/-1 per factor."""
    if k <= 3 or full:
        return [list(r) for r in itertools.product([-1, 1], repeat=k)]
    g = GEN[k]
    nb = k - len(g)
    rows = []
    for r in itertools.product([-1, 1], repeat=nb):
        row = list(r)
        for w in g:
            v = 1
            for ch in w:
                v *= row[L.index(ch)]
            row.append(v)
        rows.append(row)
    return rows


def defining_words(k, full):
    """The defining relation (sets of factor letters) of the fraction."""
    if k <= 3 or full:
        return []
    nb = k - len(GEN[k])
    gens = [frozenset(w) | {L[nb + i]} for i, w in enumerate(GEN[k])]
    words = set()
    for n in range(1, len(gens) + 1):
        for c in itertools.combinations(gens, n):
            w = frozenset()
            for x in c:
                w = w ^ x
            words.add(w)
    return sorted(words, key=lambda w: (len(w), sorted(w)))


def aliases(k, full, names):
    """Two-way interactions aliased with each other (words of length 4), as text."""
    words = defining_words(k, full)
    out = []
    seen = set()
    for a, b in itertools.combinations(range(k), 2):
        t = frozenset({L[a], L[b]})
        if t in seen:
            continue
        chain = [t]
        for w in words:
            x = t ^ w
            if len(x) == 2:
                chain.append(x)
        for x in chain:
            seen.add(x)
        if len(chain) > 1:
            out.append(' = '.join(':'.join(names[L.index(ch)] for ch in sorted(x)) for x in chain))
    res = min((len(w) for w in words), default=None)
    return res, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ideas', required=True, help='comma-separated switch names (ideas.tsv or tp_gc atoms)')
    ap.add_argument('--tier', default='S1', choices=sorted(TIERS))
    ap.add_argument('--bin', required=True, help='binary directory name under $W/jobs/gpu-pr2/bin2 (its commit-gpu must match)')
    ap.add_argument('--name', required=True, help='item name, <order>-<name>')
    ap.add_argument('--blocks', type=int, default=2)
    ap.add_argument('--full', action='store_true', help='full factorial even for k >= 4')
    ap.add_argument('--sole', action='store_true', help='FL_SOLE=1: the item is the only one on its node (one-GPU pilot, smoke node)')
    ap.add_argument('--theta', type=float, default=0.03)
    ap.add_argument('--metric', default='held', help='primary metric of the decisions (analyze.py names)')
    ap.add_argument('--out', default='.')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tag', default=None, help='override the tier\'s point shape, e.g. k1t8s1r4')
    ap.add_argument('--sample', default=None, help='override the tier\'s sample')
    o = ap.parse_args()
    names = [x for x in o.ideas.split(',') if x]
    k = len(names)
    if k > 10 or (k >= 4 and not o.full and k not in GEN):
        sys.exit('k must be 1..10')
    T = dict(TIERS[o.tier])
    if o.tag:
        T['tag'] = o.tag
    if o.sample:
        T['sample'] = o.sample
    rows = cells(k, o.full)
    res, al = aliases(k, o.full, names)
    rng = random.Random(o.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    points = []
    for b in range(1, o.blocks + 1):
        seq = order if b % 2 == 1 else order[::-1]
        for i in seq:
            on = [names[j] for j in range(k) if rows[i][j] > 0] + T['extra']
            points.append('%s@%s#b%d' % (T['tag'], '+'.join(on) if on else 'base', b))
    minutes = len(points) * T['min_per_point']
    print('# %s: %d ideas (%s), tier %s (%s)' % (o.name, k, ', '.join(names), o.tier, T['note']))
    print('# %d cells x %d blocks = %d points, resolution %s, about %.0f min on an H100' % (len(rows), o.blocks, len(points),
          'full' if res is None else ['', 'I', 'II', 'III', 'IV', 'V', 'VI'][min(res, 6)], minutes))
    for a in al:
        print('# aliased two-way interactions: ' + a)
    item = os.path.join(o.out, o.name + '.g1.sh')
    W = '/mnt/projects/debruinz_project/debruinz/singlet'
    commit = o.bin.split('-')[0]
    with open(item, 'w', newline='\n') as f:
        f.write('#!/bin/bash\n# PILOT-MINUTES: %d\n' % int(minutes * 1.3 + 10))
        f.write('# fastloop item (bench/gpu/fastloop, design.py): %d ideas %s, tier %s: %s.\n' % (k, ','.join(names), o.tier, T['note']))
        f.write('# %d cells x %d blocks, resolution %s. Effects, CIs and stages: fastloop.md in the results directory.\n' % (
            len(rows), o.blocks, 'full' if res is None else res))
        f.write('W=%s; P2=$W/jobs/gpu-pr2; FL=$W/jobs/gpu-fastloop; N=%s\n' % (W, commit))
        f.write('export BINDIR=$P2/bin2/%s EXPECT_COMMIT=$N REFDIR=$P2/bin2/9560b8db REFCOMMIT=9560b8db ALLOW_NO_EMBED=1 IDENT=0 PINDEV=${PINDEV:-0}\n' % o.bin)
        f.write('export FL_IDEAS=$FL/ideas.tsv FL_SOLE=%d SAMPLE=%s\n' % (1 if o.sole else 0, T['sample']))
        f.write('export POINTS="%s"\n' % ' '.join(points))
        f.write('bash $FL/tp_gc.sbatch; rc=$?\n')
        f.write('RES=$W/jobs/gpu-integ/results/tp-${RUNID:-$SLURM_JOB_ID}\n')
        f.write('$W/envs/pbench/bin/python3 $FL/analyze.py --res $RES --ideas %s --drop-first %d --metric %s --theta %s > $RES/fastloop.md 2>&1\n' % (
            ','.join(names), T['drop_first'], o.metric, o.theta))
        f.write('cat $RES/fastloop.md\nexit $rc\n')
    print('# wrote ' + item)


if __name__ == '__main__':
    main()
