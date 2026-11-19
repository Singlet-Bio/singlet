#!/usr/bin/env python3
# hostcut round 3: the cgroup's anon+shmem peak of an rt_mix/rt_node run (mon.tsv, col 2, GB) and, at that moment, the sum of the pool's per-item estimates
# (A + B x pairs(M), the env constants or the code's 61.8 / 0.2235) of the items in flight (samples.tsv start/end), against the cgroup value.
# usage: t2-inflight.py <results dir> [A B]
import sys, os
d = sys.argv[1]
A = float(sys.argv[2]) if len(sys.argv) > 2 else 61.8
B = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2235
mon = [l.split('\t') for l in open(os.path.join(d, 'mon.tsv')).read().splitlines() if l.strip()]
pk = max(mon, key=lambda r: float(r[1])); t = float(pk[0])
tot = 0; n = 0; lines = []
for p in sorted(x for x in os.listdir(d) if x.startswith('pool') and os.path.isdir(os.path.join(d, x))):
    for ln in open(os.path.join(d, p, 'samples.tsv')).read().splitlines()[1:]:
        f = ln.split('\t')
        if len(f) < 15: continue
        s, e = float(f[4]), float(f[5])
        if s <= t <= e:
            est = A + B * float(f[13]) / 1e6; tot += est; n += 1
            lines.append((os.path.basename(f[2]).replace('.1fq', ''), float(f[13]) / 1e6, est))
print("cgroup anon+shmem peak %.1f GB at %d (memory.current %.1f GB); %d items in flight, sum of the pool's estimates %.0f GB (A=%.1f, B=%.4f); ratio estimates/cgroup %.2f" % (float(pk[1]), t, float(pk[2]), n, tot, A, B, tot / float(pk[1])))
for l in sorted(lines, key=lambda x: -x[2]): print("   %-16s %8.1f M  est %.0f GB" % l)
