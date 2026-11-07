# Sequential A/B timing

The campaign tests each idea in tiers and stops at the first tier that settles it:

1. build plus ctest;
2. the pbmc2m identity check (about 35 s);
3. kernel replay against the STAR-side oracle, for kernel changes (`bench/gpu/replay/`);
4. paired timing with sequential stopping (this folder);
5. the 9-sample identity gate, only for an idea being adopted.

Tier 4 runs the control arm A and the candidate arm B in blocks A B B A, one point of 8 runs each (8 in flight, 7 threads, 2 slots on
one pinned GPU). After every block `seqstop.py` reads the finished points' `point.json` and bootstraps the relative change of B's mean
against A's. It resamples runs within each point, so a point's level stays its own. The primary metric is core-seconds per million read
pairs (`core`) or the seconds a run holds a device lock (`held`).

| Decision | Rule (theta: the smallest improvement worth adopting) |
|---|---|
| WIN | the interval's upper end is below -theta |
| RULED-OUT | the interval's lower end is above -theta |
| CONTINUE | neither, and both arms are below the cap |
| INCONCLUSIVE | neither at the cap, which counts as a rejection |

Interim looks use a 99 % interval and the look at the cap a 95 % interval. The cap is 40 runs per arm: two full blocks (16 runs per arm
each), then a final A B block. Each experiment card states its metric, theta and cap.

```bash
# in a pilot item or a job: the harness builds the blocks and calls seqstop.py after each one
SEQAB=base:wshrink SEQTAG=k8t7s2r1 SEQMETRIC=core SEQTHETA=0.03 SEQCAP=40 bash tp_gc.sbatch
# afterwards, or on any finished points
python3 seqstop.py --res <results dir> --a base --b wshrink --metric core --theta 0.03 --cap 40
```

`tp_gc.sbatch` (in `bench/gpu/pilot/`) is the throughput harness. It records the pilot job, node and GPU of every point and checks the
GPU pin after each one.
