# Host-thread sweep at 2 runs in flight (job 802089, candidate c8849bb2)

Requested by the Beta orchestrator (round-3 referee: is the pooled rate thread-starved?). Exclusive node g051 with all 54 schedulable cores, k=2 samples in flight, 2 device slots, WSHRINK on, pbmc1k_v3 (66.6 M pairs), 12 runs per point, single replicate. `ALLOW_NO_EMBED=1`. The runs of this job wrote their results to `$W/jobs/gpu-integ/results/tp-802089/` (script default).

| T per sample | Host threads in flight | M pairs/s (span) | core-s per M (cgroup) | core-s per M (clean runs) | Lock held (clean runs) | Wall per run (clean) | Cores busy (mean over span) | Fallback runs |
|---|---|---|---|---|---|---|---|---|
| 7 | 14 | 0.911 | 8.54 | 8.15 | 27.5 s | 143 s | 7.8 | 1 of 12 |
| 14 | 28 | 0.986 | 9.28 | 8.91 | 27.0 s | 133 s | 9.2 | 1 of 12 |
| 21 | 42 | 0.948 | 10.24 | 9.91 | 31.9 s | 139 s | 9.7 | 2 of 12 |
| 27 | 54 | 0.939 | 10.52 | 10.12 | 28.1 s | 140 s | 9.9 | 1 of 12 |

**Reading.** At 2 runs in flight, quadrupling the host threads (14 to 54) leaves the rate flat (0.91 to 0.99 M pairs/s, a spread of 8%, about the run-to-run noise seen between b0 repeats) while core-seconds per M rise from 8.5 to 10.5. Cores busy stay at 8 to 10 of 54. The wall of one run barely moves (133 to 143 s). With two samples in flight the pipeline is not limited by host threads. Rate is about k times the pairs per run divided by the wall of one run (2 x 66.6 M / 140 s = 0.95). Higher rates need more runs in flight, which the k=4 and k=8 points of job 802672 test. This sweep does not show thread starvation at 14 threads. It does not exclude it at k=8.
Single replicate, 12 runs per point, one or two fallback runs per point.
