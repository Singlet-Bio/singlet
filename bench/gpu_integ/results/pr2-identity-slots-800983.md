# PR-2 gate G5: identity at 2 slots and at 3 slots with MPS (candidate c8849bb2)

Job 800983 (GPU, g051, T=7, k=8, pbmc1k_v3, 24 runs per point, WSHRINK on, `IDENT=1`). References: the CPU engine of this binary (job 800980, T=7) and each point's alone run. Script `tp_integ.sbatch` of this branch. Results were written by that script to `$W/jobs/gpu-integ/results/tp-800983/` (a new folder, nothing existing overwritten).

**Identity: 48 of 48 runs IDENTICAL**, against the alone run and against the CPU engine. 24 of 24 at 2 slots, 24 of 24 at 3 slots with MPS. Every replicate was checked.

**Fallbacks: the gate FAILS the "no fallback" rule.**

| Point | Runs with a fallback | Cause in the log |
|---|---|---|
| 2 slots, no MPS | 5 of 24 (c6, c8, c10, c15, c19) | non-host GPU screen: `cudaMalloc ... resource already mapped` or `cudaSetDevice ... CUDA-capable device(s) is/are busy or unavailable`, then `screening on the CPU` |
| 3 slots with MPS | 10 of 24 | the same two errors |

The fallback is the non-host screen only. The run finishes on the CPU for that stage and its output is identical. The admission log shows 0 runs with a fallback line (admission control is not in use).

Timing in this job (not a decision): 2 slots 2.719 M pairs/s over the span, core-s per M 9.21, lock held 33.7 s. 3 slots with MPS 2.792, 9.45, 41.4 s. Fallback runs must be dropped from timing points. These points are not clean.

Open: whether the fallbacks come from E20 (the pileup unlocks the device before it resets its context, so the next sample's non-host screen meets a context in teardown). The b0 arm of the timing jobs and a reset-then-unlock arm (job 802058) test this.
