# HP: the host phases of a GPU run

**Hypothesis.** Once STAR's child exits, a GPU run spends most of its remaining time on the host: the non-host screen's EM, the export, and the donor demux. On pbmc1k_v3 at one slot (RTX gate 803937, stage table of the GPU run), these phases cost 177 core-s ("demux/ancestry/writers", 46 s wall), 50 core-s ("non-host", 27 s wall) and 42 core-s ("export, QC, callers", 19 s wall) out of 652 core-s and 198 s. That is 41% of the run's CPU and 46% of its wall time. Much of it is serial work, or work repeated for no reason, so it can shrink without changing any output.

**Where the time goes.**

- **Per-thread CPU by role.** Pilot item 21 sampled every thread of two runs (gpu/frontier 654109ad, k1 T27 s1) with `bench/gpu/profile/threadsampler.py`. The table is per run, summed over a role's processes. The sampler reads only live threads, so threads that exit between samples are missed. It accounts for 62% of the cgroup's core-s.

  | thread role | core-s per run |
  |---|---|
  | sg-scan (E13 ingest decode) | 65 |
  | gpu-map (mapDistinct host threads) | 61 |
  | sg-demux (export Phase 3 and its writers) | 53 |
  | sg-nonhost (non-host screen) | 43 |
  | sg-refs (pileup references) | 38 |
  | sg-stream (pileup stream from STAR) | 28 |
  | sg-gpreload (genome preload) | 18 |
  | sg-cellcall (cell calling) | 16 |

- **The non-host phase on pbmc1k_v3 (27 s wall).** The device screen takes 7 s: the index upload, then 0.7 s to classify. The main EM then runs its 500 iterations single-threaded in 8 s. The candidates table (#211) repeated the same 500-iteration EM over the same reads for about 9 s more.
- **The demux thread (38 s wall)** is the critical path of export Phase 3. auto-K takes 10 s. The K=1 fit runs one restart, with 100 pre-iterations on one thread (17 s), then 200 threaded iterations (11 s). Every fit runs to its iteration cap. The convergence test asks for an absolute ELBO change below 0.01, but the ELBO is near −1e6 and is held in float precision, so the test never passes. Fixing that test would change the genotype outputs, so it stays as it is.

**Changes, each output-identical.**

1. `8812402c` gives each demux VB restart its share of the threads. The fit's reductions use fixed chunks (`parallel_det_chunks`), so the result does not depend on the thread count; `test_demux_doublet_autok` checks this.
2. `1670c281` makes one non-host EM fit serve both reports, the abundance table and the candidates. The iterations never read the report thresholds. The E-step runs on the run's threads and stays bit-identical: each pattern's denominator is its own sum, and each species adds its terms in pattern order through an inverted list. Unit test 8 checks 1 against 8 threads.
3. `f7f23fa4` adds `SINGLET_EXPORT_TIMING=1`, which prints wall and CPU per export phase and per writer thread, to cost the rest of Phase 3.

**Measure.** The 9-sample identity gate against the CPU references of 654109ad, which catches any change to the demux or non-host outputs. Then the stage table of the gate's GPU runs for the wall and core-s of "non-host" and "demux/ancestry/writers", and the per-writer lines from `SINGLET_EXPORT_TIMING=1`. Throughput effect: a sequential A/B at k8 once the k-scan (item 27) shows whether these phases bound the in-flight count.

**Falsifier.** Any gate difference ends a change. A change that saves under 2 s of wall per pbmc1k_v3 run is reverted as not worth its code.

**Result.** Pending: gate 804095 with hgmm as pilot item 29, and timing run 804091.
