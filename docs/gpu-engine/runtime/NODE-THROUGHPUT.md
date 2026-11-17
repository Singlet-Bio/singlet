# GPU pipeline node throughput: changes on top of v1.3.8 / v1.3.9

This page records every change to the GPU pipeline (`--engine gpu`, `singlet gpu-pool`) made by the node-throughput track of
2026-11-12 to 2026-11-17. For each change it gives what changed, the switch that restores the previous behaviour, and what was
measured. It is the reference for the manuscript's methods and for operators. The measurements and the experiments that did not
ship are in `bench/gpu-runtime/results-2026-11-12.md` and `results-2026-11-13.md`. The harness is in `bench/gpu-runtime/jobs/`.

## What is measured, and how

- **Metric: node throughput.** Million read pairs per second for one Clipper H100 NVL node: 2 GPUs, 54 usable cores, 735 GB
  allocatable. Each GPU runs one `singlet gpu-pool` with K samples in flight. The pipeline is run with the campaign's arguments.
  HLA typing is off unless stated, because 3' libraries are not typed (see Settings).
- **Single-sample benchmark** (`rt_node.sbatch`): N copies of `d300_3p`, a 300 M-pair 3' library, with K=3 per GPU and T=8.
  - RUN is the whole-run rate.
  - NODE is the steady-state rate after the ramp, which is noisy at about ±4-5 %.
- **Mixed benchmark "mix4"** (`rt_mix.sbatch`): 110 items. The samples are 2.0 M to 1.30 G pairs; 3' and 5'; 90 and 149 bp
  cDNA; v2/v3 chemistries; CITE; human and human-mouse. The mix is weighted like the campaign: 46 % of pairs in samples of 500 M
  pairs or more, as in the 235-sample stratified dry run of the 119 k-sample corpus. Results are reported by class:
  - shallow: up to about 80 M pairs;
  - mid;
  - deep: 700 M pairs and up.
- **Identity.** No change on this page changes a count, call or table.
  - Every merged change passed `ctest` (179 tests).
  - Every merged change passed the identity gate (Clipper `jobs/gpu-pr2/pr2-gate.sbatch`). The gate runs 8 public samples, each
    on the CPU engine and the GPU engine of the same build. The required verdict is 8/8 `cpu:gpu IDENTICAL` (by
    `bench/star_accel/out_equiv.py`), with no fallback line other than the R11 info line.
  - Changes with a forced path (e.g. chunking that small samples never reach) were also gated with that path forced.
  - Node runs check one output set hash per sample across all items, and the same hash as the reference run.

## Results

| | v1.3.8 | this branch |
|---|---|---|
| d300_3p, K=3 T=8, NODE / RUN (Mpairs/s) | 6.85 / 6.29 (g051, A2) | 10.30 / 9.07 (g051, 816992) |
| d300_3p, paired A/B/B/A on g052 at 282e579a | 6.74 / 6.17 | 8.77 / 8.03 (+31 %, outputs identical) |
| mix4, 110 items, NODE / RUN | not run (pre-admission builds livelock) | 4.31 / 3.99, 110/110 ok, all identical (817205) |

- The node is not idle-CPU bound on the single-sample benchmark. Slots are 80-90 % used; host cores are 38-45 of 54.
- On mix4 it is bound by host memory per deep sample. A deep sample uses about 300 GB during mapping, so only 3.4 of 6
  samples are in flight on average and about 28 of 54 cores are busy.
- At full occupancy the mid and deep classes would run at about 7.6-7.8 Mpairs/s per node.
- The CPU ceiling for a campaign-weighted mix is about 8.5 Mpairs/s at 100 % core use. It is set by the donor demultiplexer on
  multi-donor samples, by 149 bp reads, and by the records-phase CPU fallback described below.

## Changes, by area

Each switch restores the earlier behaviour when set as shown. All are on by default unless marked opt-in.

### Node scheduling and memory (gpu-pool)

- **Host-memory admission** (`SINGLET_POOL_MEMADMIT=0`).
  - A node-wide registry in /dev/shm, under flock, of what each pool's items can still allocate.
  - Each item's estimate is 61.8 GB + 0.2235 GB per M reads. The read count comes from the `.1fq` header.
  - Reserve 40 GB (`SINGLET_POOL_MEM_RESERVE_GB`). Skip-ahead of 16 items, limited to 900 s of the head item's wait.
  - Stagger of 40 s after items estimated at 100 GB or more. Bounded wait (`SINGLET_POOL_MEMADMIT_MAXWAIT_S`, 1800 s).
  - An item is charged its estimate until its mapping level is set, 0.4 × the estimate after that, and what it actually holds if
    that is more.
  - Without admission, mix runs livelocked in THP direct compaction: six deep items in the scan phase at once, pressure 99 %,
    no item done.
- **Phase charge** (`SINGLET_POOL_PHASE_ADMIT=0`; `SINGLET_POOL_TAIL_GB`, default 80). After the async all-site pass is
  joined, an item's charge drops to the tail charge. The tail plateau measured on mix4 is 65 GB at most.
- **THP policy** (`include/singlet/engine_gpu.h` thpFreeOk). No `MADV_HUGEPAGE` when MemFree is below 48 GB or memory pressure
  (full avg10) is above 1 %.
  - The page-cache rule is off by default (`SINGLET_THP_MAXCACHE_GB` > 0 enables it). It cost 12-15 % on a cache-heavy node.
- **One shared non-host index per node** (`SINGLET_NONHOST_SHARE=0`). Genome files are dropped from the page cache after the
  pool's copy is made. `malloc_trim` runs after the pileup (`SINGLET_MALLOC_TRIM`).

### Device memory robustness

- **Hand-off pass.**
  - When the reads do not fit at once, the chunk is capped by the slot budget, not by the card's free memory. Other samples, the
    shared index and the non-host server use the card at the same time.
  - Every allocation of the pass recovers: per-read buffers halve the chunk down to 2^20 reads; fixed-size buffers wait (bounded,
    `SINGLET_E2E_OOM_WAIT_S`, 300 s).
  - The distinct-CR sort reports OOM instead of throwing.
  - Test hook: `SINGLET_E2E_OOM_INJECT=k`.
  - Result: a mix of 10 × 700 M + 2 × 1.0 G-pair samples went from 1/12 to 12/12 completed, identical to K=1 references.
- **Mapping lanes and clip kernels.** Allocations wait and retry, and launches that fail for memory are repeated
  (`SINGLET_ENGINE_GPU_OOM_WAIT_S`, 120 s).
- **Solo two-pass.** Waits for memory before the host fallback (`SINGLET_SOLO_OOM_WAIT_S`, 120 s).
- Every recovery writes one stderr line. The chunk size never changes a result.

### Mapping phase (device slot hold)

- Reads uploaded packed and decoded on the device (`SINGLET_ENGINE_GPU_PACKUP=0|check`). rc.reads 11.5 → 8.4 s.
- Dedup groups on the device (`SINGLET_ENGINE_GPU_DEDUP_GROUPS=host|check`) and dedup hashes on the device (H1DEV).
  rc.dedup 9.6 → 8.1 s.
- Record counts on the host, then one device pass over the records (`SINGLET_ENGINE_GPU_REC2PASS=1|check` restores two passes).
  hold2 7.7 → 5.6 s.
- Lane results copied D2H on a thread with pooled temporaries (LANED2H). Lane slots on 2 streams, batches in cost order,
  FASTCOPY/FASTSJ on.
- Lean mode compacts distinct sequences on a thread (`SINGLET_ENGINE_GPU_LEANFILL=0`).
- Stitch clock counters compiled out (CMake `SINGLET_ENGINE_GPU_CLK`, default OFF).
- CR4 probe arms start beside the `.1fq` scan (`SINGLET_PROBE_EARLY=0`). Mapping starts 4-5 s earlier.
- Region prefault before staged D2H. Non-host upload overlapped with reads. Zero-copy index (opt-in).
- Hand-off region registered before the slot lock. Solo pinned frees after unlock. The map2 device pool is opt-in
  (`SINGLET_MAP2_POOL=1`).
- Packed read planes released before STAR's loop, and unmapped reads served from a compact store
  (`SINGLET_ENGINE_GPU_EARLY_RELEASE=0|check`). Host peak 451 → 430 GB at K=3.

### Pileup and all-site

- Pileup streams reads from the host into a transient top-arena region (`SINGLET_PILEUP_STREAM=0`).
- Blocked signature tables (SIGBLOCK=auto), cell-table sizing, and pinned, pipelined partition transfers. 1.0-1.3 G-pair samples
  fit one 28 GB slot arena.
- All-site gather: chunks gathered beside the device, the first before the lock. Look-ahead prefetch in the gather walk
  (`SINGLET_ALLSITE_GATHER_PF=0`).
- The all-site pass runs beside applyOut and the host pass (`SINGLET_ALLSITE_OVERLAP=0`). It is joined at the first use of its
  result (`SINGLET_ALLSITE_ASYNC=0`).

### End-of-run (tail)

- The export tail runs as a stage graph (`SINGLET_EXPORT_GRAPH=0`).
- Donor demux variational Bayes: exact fixed-point exit (`SINGLET_DEMUX_FIXEDPOINT=0|1|check`).
  - A one-donor fit used to run all 300 iterations, because a float rounding residual never fell below the convergence epsilon.
  - Demux time fell from about 49 to 10 s per item.
- Auto-K fast path: progressive PCA, folds side by side, a wave of 1 + patience, and a worker pool
  (`SINGLET_DEMUX_AUTOK_FAST=0`). Restarts get ceil(threads/restarts) threads (`SINGLET_DEMUX_AUTOK_SPREAD=0`).
- The ambient ensemble shares the salt-independent cluster preparation (`SINGLET_AMBIENT_SHARED_PREP=0`).
- `SINGLET_INGEST_THREADS` sets the thread count of the read ingest's pass alone.

### HLA

- The HLA prefilter runs inside the read ingest's fused pass (`SINGLET_HLA_PRESIFT=0`). It saves 120 core-s per sample.
- Per-chemistry off on the GPU route: `--hla-gpu-3p off`, `--hla-gpu-5p off`.
- `--hla-max-reads N`: an even-stride sample of the candidate reads (default off).
  - 5' main-five calls were unchanged down to 1/64 (pbmc5p_v1) and 1/256 (d300_5p) of the candidates.
  - 3' calls move with depth: Oetjen truth, 2-field accuracy 0.950 at 361 M pairs, 0.900 at 180 M. So 3' gets no cap.
  - Only concordance counts are reported here, never alleles.

## Experiments that did not ship (data in the results files)

| Experiment | Reason not shipped |
|---|---|
| MPS | no gain |
| K=4 per GPU | memory-bound, then slot- and CPU-bound |
| 3 slots per GPU | no gain |
| 34 GB slot budget | no gain |
| slot priority | lost |
| lane COOP warp path | lost |
| all-site kernel rewrites | no gain |
| all-site early start | no gain |
| tail-end and exit trims | no gain |
| demux worker pool alone | no gain |
| early non-host request | neutral |
| NVMe spill of the per-read host arrays (rt/hostbound) | one item's peak 327 → 105 GB, but wall +53 %, and the node stalled on dirty page cache at the cgroup limit |

## Settings for production (campaign), chosen from the data

| Setting | Value | Evidence |
|---|---|---|
| samples in flight per H100 | 3 | K=4: no gain, memory cliff |
| threads per sample | 8 | T=10 was +4 % in one early v1.3.8 run and was not re-measured at head; T=8 kept as the campaign value |
| engine slots per H100 | 2 | 3 slots no gain |
| per-slot device budget | 31 GB | 34 GB no gain |
| pool CPU/memory binding | `numactl --cpunodebind=d --preferred=d` per GPU's NUMA node | +5-6 % steady, -7 % core-s |
| `SINGLET_TAIL_THREADS`, `SINGLET_INGEST_THREADS` | 16, 16 | +2-3 % (single runs) |
| host-memory admission | on (default) | prevents the livelock |
| HLA, 3' libraries | off (`--hla-gpu-3p off`) | policy |
| HLA, 5' libraries | on, `--hla-max-reads 500000` | calls unchanged far below this |
| depth cap on H100 | 1.4 G pairs (largest measured in a mix: 1.30 G) | mix4 |
| "alone above" rule | removed; admission handles it | mix4 |
| planning rate | 2.0 M pairs/s per H100 on a campaign-like mix; 4.5 on 300 M 3' samples | 817205, 816992 |

## Known limits and next steps

- **Records-phase CPU fallback.** A sample whose records phase needs more device memory than one slot's budget falls back to
  CPU STAR for the whole sample. This happened to a 100 M-pair, 149 bp library with an 18-27 % unique rate, and to a mouse
  library mapped against GRCh38. That sample class costs about 26 core-s per M pairs against about 4. A campaign scheduler
  should route such samples to CPU lanes, or give them two slots (`SINGLET_ENGINE_GPU_ADMIT=1`, untested at node scale).
- **Deep-sample host memory during mapping** (about 300 GB at 1.3 G reads) limits occupancy. The inventory is in
  results-2026-11-13.md.
- **A100 and other GPU models.** The identity gate runs on RTX 8000 (Turing) and the timing on H100 NVL. Other models are
  untested.
