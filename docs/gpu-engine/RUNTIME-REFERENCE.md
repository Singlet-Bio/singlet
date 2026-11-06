# GPU engine run-time reference

This page describes how `singlet --engine gpu` shares a GPU, which switches it reads, how it builds and how a change is gated. Facts come from the source files and from the measured record. The kernels are in `KERNEL-REFERENCE.md`. The operator page is `fable/GPU-PATH.md`.

**Status of this page.** First draft. It covers the concurrency model, the switches that `GPU-PATH.md` does not list, the build matrix, the strand-restart fixes, the gate harness and a profiling recipe. The accepted-difference list per entry and the performance method need a longer write-up.

## 1. Concurrency model

### 1.1 The device lock

Each sample takes an exclusive `flock` on `/tmp/singlet_engine_gpu_<PCI bus id>[.k].lock` while it uses the device. The suffix `.k` appears for permits above the first. `SINGLET_ENGINE_GPU_SLOTS=K` sets K permits per GPU from 1 to 8. The default is 1.

A process first tries each permit of each device without waiting. If none is free it waits on one permit chosen from its process id. `SINGLET_ENGINE_GPU_DEVICE` pins the device and the process then tries only the permits of that device.

The lock files open with `O_CLOEXEC`. A process that execs itself, as the strand restart does, no longer carries a held lock into its successor. `unlockDevice` calls `flock(LOCK_UN)` before `close`, so a forked child keeps no lock.

### 1.2 What the lock covers

The lock covers five phases of a sample.

| Phase | Where | Notes |
|---|---|---|
| Prefill | `mapDistinct`, `SeqCacheGpuPrefill.cpp` | Two holds when E8 records are built. Hold 1 runs the clip, entries, dedup and lanes. The host lane and the host assembly of records run unlocked. Hold 2 uploads the records. `SINGLET_ENGINE_GPU_SPLIT=0` gives one hold |
| Hand-off pass | pileup engine | Part of the pileup hold |
| Pileup | `gpu_pileup_apply` | The device unlocks right after `pileupRun`. The context reset runs on a thread after the unlock |
| Non-host screen | `nonhostClassify` in `singlet.cpp` | Covers the index load, upload, classify and download |
| Solo | `seqCacheGpuAfterMapping` | Short, 0.6 to 1.0 s |

Treat the E20 split as part of PR-2. Before it, one prefill hold ran from the index upload to the end of the record build.

### 1.3 Device memory budget

The budget of one sample is the device size minus 2 GB, divided by the slots. `SINGLET_ENGINE_GPU_MEM_GB` overrides it. The plan holds the index (16.4 GB), reads (6.1 GB), the CR4 clip (7.3 GB), E10 auxiliary data (2.0 GB), entries and masks (2.5 GB), E11 kept data (1.0 GB) and context (0.5 GB). The lanes take the rest. The non-host screen index (about 25 GB) sits outside the budget.

A sample that does not fit stops with `EXITING because of FATAL ERROR (singlet --engine gpu, E17)` unless `SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu` is set. With that switch STAR maps the sample on the CPU.

**Known risk.** With all slots held the device had 2.2 GB of headroom on the E20 build and 3.4 GB on `s3` (e42c0636). One of 16 loaded runs of the E20 build fell back to the CPU after a failed allocation in the bit set of `k_p2`. PR-2 tests whether WSHRINK removes this.

### 1.4 Streams and threads

- All kernels launch on the legacy default stream. The code has no other stream since BSTREAM was removed. The `st` argument of `runLane` keeps its default of 0.
- Host threads do three jobs next to the device. The host lane maps lane B overflows on a pool of T minus 1 threads. A thread resets the pileup context after the unlock. Ingest threads decode the staged read file into prefill buffers.
- The device-memory tally (`g_bud`), the step times and the lane log are guarded by one mutex (`g_budMu`). The mutex has no contention now that only one thread launches lane work.

### 1.5 CUDA MPS

MPS lets kernels of different processes run at the same time. It is an operations setting, not code. The timing harness starts and stops the daemon (`start_mps` and `stop_mps` in `bench/gpu_integ/jobs/tp_integ.sbatch`). The slot lock stays in place under MPS. Run identity and timing at 2 and 3 slots under MPS before relying on it.

## 2. Switches added or changed by this line

`fable/GPU-PATH.md` section 2 lists the switches of the release path. This table lists the others that the source reads. Defaults come from the code. A dash in the default column means the source treats presence of the variable as the switch.

| Switch | Default | Effect |
|---|---|---|
| `SINGLET_BOLD_WSHRINK` | off | `1`: lane B on `LaneBc` with the align pool (`KERNEL-REFERENCE.md` 2.7) |
| `SINGLET_BOLD_WASLICE` | 160 | Aligns of pool per read in flight for WSHRINK, at least 8 |
| `SINGLET_ENGINE_GPU_SPLIT` | on | `0`: one device hold in the prefill |
| `SINGLET_ENGINE_GPU_CR4_DEVICE` | on | `0`: CellRanger4 clip on the host |
| `SINGLET_ENGINE_GPU_CR4_CHECK` | off | `1`: run both clips and compare byte for byte |
| `SINGLET_GPU_PILEUP_UNLOCK` | unlock, then reset | `reset`: reset then unlock. `late`: reset and unlock after the host pass |
| `SINGLET_GPU_PILEUP_E23` | on | `0`: the per-group R9 kernels `k_r9` and `k_cr_roots` |
| `SINGLET_GPU_PILEUP_PARTS` | 0 | Force the pileup to run in this many parts |
| `SINGLET_GPU_PILEUP_ARENA_GB` | budget minus 3 GB | Cap the pileup arena |
| `SINGLET_GPU_PILEUP_QUAL` | auto | `force`: always carry base-quality masks |
| `SINGLET_GPU_PILEUP_FORCE_DECLINE` | off | Test hook: the GPU pileup declines |
| `SINGLET_GPU_PILEUP_PROF` | off | Log stage timings of the pileup |
| `SINGLET_ENGINE_GPU_CFG` | new | `old`: the mapping kernels before E12 (`k_p1`, `k_p3`) |
| `SINGLET_ENGINE_GPU_NB`, `_NBM`, `_NBB` | 524288, 393216, 131072 | Reads in flight per batch in lanes A, M and B |
| `SINGLET_ENGINE_GPU_CHUNK` | 16 M | Distinct sequences per chunk. The budget may shrink it |
| `SINGLET_ENGINE_GPU_WARPM` | on when it fits | Lane M warp per read |
| `SINGLET_ENGINE_GPU_HOSTLANE`, `_LEAN`, `_DEDUP`, `_D2H`, `_UPLOAD`, `_PIN` | see source | Tuning and debug switches of the prefill (`engine_gpu.cu` lines 1062 to 1622) |
| `SINGLET_TEST_FORCE_STRAND_RESTART` | off | `1`: restart on the sorted path whatever the strand counts. Used by `test_strand_restart_e2e` |
| `SINGLET_ALLOW_PILEUP_SHORTFALL` | off | `1`: exit code 10 becomes a warning |

**Dead switches.** The plan asks for a table with no dead switches. This draft lists what the source reads. It has not checked which switches no gate or operator uses. That check is open.

## 3. Strand restart

The stream path counts reads in the protocol orientation. When the library lies far from it, `restart_on_sorted_path` restarts the same command with `--sorted-bam`.

PR-1 fixes three faults of that restart.

1. The restarted process inherited `SINGLET_ENGINE_GPU_HANDOFF`. STAR then left the hand-off reads out of the sorted BAM and the pileup read 143,514 records instead of 39.5 M on pbmc2tvdj. The run still exited 0. The restart now builds its environment from a snapshot taken at the start of `main()`.
2. The hand-off files `.gpu_handoff.bin`, `.reads` and `.reads.q` were left behind. They are removed at the restart, at the end of the run and at exit. They never appear in `provenance.json`.
3. The device lock was inherited across the exec. The restarted process waited for it until a timeout. The lock files now open with `O_CLOEXEC`.

A new exit code 10 (`pileup_shortfall`) fires when the pileup sees fewer than half of STAR's uniquely mapped reads.

Exit codes of the pipeline: 3 star_fatal, 4 zero_cells, 5 stalled, 8 strand_contradiction, 9 GPU pileup failed after the CPU skipped its steps, 10 pileup_shortfall.

## 4. Build matrix

| Item | Value | Source |
|---|---|---|
| CMake option | `SINGLET_ENGINE_GPU` = `AUTO`, `ON` or `OFF` | `CMakeLists.txt` |
| Target | `singlet_engine_gpu`, six `.cu` files | `CMakeLists.txt` |
| CUDA standard | 17 | `CMakeLists.txt` |
| Architectures | `80;90` unless `CMAKE_CUDA_ARCHITECTURES` is set | `CMakeLists.txt` |
| Flags | `-O3 -fmad=false` on every CUDA file of the target. `solo_gpu.cu` adds `--extended-lambda` and OpenMP | `CMakeLists.txt` |
| Fast math | Not used. The separate `singlet_gpu` library (`SINGLET_BUILD_GPU`) uses `--use_fast_math` and is not part of the engine | `CMakeLists.txt` |
| Toolchain on Clipper | `nvidia/cuda:12.4.1-devel-ubuntu22.04` image, conda gcc, `-allow-unsupported-compiler` | `bench/fable/lockclip/jobs/build.sbatch` |
| Minimum device | compute capability 8.0 | `GPU-PATH.md` |
| Not built | `sm_70`. A V100S node has no kernel image. Determinism across GPU models has never been tested | record |
| Commit embed | Binaries built on Clipper have embedded only a short hash and the commit check refuses them, so timing and corpus jobs set `ALLOW_NO_EMBED=1`. The embed fix is due before the freeze | record |

The sm_70 probe in the plan needs a build with `-DCMAKE_CUDA_ARCHITECTURES="70;80;90"` and one run of pbmc2m on a V100S node. It has not run.

## 5. Gate harness

A green GitHub check proves nothing about CUDA. GitHub CI runs R CMD check. Each GPU change needs Clipper jobs and a committed `summary.md`.

| Gate | What it checks | Where |
|---|---|---|
| Identity | Every output file hashed by `bench/star_accel/out_equiv.py`, CPU engine against GPU path, same binary, unsorted-stream path | `bench/fable/lockclip/jobs/gate.sbatch` |
| Ten-sample gate | pbmc2m, pbmc10m, pbmc1k_v3, pbmc1k_v2, pbmc1k_cite_v3 (gene and antibody libraries), pbmc5p_v1, neuron1k_v3, hgmm1k_v3, nuclei2k_v2 | same |
| Code-path corpus | 16 inputs of about 1 M pairs. Identity, path assertions, invariants and fallback rules | `bench/gpu-paths` on its own branch |
| Timing | A/B/B/A inside one exclusive job with a b0 reference arm. Decide on core-seconds per M pairs and lock held | `bench/gpu_integ/jobs/tp_integ.sbatch` |

Accepted differences between two runs are the files whose content depends on thread arrival order or timing.

| File | Reason |
|---|---|
| `nonhost_reads.fq`, `star_Unmapped.out.mate1`, `mate2` | Same records. STAR threads write in arrival order. The gate compares sorted records |
| `Gene/CellReads.stats` | Same rows in another order |
| `star_Log.std.out`, `read_stats.tsv` stage rows | Timings and paths |
| `pileup_stats.json` | Provenance counters and byte totals |
| `summary.json` field `memory_tier` | Differs between engines |
| `provenance.json`, `ambient_contamination.json` | Timings, binary directory, whitelist path |

One known exception: for nuclei2k_v2 the file `nonhost_reads.fq` holds other records in each run. The non-host screen subsamples more than 20 M unmapped reads and the writer indexes the subsample against the whole file.

## 6. Profiling recipe

- `ncu --target-processes all` follows STAR's forked child. Use `--set full` for one kernel class. Kernel names are in `KERNEL-REFERENCE.md`.
- `nsys` does not follow the fork. Profile the child process directly.
- `perf` is refused on Clipper (`perf_event_paranoid=4`).
- Set `SINGLET_NVTX=1` for NVTX ranges. The device lock wait and hold are ranges of their own.
- `SINGLET_E16_TRACE=1` prints the `[e16]` timing marks. `SINGLET_GPU_PILEUP_PROF=1` prints pileup stage times.
- The `--tail` timing code adds 14 registers and 18 percent to the seed-and-stitch harness time. Use the build without it for timing.

## 7. Errata applied in this PR

| Statement | Fixed in | Correct reading |
|---|---|---|
| E20 report has unfilled placeholders (E-087) | `fable/E20-lockclip.md` | Filled from the record. The report still names the ungated samples |
| The 28 s pileup hold on neuron1k_v3 is the mitochondrial pass (E-037) | `KERNEL-REFERENCE.md` section 3 | The junction R9 collapse `k_r9` took 25.8 s of the 27 s. The chrM stages take 16 ms |
| `engine-gpu.md` says every stage except mapping is CPU code and 5' samples do not qualify (E-086) | `docs/engine-gpu.md` | Banner added. The GPU also builds records, counts, pileup and the non-host screen. The CR4 clip runs on the device |
| Handoff says e20w has no fallbacks (E-021) | `fable/E20-lockclip.md` | One device-out-of-memory fallback in 16 loaded runs |

Still open from the plan: the README GPU section, `gpu-cr4-clip.md`, `staged-format-1fg2.md` and the study ceilings. They live on `prod/campaign` or in other files and need their own pass.
