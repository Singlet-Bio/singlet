# `--engine gpu`: where the time goes (S8, Phase A, as it stands)

Profiling only, no optimisation. Every number below is a Clipper measurement (H100 NVL nodes g050-g052, one GPU per run, 16 threads, a quiet node unless
stated; job ids and result paths are in the last section). Gaps are listed at the end instead of being filled with estimates.

## 1. Method, and what each tool can and cannot see

| Tool | Sees | Does not see |
|---|---|---|
| `nsys` (CUDA API, kernels, copies, NVTX, OS runtime) | the parent process: pileup and non-host GPU phases; NVTX ranges of all processes | the CUDA activity of the forked STAR child, where the prefill runs (nsys follows exec, not fork); CPU sampling is blocked by `perf_event_paranoid=4` |
| Nsight Compute (`ncu`) | per-kernel counters of every process, including the child (counters are permitted on these nodes) | overlap and gaps (it serialises kernels) |
| engine ledger, `SINGLET_ENGINE_GPU_TRACE=1` (`perf/nvtx-host-ranges`) | the prefill: every `cudaMemcpy` by source line (calls, bytes, host seconds, GB/s, pinned or pageable), the GPU time of each kernel launch (cudaEvents) | kernels launched from other files than `engine_gpu.cu` |
| STAR phase marks, `SINGLET_STAR_PHASE_LOG=1` (`prof/star-phase-marks`) | wall, process CPU time and resident memory of the STAR child at each phase boundary | |
| NVTX ranges (`include/singlet/nvtx.h`, `SINGLET_NVTX=1`) | named host phases next to CUDA activity; compiled out without the toolkit headers, one branch per range when off | |
| cgroup `memory.stat`, PSI, `/usr/bin/time -v` | job memory by kind, stalls, process peak | |

CPU per thread-pool task is **not** available as a profile (no `perf`); the CPU inventory below comes from process CPU time at phase marks and from
STAR's own `STAR_PROFILE` counters.

## 2. One sample, wall-clock budget (pbmc1k_v3, 66.6 M reads, 16 threads, `nsys` run 794901, wall 288 s with the profiler, 252 s without)

Seconds from the start of the process, from the run's timestamped log; `(L)` = under the device lock.

| t (s) | segment | notes |
|---|---|---|
| 0 - 22 | decode `.1fq` to temp FASTQ (16 threads) | 21 s; the references load (GTF, 7.35 M-site SNP panel) runs beside it, done at 26 s |
| 26 - 55 | CR4 probe: a separate STAR process maps 200 k reads on the CPU | 29 s, loads the genome a second time |
| 55 - 158 | STAR main start: genome load 7.7 s, prefill read pass 15.6 s, prefill under the lock **24.9 s (L)**, host assembly and hand-off records, STAR start-up until its first BAM header | child marks (794855): see section 3 |
| 158 - 176 | STAR maps the remaining reads and streams to the pileup | 32.6 s in the child's marks |
| 176 - 198 | hand-off pass 6 s, pileup input, pileup run **9.7 s (L)** | pileup GPU kernels 2.2 s of it |
| 198 - 221 | non-host screen 17.7 s, of which the GPU phase 4.9 s **(L)** | index + Bloom load 4.1 s |
| 221 - 238 | export: CSC 0.7 s, cell calling 1.3 s, ambient decontx 11 s, doublets 1.3 s, cell QC | single-threaded stretches |
| 238 - 288 | demultiplexing (auto-K 10 s, VB fit), writers | ~50 s |

Under the lock per sample: 4.9 + 9.7 + 24.9 = **39.4 s**. Not attributed in this table: the host-side part of 55 - 158 s outside the prefill hold (section 3
splits the child from its own marks, which I have for this sample from the memory-profile run 794855, not from the same run as the table), so this table
is a budget of the sample with about **50 s (17%) unattributed**; the 5% target is not met yet.

## 3. The STAR child (where the prefill runs), phase marks, same sample (794855, no profiler)

| phase | wall s | process CPU core-s |
|---|---|---|
| genome load (17.7 GB, once per process) | 7.7 | 7.6 |
| prefill read pass (`readSequences`: parse the temp FASTQ, **CR4 opal clip per read**) | 15.6 | **162.5** |
| prefill under the device lock (`mapDistinct`) | 35.6 (hold 24.9) | 88.7 |
| cache fill, per-record hand-off (alignment records and summaries built on the host) | 1.3 + 6.6 | 1.3 + 55.0 |
| cache inserts | 0.4 | 5.9 |
| STAR mapping pass (per read: readLoad, summary replay, hand-off entries, Solo records) | 32.6 | 218.6 |
| solo post-map: records read back 36.3, UMI collapse 8.5, matrix 0.7, filtering 0.3, stats 0.5 | 13.4 | 46.5 |
| total | 115.8 | 589.7 |

The 163 core-s of the read pass is the CR4 clip (opal alignment of the TSO on the CPU): 2.4 core-s per million reads, 28% of the child. On an 11 M-read
sample whose CR4 profile is off the same pass is 7.9 core-s. For 263 M reads (794854) it is 648 core-s.

## 4. The prefill hold, attributed (transfer ledger, kernel events)

| | pbmc1k_v3, 66.6 M reads (31.7 M distinct) | GSM7656149, 263.4 M reads (135.9 M distinct) |
|---|---|---|
| hold (device lock) | 24.9 s | 95.8 s |
| GPU kernels (sum of the event times) | **12.8 s (51%)** | **53.7 s (56%)** |
| - `k_p3` stitch | 8.9 s (70% of kernel time) | 37.8 s |
| - `k_p1` seed | 3.2 s | 13.5 s |
| - `k_p2` | 0.54 s | 2.2 s |
| - clip, gather, hash, groups, pack, prefix key | 0.18 s | 0.7 s |
| H2D copies, all pageable, 5.73 GB/s | 24.4 GB in 4.26 s (17%) | 44.6 GB in 7.8 s (8%) |
| D2H copies, pageable, 6.4-6.8 GB/s | 3.9 GB in 0.7 s | 15.7 GB in 2.4 s |
| dedup: host loops after 0.04 s / 0.16 s of kernels | **3.8 s (15%)** | **17.4 s (18%)** |
| host lane (reads past the GPU lanes' capacity, STAR's CPU aligner) | 2.7 s (86 k reads) | 7.7 s (255 k reads) |
| launch and sync gaps, remainder | ~2 s | ~7 s |

GPU busy inside the hold: **51-56%**. The PCIe link (Gen5 x16, above 50 GB/s) is used at **11%** of its rate. The 16.9 GB suffix-array index is
3.05 s of the uploads, every sample.

## 5. Kernel rooflines (Nsight Compute, lane A, 2048 blocks x 128 threads, representative launches, pbmc1k_v3)

| kernel | time per launch | SM throughput | DRAM | L2 | issue active | IPC | warps active | active threads per warp instruction | registers (blocks/SM) |
|---|---|---|---|---|---|---|---|---|---|
| `k_p3` (stitch) | 19.2-22.8 ms | 9% | 285 GB/s (7% of ~3.9 TB/s) | 16-18% | 13% | 0.49 | 22% | **2.07 of 32** | 96 (5) |
| `k_p1` (seed) | 14.0-14.7 ms | 21% | 505 GB/s (13%) | 24.5% | 21% | 0.82 | 21% | **3.9 of 32** | 98 (4) |

DRAM traffic per `k_p3` launch: 3.6 GB read + 1.9 GB written, i.e. ~13 KB read and ~7 KB written **per read**: the per-thread `Work` state lives in global
memory and does not stay in cache. Both kernels are limited by divergence, occupancy and latency, not by any hardware bound.
Not profiled yet: lanes M, B and heavy, and the pileup kernels (`k_r9` 1.4 s and `k_cr_roots` 0.49 s of the parent's kernel time on this sample; the
Nsight job is queued).

## 6. The parent process (pileup, non-host), `nsys` 794901

* Kernel time (pbmc1k_v3): `k_r9` 1.43 s, `k_cr_roots` 0.49 s, `k_binsum` 50 ms, `k_hits` 32 ms, `k_dir` 27 ms, sorts 16 ms: ~2.2 s in all.
* H2D: 30.5 GB in 2.96 s (10.3 GB/s average; 439 copies), mostly the non-host screen's three indexes: **24.9 GB re-uploaded for every sample**, 4.1 s.
* `cudaHostAlloc` 8.3 s over 42 calls and `cudaStreamCreate` 7.2 s over 40 calls in the 13 M-read run: pinned buffers and streams made per sample by the
  loader threads.
* Pileup hold on the 263 M-read sample (earlier run): 33.9 s of which GPU kernels 2.1 s, host fetch between partitions 9.4 s, the rest result copies and sorts
  (539 M chrM tuples per partition are copied back to the host).

## 7. Memory (STAR child, resident set by phase) and disk

| | 66.6 M reads | 263.4 M reads (FASTQ route) | 263.4 M reads (`.1fg2` route) |
|---|---|---|---|
| high-water mark | 43.6 GB | 120.3 GB | 84.4 GB |
| level after the read pass | 25.7 GB | 47.9 GB | |
| highest steady level (end of prefill) | 35.6 GB | 83.4 GB | |
| whole job (cgroup peak) | 57.6 GB | 180.9 GB | 100.6 GB |
| temp FASTQ in `/tmp` | ~21 GB (320 B per read, not measured at this size) | 84 GB (measured in the page cache) | none (258 MB left in `/tmp`) |

The peak is a transient in the read pass (file pages, per-thread buffers and the concatenated copy alive together), plus whole-sample arrays that live
through the prefill (all reads' numeric sequences: 91 B per read; the mapping output and the hand-off records). Fit for the FASTQ route: 17.5 GB + 0.39 GB per
million reads (four points, within 4%); `.1fg2` route: ~17.5 GB + 0.26 GB per million reads (two points).

## 8. Concurrency (campaign configuration)

Measured with the queue benchmark (8 samples per GPU x 12 threads, 1 device-lock permit): 2.0-2.2 M pairs/s steady per node, 16 core-s per million reads,
28 of 56 cores busy, GPU utilisation 28%, device lock busy 57-72%. More samples per GPU (12) is slower; a second permit gives +4% at 2x the CPU; NUMA pinning,
OpenMP passive waiting and storage staging do not help (PSI: I/O stall 9%, memory stall 0). A read-weighted mix (3,842 M pairs) ran at 1.97 M pairs/s steady, peak
cgroup memory 590 GB. A one-sample worker with 16 threads reaches 0.26-0.27 M pairs/s per GPU.

## 9. CPU work on the GPU path: inventory

| host stage | core-s / M reads (pbmc1k_v3) | input already on the device? | could run on the device? |
|---|---|---|---|
| CR4 opal clip per read (`readSequences`) | 2.4 | no (FASTQ text) | yes: an alignment kernel (the engine already clips Hamming adapters on the device) |
| STAR mapping pass for hand-off reads | 3.3 (child) | the summaries are | yes: a bypass from the planes and summaries (scoped in `star-handoff-bypass.md`) |
| STAR solo records read back, UMI collapse | 0.7 | no | yes, with the bypass |
| hand-off records and summaries per distinct sequence (host) | 0.83 | the alignments are produced on the GPU | yes: assemble from the lane outputs on the device |
| dedup per-read pass | (3.8 s of hold) | yes | yes: kernels written (Phase B, pending gate) |
| host lane | (2.7 s of hold) | yes | partly (STAR's limits) |
| genome load (once per process) | 7.6 s fixed | no | not needed with a resident index |
| decode `.1fq` to temp FASTQ | ~1.3 | | not needed on the `.1fg2` route |
| non-host EM, ambient, doublets, demux (host, mostly single-threaded) | 5-50 s per sample | partly | partly |

## 10. The criterion, and the verdict on each segment so far

Every remaining segment must be at a hardware bound within ~20%, or justified in writing.

| segment | at a bound? | verdict |
|---|---|---|
| H2D copies (11% of PCIe) | no | fix: pinned staging buffers filled by several threads (written, under test) |
| `k_p3`, `k_p1` (7-13% of HBM, 9-21% of SM) | no | the SIMT divergence of the recursive stitch bounds it (section 10.1); cheap knobs tried and exhausted; the next step is a rewrite (warp-cooperative stitch) |
| dedup (3.8-17 s of host loops) | no | fix: per-read pass on the device (written, under test) |
| host lane | no | open |
| non-host index upload (24.9 GB per sample) | no | resident index |
| suffix-array upload (16.9 GB per sample) | no | resident index |
| CR4 opal clip (163 core-s of a 116 s child) | no | move to the device |
| STAR per-read pass | no | bypass |
| input from disk | not measured | |

### 10.1 `k_p3` (stage 3, the stitch): what bounds it, and what was tried (pbmc1k_v3, 66.6 M reads, 40.1 M in lane A)

Measured with the in-engine ledger (`SINGLET_ENGINE_GPU_TRACE=1`, now with a stage-3 warp-balance line) and Nsight Compute:

* 8.7 s of GPU time over 171 launches; 3.29e9 stitching steps in total. Thread cycles by phase: stitching about 49%, seed search 39%, windows 11%, selection and encoding under 1% (proportions only; the counters are not reset per batch).
* Warp efficiency in steps (total / (32 x the sum over warps of the largest step count)): 28.7% in the order the reads run, 43.9% sorted by stage-1 hit count, 50.5% by window count, 99.9% sorted by the true step count (oracle). Only 0.9% of reads take 0 steps; the top 1% of reads hold 5.2% of the steps, so the imbalance is broad.
* Nsight Compute: 1.9-2.1 active threads per instruction of 32, 94 registers (5 blocks of 128 per SM, 25.7% of warps active), issue slots 13.6% busy, SM 7.7%, DRAM 7%.

Tried, all on the same sample and reads, k_p3 time:

| change | k_p3 | result |
|---|---|---|
| baseline | 8.72 s | |
| bucketing, ascending by hit count | +2.7 s | worse |
| bucketing, descending by hit count | 10.91 s | worse (hit count predicts the step count poorly; the sort scatters the per-read state) |
| persistent threads (one atomic read counter per batch, every lane refills when its read ends) | 8.64 s | no change; ncu still 1.93 active threads per instruction |
| launch bounds for 6 / 8 / 12 blocks per SM | 8.70 / 8.61 / 9.97 s (default build 9.13 s in the same sweep) | no change within the run-to-run spread, worse at 12 (spills) |

Reading: the idle lanes are not waiting for the slowest read of the warp (the persistent version removes that wait and keeps the same 1.9). They sit at different code points of the recursive stitch at the same moment. More resident warps do not help either. The kernel is bound by the divergence of its control flow, not by memory or by SM throughput (7% / 8%). The remaining lever is to restructure the stitch so that a warp works on one read, or so that its lanes take the same branches (warp-cooperative stitch, branches turned into selects), which is a rewrite of a 1.5 k-line port with a byte-identity gate; the ceiling is about 8.7 s of the 25 s the device is held for a 66.6 M-read sample, and the sample's wall time is 255 s, dominated by the STAR child on the host.

## 11. Gaps

* The wall-clock budget has ~50 s (17%) unattributed on the 66 M-read sample; a run with marks and NVTX in the same process would close it.
* Lanes M, B and heavy, and the pileup kernels: no roofline yet.
* No `nsys` trace of the campaign concurrency configuration; no trace of the child's CUDA timeline (overlap, sync stalls) beyond the ledger's host times.
* The 263 M-read FASTQ route has the ledger but not the phase marks per sample on the same run; the `.1fg2` route has no ledger (older binary).

## 12. Runs

Ledger, 66 M: 794901 (`gate-794901-0`), 263 M: 794902; child marks 66 M: 794855; 263 M: 794854; `.1fg2` route 263 M: 794935; Nsight Compute: 794942 (`k_p3`),
794943 (`k_p1`), numbers extracted by 794974; nsys parent trace: 794901/794880; queue and concurrency runs 794059, 794277, 794278, 794517, 794581, 794608, 794736, 794716;
tools: `bench/engine_gpu_gate/s8_phases.sbatch` (`WRAPMODE=nsys|ncu`, `S8_TRACE=1`), `s8_queue.sbatch`, `s8_ncuq.sbatch`.
