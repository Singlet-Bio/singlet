# Ideas from the unmerged GPU branches: verdicts

Each idea below was developed on a branch that never merged. It was redeveloped on `gpu/frontier`, tested, and adopted, kept off by default,
or rejected. Rates name the sample, binary, node, GPU count and baseline. Timing runs in an A/B/B/A design inside one Slurm job, with the
fixed reference binary b0 = gpu/frontier at 9560b8db where a b0 arm is named. From the GPU pilot (`bench/gpu/pilot/`) on, each point runs
on one pinned H100 NVL and its log shows the pin check. Earlier timing jobs exported `CUDA_VISIBLE_DEVICES=0`, but it never reached the
runs, so their rates are per node with two GPUs.
Identity means the 9-sample gate (`pr2-gate.sbatch`: out_equiv over every output, the ten Solo files, the mapping statistics, SJ.out.tab,
the read files by sorted records, CellReads.stats by sorted rows) of the GPU engine against the CPU engine. Job logs are on Clipper under
`$WORK/logs/`, harness results under `$WORK/jobs/gpu-integ/results/tp-<job>/` and `$WORK/jobs/gpu-campaign/results/`.

The source branches are deleted once their verdict is recorded; each tip stays pinned by its archive tag.

| # | Idea | Source branch, tip | Verdict | Switch |
|---|---|---|---|---|
| 1 | Non-host GPU screen falls back to the CPU | bug, no branch | resolved: device wandering, gone on a pinned GPU; retry kept on | `SINGLET_NONHOST_GPU_TRIES` |
| 2 | Transfer ledger and NVTX ranges | `perf/nvtx-host-ranges`, cacf6435 | adopted, off by default (diagnostic) | `SINGLET_ENGINE_GPU_TRACE=1`, `SINGLET_NVTX=1` |
| 3 | 512-base lift and lean-path memory (P3-MEM-03) | `perf/gpu-device`, a462be4d; `gpu/frontier-fold-device`, 7cc07dd4 | adopted: identical, cost-neutral, 1.2 G reads fit one H100 | default |
| 4 | VMM shared index | `gpu/bold-shidx-fix`, 84be99f8 | pending | `SINGLET_GPU_SHARED_INDEX=1` |
| 5 | E21 resident pool | `gpu/fable/pool`, 719fd650 (tag only) | pending | `singlet gpu-pool` |
| 6 | Heavy-read split | `gpu/bold-split`, 3c0ac3c2 (tag only) | pending | `SINGLET_ENGINE_GPU_STEPS_B=<n>\|auto` |
| 7 | Single-donor demux screen | `gpu/bold-hostcut`, f04a64c7 (tag only) | pending | `SINGLET_DEMUX_SCREEN=1` |
| 8 | WSHRINK | in the tree | pending | `SINGLET_BOLD_WSHRINK=1` |
| 9 | k_resolve exon-multimapper fix | no source | pending | |
| 10 | One device dedup | E12 and #145 | resolved: one remains | |
| 11 | Prefill out of memory at 2 slots per GPU | bug, found by item 1 | open: confounds every 2-slot A/B | |

## 1. Non-host GPU screen falls back to the CPU

On the unpinned timing job 803458 (80 runs per arm on both GPUs of g051) the non-host screen fell back to the CPU in 14 runs with its OOM
retry on, 31 with it off and 26 on b0. Every recovered run succeeded on its third attempt. Those jobs exported `CUDA_VISIBLE_DEVICES=0`,
but the variable never reached the runs, so their processes spread over both GPUs and the screen's 26.5 GB index upload could land on a
device that another process had just filled.

On one truly pinned GPU the screen never falls back. Pilot 803591 (g051, GPU 0, binary 4ea9d1e9, item 10-fb-a) ran four points of 40
pbmc1k_v3 runs at 8 in flight and 2 slots, A/B/B/A with the retry on (`base`) and off (`SINGLET_NONHOST_GPU_TRIES=1`). The pin check
confirmed that every copy used GPU 0, and none of the 160 runs had a non-host fallback. The retry costs nothing measurable: on 803458,
core-seconds per million read pairs with the retry off against on are +0.17 % [-0.75, +1.06] (paired bootstrap, 95 %), ruled out at a 3 %
threshold.

Verdict: the bug was device wandering on a two-GPU node, fixed by pinning. The retry stays on by default as a guard for a shared GPU.

The same points expose a different fallback, which is item 11.

## 11. Prefill out of memory at 2 slots per GPU

In pilot item 10-fb-a, 8 of the 160 runs (2, 1, 2 and 3 per point) lost the GPU in the prefill, before mapping started: `cudaDeviceSetLimit
(cudaLimitStackSize, ...)`, `dmal(&dWork, ...)` or an empty CUDA message, all out of memory. STAR then maps every read of that sample on
the CPU, which takes 620 to 700 s against a mean of 265 s, so one such run sets the span of its point. The span rate per GPU tracks the
count: 1.887 and 1.847 M read pairs/s for the points with 2 and 3 fallbacks, 1.974 and 1.975 for those with 1 and 2. In the 8-run blocks
of the WSHRINK A/B (item 20-wsh-seq) a single fallback halves the span rate, and the host copy of the genome that a fallback run loads
pushed the 340 GB step over its memory limit five times. Every A/B at 2 slots on a pinned GPU is confounded until this is fixed.

The likely cause is the E17 budget: each of the 2 slots gets (total - 2 GB)/2 of device memory, while the non-host screen's 24.9 GB index
sits outside the budget. With two samples in flight, a prefill that starts while the other slot holds the screen's index finds less memory
than its budget promised.

## 2. Transfer ledger and NVTX ranges

`perf/nvtx-host-ranges` (cacf6435) carried a ledger of every copy in engine_gpu.cu and NVTX ranges for mapDistinct's sections. The staging
merge of the Fable line (82a38f53) dropped both. The redevelopment (5f7f2f61) puts the ledger in its own header, `gpu_ledger.h`, which
wraps cudaMemcpy and cudaMemcpyAsync in every .cu file of the GPU path. It records file:line, bytes, host seconds and whether the host side
is pinned, and prints one block per phase (prefill, hand-off, pileup, nonhost, solo) when `SINGLET_ENGINE_GPU_TRACE=1`. The NVTX ranges
return for mapDistinct's index, reads, dedup and chunk sections and for the GPU and host lanes of each chunk (`SINGLET_NVTX=1`). Off, each
copy costs one branch.

Identity and cost: the gate 803456 runs pbmc1k_v3 twice, untraced and traced, and both are IDENTICAL to the CPU reference; wall is 124.4 s
untraced and 125.4 s traced, one run each. What it shows on that run: the prefill uploads the 31.8 GB genome index (pinned, async), the
non-host screen uploads its 26.5 GB index at 6.4 GB/s with the host blocked for 4.2 s, and the hand-off and pileup move 6.4 GB and 3.6 GB
at 12.3 GB/s from pinned memory. The two index uploads happen once per sample, which makes a resident index (the VMM shared index, the E21
server) the largest transfer saving on the table.

## 3. 512-base lift and lean-path memory (P3-MEM-03)

All nine commits of `perf/gpu-device` are on `gpu/frontier` (`git cherry` finds an equivalent patch for each): the eight of the fold
branch, fast-forwarded as 7cc07dd4, and the chunked hand-off pass, cherry-picked as b7f9244d. The lift (5b8afcbd) replaces the fixed
128-base low-quality mask with variable-width masks, so cDNA reads up to 512 bases stay on the GPU path. P3-MEM-03 runs the reads phase by
read chunks with the dedup by hash partition (step A), the records phase by record chunks (step B), frees the lean path's host copies once
used, and streams the reads of the E16 hand-off pass in chunks sized from free device memory while the records stay resident.

Identity: the 9-sample gate is IDENTICAL on the fold (803425, g051) and on gpu/frontier 6b4a6a25 (803456). Gate 803427 (a462be4d) is
IDENTICAL on pbmc1k_v3, neuron1k_v3 and a slice of GSM7064234 with every chunk forced small (`RCHUNK=8000000 RPARTS=3 RECCHUNK=4000000
SINGLET_E2E_PASS_CHUNK=3000000`), which exercises every chunk boundary on small inputs.

Capability: the full GSM7064234 (1.198 G reads, 90.7% unique) runs end to end on one H100 (803428, a462be4d, 728 GB allocation): exit 0
in 2647 s, device peak 93.1 GB, host peak 629 GB. Before the chunked hand-off this sample uploaded about 86 GB in that pass and died of a
thrust bad_alloc. STAR plus the GPU engine end at about 1130 s and the pileup at 1435 s; demux auto-K (about 430 s), ambient (280 s) and
mt plus export (220 s) are the rest. The host peak comes from the STAR caller keeping the read planes (about 240 GB at 1.2 G reads) for the
whole run, because the rRNA hook, ingRecord and E13 read them after mapDistinct; samples over about 800 M pairs need host-streaming
prefill (P3-MEM-15) to fit in 600 GB.

Cost on small samples: none. The baseline gate on 9560b8db (803435, g052, all nine IDENTICAL) and the fold gate (803425) agree on the
device peak to within 0.4 GB on every sample and on wall and core-seconds to within noise (pbmc1k_v3: 126.3 s, 567 core-s against
126.2 s, 566 core-s). The peak near 90 GB in both is the E17 arena filling the card (the gate's `dev_peak_gb` samples the whole GPU),
not the fold.

## 10. One device dedup

Only one device dedup is in the tree. #145's per-read dedup pass (`k_dd_flags`, `k_dd_repofgroup`, `k_dd_readrep`, switch
`SINGLET_ENGINE_GPU_HOSTDEDUP`, commit e51d081e on `perf/gpu-pinned-dedup`) and its pinned staging uploads (edbfbd97,
`SINGLET_ENGINE_GPU_FASTUP`) were dropped by the staging merge of the Fable line (82a38f53, 2026-11-04), which kept E12's dedup
(`k_heads`, `k_readrep`, the dedup's products on the device) and E12's staged uploads. E12's host loops stay behind
`SINGLET_ENGINE_GPU_DEDUP=host` and `SINGLET_ENGINE_GPU_CFG=old` as the opt-out. P3-MEM-03 adds the lean path's dedup by hash
partition, which is a memory plan of the same dedup, not a second one. Nothing to delete.

## Second architecture (T10): V100S and RTX 8000

The engine builds unchanged for sm_70 and sm_75 (`CMAKE_CUDA_ARCHITECTURES="70;75;80;90"`, build 803592; no kernel uses an sm_80-only
feature). At run time it declined every device below compute capability 8.0, so 8aa096b2 adds `SINGLET_ENGINE_GPU_MIN_CC` (default 80;
70 admits Clipper's V100S and RTX 8000). With it, pbmc2m runs the whole GPU path (level 5, ingest, GPU pileup and non-host screen, no
fallback) and is IDENTICAL to the CPU engine on both: RTX 8000 39.1 s and 126 core-s (803615), V100S 39.5 s and 157 core-s, device peak
28.6 GB of 32 GB (803616), against 34.2 s and 121 core-s on an H100 NVL for the same engine (803549). The same outputs on three GPU
architectures are evidence for T10. The 9-sample gates (803632 on the RTX 8000, 803633 on the V100S) gave IDENTICAL for seven samples. hgmm1k_v3 does not fit: its barnyard
index is 52.7 GB, and E17 stops the run as designed (budgets 42.3 and 29.7 GB). pbmc1k_cite_v3 differed in nonhost_reads.fq and
Unmapped.out.mate1 with the same record counts. The cause was the input: the T18 reference had read an older encoding of the sample's .1fq,
and the file was rewritten before these gates. With CPU and GPU reading the same file on the RTX 8000 (803808), cite is IDENTICAL even
though the non-host screen ran out of device memory there and screened on the CPU. Identity checks of samples whose index fits 30 GB can
therefore run on the V100S and RTX 8000 nodes; barnyard samples stay on the H100s.
