# Fable GPU aligner: handoff (2026-11-02)

The program stopped on a usage limit, on Zach's instruction to wrap up. No new work was run for this document. It reads results of jobs that
had already finished.

- **In this branch (`gpu/fable-engine`, E1-E19 + E23):** the GPU path from `.1fq` to count outputs, on by default under `--engine gpu`, byte-identical
  to the CPU engine on all 10 benchmark runs.
- **Not in this branch:** E20 (`gpu/fable/lockclip`) and E21 (`gpu/fable/pool`). Both are pushed and both measured well, but neither has the all-sample gate.
- **Not mergeable yet:** a trial merge into `prod/campaign` conflicts in 10 files. It needs a staging merge and a full re-gate.

Reading order: this page, `GPU-PATH.md` (operator's page), `GPU_ALIGNER_STUDY.md` (the north-star study that ordered the program; Part II has every
result in order), `E<N>-*.md` (one report per experiment). The study stops at E23. Where it and this page disagree, this page is newer.

## 1. Key results

Whole-sample gate 798027 (binary ced7f553, T = 18, H100 NVL, against CPU references): all 10 runs identical, each at level 5 + ingest with no fallback.

| 10 runs, 548 M pairs | wall s | core-s |
|---|---|---|
| CPU engine | 3,303 | 19,388 |
| GPU path | 971 | 4,603 |
| ratio | 3.40 | 4.21 |

| sample | pairs | CPU wall / core-s | GPU wall / core-s |
|---|---|---|---|
| pbmc2m | 2.0 M | 49.6 / 161 | 32.4 / 121 |
| pbmc10m | 10.0 M | 100.5 / 482 | 45.2 / 193 |
| pbmc1k_v3 | 66.6 M | 403.9 / 2,507 | 101.5 / 548 |
| pbmc1k_v2 | 76.9 M | 380.8 / 2,798 | 104.4 / 536 |
| pbmc1k_cite_v3 (GEX) | 45.4 M | 278.0 / 1,713 | 78.4 / 411 |
| pbmc1k_cite_v3 antibody library, mapped as RNA | 12.6 M | 129.9 / 1,255 | 65.0 / 274 |
| pbmc5p_v1 (5') | 60.0 M | 495.4 / 2,123 | 123.6 / 509 |
| neuron1k_v3 (mouse) | 92.9 M | 491.5 / 2,148 | 131.5 / 615 |
| hgmm1k_v3 (barnyard) | 63.1 M | 363.1 / 1,597 | 90.5 / 443 |
| nuclei2k_v2 | 118.7 M | 609.9 / 4,604 | 198.7 / 954 |

E23 then cut neuron1k_v3 from 127.7 to 100.2 s (gate 798081: 21 of 21 verdicts identical, all 10 samples). Host memory on the GPU path is higher on large
samples: nuclei2k_v2 peaks at 84.4 GB of RSS against 34.9 GB.

Node throughput, 2 H100 NVL, 54 cores, pbmc1k_v3 streams, steady state unless marked:

| configuration | M pairs/s | core-s/M | host peak GB | head |
|---|---|---|---|---|
| CPU engine (same node) | 1.38-1.44 | | | |
| E18: 2 slots, 8 per GPU, 7 threads, pinned | 4.39 (3.24 over the span) | 9.1 | 607 | E17 |
| E21 pool: 2 slots, 8 per GPU | 4.56 (3.54 over the span) | 8.63 | 442 | E23 + pool |
| E21 pool: 2 slots, 12 per GPU | **5.44** (3.65 over the span) | 8.63 | 626 | E23 + pool |

Campaign arithmetic from E18: about 20.6 T pairs is ~54 node-days at 4.39 M pairs/s, against ~170 on the CPU engine. At 5.44 it is ~44 node-days, which is a
calculation and assumes that rate holds on real sample mixes.

The path to pbmc1k_v3 at level 5:

| step | change | effect |
|---|---|---|
| E11 | STAR's per-read loop off the path (level 5) | 211 s, identical |
| E14 | genome preload, concurrent probes | -32 s |
| E12 | prefill's device time | 38 -> 16 s |
| E13 | no FASTQ text | -26 s, RSS 61 -> 46 GB |
| E16 | tail | 36 -> 8.8 s |
| E17 | device budget | 91 / 80 / 40 / 24 GB: 103 / 105 / 108 / 121 s, identical. A100 fit is a budget setting. Barnyard 89 s against 375 s |
| E19 | defaults, loud fallbacks, GPU probe | no flags needed beyond `SINGLET_GPU_PILEUP=on` |
| E23 | R9 collapse no longer quadratic in the deepest group | pileup GPU time <= 0.32 s on every sample |

Ceilings:
- The exact port's kernels reach 3.64 M distinct/s against a ceiling of ~4.3-4.6.
- The node is host-core-bound, not GPU-bound. Pure kernel time would allow ~10 M pairs/s per node.
- At 9.1 core-s/M the process model tops out at ~5.9 M pairs/s, and each core-s/M removed is worth ~11%.

## 2. What was tried

| exp | what | outcome |
|---|---|---|
| E1 profile, E2 census | where kernel time goes; workload census | set the program's priorities |
| E3 warp-window, E4 order, E5 compact, E7 seeds | kernel parallelism, batch order, compact state, seed stage | merged, 100% identical. Reports list the variants tried (E4 81, E7 55) |
| E6 records, E8 e2e, E9 solo, E10 hostloop, E11 solowire | device records and Solo, then no per-read host loop | merged, level 5 |
| E12-E14, E16, E17, E19, E23 | see section 1 | merged |
| E15 posterior | R1 qualities in `.1fq` | `.1fq` writes constant 'F'. The constant changes 3.3% of multi-match CB decisions against STARsolo on FASTQ. Format question, open |
| E18 throughput | node configuration | merged (bench only). Queue model matches one-slot points within 1-5% |
| E20 lockclip | see section 3 | measured, not merged |
| E21 pool | see section 3 | measured, not merged |

## 3. Unmerged work, on pushed branches

**E20, `gpu/fable/lockclip`, code head de5d85aa, on top of f73938b2.** The device lock is held only around device work. STAR's CellRanger4 clip runs on the device
(`k_cr4`, 66.6 M reads in 0.093 s). The report `E20-lockclip.md` is an unfinished draft: sections 1-3 are written, and results sections hold `[[...]]` placeholders.

| pbmc1k_v3, one slot, T = 7 (job 798129) | M pairs/s | core-s/M | lock held s per run |
|---|---|---|---|
| f73938b2, k = 1 | 0.532 | 8.73 | 35.2 |
| E20, k = 1 | 0.555 | 7.50 | 26.2 |
| f73938b2, k = 8, steady | 1.599 | 8.11 | 31.9 |
| E20, k = 8, steady | **2.177** | 6.90 | 24.4 |

- **Identity:** pbmc1k_v3, neuron1k_v3 and pbmc5p_v1 identical to CPU, and the default path identical to E11's gpuD. The device clip matches the host's on neuron1k_v3's
  92.9 M reads. The stream runs' 4 mapDistinct dumps match. **The other 7 runs are not gated.**
- **Targets:** one-slot k = 8 >= 2.0 is met. Host core-s/M <= 6.5 is not (6.90). Node >= 5 M pairs/s was never measured on this branch.
- **CR4 probe:** deliberately stays on the host. It costs 19-27 core-s per sample, and a device version needs the index resident, so it belongs with E21. Reasons in E20 §3.

**E21, `gpu/fable/pool`, head 719fd650, on top of f73938b2.** `singlet gpu-pool`: a resident process per GPU that holds the genome and the non-host index, then forks
one worker per sample from a queue. It has no report and no `GPU-PATH.md` section yet. Results are in `jobs/fable/pool/results/run-798109/` on Clipper (log `logs/e21-node-798109.out`).

- **Throughput:** node table in section 1.
- **Identity:** all 144 pbmc1k_v3 runs in the single-sample points identical to CPU and to the same job's run alone. In a mixed run, pbmc1k_v3, pbmc5p_v1, neuron1k_v3 and nuclei2k_v2 are identical to CPU.
- **Memory:** fixed 82 GB for two pools, then 11.5-14.1 GB per sample in flight on average and 22 GB at peak. Peak private per run 25-29 GB, against 38-43 GB per sample before.
- **Components:** turning off the non-host server and the genome preload lowers the rate over the span from 3.54 to 3.44 and 3.26, and raises the host peak to 685 GB.
- **Open defect, not diagnosed:** hgmm1k_v3 in the mixed pool stops after ~30 s, with `STAR exited 102 (streaming path)`, pool exit 3. The barnyard genome differs from the other samples',
  so check the resident genome first. Do not use the pool on mixed-genome queues until this is fixed.

E20 and E21 have never run together, and both touch the start path. Expect them to compound, but measure it.

## 4. What still needs to be tried

In order:
1. **Bring in `prod/campaign`.** Merge base cf92cdb9, 291 commits behind. Conflicts: `CMakeLists.txt`, `engine_gpu.h`, `cb_correct.h`, `pileup_engine.h`, `engine_gpu.cu`, `m6.cu`, `pileup.cu`,
   `pileup_gpu.cu`, `singlet.cpp`, `SeqCacheGpuPrefill.cpp`. Campaign changed output definitions, so refresh the CPU references with `REFRESH_REF=1` and re-run
   `bench/fable/release/jobs/gate.sbatch`. Device dedup exists on both lines (E12 and #145): pick one.
2. **Finish E20:** gate the remaining samples, fill in the report, measure the node, merge.
3. **Finish E21:** fix hgmm1k_v3, write the report and the operator's section, merge, measure E20 + E21 on the node.
4. **Check the gaps listed in `FROM-BETA-SESSIONS.md` (#180):**
   - Deep samples are not gated: GSM7656149 has 263.4 M reads, and the largest gated here is 118.7 M.
   - 5' GSM6470208 writes 2.7 alignment records per read.
   - The drain worker's memory model needs refitting for the ingest.
   - The `.1fg2` table-fed prefill is not used by E13's `.1fq` ingest.
   - Chimeric, TE, viral and reads-per-UMI outputs at level 5 were not verified. `SeqCache.h` says its exactness assumes no chimeric detection.
5. **The largest remaining host costs:** the host track's tail is now the largest piece of the sample's wall and the node's core budget: demux auto-K + VB ~16 s / ~94 core-s, export/QC/callers
   17 s / 36 core-s, non-host's EM 7 s. It was not taken, pending Zach's decision.
6. **Smaller items:**
   - the device CR4 probe once E21's index is resident;
   - pbmc5p_v1's 3.06 s results fetch in the pileup;
   - the ingest's decline conditions (E19 §4b);
   - a real A100 run, since the fit so far is a budget setting on an H100.

## 5. Known defects and decisions

- **Non-host writer bug:** nuclei2k_v2's `nonhost_reads.fq` differs per run, with the same count. It indexes the screen's subsample against the whole unmapped file. It is not specific to the GPU path.
- **g050:** its GPU is invisible to jobs or fails CUDA initialisation. A cluster-admin item. Gates use `--exclude=g050`.
- **Unpinned workers** wander between GPUs and sometimes lose the device in the non-host screen. Outputs stay identical, but pin them.
- **`provenance.json`** records the binary's directory in the whitelist path.
- **Decisions for Zach:**
  - contract-2 experiments (output changes beyond CPU identity);
  - whether `.1fq` should keep R1 qualities;
  - whether to take on the host-side tail;
  - whether #180 should be merged into this branch. It is docs only and still open.

## 6. How to reproduce

```bash
sbatch --export=ALL,REFRESH_REF=1 bench/fable/release/jobs/gate.sbatch
sbatch --gres=gpu:nvidia_h100_nvl:2 -c 54 --mem=730G --export=ALL,POINTS="at7s2:pbmc1k_v3 p2k12t7s2:pbmc1k_v3*48" bench/fable/pool/jobs/run.sbatch
```

The second command needs `gpu/fable/pool` checked out. Job scripts for E20 are in `bench/fable/lockclip/jobs/` on its branch. CPU references are in
`$W/jobs/fable/release/ref/` with `W=/mnt/projects/debruinz_project/debruinz/singlet`.
