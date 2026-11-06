# The singlet GPU aligner: context, criteria, blind-vs-measured profile, and a GPU-native redesign

Written 2026-10-31 by Claude Fable 5.1 from `Singlet-Bio/singlet` (branch `gpu/gpu-slots2`, tip cf92cdb9, the latest of the
`gpu/*` stack) and `Singlet-Bio/singlet-agents` (`main` and `bench/ground-truth`). Thought experiments only; nothing was run.
Companion file: `engine/notes/gpu-aligner-blind-predictions.md` (written before any measurement was read).

Hardware assumed throughout: Clipper H100 NVL (94 GB HBM3, 132 SMs, 60 MB L2 as measured by M1, PCIe Gen5 measured at
12.3 GB/s pinned H2D on that node) and A100 40/80 GB (108 SMs, 40 MB L2, HBM2e 1.55/2.0 TB/s).

---

## 1. What the aligner is

`singlet --engine gpu` is not a GPU aligner in the usual sense. It is STAR 2.7.11b, vendored and modified, in which the
**mapping of each distinct cDNA sequence** (seed search, windows, stitching, extension, multimapper selection) is done once on
the GPU before STAR starts, and STAR's own sequence cache is pre-filled with those results. STAR then runs unchanged over all
reads and finds ~99.99% of them in the cache. Later stages (barcodes, the pileup, counting) were subsequently moved to the GPU
too, with STAR's per-read pass reduced to replaying per-sequence summaries.

### 1.1 Code map (branch `gpu/gpu-slots2`)

| piece | file | lines | role |
|---|---|---|---|
| public API | `include/singlet/engine_gpu.h` | 170 | `mapDistinct`, `nonhostClassify`, `pileupRun`, device file locks (`SINGLET_ENGINE_GPU_SLOTS`) |
| host orchestration | `src/engine_gpu/engine_gpu.cu` | 529 | index upload, read upload, clip, dedup (hash sort), prefix-sort, lane cascade A→M→B→host, record assembly |
| the port | `src/engine_gpu/stitch.cuh` | 1553 | line-by-line port of `qualitySplit`, `maxMappableLength2strands`, `storeAligns`, `createExtendWindowsWithAlign`, `assignAlignToWindow`, `extendAlign`, `stitchAlignToTranscript`, `stitchWindowAligns` (explicit stack), `multMapSelect`, `mappedFilter`, cache-blob encoding |
| STAR glue | `src/star/SeqCacheGpuPrefill.cpp` | 488 | reads the FASTQ text (mmap, 24 threads), CR4/Hamming clip, calls `mapDistinct`, inserts into `SeqCache`, writes the pileup hand-off |
| GPU pileup | `src/engine_gpu/pileup_gpu.cu`, `gpu_pileup/{m6,pileup}/*.cu` | ~1800 | sort/reduce counting under the order-free rules R1–R12 |
| non-host | `src/engine_gpu/nonhost_gpu.cu` | 400 | minimizer screen with the 30 GB index resident |

Design and evidence: `docs/engine-gpu.md`, `docs/engine-gpu-results.md`, `docs/gpu-engine/exploration.md` (W11, 1934 lines),
`docs/research-beta/{gpu-host-offload,host-cost-model,gpu-pileup-design,align-and-count-design,gpu-cr4-clip}.md`.

### 1.2 Kernel structure (what actually runs per sample)

1. Upload STAR's index unchanged: byte genome G (3.2 GB, 1 byte per base, both strands + sjdb inserts), packed 33-bit SA
   (12.8 GB for GRCh38 at `genomeSAsparseD 2`; ~22.5 GB dense for GRCm39), packed SAindex (1.6 GB). **Per process, i.e. per
   sample.** ~4–6 s pageable.
2. Upload all read sequences as 1 byte per base (~6 GB for 66 M reads), clip on the GPU (`k_clip`), hash (two 64-bit hashes),
   `thrust::sort_by_key`, group, byte-compare every member against its group head (a collision aborts the GPU path), keep one
   representative per distinct sequence, drop sequences with N, then **stable-sort the representatives by their first 32 bases**
   (2.7× faster mapping from locality).
3. Map distinct sequences in chunks of 16 M through a **lane cascade**. Each lane is a `Caps<>` template instantiation with fixed
   per-read capacities (windows, aligns per window, exons, transcripts kept, blob bytes) and a step budget for the stitching
   recursion:

   | lane | windows / aligns per window / exons / kept | step budget | reads in flight | where |
   |---|---|---|---|---|
   | A | 16 / 16 / 8 / 8 | 1024 | 262,144 | GPU |
   | M | 64 / 8 / 8 / 32 | 2048 | 262,144 | GPU |
   | B | 128 / 50 / 20 / 64 | 2048 | 65,536 | GPU |
   | host | STAR's limits (10,000 / 50 / 20 / 4096) | none | 24 threads | CPU, same source |

   A read that overflows any capacity or its step budget returns `Status::Overflow` and is re-run in the next lane. Measured
   shares on pbmc1k_v3: A 100%, M 26%, B 4.9%, host 0.27%.
4. Each lane batch runs three kernels: `k_p1` (thread per read: read set-up, seed loop, windows from anchors, flanks),
   `k_p2` (warp per read: scan every SA entry of every seed, coalesced, mark the entries whose bin holds a window), `k_p3`
   (thread per read: assign the marked entries in STAR's order, stitch each window with the explicit-stack recursion, select,
   encode). Between them: a `thrust::exclusive_scan`, two synchronous D2H copies of one word each, a `cudaMemset`.
5. Per-thread state is a `Work<Caps>` struct in **global memory, array-of-structs, one struct per thread** (the comment in
   `engine_gpu.cu` says ~150 KB per read in lane A including `ReadBuf`: 262,144 × 150 KB ≈ 40 GB). Transcripts (`Tr<NEX>`,
   ~770 B each) are copied whole on every include/exclude step of the recursion (`copyFrom`, live rows only).
6. Results are packed (`k_pack`) and copied back; the host assembles records, then STAR's threads insert ~32 M records into the
   sequence cache and build the pileup hand-off.

Everything is one CUDA stream, synchronous `cudaMemcpy`, `cudaMalloc`/`cudaFree` per lane per chunk, `cudaDeviceSynchronize`
after each lane. Samples share a GPU through `flock` on a per-device lock file (1–8 "slots"), each process holding its own copy
of the index.

### 1.3 Numbers that define the current state (all measured on Clipper, from the docs)

| quantity | value | source |
|---|---|---|
| identity of the port to STAR | 100% of 1.99 M sampled distinct sequences byte-identical; whole samples: 0 verify mismatches on 64–66 M reads; 7 of 8 release-gate datasets bit-identical on every data file | engine-gpu-results §1, release-gate §7 |
| GPU mapping rate | **2.40 M distinct sequences/s** per H100 (lane A alone: 6.4 M/s); 31.9 M distinct sequences of pbmc1k_v3 in 17.1 s | exploration §12.12 |
| where M5 time goes | lanes M+B (31% of sequences, > 16 windows) = **70%** of GPU time; lane B's 4.5% = 63%, of which 81% stitching, ~3,110 SA entries per read | §12.12 |
| seed search alone (M3) | ~1.0 G seed records/s = 78–105 M distinct reads/s; 323.7 M records identical to STAR; ~10.5 dependent steps per record; SIMT 0.39 unsorted / 0.61 prefix-sorted (only 1.03–1.14× faster) | §12.6 |
| random-access roofline (M1) | 28–29 G 32-byte sectors/s, flat from 8 to 74 GB footprint; loaded dependent latency 1.17 µs; 34k chains in flight reach 90% of the rate | §12.1 |
| distinct fraction d | 0.30–0.52 for 3′, 0.30 for 5′, **0.66 for nuclei**; cross-sample reuse 19% of distinct / 49% of reads for same-length same-chemistry pairs | §12.2, §12.16 |
| per-sample wall, 66.6 M reads, one H100 node, 18 threads | CPU engine 480–494 s; `--engine gpu` + GPU pileup **298–373 s**; **GPU busy 6–8% of wall** (~20 s) | gpu-host-offload |
| host cost | CPU engine 36–40 core-s per M reads; GPU path 12.5–14.4 (3′), 18.7→~14.5 (5′); cloud break-even 12.7 | host-cost-model, gpu-host-offload |
| device memory | human index 17.6 GB; lane A work ~40 GB; pileup 25–31 GB at 60–105 M reads, ~100 GB at 223 M (needed K=2 cell partitions) | engine_gpu.cu, align-and-count-design §4.4 |
| node throughput (projected if cores bind) | CPU engine 1.3–1.5 M pairs/s; GPU path 2.65–3.07 M pairs/s | engine-gpu-results §2 |

The campaign is 70k samples, ~20.6 T read pairs. Input supply (download/convert) is ~1 GB/s aggregate on Clipper, so beyond
~2 GPUs at 10 M pairs/s the campaign is input-bound, not compute-bound (exploration §7.4, §12.10).

---

## 2. Success criteria

### 2.1 As the repositories define them (verbatim intent, with file pointers)

1. **Counting is exact.** Every count output (matrices, junctions, SNP AD/DP, chrM, cell calls, …) must be byte-identical to the
   CPU engine under the order-free rules R1–R12 (`order-free-rules.md`), given the same alignments. Gate (a) =
   `gpu_pileup_gate.sbatch` / `out_equiv.py` / `gpu_diff.py`; the known 6–7 run-dependent files (unmapped-read order, logs,
   timings) are the only allowed differences.
2. **Alignment: no regression on M1–M7** (Zach, 2026-10-29). Byte identity to STAR is no longer required. A new aligner is
   accepted only with its scores on the frozen ground-truth harness (`bench/ground-truth/PREREGISTRATION.md`, frozen
   2026-10-28; `gate/GATE.md` validated: STAR production passes with every difference 0; best-practice STAR and HISAT2 fail):
   - M1 locus-correct share at ≤ 0.5% wrong (headline; STAR cannot reach the operating point, so default operating points are
     also reported); M2 junction recall/precision per read and per junction; M3 UMI-to-gene assignment; M4
     spliced/unspliced/ambiguous accuracy; M5 calibration (ECE) and multimapper set recall; M6 allele-dependent loss; M7
     foreign-read acceptance.
   - Non-inferiority margins: M1 0.1 pp, M2 0.5 pp, M3 0.1 pp, M4 0.5 pp, M5 ECE 0.01, M6 1 pp, M7 0.05 pp, at the paired
     bootstrap interval bound (1,000 replicates over cells).
   - Guard rails vs STAR production within 2× STAR's own noise floor: cells called, UMIs per cell, per-cell exon correlation.
   - The **win rule** (for calling the aligner *better*): at least one measure better than STAR best practice with the CI
     excluding zero, none worse beyond its margin, guard rails intact. Test split untouched until phase 6.
3. **Determinism.** Byte-identical outputs across runs, thread counts, GPU counts, batch sizes and cell partitions K (gate (c):
   K = 1, 2, 7 identical). No float atomics; sort-based reductions; `-fmad=false` where host floats are replicated.
4. **Throughput and cost.** Brief: ≥ 10 M pairs/s end to end per H100, P1 kill at < 5 M/s (modelled; passed). Engine gates:
   host core-s per M pairs not worse than the CPU path, cloud break-even 12.7 core-s/M on a 54-core GPU node; M5 requirement
   ≥ 10 M distinct reads/s per GPU (currently 2.4, "FAIL by 4.2×"); node target ≥ 20 M pairs/s.
5. **Fit and reliability.** Single GPU; the largest sample per class runs on one H100 without the unified-memory retry
   (gate (f)); sizes estimated up front, partition by cell if needed; every GPU failure falls back to the CPU path with a logged
   reason (the nuclei2k_v2 hard failure in the release gate is the counter-example to fix).
6. **Generality** (align-and-count design §1): every protocol singlet supports (3′, 5′, paired, plate, spatial, ATAC/multiome,
   feature barcodes), all species, through one engine configured from the protocol table.

### 2.2 Proposed thresholds for a change to the GPU aligner

The current engine's own number is the reference ("status quo"). For any change, measured on the six gate samples
(pbmc1k_v3, GSM4006843, GSM7036403, GSM7656149, GSM6470208, neuron1k_v3) plus nuclei2k_v2, three runs each:

| dimension | demonstrable improvement | status quo | acceptable regression | hard floor (reject) |
|---|---|---|---|---|
| M5 rate (distinct sequences/s per GPU, whole sample) | ≥ +25% on every sample and ≥ +50% on at least one (the noise across 3 runs is ~5%) | within ±10% | up to −10% only if it buys a hard-floor property (e.g. A100-40 fit, determinism) | — |
| per-sample wall, one H100 node, 18 threads | ≥ −15% on every sample | ±5% | ≤ +5% | > +15% |
| host core-s per M reads | ≥ −1.5 /M | ±0.5 | ≤ +0.5 | above 12.7 (break-even) on 3′, above the CPU path on any sample |
| GPU busy share of wall | rises (the goal is a GPU-bound node) | — | — | — |
| peak device memory (human 3′, 66 M reads) | ≤ 30 GB (two samples per H100 NVL, one per A100-80) | 50–75 GB | — | > 94 GB; any unified-memory retry |
| counting outputs | identical | identical | none | any byte differs (outside the 7 known files) |
| alignment, if the port is kept exact | 0 verify mismatches | 0 | none | ≥ 1 |
| alignment, if the algorithm changes | win rule met on dev; test untouched | all M1–M7 within margin | within margin, with the M1–M7 table shipped | any measure beyond its margin; any guard rail beyond 2× noise |
| determinism | identical across runs / threads / K / batch sizes / GPU model (H100 vs A100) | identical | none | any difference |
| failure handling | every GPU error → CPU fallback with reason; no silent fallbacks in production (log a counter) | — | — | a hard failure with no outputs |

"Acceptable regression" exists only for speed and memory, and only in exchange for a floor property. Accuracy, counting identity
and determinism have no regression budget.

---

## 3. Blind predictions against the measurements

My predictions (`engine/notes/gpu-aligner-blind-predictions.md`) were written from the code before reading any result. One
headline had leaked (GPU busy ~6%, host core-s ~40× GPU busy-s); everything else was unprimed.

| item | predicted | measured | verdict |
|---|---|---|---|
| end-to-end split | GPU < 10% of wall; sample host-bound; ~2× end to end despite > 50× on mapping | GPU 6–8% busy; 494 → 298–373 s wall (1.3–1.7×); host 36–40 → 12.5–14.4 core-s/M | right (leaked headline; the per-stage ranking was not) |
| index upload | 2–4 s pageable, once per process | 4–6 s pageable; "index upload" timed, `SINGLET_ENGINE_GPU_PIN` added later | right |
| dedup | 1.5–3 s | 1.6 s GPU + a serial host stretch | right |
| lane A | 6–12 s GPU, ~15–40 ms per batch | lane A alone runs at 6.4 M distinct/s → ~5 s for 31.9 M | right |
| lanes M and B | 1–3 s, "a few % of reads" | **70% of GPU time**, 31% of sequences overflow lane A (> 16 windows) | **wrong by 4–6×**: I under-estimated how many 10x reads are multimappers with up to 50 anchor loci, and the cost of visiting ~3,000 SA entries per read in the wide lanes |
| the longest kernel | `k_p3` (stitching state traffic) | stitching is 81% of lane B; cycle counters attribute 54–85% to window assignment + stitching of > 16-window reads | right in kind, wrong in which lane |
| `k_p1` seed stage | latency-bound, L2 hit 40–60%, DRAM 10–25% of peak, SIMT 30–50% | M3: ~20 G accesses/s = **70% of the gather roofline**, SIMT 0.39 → 0.61 sorted; the seed stage is not the bottleneck at all (78–105 M distinct reads/s) | **wrong**: the seed search is near its roofline, not under-occupied; sorting barely helps (1.03–1.14×) |
| per-thread AoS Work state as the main kernel inefficiency | yes | not measurable: Nsight Compute could not run on Clipper's nodes; only `clock64()` counters exist. The W11 team names the same lever (compact transcripts, shared-memory state) | unverified, consistent |
| memory footprint | 60–75 GB peak; A100-40 impossible without shrinking NB | lane A Work ≈ 40 GB; knobs `SINGLET_ENGINE_GPU_NB/NBB/CHUNK` added on 2026-10-31 for multi-slot use; 2 samples per 94 GB only by shrinking chunks | right |
| no overlap / single stream / per-lane malloc | a measurable loss | the serial 60 s single-core stretch in `mapDistinct` was found and attributed to pageable copies, dedup loop and record assembly (≈ 1 core-s/M, up to 1 min of wall) | right |
| STAR per-read pass dominates | 150–300 s at 18 threads | 34–47 s wall at 18 threads, 3.8–4.1 core-s/M (readLoad 51, map 24, output 109 thread-s) | **wrong by 4×**: the cache hit path is cheaper than I assumed (1.45 µs per hit) |
| the step budget evicting the worst reads | yes | yes: budget 1024 → +1.7× ; lane M → +1.14× | right |
| sorting by prefix | helps locality | 2.7× on M5 (not 1.1×): neighbouring reads share windows and loci in stitching, not only SAi lines | under-estimated |

Two lessons for the next round:
1. **The expensive reads are the multimappers, not the "too-short" or repeat reads I had in mind.** In 10x 3′ data 26–31% of
   distinct sequences create more than 16 windows. Any design must treat "many windows, few aligns each" as the common heavy
   case, with a parallel decomposition across windows, not across seeds.
2. **Seeding is solved; stitching is not.** The exact STAR seed search already runs at ~1 G records/s (≈ 40× faster per read
   than the stitch stage). Certificates (U-array) were correct but only 16–23% coverage and were dropped. Effort on seeding has
   almost no return; effort on windows/stitching/selection has all of it.

---

## 4. Obvious improvements to the current engine (kept exact, scored against §2.2)

Ranked by expected gain on the binding quantity for each phase. The node is host-bound today, so items 1–3 move wall and
core-s; items 4–8 move M5, which binds only once host work is below ~7 core-s/M.

### 4.1 Make the GPU path a service, not a per-sample process (wall −60 to −90 s per sample, memory −18 GB per extra sample)
- One persistent process per GPU holds G/SA/SAi (and the non-host index) resident; samples arrive as `.1fq` byte ranges over
  a Unix socket or shared memory; results return as the hand-off record. Removes the 4–6 s index upload, the per-process
  context, the file-lock scheme (`lockDevice`) and the duplicate 17.6 GB per concurrent sample. Both species fit (17.6 + ~27
  GB) with room for two in-flight batches on 94 GB; on A100-80 one species at a time.
- Exact by construction. Also the precondition for cross-sample reuse (19% of distinct sequences, 49% of reads on
  same-length same-chemistry pairs; §12.16): a per-species, per-read-length result table keyed by the full sequence.

### 4.2 Remove the FASTQ text round trip and the cache inserts (host −3 to −4 core-s/M, wall −40 s)
- Today: `.1fq` → FASTQ text (26 GB for pbmc1k_v3) → STAR parses it → the prefill mmaps and parses R2 again → GPU → 32 M
  `SeqCache::insert` calls → STAR looks every read up again. The hand-off already carries per-read record ids
  (`readRecord`), so STAR's per-read pass only needs the record id, barcode, UMI and qualities per read: feed it 2-bit reads
  from `fq::RunSet` directly and skip the cache for hand-off reads entirely (the summaries already replace it for mapped
  sequences; "deferInsert" is halfway there).
- This is S8 stages 2–5 and 8 in `gpu-host-offload.md` (≈ 9 of 14 core-s/M). The first step (three serial passes parallelised)
  already took wall from 375 to 298 s.

### 4.3 Overlap, streams and pinned memory inside `mapDistinct` (wall −30 to −60 s per sample, GPU busy ↑)
- Three streams: copy (pinned ring, `cudaMemcpyAsync`), lane A, lanes M/B. Lane M/B of batch *i* runs concurrently with lane A
  of batch *i+1*; the tail of a wide-lane batch no longer idles the GPU. The W11 doc lists this ("lanes overlapped") but it
  is not in the code: `runLane` allocates, runs, synchronises and frees per lane per chunk.
- Replace per-lane `cudaMalloc/cudaFree` with one arena sized up front (the pileup already does this); use a CUDA Graph per
  batch (3 kernels + scan + memset = ~10 launches, ~2 µs each is small, but the graph also removes the two blocking D2H
  copies of `lastB/lastN` by sizing the bit array to a worst case).
- The host loops between lanes (`for k < repc.size()` building `repM`, `lane`, `lpos`) are O(16 M) serial scans; move the
  partition to the device (`DevicePartition`) and copy only indices.

### 4.4 Lanes M and B: parallelise across windows (M5 ×2–3; this is 70% of GPU time)
- STAR's per-read loop stitches windows one after another; each window's recursion is independent except for
  `maxScoreMate` (used in the `Score + range >= maxScoreMate` recording test) and the running `trBest`. The W11 note's two-pass
  trick is correct: pass 1 computes each window's maximum finalised score independently (the maximum does not depend on
  recording), a prefix max over windows gives STAR's `maxScoreMate` at each window's start, pass 2 re-runs the recording with
  that value. Both passes are warp-parallel over windows (one lane per window; a warp handles 32 windows; a 128-window read
  takes 4 iterations). The recursion stack per lane stays in registers/local memory because lane-M/B windows have ≤ 8–50
  aligns but typically 1–3.
- Keep thread-per-read only for the selection step. This turns a 940 ns-per-heavy-read serial tail into ~32-way parallel work;
  the expected lane B time falls from 63% to ~15–20% of M5. Exactness: the enumeration order within a window is unchanged;
  only the cross-window dependency is replayed exactly.

### 4.5 Lane A: compact state and shared-memory transcripts (M5 lane A ×1.5–2.5)
- `Tr<8>` is ~770 B with 64-bit fields; every exon field fits in 32 bits (read offsets < 2^16, genome < 2^33 but `EX_G` can be
  stored as a 32-bit offset from the window start plus a 64-bit window base). A 32-bit `Tr<8>` is ~250 B; the stack of 17 + 9
  slots + 8 kept becomes ~8.5 KB per thread, still too big for shared memory at 128 threads/block (1.1 MB) but 3× less global
  traffic. Alternative: put only the active frame's transcript in registers and the stack in local memory with a
  **structure-of-arrays layout** so that lane *t*'s field *f* sits at `base + f*NB + t` (coalesced across the warp; today each
  field access of a warp touches 32 sectors).
- `PC`, `WC`, `WA` and the winBin map likewise SoA. This is a mechanical change; identity is testable by the existing
  verify run.

### 4.6 A persistent "grab the next read" kernel instead of fixed batches (M5 ×1.2–1.5 from tails)
- Each lane batch ends with its slowest read. A persistent kernel with an atomic work counter keeps every SM busy until the
  queue empties, and lets the next batch's reads start while the previous tail finishes. Deterministic because results are
  written by read index, not by completion order.

### 4.7 Two-bit genome compare in `compareSeqToGenome` (seed stage ×1.3; small overall)
- The byte genome costs one load per base compared; a 2-bit text with an N bitmap compares 32 bases per 8-byte load (XOR, `clz`).
  The seed stage is ~1/40 of M5 so this is low priority; it matters only for the GPU-native design where seeding is a
  larger share.

### 4.8 Cross-sample result table (M5 work −19% now, more as the campaign runs)
- Key = (species, read length, clip settings, full sequence). Exact. A GPU-side probe of a host-resident table (pinned, sorted
  keys, merge-join after the dedup sort) costs ~1 s per batch per §4.2b of the exploration.

What these add up to, kept exact: M5 from 2.4 to ~8–12 M distinct sequences/s per GPU (items 4.4–4.6), wall per 66 M
sample from ~300 s to ~150–180 s (4.1–4.3), host from 14 to ~8–9 core-s/M (4.2). That reaches the ≥ 10 M distinct/s
requirement and the break-even, and leaves the node host-bound at ~20–25 M pairs/s. It does not reach "the GPU is the
bottleneck"; §5 is what does.

---

## 5. Thought experiment: the GPU-native paradigm

### 5.1 What the physics says

- **Random access is cheap and already saturated by the seed search.** 29 G sectors/s with 34k chains in flight; a STAR seed
  call costs ~10 dependent steps × 2 accesses. The port issues ~20 G accesses/s. There is at most ~1.4× left in seeding from
  occupancy and nothing from bandwidth. The only seeding levers are *fewer accesses per read* (reuse, locus propagation) and
  *fewer reads* (dedup, cross-sample tables).
- **The stitch stage is not memory-bound and not instruction-bound; it is structure-bound.** Lane A at 6.4 M reads/s is
  ~156 ns per read; at 4.5 T issued int-ops/s (30% of peak) that is ~700k instruction slots per read against ~10–40k of real
  work. The gap is divergence (one read per thread with data-dependent recursion), per-thread state traffic (32 sectors per
  warp access), and batch tails. These are decomposition problems, so the redesign is about decomposition.
- **Everything after mapping is sort and reduce** and already runs at 1–3 G reads/s on the GPU (M6, GPU pileup, M13). Those
  stages cost < 3% of GPU time at 50 M pairs/s and need no redesign.
- **Little's law for the whole pipeline.** At the loaded latency of 1.17 µs and 29 G sectors/s the GPU needs ~34k accesses in
  flight. A wavefront design with 4–8 M reads per batch and thread-per-seed kernels is two orders of magnitude above that; the
  batches can be *smaller* than today (262k) if the kernels are short and independent, which is what makes multi-sample
  interleaving and A100-40 fit possible.
- **Hopper vs Ampere.** Nothing in this workload needs tensor cores, TMA or clusters. What helps on both: `cp.async` prefetch of
  SA ranges into shared memory (Ampere+), L2 residency control for the top SAindex levels (+4–11% measured in M1, not worth
  engineering now), independent thread scheduling (Volta+) for warp-cooperative loops with early exits. Hopper-only wins:
  distributed shared memory to share a read's windows across a 2–4-block cluster when a read has > 128 windows (rare), and the
  larger L2 (60 vs 40 MB). The design below should be CC 8.0+ with optional 9.0 paths, as the current build already is.

### 5.2 The design: a persistent per-GPU engine, wavefront kernels over SoA queues, CPU as a byte pump

```
host (1–3 cores per GPU)                                GPU (index resident; batches of 2–8 M reads from ≥ 1 sample)
.1fq blocks ──zstd decode (0.085 core-s/M)──► pinned ring ══► K0 unpack: 2-bit R2, CB+UMI key, clip spans, qual mask
                                                             K1 dedup: hash → radix sort → compare → seq_id; cross-sample table probe
                                                             K2 seeds: STAR's seed loop as rounds: emit queries → prefix sort →
                                                                 LCP leader election → thread-per-query SA search → storeAligns replay
                                                             K3 windows: thread-per-read createExtendWindows from anchors (≤ 50 loci)
                                                             K4 assignment: warp-per-read coalesced SA scan (today's k_p2)
                                                             K5 stitch: ONE LANE PER WINDOW, warp-cooperative; two-pass maxScoreMate
                                                             K6 select: thread-per-read multMapSelect/mappedFilter → per-sequence record
                                                             K7 annotate: per distinct sequence → gene / junction / SNP / chrM features
                                                             K8 expand to reads, radix sort (cell, feature, UMI), collapse, reduce → CSR
                                                             K9 non-host on unmapped distinct; barcode correction; reducers
host: callers (cell calling, ambient, doublets, demux), export; .s1 writer         ◄══ D2H ~100 MB per sample
```

Specific design decisions and why:

1. **Index.** Keep STAR's SA and SAindex (the contract for the exact path, and the seed search is already at its roofline).
   Replace the byte genome by a 2-bit text + N bitmap (3.2 → 0.9 GB) and add the sampled SA bucket keys (SBK, 1.5 GB for the
   sparse human SA) so that most binary-search steps compare against contiguous 8-byte keys instead of dereferencing the
   genome. Human core index ≈ 16 GB; mouse dense ≈ 25 GB.
2. **Per-read state ≤ 512 B, SoA, fixed caps; overflow to a wide lane.** Caps from the measured distributions: 24 seeds, 8
   windows in the narrow lane. The wide lane is the warp-per-read path, with windows spread over lanes and aligns in shared
   memory (≤ 50 × 16 B × 32 windows = 25 KB per warp-read; fits with 4 warps per block on both architectures). The host lane
   stays as the exact fallback for the ~0.3% that exceed everything.
3. **Stitching as a parallel decomposition over windows** (§4.4), and inside a window either STAR's recursion (exact) or a
   chaining DP over the window's aligns (O(n²) for n ≤ 16, trivial per lane) under the new contract. The DP finds STAR's
   best transcript when scores are additive along the chain; STAR's score is additive except the genomic-length term, which is
   applied at finalisation and can be applied to the DP's k-best. The exactness-preserving choice is the recursion; the
   speed choice is the DP; the gate decides.
4. **No batch tails: persistent kernels with work queues** for K2/K5; results indexed by read id, so determinism holds.
5. **Memory plan (per GPU).** Index 16–25 GB; batch of 4 M distinct reads × 512 B = 2 GB; seeds 4 M × 24 × 16 B = 1.5 GB;
   wide-lane scratch 64k reads × 32 KB = 2 GB; per-sample count tuples (worst 263 M reads × 16 B, double-buffered) 8 GB;
   pileup working set sized up front with cell partitioning (K from the estimate). Total **≈ 30–40 GB on H100 NVL for one
   species, two batches in flight**; **A100-40 fits the human sparse index with one batch and cell partitioning (≈ 28 GB)**;
   mouse dense (25 GB) on A100-40 needs either a sparse-D 2 mouse index (allowed under the new contract if M1–M7 hold) or
   streaming the pileup to the host. A100-80 fits everything.
6. **Occupancy plan.** K2 thread-per-query, 64 registers, 50% occupancy is enough (M3 shows 70% of roofline already). K5
   warp-per-read with 25 KB shared per warp-read → 4–8 warp-reads per SM → ~1,000 reads in flight per GPU; each read's windows
   are independent so the warp is never waiting on one lane's recursion except the longest window. K8 is CUB.
7. **CPU role.** Decode `.1fq` (measured 12–15 M pairs/s per thread), pinned-ring producer, orchestration, callers and export
   (W ≈ 1.4–2.2 core-s/M today; demux auto-K and the non-host EM are 74% of it). To make the CPU "extremely lightweight" the
   demux fits and the EM must also move (bit-exact ports are feasible because the CPU sums are serial per accumulator, §12.9
   step 3), after which the host is ~0.6 core-s/M and a 56-core node supports ~80 M pairs/s.
8. **Multi-sample batching** by (species, read length, clip, chemistry): same-study samples share 17% of distinct sequences;
   small samples no longer pay fixed costs; the GPU stays fed while a sample's callers run on the host.

### 5.3 Throughput arithmetic (per H100 NVL, 3′ human, d = 0.47)

| stage | cost per distinct read | bound | M pairs/s |
|---|---|---|---|
| seeds (exact STAR, no certificates) | 13 records × 10 steps × 2 accesses = 260 sectors | 29 G/s × 0.7 | 78 M distinct → **166** |
| windows + assignment | ~50 sectors + 3k coalesced entries for 30% of reads | bandwidth | > 300 |
| stitch, narrow lane (69%), warp-per-window | ~30 k instr, ~100 sectors | 4.5 T instr/s | ~150 M distinct → 320 |
| stitch, wide lane (31%), 32-way window parallel | ~300 k instr per read / 32 lanes + 3k entries | instr + tails | ~40 M distinct → 85 |
| select + annotate + expand + sort + collapse | ~300 B + 2 sectors per pair | M6: 1.6–2 G pairs/s | > 1,000 |
| **mapping, combined** (harmonic over distinct-read stages, then ÷ d) | | | **~45–60** |
| host at W = 1.4 core-s/M, 26 cores | | | 19 |
| host at W = 0.6 (demux + EM ported) | | | 43 |

So a GPU-native engine lands at **~20 M pairs/s per GPU host-bound today, ~40–45 M once the last callers move**, against 2.4 M
distinct/s ≈ 5 M pairs/s of GPU mapping now and ~3 M pairs/s per node end to end. Six H100s at 40 M/s finish the 20.6 T-pair
campaign in ~1 day of GPU time; the download tier (~1 GB/s) then needs ~18 days for the same data. **Past ~10–20 M pairs/s per
GPU the campaign is input-bound.** "World's fastest" is therefore settled by the pipeline, not by the kernel: the right
engineering target is a GPU engine that is *never* the bottleneck at the input rate, is deterministic, fits on A100-40, and
leaves the SMs free for the resolver (§5.5).

### 5.4 Staged migration (each step gated as in §2)

| step | what | gate | keeps exactness |
|---|---|---|---|
| S1 | persistent per-GPU service; pinned ring; streams; arena; 2-bit reads from `fq::RunSet`; no FASTQ text; no cache inserts for hand-off reads | identity test + gate (a); wall, core-s | yes |
| S2 | wide lanes warp-parallel over windows with two-pass `maxScoreMate`; persistent work queue | 0 verify mismatches; M5 ≥ 8 M distinct/s | yes |
| S3 | SoA/compact state in the narrow lane; 2-bit genome + SBK in the seed search | identity; M5 ≥ 12 M/s; A100-40 fit on human 3′ | yes |
| S4 | multi-sample batching + cross-sample result table | identity; same outputs for any batch composition | yes |
| S5 | demux fits and non-host EM on the GPU (bit-exact serial-order ports) | gate (a) | yes |
| S6 | chaining-DP stitcher and the sample-level resolver (§5.5) as an *alternative* aligner mode | M1–M7 on dev; win rule; test once | no (contract 2) |

### 5.5 Beyond STAR: where "better than STAR" can come from

The ground-truth census (`bench/ground-truth/SUMMARY.md`) is the most important input to any aligner redesign, and it says the
room is not in search but in **decision**: 83–91% of STAR's wrong unique placements score *higher* than the truth under
STAR's objective; 53% of wrong junctions have a truth block ≤ 15 nt; the true locus is already in the multimapper set for
2.7–3.1% of reads; HISAT2 gets half of STAR's wrong junctions right by accepting fewer junctions. A faster port of the same
objective "would place the same reads in the same wrong places."

The GPU-native engine is the natural home for the proposed resolver, because it is per-molecule and per-locus parallel and
sort-based: after K6 emits *all* candidates per distinct sequence (multimapper set plus unspliced/clipped alternatives of every
spliced candidate), a sample-level pass (a) tallies junction support with long anchors across cells, (b) groups reads by
(CB, UMI) and chooses placements jointly per molecule, (c) runs an EM over loci, and (d) outputs a calibrated posterior instead
of MAPQ. All four are radix-sort + segmented-reduce workloads that run at the GPU pileup's rate (~1 G reads/s). This is the
only path to satisfying the win rule, and it needs no new index. It should be designed in parallel with S1–S3 and scored on
dev data only, as the pre-registration requires.

---

## 6. Risks and open questions

- **Profiling is blind, and the cause is a version mismatch, not permissions.** Probe job 795199 on g050 (2026-10-31): a trivial
  kernel runs, but `ncu` 2024.1.1 from the CUDA 12.4 image fails with "Failed to prepare kernel for profiling / Unknown Error" while
  `RmProfilingAdminOnly` is unset; the node runs driver 610.43.02 with CUDA UMD 13.3. The fix is a CUDA 13.x devel image (pull job
  795202, re-probe 795203). `nsys` is not in the 12.4 image either. Nsight Compute aborting on Clipper's GPU nodes (counter permissions) and Nsight Systems is not in the
  image. Every kernel-level claim above, mine and W11's, rests on `clock64()` counters. Getting `CUPTI` permissions (or an
  image with `nsys`) is worth more than any single optimisation; it would settle the AoS-state question in an hour.
- **Nuclei (d = 0.66) and 5′ (3.4 alignments per record, adapter clipping)** are the cost outliers. The wide-lane fix helps
  5′ most; nuclei need the cross-sample table or simply more GPU time.
- **A100-40 for mouse** needs a decision on the dense index.
- **GPU failures today are hard failures in one path** (nuclei2k_v2 release-gate finding). The service design must make every
  GPU-side exception a logged CPU fallback at sample granularity.
- **Determinism across GPU models**: the port is integer except the host-replicated floats; a H100 and an A100 run must be
  byte-identical and should be a standing gate.

## 7. Literature and hardware notes

### 7.1 What other GPU aligners did, and what transfers (research pass; sources at the end)

| tool | decomposition | what it teaches here |
|---|---|---|
| Parabricks rna_fq2bam (STAR 2.7.2a) | undocumented; "Stage0/1/2" GPU queues | ~382 M × 98 nt reads in 7.5–7.8 min on 2× H100 ≈ **0.85 M reads/s**, *slower* on 4× H100; users report no speedup at 150 bp two-pass. Host-bound. singlet's 3.9 M reads/s of GPU mapping per H100 is already ~4–5× that per GPU; the lesson is Parabricks' failure mode, not its design |
| Parabricks fq2bam (BWA-MEM) | seeding on CPU until v4.8 | measured slower than BWA-MEM2 on 64 cores in 2023; CPU-side seeding is the trap singlet already avoided |
| BarraCUDA (2012) | thread per read, BWT DFS | one GPU ≈ 6 Xeon cores: thread-per-read on a branchy backtracking search gives ~nothing |
| SOAP3-dp (2013) | 64-B BWT chunks, reads interleaved by 32, **divergence predicted from SA-range counts, 3 classes, hardest to CPU** | the lane cascade singlet has, but decided *before* the kernel from the index, not after an overflow; worth adopting for the wide lanes |
| nvBowtie / NVBIO | Map → Select → Locate → Score → Reduce → Traceback over queues, parallelism *below* read level | the wavefront shape of §5.2; 12.8× over Bowtie2 on a K80 |
| Chacón FM-index (2015) | k-step FM, thread cooperation on wide counter blocks | ~2 G bases/s; FM is 1 base per dependent load, which is why STAR's SA + 14-mer table (≈ 10 steps per seed) already beats it |
| GPU-BWA-MEM (ICS 2023, A100) | **thread-per-read SMEM 14% warp efficiency → warp-per-read 49% (4.1×)**; sorting reads in 1–5 M batches +40%; chaining as sort + binary-search predecessor + thread-per-chain; bump allocator 30× over device malloc | end to end 3.8× BWA-MEM2 on 64 cores only after *every* stage moved; the per-stage shares shift (chaining 6% CPU → 17% GPU) |
| GASAL2 / SaLoBa / AGAThA | thread-per-pair DP vs subwarp-per-alignment wavefront with register-resident slices, length bucketing, subwarp rejoining | AGAThA: balancing alone 2.2×, 18.8× total; GASAL2's 20× extension gave 1.3× end to end (Amdahl) |
| mm2-ax / mm2-gb | chaining on GPU, segments by size class, persistent blocks longest-first | stuck at 33% active warps, >70% cycles stalled on dependencies with 98% L2 hits: *latency*, not bandwidth; cross-batch aggregation of long items raised occupancy 20% → 86% |
| BWA-MEM2 / ERT / LISA / BWA-MEME / Sapling | enumerated radix trees (60 GB), learned SA models (Sapling: 32 → ~10 probes, < 1% memory; MEME: 8.8× fewer memory accesses) | the SBK idea in §5.2 is Sapling's sampled form; a PWL model over STAR's SA buckets is the cheapest next step if seeding ever binds |
| p-ary / Eytzinger GPU search | a warp probes 32 pivots per step; Eytzinger layout up to 3.8× over naive binary search | for the wide-lane SA ranges (Nrep up to 10,000) and the bucket search; no paper batches sorted genomic queries against a GPU suffix array, so §5.2's sorted-wavefront search is unexplored territory |
| cuSBF / warpcore / Hive | warp-cooperative minimizer tables, ~85% SM utilisation out of cache | the non-host screen and any future minimizer locator |
| DRAGEN (FPGA) | fixed-length seed hash, **seed-extension trees precomputed for seeds above frequency 16**, one DRAM burst per lookup | the hardware-native answer to repeats: precompute, do not search. STAR's `winAnchorMultimapNmax 50` loci per anchor are the analogue; a per-index table of extension outcomes for the top repeat k-mers would remove most wide-lane work (rejected in W11 for exactness; allowed under contract 2 if M1–M7 hold) |
| GenAx / Darwin | automata/tiled DP co-processors, 24–125× | out of scope but confirm that filter-before-DP is where the wins are |

Corrections these lessons force on §4–§5:
- Lesson "never thread-per-read for seeding" is about FM backward search. M3 shows STAR's SA search at 70% of the random-access
  roofline with thread-per-read, because a seed costs ~10 steps, not ~100. Keep thread-per-query for the narrow case; use the
  warp-cooperative p-ary search only for large intervals (wide lanes, `findMultRange`).
- SOAP3-dp's pre-classification is better than singlet's overflow-then-retry: the number of windows a read will create is known
  after the seed stage (anchors with Nrep ≤ 50), so reads can be routed to the warp-per-read lane *before* lane A wastes a
  pass on them. Today 31% of reads run lane A to overflow first.
- AGAThA's uneven bucketing and subwarp rejoining are the right scheduling for the wide lane: sort by window count, give each
  warp one of the longest reads, let idle subwarps join.
- No GPU spliced short-read aligner has been published in 2024–2026; the only production one (Parabricks STAR) is host-bound
  and slower per GPU than this engine already is. The field is open.

Sources: Parabricks docs and forum threads (rna_fq2bam, 4.7.1/4.8 release notes), NVIDIA blog on Parabricks timings; ICS 2023
GPU-BWA-MEM (PMC10425913); BarraCUDA (PMC3278344); SOAP3-dp (PMC3669295); NVBIO/nvBowtie docs; Chacón et al. TCBB 2015; GASAL2
(PMC6815017); SaLoBa (arXiv 2301.09310); AGAThA (arXiv 2403.06478); mm2-ax (PMC10018915); mm2-gb (bioRxiv 2024.03.23.586366);
BWA-MEM2 (arXiv 1907.12931); ERT (bioRxiv 2020.03.23.003897); LISA (arXiv 1910.04728); BWA-MEME (bioRxiv 2021.09.01.457579);
Sapling (bioRxiv 2020.01.29.925768); Kaldewey p-ary search (HotPar 09); Eytzinger GPU index benchmark (arXiv 2506.01576);
cuSBF (arXiv 2606.24417); Hive (arXiv 2510.15095); DRAGEN hash-table docs and US11803554; GenAx (ISCA 2018); Darwin (bioRxiv 092171).


### 7.2 Hardware facts that bound the design (research pass, sources at the end)

| fact | H100 (SXM/NVL) | A100 (40/80) | consequence for the aligner |
|---|---|---|---|
| SMs; max resident warps/SM; registers/SM | 132; 64; 64K | 108; 64; 64K | 64 regs/thread → 32 warps/SM; 128 → 16; the lookup stages must stay ≤ 64 regs or occupancy halves |
| shared memory per SM / per block | 228 KB / 227 KB | 164 KB / 163 KB | a warp-per-read window set of 25 KB gives 8 (H100) or 6 (A100) warp-reads per SM |
| L2 | 50 MB enabled (60 on die), two partitions; far hit ≈ 1.6× near | 40 MB, two partitions | hot tables must be < ~20–25 MB to be "near" everywhere; SAi's top 10–12 levels fit, no more |
| DRAM latency, loaded pointer chase | ~330 ns idle; M1 measured 1.17 µs loaded on the NVL part | ~400 ns idle | Little: ~35–45k sectors in flight chip-wide; ~100–265 per SM |
| random 32-B sector rate | ~30–32 G/s (cuco GUPS 256 GB/s of 8 B; M1 28–29 G/s) | ~17 G/s (141 GB/s) | **H100/A100 ≈ 1.8×, not 2.2× (HBM) or 3× (FLOPS)**; plan A100 at 55% of H100 for seeds |
| per-thread outstanding loads | plain `ld.global` bandwidth peaks at ~2 outstanding loads per thread and falls ~35% by 8 | — | breadth (more warps) beats depth (unrolled chains): keep 2–4 queries per lane, not 8+ |
| TLB reach | no H100 measurement; A100 random access is flat below a ~64 GB window per warp | measured | the 17–25 GB index is safe; keep index + reads < 64 GB per warp's footprint; `cudaMalloc`, never UVM |
| divergence | linear tax: k distinct paths cost k×; no reconvergence until `__syncwarp` | same | sort reads/windows by expected cost before each stage; explicit masks after data-dependent loops |
| Hopper-only | clusters + DSMEM (181–213 cycles vs 258–744 from L2/DRAM), TMA, `cp.async.bulk`, DPX (≥ 4.75× Smith-Waterman) | `cp.async` only | DSMEM for a per-cluster hot slab; DPX only matters if a DP extension stage exists |
| launch costs | ~2 µs stream launch, ~1.3 µs in a graph | similar | irrelevant above ~50 µs per kernel; the current lanes are seconds, so graphs buy little except removing blocking D2H syncs |
| PCIe copies | pinned ~25–50 GB/s theoretical, 12.3 GB/s measured on Clipper's NVL node, pageable ~6 GB/s | — | index upload must be pinned or, better, done once per service lifetime |
| multi-process sharing | time-slicing serialises contexts (4 jobs 49.5 s vs 27.2 s under MPS) | same | the file-lock slots scheme loses ~2× to context switching; one service process per GPU, or MPS |
| sort | Onesweep 29 G keys/s (32-bit, A100); ~45–50 est. H100 | measured | seed/tuple sorts are ~1–2% of the budget |
| decompression | nvCOMP LZ4/Snappy 70–160 GB/s (A100); zstd "moderate", no H100 figure | measured | `.1fq` is zstd: decode on the CPU (0.085 core-s/M, 12–15 M pairs/s per thread) unless re-encoded as LZ4 |
| wavefront vs megakernel | wavefront wins when stages differ in registers/divergence/memory behaviour and per-item state is small; megakernels win for uniform short stages (LLM inference) | — | this workload is the wavefront case; a persistent kernel with per-stage queues gets both |

Two notes against the current code: (1) `k_p3` inherits the register footprint of the whole stitcher, so the cheap assignment
and encoding steps run at the stitcher's occupancy (the Laine et al. argument); (2) the A100 random-sector rate makes the
seed stage ~1.8× slower per GPU, so on A100 the design should keep the seed stage's share small (reuse, dedup, cross-sample
tables) rather than expect the H100 ratio.

Sources: Luo et al. 2025 (Dissecting Hopper, arXiv 2501.12084); Chips and Cheese H100 and GH200 articles; NVIDIA Hopper tuning
guide; cuCollections blog; arXiv 2608.15764 (H100 load concurrency); arXiv 2607.23402 (warp divergence Pascal→Blackwell); arXiv
2405.11425 (A100 64 GB random-access window); FlashAttention-3 (arXiv 2407.08608); Onesweep (arXiv 2206.01784); nvCOMP
algorithm selection; GDS design guide; Laine, Karras, Aila 2013; Hazy Research "No Bubbles"; ORNL GPU concurrency overview.

---

# Part II. North star: target state, development strategy, and the experiment program (2026-10-31, evening)

This part is the operating plan. Part I is the evidence it rests on. It is revised at the end of every wave.

## II.1 The target state

One persistent engine process per GPU. The CPU decodes `.1fq` into a pinned ring (measured 12-15 M pairs/s per thread),
orchestrates batches, and runs what is left of the callers and export; everything from unpack to the count matrices, pileup
tallies, non-host screen and, later, the demux fits and the resolver runs on the GPU as wavefront kernels over SoA queues,
sort-based and deterministic. Multi-sample batches by (species, read length, clip, chemistry). Fits A100-40 for human.

**Ceilings per H100 NVL (what the physics allows, so that every result can be scored against a bound):**

| stage | bound | basis | today |
|---|---|---|---|
| random 32-B sectors | 29 G/s | M1, flat to 74 GB | seed stage issues ~20 G/s (70%) |
| seeds, exact STAR search | ~110 M distinct reads/s (260 sectors per read at the roofline); 78-105 measured | M3 | 78-105 M/s |
| windows + assignment | > 300 M distinct/s (coalesced SA scans, ~50 sectors) | Part I 5.3 | inside M5 |
| stitch, narrow lane (<= 16 windows) | ~150 M distinct/s instruction-bound at 30% issue efficiency, ~30 k instr/read | Part I 5.1 | 6.4 M/s (lane A) |
| stitch, wide lane (> 16 windows, 31% of reads) | ~40 M distinct/s with 32-way window parallelism | Part I 5.3 | ~1 M/s effective (70% of M5 time) |
| **M5 combined, practical** | **20-30 M distinct/s** (~40-60 M pairs/s at d = 0.47) | harmonic sum, tails, overflow | **2.4 M/s** |
| counting, pileup, non-host | > 1 G reads/s | M6, exploration 12.13, M13 | done |
| PCIe (pinned, this node) | 12.3 GB/s ~ 200 M pairs/s at 56 B per pair | M8 | pageable, per sample |
| host decode | 12-15 M pairs/s per thread | M8 | FASTQ text round trip |
| end to end per GPU | host-bound: 19 M pairs/s at W = 1.4 core-s/M, 43 M at W = 0.6 | Part I 5.3 | ~3 M pairs/s per node |
| campaign input supply | ~1 GB/s download ~ 10-20 M pairs/s for 1-2 GPUs | exploration 12.10 | - |

**Definition of done for the mapping engine (this branch):** M5 >= 15 M distinct sequences/s on pbmc1k_v3 (6x today) at
100% identity to STAR on the 2 M-read dump set; lane device memory <= 10 GB at full rate (so A100-40 is feasible); batch
tail < 10% of lane time; GPU busy > 60% of the mapping phase; byte-identical outputs on H100 and on an sm_70 build run on
V100S (Clipper has no A100; V100S at 32 GB is the memory-fit proxy). Beyond that, the remaining headroom (to ~25 M/s)
costs more than it returns because the campaign is input-bound (last row above); effort then goes to the host path and
to contract 2.

## II.2 Development strategy: stepwise architecture changes against per-stage oracles, not a big bang

Why not "hammer everything at once":
1. **Exactness is only debuggable with a per-stage oracle.** The stardump blobs give a byte-level oracle for the mapping
   of every distinct read (m5 harness, minutes per run); the pileup gate gives one for counting. A rewritten engine has
   no oracle until it is complete, and the port is 1,500 lines of enumeration-order-sensitive logic: a single tie rule
   wrong is found in hours stepwise and in weeks big-bang.
2. **The model has already been wrong three times** (seeds at roofline, multimappers dominant, cache hits cheap). Each
   step must re-measure before the next is designed; a big bang bakes the current model in.
3. **Another agent is changing the host side concurrently.** A stepwise branch merges; a rewrite diverges.

Why not timid either: each step is an *architectural* change to one stage (decomposition, memory layout, scheduling),
gated in hours on the m5 harness by identity and speed. The rule: **never optimise the exact port and change the
algorithm in the same step.** "Hammer it" is reserved for the contract-2 engine (resolver + chaining-DP stitcher), which
is a different product with a different oracle (M1-M7 on dev data) and is prototyped in isolation once the exact engine
is near its ceiling.

Mechanics: waves of independent experiments on sub-branches `gpu/fable/<name>` off `gpu/fable-engine` (from
`gpu/gpu-slots2` cf92cdb9), each returning a number, a go/no-go against II.1, and a mergeable kernel if go. The
orchestrating session merges, re-measures the merged head, updates this file, and plans the next wave. Nothing lands in
`gpu/fable-engine` without 100% identity on the dump set and a speed number from the same job.

## II.3 What has to be known first (ranked by how much it changes the plan)

| # | question | why it decides the plan | experiment |
|---|---|---|---|
| Q1 | Where do lane A / M / B cycles go: stall reasons, sectors per request on the Work state, registers and occupancy, L2 hit, per-kernel time | decides whether SoA state, persistent scheduling or window parallelism comes first; the Part I 4 ranking rests on an untested hypothesis | E1: ncu on k_p1/k_p2/k_p3 per lane with the CUDA 13.3 image (ncu 2026.2 and nsys verified working on g050, job 795203) |
| Q2 | The distributions: windows per read, aligns per window, SA entries scanned, recursion steps, kept transcripts, overflow reasons, per lane and per read class | fixes the caps of compact state and the warp-window lane; predicts pre-classification gains | E2: census from the m5 harness on the 2 M dump |
| Q3 | Batch-tail share: lane time vs the sum of per-read cycles / concurrency | the gain of persistent work-queue kernels | E1 (nsys per-batch timeline + clk counters), E4 |
| Q4 | Occupancy vs reads in flight: NB sweep 16k..524k per lane; local vs global Work | whether the 40 GB Work buys anything; A100-40 feasibility; the AoS hypothesis | E1 (`--batch-a/m/b`, `--blocks-per-sm`, `--local`) |
| Q5 | Ordering of phase 3: prefix order vs first-anchor-locus order vs window-count bins | cheap locality gains; the earlier "no gain" result was void (stale binary) | E4 |
| Q6 | Window-parallel stitching: exact and how fast | the 70% item | E3 |
| Q7 | Steps per window distribution and how often one window dominates | whether the chaining DP (contract 2) is worth building; how heavy the exact overflow lane must be | E2 |
| Q8 | The host end state after the other agent's S8 work | sets the M5 rate actually needed (the node is host-bound until M5 < ~7 core-s/M equivalent) | read their results; do not duplicate |

## II.4 Wave 1 (parallel, one Opus session each, own sub-branch and worktree, m5 harness on Clipper)

| exp | branch | question | deliverable | gate |
|---|---|---|---|---|
| E1 profile | `gpu/fable/profile` | Q1, Q3, Q4 | `docs/gpu-engine/fable/E1-profile.md`: per-lane ncu table (time, registers, occupancy, stall mix, sectors/request for Work loads/stores, L2 hit), NB sweep curve, local-vs-global, tail share; ranked bottlenecks with numbers | numbers from real runs; no change to the port |
| E2 census | `gpu/fable/census` | Q2, Q7 | `E2-census.md` + JSON: histograms and cost correlations; recommended caps and a cost predictor from phase-1 state | identity unchanged (instrumentation only) |
| E3 warp-window | `gpu/fable/warp-window` | Q6 | lanes M/B stitched warp-per-read with windows across lanes and the two-pass maxScoreMate; `E3-warp-window.md` | 100% identity on the 2 M set; lane M+B time <= 0.5x baseline, target <= 0.3x |
| E4 order + queue | `gpu/fable/order` | Q5, Q3 | phase-3 ordering variants and a persistent work-queue variant; `E4-order.md` | identity; report every variant's time, keep the best |
| E5 compact | `gpu/fable/compact` | Q1 hypothesis | fewer and smaller transcript copies in the recursion (exclude branch by reference), 32-bit fields where exact; `E5-compact.md` | identity; lane A time <= 0.7x baseline |

Clipper rules for every experiment: `short-gpu` or `debug-gpu`, `--nice=10000`, at most two GPU jobs per experiment in
flight, `squeue -u $USER` before submitting, nothing but `sbatch`/`squeue`/`sacct`/log reads on the login node, read
sequences only in job `/tmp`, results (small JSON and logs) under `$W/jobs/fable/<exp>/`.

## II.5 Wave 2 candidates (chosen after wave 1's numbers)

- Pre-classification from the seed stage (SOAP3-dp): route reads to the narrow or warp lane before phase 3.
- 2-bit genome plus sampled SA bucket keys in the seed search (only if E1 shows the seed stage above ~15% of M5).
- Persistent service, pinned ring, streams and lane overlap (coordinate with the host-offload agent; shared files).
- Cross-sample result table.
- Contract-2 prototype: chaining-DP stitcher for single-window reads and the candidate audit (4a) on the ground-truth dev set.
- sm_70 build and a V100S run as the 32 GB memory-fit proxy for A100-40.

## II.6 Branch, scope and collaboration

- `gpu/fable-engine` (pushed) is this track's integration branch; sub-branches `gpu/fable/<exp>`; only the orchestrating
  session merges. The other agent's branches (`gpu/gpu-slots2`, `gpu/fg2-engine`, `gpu/host-offload`, `perf/*`) are not
  touched; their results are read, not re-derived.
- Scope of this track: `src/engine_gpu/`, `bench/gpu/m5/`, new `bench/fable/`, `docs/gpu-engine/fable/`. `src/star/`,
  `src/pipeline/`, `include/singlet/pileup/` only through the other agent.
- The m5 harness (`bench/gpu/m5/m5.cu`, `jobs/m5_compile.sbatch`, `jobs/m5_run.sbatch`, `jobs/m5_ncu.sbatch`) is the
  experiment vehicle: it regenerates the truth per job (decode, distinct, stardump, compare), compares byte for byte, and
  times lanes with `--batch-*`, `--budget-*`, `--bin`, `--local`, `--blocks-per-sm`. `src/engine_gpu/stitch.cuh` and
  `bench/gpu/m5/stitch.cuh` are identical today; experiments edit the `bench/gpu/m5` copy and the merge brings the engine
  copy along.
- Toolchain: `$W/images/cuda-13-devel.sif` (CUDA 13.3, ncu 2026.2, nsys 2026.2; verified on g050, job 795203) for all
  experiment builds; `cuda-12.4.1` for the production binary until the other agent moves. Compile with
  `-arch=sm_90 -fmad=false -lineinfo`.

## II.7 Wave 1 results (updated as reports land)

### E1 profile (done 2026-10-31 ~21:40; branch gpu/fable/profile 26f0525f, merged into gpu/fable-engine 518e4163; report docs/gpu-engine/fable/E1-profile.md)

Baseline on the 2 M set: **860.7 ms = 2.31 M distinct/s**; lanes A 268.6 / M 337.4 / B 254.7 ms; k_p1 26%, k_p2 4.7%, k_p3 69%; host gaps 0.15%. Identity 100% in every job.

| hypothesis (Part I 3-4) | verdict | evidence |
|---|---|---|
| per-thread AoS Work state dominates k_p3 | **partly, and not as a layout problem** | sectors per request 2.2-2.4 (loads) / 2.8-3.6 (stores) = ~1 per *active* thread, because only **1.94 of 32 lanes are active per instruction** (SIMT ~6%) in every lane's k_p3; Tr::copyFrom is 19-22% of stall samples and ~half of k_p3's L2 sectors; k_p3 writes 2.4-2.7x the sectors it reads |
| batch tails are large | **true** | 39% of k_p1+k_p3 warp-slot time is empty; 23% lost to finished warps held in 128-thread blocks, 19% to batch drain; the longest warp is 67-100% of every batch; lane B 62% under-filled; a persistent queue's floor is ~277 ms (32%) lower |
| occupancy is register-limited | true as limiter (80-95 regs -> 31-37% theoretical, 19-30% achieved), secondary as cost (5-15%) | ncu |
| lanes M/B are 70% because of divergence | **true share, wrong cause**: serial work per read (M 8.3x, B 27.5x lane A's instructions per read), memory-stalled (DRAM per read 26 KB A, 285 KB M, 777 KB B); instructions cost the same everywhere (~7-9 ns per 1k warp instructions) | ncu, nsys |
| the seed stage is small | **false**: ~17% of M5, and lanes M/B redo phase 1 for the 26% of reads lane A already seeded (13.7%); inside k_p1 seeds run at 21.9 M reads/s vs 78-105 M/s standalone (occupancy + divergence) | nsys, clk |
| lane A k_p3 is memory-bound | **false**: top stall is instruction fetch (51%, i-cache hit 54%); the byte-wise blob encoder alone is 9.4% of its stall samples | ncu |
| local (interleaved) Work is faster | 1.60x in mono mode, but with 4.3x the DRAM bytes and 25% fewer warp instructions: the gain is convergence and i-fetch, not traffic | ncu |

**What this changes in the north star.** The thread-per-read megakernel runs at ~2 active lanes per warp: the GPU
is executing STAR's serial control flow at ~17k-way scalar parallelism. The ceilings in II.1 stand, but the order of
levers changes: convergence first (warp-cooperative lanes for wide reads, signature-sorted warps and a persistent
queue for narrow reads), instruction count second (copies, encoder, register caps), memory layout last (not a lever at
this SIMT level). Immediate no-code gain: batch sizes (1M / 524k / 131k) -11.9%. E1's ranked projection: items
1-3 and 5-7 reach ~500-550 ms (3.6-4.0 M/s); 10 M/s needs window/entry parallelism inside a read (E3) or the
wavefront redesign.

## II.8 The whole path, disk to count matrices, with nothing temporary (the big picture the mapping work serves)

Zach's note (2026-10-31): never lose the end-to-end view: the `.1fq` on disk is read on the CPU, goes to the GPU, and the
entire alignment and pileup runs through to the output count matrices and tables. No temporary matrices, no BAM, no FASTQ
text, no cache inserts, no per-read host pass. The mapping kernels are one stage of this; a 6x faster M5 that still feeds
STAR's per-read pass changes nothing for the sample's wall time (GPU busy is 6-8% today).

**The path, per sample batch (4-16 M read pairs), with what exists and what is missing:**

| hop | bytes per pair | mechanism | exists | measured |
|---|---|---|---|---|
| 1. `.1fq` blocks from disk | ~20 B | CPU reads zstd blocks (local NVMe or NFS; M8: NFS no slower than cold NVMe) | yes (`fq::RunSet`, `.1fg2` staging by the other track) | 0.2 GB/s per GPU at 10 M pairs/s |
| 2. decode + pack on the CPU | 56 B out | zstd decode, 2-bit R2, CB+UMI key, quality bits, into a pinned ring of 4 x 256 MB | yes as a prototype (`bench/gpu/m8/m8_ingest.cu`) | 12-15 M pairs/s per thread, 0.085 core-s/M |
| 3. H2D | 56 B | `cudaMemcpyAsync` on a copy stream, events gate compute | prototype | 12.3 GB/s pinned on this node (> 200 M pairs/s) |
| 4. clip + dedup | ~190 B of device traffic | k_clip, hash, radix sort, byte compare, `seq_id` per read, prefix sort of representatives | yes (`engine_gpu.cu`) | 3.5 G reads/s (M6), 1.6 s per 66 M in the engine |
| 5. mapping of distinct sequences | 26 KB (A) to 777 KB (B) of device traffic per distinct read today | lanes A/M/B/host, STAR-exact | yes | 2.3 M distinct/s; target >= 15 M/s (wave 1-2) |
| 6. **records from the mapping result** | ~70 B per distinct sequence | the cache blob (transcripts) -> alignment records (tid, pos, strand, primary, MAPQ, CIGAR ops, NH, nM) = STAR's `alignBAM` single-end rules | **missing on the GPU**: done on the host by `ReadAlign::seqCacheAlignRecords` / `seqCacheLeanAln` once per distinct sequence (3.7 core-s/M on 5'), then written to the hand-off file and parsed again | E6 |
| 7. per-read expansion | 16 B per read (`ReadRec`: sig, barcode column, UMI) | gather of `seq_id`, barcode lookup/correction, UMI pack | partly on the host (`gpu_handoff_pass`, `HandoffReadEntry` per read written by STAR's threads) | E6 then wave 2 |
| 8. reducers | ~300 B of device traffic per pair | annotation of distinct alignments against the gene model (merge-join), expansion, radix sort (cell, feature, UMI), directional collapse, segmented reduce -> exon/intron/ambiguous CSR; junctions, SNP AD/DP (R1), chrM, indels; R9 roots; cell partitions by K | yes (`pileup_gpu.cu`, `gpu_pileup/`) | 0.8-1.5 G reads/s; identical on 7 of 8 release-gate datasets |
| 9. D2H of results | ~100 MB per sample | CSR matrices, tallies, per-cell stats | yes | negligible |
| 10. callers and export | - | cell calling, ambient, doublets, demux, writers (`.s1`, `.1pz`) | CPU | W = 1.4-2.2 core-s/M; demux auto-K and non-host EM are 74% of it |

What "no temporary matrices" means in practice: the reducers emit tuples into device buffers sized up front from the read
counts (partitioned by cell hash when the estimate exceeds the device budget), sort and reduce in place, and write the final
CSR once; there is no per-batch matrix that is merged later, because every reducer's key starts with the cell and the batch is
the sample (or a multi-sample batch with the sample id in the key). The only state that crosses batches of one sample is
the sorted tuple buffer itself.

**Where the host core-seconds go today on the GPU path (12.5-14.4 core-s/M, 3') and which hop removes them:** STAR's
per-read pass 3.8-4.1 (hops 6-7), prefill host work 3.2-3.9 (hops 2 and 6: FASTQ text parse, cache inserts, hand-off build),
FASTQ temp files 1.3 (hop 2), the pipe consumer and hand-off passes 1.7-2.1 (hops 6-7), export and callers 1.2-2.7 (hop 10),
non-host 0.5-0.9 (done on the GPU, host part is index load + EM). Hops 2, 6 and 7 are the mapping track's business; hop 10
is the host track's.

**E6 (started 2026-10-31 evening, branch gpu/fable/records):** the GPU record builder for hop 6 and the per-read expansion
for hop 7, validated byte for byte against the engine's own hand-off records for the same sequences, and timed. With E6 in
place the chain 4 -> 5 -> 6 -> 7 -> 8 runs on the device without STAR's per-read pass; wiring it into the binary is a
small change that the host-offload track can take.

### E2 census (done 2026-10-31 ~22:30; branch gpu/fable/census 51eeb2c1, merged into gpu/fable-engine 136c9997; report docs/gpu-engine/fable/E2-census.md; three samples: pbmc1k_v3, pbmc5p_v1, nuclei2k_v2)

Identity unchanged with the counters on (pbmc5p_v1: 9 of 1,850,654 reads exceed LREAD in every lane including the host lane, a pre-existing limit). Census build +4.5%.

| fact | pbmc1k_v3 (5' / nuclei in parentheses) |
|---|---|
| reads with > 16 windows | 25.6% of reads, **72.6% of cycles** (24 / 70%; 37.5 / 80%) |
| where lane A's overflow is detected | **97% in phase 1** (windows > 16); phase-3 overflows are 0.94% of the reads that pass phase 1, 79-93% of them predictable from phase-1 state |
| cycles wasted in lanes the read later overflows | 18.3% (20.6%; 17.3%); lanes M and B re-seed: 11% and 5% of their cycles |
| phase 1 share of all thread cycles | 32% (30%; 34%); seed search is 80% of k_p1; k_p1 runs at 18.7 M reads/s vs 78-105 M/s standalone |
| stitching / assignment share | 48% / 9.5% |
| "too short" reads (STAR rejects them) | 7% of reads, **19.5% of cycles** (23 / 54.5%; 23 / 47%) |
| windows per read p50/p90/p99/p99.9/max | 3 / 42 / 91 / 133 / 238 |
| aligns summed over windows | 4 / 52 / 135 / 250 / 1,180; steps per read 24 / 285 / 864 / 2,276 / 87k; steps per window 5 / 11 / 41 / 127 / 46k |
| per-window cost | 81% of windows have one align and a fixed ~5 steps; 1 align 0.91 M cycles vs 2 aligns 1.09 M: **finalization dominates, not recursion** |
| largest window dominates? | no: > 50% of a read's steps in only 11% of multi-window reads; overall the largest windows hold 16% of steps |
| router | ranking by nWall alone = a fitted tree (74.7% vs 74.6% of cycles in the top 30%); oracle 76.2% |
| simulated warp slot use, sorted by nWall then nEntries within a batch | lane A 14.9% -> 49.1% (steps), M 31% -> 69% |
| control-flow signatures | 6 / 29 / 71 bucketed signatures cover 50 / 80 / 95% of reads |
| caps at 99 / 99.9 / 99.99% | NW 91 / 133 / 173; NWA 8 / 50 / 50; NEX 3 / 4 / 4; NWTR 5 / 13 / 34; NKEEP 14 / 38 / 74; NBLOB 323 / 1,310 / 3,333; steps 864 / 2,276 / 8,818 |
| who costs | singletons are 80% of distinct reads and **87% of cycles**: cross-sample reuse buys little cycle time |

**What this changes.** (1) The cascade is the wrong control structure: run phase 1 once, route on windows > 16 before
phase 3 (exact, phase-1 detectable), never re-seed. (2) Sorting each batch by (nWall, nEntries) is exact and is the
cheapest convergence lever (3x upper bound in lane A). (3) Window-level parallelism is balanced, so E3's design holds;
a per-window DP is not the first lever. (4) The narrow lane's cost is the fixed per-window path (finalization, copies,
encoder), E5's target. (5) Seeds are a third of cycles because the per-read seed loop runs at 18.7 M reads/s inside
k_p1: a round-synchronous seed wavefront (every lane does one search per round) is a wave-2 experiment. (6) "Too short"
reads (half the cycles on 5') need either an exact early bound or a contract-2 reject gated by M1-M7.

### E5 compact (done 2026-10-31 ~23:00; branch gpu/fable/compact ac61b3e, merged into gpu/fable-engine b42662d2 with S3+TR32+ENC1 on by default; report docs/gpu-engine/fable/E5-compact.md)

Gate (lane A <= 0.7x) **not met**; best exact configuration lane A **0.847x** (248 -> 210 ms), lanes M 0.84x, B 0.86x; **M5 2.40 -> 2.83 M distinct/s (+18%)**; identity 100% in every run.

| variant | what | lane A | note |
|---|---|---|---|
| S3 | exclude branch of the recursion by reference, one finalize copy, no slot-0 init | 0.930x | copies 19.5 -> 9.4 per read, 4.8 -> 2.4 KB |
| TR32 | 32-bit transcript fields (Tr<8> 728 -> 440 B; Work A 46 -> 36 KB) | 0.938x | 0 of the stores fail the round-trip check |
| S3 + TR32 | | 0.90x | L2 write sectors -38%, DRAM bytes -1/3 (ncu) |
| ENC1 | the byte-wise encoder called from one site (loops not unrolled) | -12 ms | same instruction count: less code, fewer i-cache misses |
| BLOBW word-wise encoder | | +2 to +20 ms | registers 80 -> 94: rejected |
| local-memory Work in the staged kernels (p1local / p3local) | | 1.10-1.31x slower | DRAM bytes 2.8x, L1/L2 hit 80% -> 57%: **the AoS hypothesis is rejected as the cost**; at ~2 active lanes per instruction the per-thread struct keeps a thread's walk in a few sectors |

What this changes: layout work is closed. Code size in k_p3 is a lever (ENC1 wins without fewer instructions). Lane A is now
phase 1 (99.5 of 210 ms): the seed stage. E5 also suggests a 4-exon transcript (99.9% cap) at ~290 B.

### E6 records (done 2026-10-31 ~23:30; branch gpu/fable/records 5223bec4, merged into gpu/fable-engine; report docs/gpu-engine/fable/E6-records.md)

Hop 6 and 7 of II.8 exist on the device now: `src/engine_gpu/records.cuh` (blob decode = inverse of encodeTr, mappedFilter,
seqCacheLeanAln single-end, the hand-off byte layout, prepend_handoff's arrays, the per-read rule of gpu_handoff_pass_entries;
one source for host and device) and `records_gpu.cu` (buildRecords: count -> scan -> fill; distinctBarcodes; expandReads).
Nothing is wired into the default path; `SINGLET_ENGINE_GPU_DUMP` (off by default) dumps mapDistinct's output for the test.

| check (whole pbmc1k_v3: 66,601,887 reads, 31,636,557 records; job 795356) | result |
|---|---|
| hand-off records vs the engine's file | 31,636,557 / 31,636,557 byte-identical (2.25 GB block and offsets) |
| SigRec / AlnRec / CIGAR ops / nt16 sequence vs `Collector::prepend_handoff` | all identical (31.6 M / 32.0 M / 42.0 M / 1.41 GB) |
| ReadRec with the engine's R11 rule | 61,294,166 / 61,294,166 (= the engine's own count); exact-CR mode 60,060,876 identical; masks and counters identical |
| mate-gap transcripts (host alignBAM fallback) | 0 on this sample |

| step | host today | GPU |
|---|---|---|
| records for 31.6 M sequences | 5 s on 18 threads (STAR hand-off loop); 3.6 s host port | **37 ms** (851 M records/s); 65 ms with the hand-off bytes too |
| per-read expansion, 66.6 M reads | 4 s | **10.5 ms** (+5.7 ms distinct CRs) |
| hand-off file + parse | 2.25 GB, 2.8 s serial | none |
| moving data if the blobs leave the device | - | blobs+sequences H2D 5.0 GB: 0.41 s pinned / 1.7 s pageable; arrays D2H 2.6 GB: 0.18 s pinned |

Lessons: the R11 histogram must cover every read (a histogram from the hand-off entries alone changes 76 posterior decisions);
exact-CR vs R11 differs by 2.0% of kept reads; writing the hand-off bytes costs more than the arrays (45 vs 17 ms) and the pileup
needs only the arrays; moving the blobs off the device costs 11-46x the kernels, so the records must be built before mapDistinct
releases the device. Wiring plan (E6's): build records inside mapDistinct from device-resident blobs (upload only the host-lane
records), a device-pointer overload of pileupRun (first step: a pinned copy into PileupInput, 0.18 s), per-read entries and
masks from the decoded R1/R2 and clip5/clip3 on the GPU, distinctBarcodes -> host R11 table -> expandReads; STAR's summaries,
Solo and the reads with N stay with the host track. This is wave-2 item E8 (end to end on the device, counts compared with the
CPU engine under gate (a)).

## II.9 Wave 2 (started 2026-10-31 ~23:45 from gpu/fable-engine 84a16500)

| exp | branch | question | gate |
|---|---|---|---|
| E7 seeds | `gpu/fable/seeds` | flatten STAR's seed-loop nest into one warp-uniform "one search per lane per iteration" loop (and the window creation likewise), so k_p1 converges; optional 2-bit genome compare; phase 1 once with lane M's capacity | identity; k_p1 lane A <= 0.5x, target 0.3x |
| E8 e2e | `gpu/fable/e2e` | wire E6 into the engine: records from device-resident blobs inside mapDistinct, CR/UR and R11 on the prefill side, device arrays into pileupRun, no hand-off file, behind SINGLET_ENGINE_GPU_E2E=1 | gate (a) byte-identical on pbmc1k_v3; default path unchanged; host core-s per M reads reported |
| (pending E3, E4) | | warp-window lanes; ordering, routing before phase 3, persistent queue | |

### Integration gate of gpu/fable-engine (2026-10-31 23:47; compile 795736, run 795737 on debug-gpu; bench/fable/engine/fable_{cc,gate}.sbatch)

All four builds compile (default, -DM5_TAIL, -DM5_CENSUS, -DM5_BASELINE). On the 2 M set, same job, median of 3, standard batches:

| build | lane A (phase 1 / scan / phase 3) | lane M | lane B | M5 | identity |
|---|---|---|---|---|---|
| baseline (-DM5_BASELINE) | 249.5 ms (99.5 / 14.7 / 135) | 320.4 | 253.5 | **2.42 M distinct/s** | 1,991,926 / 1,991,926 |
| merged default (E5 on) | 212.6 ms (99.4 / 14.7 / 98.4) | 268.2 | 222.7 | **2.83 M distinct/s** | 1,991,926 / 1,991,926 |

Work per read: A 46.1 -> 36.0 KB, M 82.6 -> 67.9 KB, B 635 -> 549 KB. Phase 1 (99 ms) is untouched and is now 47% of lane A:
E7's target. Phase 3 of lanes M and B (171 and 178 ms) is E3's.

### E4 order (done 2026-11-01 ~00:10; branch gpu/fable/order bf874649; report docs/gpu-engine/fable/E4-order.md; 81 variants, all 100% identical)

| variant (fixed kernel, job 795646; v0 = engine order and batches, 870 ms) | M5 | verdict |
|---|---|---|
| `--bin` cost order (E2's predicted 3x) | **0.76x** | active lanes 1.94 -> 3.39 and 43% fewer warp instructions, but the launch takes 2.2x longer: warps made only of heavy reads become the critical path while the rest of the GPU idles (issue slots busy 3%) |
| signature order | 0.94x | same mechanism |
| genome-locality order (V2) | 0.87-0.90x | |
| persistent queue (V3), fused kernel (V4/V5) | -2%; 1.6-1.85x slower | no batch tail, fill 0.98, but **a batch lasts as long as its longest single read**, which starts at t = 0 and runs ~2 lanes wide; handing out whole reads cannot shorten it |
| one warp per block, E2 `--compact` | 1.00x, 0.99x | |
| window-count-descending on lane M only (`--best-order`) | **1.05x** | the one exact ordering win |
| 64-register cap on lane A | 1.02x | |
| batches 512k / 384k / 128k | 1.14x | memory: device peak 64.6 -> 80.6 GB |
| **all three (`--e4-best`)** | **1.22x** (2.81 M/s vs 2.29 in the same job) | adopt: order + register cap + batches |
| 40 GB device fit | costs 1.7-2x with fixed small batches or the fused lanes | A100-40 needs intra-read parallelism or lane overlap first |

Two findings with consequences beyond ordering:
1. **The critical path of phase 3 is one read.** Phase-3 busy/duration is 0.57 (A), 0.70 (M), 0.35 (B) and the longest warp
   sets each batch; the idle-slot bound (~1.46x) is unreachable by scheduling whole reads. The only levers are inside a read
   (E3: windows across lanes; later, aligns across lanes) and overlapping lane B with the next chunk on a second stream
   (bound ~139 ms = 16%).
2. **Instrumentation costs 18% through registers.** The `--tail` timing code compiled into k_p3 raised registers 80 -> 94
   and M5 time by 18% even when off. The production stitch.cuh compiles `clock64()` phase counters (M5CLK / M5T / M5ADD,
   clk[8]) into every kernel unconditionally: an immediate experiment is a `-DM5_NOCLK` build (E4 follow-up).

Expected engine gain from ordering + queue: ~1.03x (not the 1.2-1.5x Part I assumed); from batches ~1.14x at 81 GB.

### E4 follow-up (2026-11-01 00:40; gpu/fable/order 83f942ce merged into gpu/fable-engine; job 795837 on the merged code, @STD batches)

| configuration (merged code, counters on unless stated) | A / M / B ms | M5 | vs head |
|---|---|---|---|
| merged default (E5 on) | 212.7 / 268.4 / 223.2 = 704 | 2.83 M/s | 1.00x |
| -DM5_NOCLK (no clock64 counters) | 212.2 / 305.8 / 217.0 = 735 | 2.71 | **0.96x**: lane M's stitching 22% slower; registers identical (79), so the volatile clock reads change ptxas scheduling, not pressure. Opt-in, off |
| **default + --best-order --p3-maxreg-a 64 --batch-b 131072** | 200.1 / 235.8 / 167.5 = **603** | **3.30 M/s** | **1.17x** (lane A -6%, M -12%, B -25%) |
| NOCLK + --e4-best (1M/512k/128k batches) | 188.9 / 292.2 / 160.8 = 642 | 3.10 | 1.10x; +28 GB of Work |

Registers after E5: k_p3 79 in every lane (lane B fits 6 blocks per SM instead of 5); k_p1 95; the 64-register cap on lane A
spills 100/140 B and still wins 6%. Cumulative on the 2 M set: baseline 2.42 -> 3.30 M distinct/s (1.36x), all exact, with
device peak ~65 GB. The gate (bench/fable/engine/fable_gate.sbatch) now runs this configuration as its third variant.

### E8 e2e (done 2026-11-01 ~01:00; branch gpu/fable/e2e dcb1a2b6, merged into gpu/fable-engine 7ed4e9f0; report docs/gpu-engine/fable/E8-e2e.md)

With `SINGLET_ENGINE_GPU_E2E=1`: records are built from device-resident blobs inside mapDistinct, the per-read entries go
through a memfd instead of `.reads` files, there is no hand-off file, and the device-gathered records equal the host-assembled
ones for all 31.6 M (job 796036). **Gate (a) on pbmc1k_v3 (job 796010): 60 files identical** (every count matrix, STAR Solo raw
and filtered matrices, SJ.out.tab, cell_calls, cell_qc); the 7 differences are the known run-dependent ones and are present
without E8 too. Default path unchanged.

| run (T = 8, one H100) | wall | core-s |
|---|---|---|
| CPU engine | 489 s | 2,578 |
| `--engine gpu` + GPU pileup | 280 s | 945 |
| + E8 | **278 s** | 905 |

Device steps: gather 180 ms, buildRecords 44 ms, per-read pass kernels 27 ms, pileup GPU 2.2 s; but the records still cross the
host twice (device -> memfd in STAR's child -> device again -> PileupInput) and the prefill now holds the device 39.5 s instead
of 24.8 s (the host lane and record assembly run before release: a throughput loss when samples share a GPU).

**The finding that matters:** the wall time is now STAR's host machinery, not anything the GPU does: index load, cache fill and
the per-read stream 167 s, the outputs phase 58 s, of 278 s. STAR's per-read loop still runs over all 66 M reads because the
pipeline consumes what it produces: the per-read entries (CR/UR, quality bits), `Log.final.out`, `SJ.out.tab`, the unmapped
reads for the non-host screen, and **STAR Solo's outputs** (`Solo.out/Gene/filtered/barcodes.tsv` is the pipeline's preferred
cell caller, `export.h:598`; `CellReads.stats` feeds `cell_qc_metrics.tsv`, `export.h:1005`). Everything else (the count
matrices, pileup tallies, callers) no longer needs it. So the end state needs either an exact device port of STARsolo's
Gene counting (1MM_All collapse, Solo's gene rules), CellReads.stats and the EmptyDrops_CR filtering, or a product decision
to stop depending on Solo's caller. The former keeps every output identical and is a well-defined port with a byte-level
oracle (Solo.out); it is wave 2b.

## II.10 Wave 2b (started 2026-11-01 ~01:10 from gpu/fable-engine 7ed4e9f0)

| exp | branch | question | gate |
|---|---|---|---|
| E9 solo | `gpu/fable/solo` | STARsolo Gene counting, CellReads.stats, Barcodes.stats and EmptyDrops_CR filtering on the device from the records and per-read entries, as a standalone library + driver | `Solo.out/Gene/{raw,filtered}/*`, `CellReads.stats`, `Barcodes.stats` byte-identical on pbmc1k_v3 |
| E10 hostloop | `gpu/fable/hostloop` | profile STAR's remaining per-read loop at T = 18 on the E8 path; E8 step 2 (CR/UR, quality masks, R11 histogram from R1 on the device); device-side `Log.final.out` counters, `SJ.out.tab`, the unmapped-read list; a switch that skips STAR's loop for reads with records once nothing needs it | gate (a) + identical side files; wall and core-s per M reads |

### E7 seeds (done 2026-11-01 ~01:30; branch gpu/fable/seeds 8c4f0b21; report docs/gpu-engine/fable/E7-seeds.md; 55 variant rows, all 100% identical)

Gate (phase 1 <= 0.5x) **not met**; best exact variant `flatqs8b` = phase 1 **0.79x** in every lane, M5 **+7.1%** (2.82 -> 3.02 M/s
on the E5 head): a persistent seed kernel (one maxMappableLength2strands call per lane per iteration, warp-aggregated refill,
iterator in shared memory, 64 registers with no spills -> 8 blocks per SM), windows thread-per-read in A/M and persistent in B.

| finding | number |
|---|---|
| warps do **not** converge when the call sequence is made uniform | active threads per instruction 3.38 (flat) vs 3.77 (baseline): every level of the search (calls, iDist passes, SAi probes, binary-search steps, findMultRange, compare lengths) has a data-dependent trip count |
| where the gain comes from | persistent refill (no batch tail) and occupancy: 27% -> 48% achieved; IPC 0.95 -> 1.44; long-scoreboard stalls 68-78% (latency-bound) |
| register count is the dominant knob | 72 -> 80 registers costs 9 ms in lane A; the 2-bit genome compare needs 188 registers inline or spills 1.1-1.5 KB and loses |
| stepping one memory operation per lane per iteration (SEEDSTEP) | 1.16-6.5x slower, still 2.2-3.9 active lanes |
| splitting the windows kernel | -21 ms in lane A; lane B only wins when persistent (32 vs 45 ms) |
| running phase 1 once with lane M's capacity | <= 37 ms with the old kernels, <= 14 ms with the flat ones (upper bound) |

**What this changes.** Thread-per-query SA search is at its practical ceiling (~0.8x of today) on the latency side: the only
route to 0.5x is a **cooperative search** (a warp or 8-lane group per call: k-ary search over the SA interval with coalesced
pivot loads and a group-parallel compare, finishing per lane when intervals are small; Part I 7.1's p-ary/Eytzinger evidence),
which needs its own exactness argument for STAR's path-dependent intermediates (the search returns the same interval, but
findMultRange's L1a/L1b bookkeeping must be reproduced or proven equivalent). Wave 3 candidate.

### E3 warp-window (done 2026-11-01 ~01:45; branch gpu/fable/warp-window ebc3f28a, merged with 84a16500; report docs/gpu-engine/fable/E3-warp-window.md; every row 100% identical)

Gate (lanes M+B <= 0.5x) **not met**: on the merged code (E5 on) the warp lanes give M+B **0.90x** (482.8 -> 434.7 ms), M5
2.91 -> 3.13 M/s (+7.6%); with big batches 3.25 -> 3.31. Before E5's compact state the warp kernel only tied the baseline
(3.2x more thread instructions, 124 registers, lane contexts in local memory dropping L1 hits from 78% to 58%); E5's 32-bit
rows shrank the contexts 38% and cut its stitching 210 -> 117 ms. Best per lane: lane M with a separate assignment kernel
(228.7 vs 259.9 ms), lane B with in-warp ballot assignment (189.1 vs 222.8 ms); together ~0.865x. ncu on lane M: active
lanes 2.20 -> 5.37, warp instructions 20.4 -> 19.8 G, achieved occupancy 30% -> 24%, duration 122 -> 112 ms. The two-pass
maxScoreMate argument held on the data (0 disagreements in the host emulation).

## II.11 Reassessment after waves 1-2 (2026-11-01 02:00): the exact port is near its ceiling; the wall time is elsewhere

Measured, all exact, on the 2 M set per H100: E5 +17%, E4 flags +17%, E7 +7%, E3 +8%, big batches +14% (at 81 GB). They are
only partly additive; the merged head with E4's flags is at 3.30 M distinct/s, and E3 + E7 on top should land near
**3.8-4.2 M distinct/s** (~8-9 M read pairs/s at d = 0.47). Every lever the plan named has now been measured:

| lever | hoped | measured | why |
|---|---|---|---|
| per-thread state layout (SoA/local) | 1.5-2.5x | **loss** | at 2 active lanes per instruction the AoS struct keeps a thread's walk in a few sectors |
| warps re-converged by sorting / signatures / queues | 2-3x | **0.76-1.05x** | a batch lasts as long as its longest single read; packing heavy reads together makes them the critical path |
| windows across lanes (E3) | 2-3x on the wide lanes | 1.11x | more instructions per read, lower occupancy; per-window fixed cost dominates |
| seed loop flattened / lockstepped (E7) | 2x on phase 1 | 1.27x | every level of the search has a data-dependent trip count; latency-bound; registers are the knob |
| fewer copies, smaller fields, code size (E5) | 1.4x | 1.18x | real, and the thing that made E3 pay at all |
| ordering + queue (E4) | 1.2-1.5x | 1.03x; batches 1.14x | |

**Conclusion.** STAR's exact algorithm has a per-read serial critical path with data-dependent control flow at every
level; on a GPU it runs at ~2-5 active lanes per warp whatever the decomposition, and the measured levers compound to
roughly 1.6-1.8x over the W11 port. The II.1 "practical ceiling" of 20-30 M distinct/s was wrong for the exact port; the
evidence puts it at **~4-5 M distinct/s per H100** (another ~1.3x may come from lane overlap on streams and phase-1 reuse).
That is ~8-10 M read pairs/s of mapping per GPU, which is already above the node's host bound and near the campaign's
input bound (II.1). Two things follow:
1. **The mapping kernel is not the sample's bottleneck and will not be.** E8 measured 278 s of wall per 66 M-read sample
   with ~17 s of GPU; the remaining time is STAR's per-read loop and Solo (167 s) and the outputs phase (58 s). Waves 2b
   (E9, E10) are the critical path to "everything on the GPU"; after them, the persistent per-GPU service (index resident,
   lane overlap, device released during host work) and the per-sample fixed costs.
2. **Beyond ~4-5 M distinct/s the gains are algorithmic, i.e. contract 2** (no regression on M1-M7, scored on the frozen
   dev harness): (a) multimapper reads with > 16 windows are 26% of reads and 73% of cycles: prune windows that cannot
   reach the score range (or cap windows per read below STAR's 10,000 and report the multimapper set differently), a
   change that only affects reads STAR reports as multimappers (NH > 1, mostly not counted); (b) "too short" reads STAR
   rejects are 7-23% of reads and 20-55% of cycles: an early bound from the seed pass; (c) the resolver (Part I 5.5),
   which is the only route to being better than STAR, not just equal. These need the ground-truth harness runs, not the
   m5 identity harness, and a decision that the alignment contract may change for those read classes.

### E7 follow-up (2026-11-01 02:30; gpu/fable/seeds cce8514e merged into gpu/fable-engine; gate 796082 on the merged code)

| build @STD, E4 flags where marked | A / M / B ms | M5 |
|---|---|---|
| -DM5_BASELINE | 248.4 / 324.8 / 253.3 | 2.41 |
| -DM5_P1NEST (old phase 1, E5 on) | 211.7 / 274.0 / 222.2 | 2.81 |
| default (E5 + flatqs8b) | 190.3 / 255.0 / 214.1 | 3.02 |
| P1NEST + E4 flags | 201 / 237 / 167 | 3.29 |
| **default + E4 flags** | **181 / 224 / 164 = 570 ms** | **3.50 M distinct/s** |

Phase 1 with E4's flags: lane A 78.8 vs 98.7 ms, M 62.5 vs 74.8, B 28.8 vs 32.1. Correction from E7: the strided compare
(M5_CMPU, 0.92x on the old phase 1, exact) was never active in the flat builds; it is an untested next step. Cumulative on the
2 M set: 2.42 -> 3.50 M distinct/s (1.45x), all exact. E3's warp lanes (~0.865x on M+B) come next on top of this head.

### E9 solo (done 2026-11-01 ~03:00; branch gpu/fable/solo 5aab9e37, merged into gpu/fable-engine; report docs/gpu-engine/fable/E9-solo.md)

`src/engine_gpu/solo.cuh` + `solo_gpu.cu` (one source, device and OpenMP host): Solo's gene annotation from the alignment
records, CB correction (whitelist, UMI check, the float posterior with a host pow10 table), the per-read rules and keys,
(CB, gene, UMI) sort, 1MM_All as Hamming-1 connected components, CellReads.stats / Barcodes.stats / Features.stats, the
CellRanger2.2 filtering, and writers that reproduce STAR's bytes.

| check (pbmc1k_v3, truth job 796074, run 796095) | result |
|---|---|
| Barcodes.stats, Features.stats, raw and filtered barcodes/features/matrix.mtx, UMIperCellSorted, CellReads.stats | **10 of 10 byte-identical**, GPU and host paths |
| per-read annotation (nTr, gene set, gene, mito, exonic) on the 66.44 M reads with a record; cbMatch and whitelist match on all 66.6 M | 0 differences |
| Summary.csv | not ported (nothing in the pipeline reads it; the E8 gate hashes it) |

| stage | GPU | host (18 thr) | STAR post-map (T = 8) |
|---|---|---|---|
| annotation / CB / rules and keys / sort+unique / union-find / compaction / download | 34 / 15 / 16 / 8 / 1.3 / 0.5 / 35-63 ms | 1.2 / 4.0 / 2.4 / 0.7 / 0.08 s | 15-16 s (plus the per-read Solo work inside STAR's mapping threads) |
| **compute total** | **~130 ms** | ~8.5 s | |
| filtering + 10 files | 0.77 s | 0.75 s | ~2 s |

Lessons: STAR's CellReads.stats row order is run-dependent (thread/chunk order); the pipeline reads it by barcode, so sorted
rows are safe, and any gate should hash sorted rows. Wiring: run `soloRun` on the device right after E8's records (no upload),
CR/UR from E8's pass, the transcript model (~20 MB) and whitelist (30 MB) uploaded once per index, host-lane and N-read
annotation handed over by STAR (8 B per read); then STAR skips getCBandUMI, soloRead->record, the Solo temp files and the
post-map step. Limits: CB_UMI_Simple / Gene / Unique / 1MM_All / CellRanger2.2; whitelist <= 2^22, genes <= 2^17, UMI <= 12.
With E9 and E10 together nothing of STAR's per-read loop is needed for reads with a record.

### E3 follow-up (2026-11-01 03:20; gpu/fable/warp-window d6230c5e merged into gpu/fable-engine; gate 796113 on the merged code)

| build @STD | A / M / B ms | GPU ms | M5 |
|---|---|---|---|
| -DM5_BASELINE | 233 / 306 / 253 | 791 | 2.52 |
| default (E5 + E7) | 186 / 243 / 220 | 649 | 3.07 |
| + E4 flags | 178 / 222 / 163 | 563 | 3.54 |
| + E4 flags + warp lanes M and B (in-kernel assignment on B) | 178 / 214 / 162 | 554 | 3.59 |
| **+ E4 flags + warp lane M only (`--warp-m --wsort --wthreads 128`)** | **177 / 214 / 156 = 548** | | **3.64 M distinct/s** |

Lane M's phase 3: 142.7 -> 134.3 ms (k_cost + sort + k_p3a 16.8 ms, k_p3w 117.5 ms); lane B's warp kernel loses to the plain
kernel once lane B runs as one 131,072 batch (127.5 vs 121.6 ms), and 351 reads that the warp merge keeps in lane M make
lane B 6.7 ms faster (both routings exact). Registers: k_p3w 95 (lane M stack 8.9 KB), k_p3a 64, k_p3 79-80.
**Cumulative on the 2 M set: 2.42 -> 3.64 M distinct/s (1.50x), every row identical; device peak ~65 GB.** The gate's fourth
variant is this configuration. Remaining exact levers, each measured or bounded: lane overlap on a second stream (~16%),
the strided compare in the flat seed kernel (~5-8%), phase-1 reuse for the wide lanes (<= 14 ms), bigger A/M batches (+5-10%
at +28 GB). Together they bound the exact port at ~4.3-4.6 M distinct/s on this hardware, as II.11 says.

### E10 hostloop (done 2026-11-01 ~03:40; branch gpu/fable/hostloop e973e6e0, merged into gpu/fable-engine; report docs/gpu-engine/fable/E10-hostloop.md)

**Profile of the E8 path at T = 18 (job 796069):** before the loop: CR4 probe + STAR start 32 s (one core), index load 9 s,
prefill reads 15 s, prefill GPU 37 s (index upload 3.2, dedup 3.6, mapping 15.2, host lane 2.5, records 1), summaries +
cache inserts 6 s. The loop: 32.7 s, **bound by chunkRead under the input mutex (28.5 thread-s serial; 231 thread-s waiting)**;
inside: readLoad 52, map 25 (cache restores + 157 k N reads), summary replay 11, outBarcode 31, outSolo 26, outEntry 25
thread-s. After the loop: Solo counting + exit 16 s, hand-off pass 6, GPU pileup 16 (input build 5.2, pileupRun 9.0), non-host 20.

| level (SINGLET_ENGINE_GPU_E2E=) | what moves to the device | wall s | core-s | identity |
|---|---|---|---|---|
| cpu | - | 432.8 | 2,617 | reference |
| gpuD (GPU pileup) | | 255.2 | 991 | 60 files |
| E8 | records | 250-255 | 943-951 | 60 files |
| 2 | per-read entries, quality masks, R11 histogram (0 of 66.44 M differ; histogram byte-identical) | 258 | 969 | 60 files; no wall gain: the loop is input-bound |
| 3 | + Log.final.out stats, SJ.out.tab (126,996 junctions), Unmapped.out, nonhost reads from the records | 254 | 941 | all identical |
| **4** | **+ STAR's loop skipped for reads with a record** (only 157 k N reads and the host lane remain) | **214.7 (-14%)** | **676 (-28%)** | 51 files; only the 7 Solo files and the two QC files that read Solo's stats differ |

Default paths unchanged (job 796116). Device kernels: entries 20-24 ms, R11 2.8 ms, products 53-56 ms; uploads ~1.2 s.
Costs: the E10 decode adds 3-5 s / 30 core-s to the prefill (serial zero-fill, R1 parse); peak RSS 43.6 -> 61 GB.

**What this changes.** With E9's Solo library wired at level 4 (E11, next), every output of the sample is produced with
STAR's loop skipped; the sample is then ~205 s at T = 18, of which the GPU holds ~38 s. The remaining pieces are per-sample
fixed costs of the host process model: the CR4 probe (32 s on one core), index load and upload (12 s), reading and decoding
the reads (15 s + the E10 decode), Solo/exit (9 s), the pileup input build (5 s), non-host (20 s, index load each sample), and
the export/callers tail. These are the persistent-service items of II.1 and II.5, not kernel work.

### Integration gate of the fully merged head (2026-11-01 ~04:10; gpu/fable-engine 1d1bbac4 + E10 = 4439fc8b; compile 796129, run 796130)

| build @STD | A / M / B ms | M5 | identity |
|---|---|---|---|
| -DM5_BASELINE | 231 / 307 / 254 | 2.51 | 1,991,926 / 1,991,926 |
| default (E5 + E7) | 185 / 246 / 218 | 3.07 | identical |
| + E4 flags | 177 / 223 / 163 | 3.54 | identical |
| **+ E4 flags + warp lane M** | **177 / 214 / 156** | **3.64 M distinct/s** | identical |

The merged head reproduces every experiment's number; nothing was lost in the merges. State of the track at this point:
mapping 1.50x exact; records, Solo, per-read products and the loop skip on the device (E6, E8, E9, E10); E11 wiring Solo at
level 4 is the last step to a sample whose every output comes out identical with STAR's per-read loop gone.

### E11 solowire (done 2026-11-01 ~04:40; branch gpu/fable/solowire 8e54be40, fast-forwarded into gpu/fable-engine; report docs/gpu-engine/fable/E11-solowire.md)

`SINGLET_ENGINE_GPU_E2E=5`: level 4 plus STARsolo's Gene counting on the device after STAR's loop (E9's library on inputs kept on
the device since the prefill: CR/UR of every read, the records' Gene annotation, the read -> record map, 1.59 GB), and
`star_Solo.out/` written from it; STAR's Solo step does not run. STAR's loop sees only the 156,931 reads without a record (0.24%).

| gate (a), pbmc1k_v3, T = 18 (jobs 796377, 796433) | result |
|---|---|
| count matrices, tables, callers, `star_SJ.out.tab`, `Log.final.out` statistics | identical (60 files, the same 60 as E8) |
| the 10 STAR Solo files, incl. `Gene/Summary.csv` (new port, identical on its first run) | **10 of 10 identical** |
| `Gene/CellReads.stats` | same 323,542 rows; the device's file is sorted by whitelist index, so it is deterministic |
| `cell_qc_metrics.tsv`, `read_qc_summary.json` (lost at level 4) | identical |
| the 7 run-dependent files | the same 7 that differ between the CPU engine and gpuD |

| run | wall s | core-s |
|---|---|---|
| CPU engine | 424.5 | 2,607 |
| gpuD | 253.4 | 997 |
| level 4 | 209.5 | 672 |
| **level 5** (two runs) | **210.8 / 210.3** | **680 / 678** |

Device: keep 41 ms in the prefill; after the loop `soloRunKept` 0.15 s (download 89 ms of it), filtering and 11 files 0.65 s on
the host, against STAR's Solo step of 43 s (CPU engine) / 13-14 s (GPU paths) plus 57 thread-s of per-read Solo work inside the loop.
Two bugs the gate caught: STAR never stores "Unique" in `MultiMappers::types` (the guard ran level 5 as level 3 once, which at least
proved the fallback identical); false sharing in per-thread Q30 counters (+620 core-s, fixed with thread-local counters).
**Limits:** level 5 runs only for CB_UMI_Simple / Gene / Unique / 1MM_multi_Nbase_pseudocounts / 1MM_All / CellRanger2.2, and only
when every CB quality gives the same `qin` (pbmc1k_v3's constant 'F'); binned NovaSeq qualities fall back to level 3. So level 5 is
not yet general: E15 below.

## II.12 The sample at level 5: where 211 s go, the II.8 table today, and wave 3 (2026-11-01 ~05:00)

**Timeline of the level-5 run (796433, node g050, T = 18, 66.6 M read pairs), from the timestamped log:**

| t (s) | stage | wall s | core-s | what it is |
|---|---|---|---|---|
| 0-21 | `.1fq` decode -> FASTQ temp files + R11 histogram (parent, 18 threads); references in parallel | 20 | 87 | text that the prefill parses again |
| 21-25 | pileup references, whitelist, STAR launch | 4 | 7 | |
| 25-54 | **CR4 probe**: two STAR children in sequence (plain, cr4), each loads the private genome (8 s) and maps 200 k reads | 29 | ~35 | one core most of the time |
| 54-62 | main STAR loads the genome (8 s, NoSharedMemory; the parent prefetched the files into the page cache) | 8 | | third genome load of the run |
| 62-120 | **prefill**: reading reads 16 s (FASTQ parse, E10/E11 decode), GPU 38 s (index upload 3.1, reads + clip upload 5.8 pageable, clip 1.2, dedup 3.7, mapping 15.2, host lane 2.5, records 1), bookkeeping 4 s | 58 | 297 | GPU busy ~27% of the stage |
| 120-129 | STAR's loop on 157 k reads | 9 | 21 | startup and exit of an 18-thread loop for 0.24% of the reads |
| 129-137 | stream into the pileup, Solo on the device 0.8 s, STAR exit | 8 | 16 | |
| 137-153 | hand-off pass (region -> parent -> device: upload 1.3 s), build input 5.0 s, pileupRun 8.8 s (GPU 2.15 s), host pass 1.1 s | 16 | 22 | the records cross the host twice (5.3 GB region) |
| 153-173 | non-host: index 24.9 GB loaded to the device in 4 s, classify 0.4 s, EM 500 iterations 7 s, write reads | 20 | 45 | index load each sample |
| 173-211 | export, QC, callers 17 s; demux auto-K + VB 16 s, writers | 38 | 130 | the host track's tail |

**The II.8 hop table today (level 5):** hops 4-9 exist on the device (clip, dedup, mapping, records, per-read products, Solo,
pileup reducers, D2H); hops 1-3 are still text (the `.1fq` is decoded to FASTQ by the parent, parsed by the prefill, uploaded
pageable); hop 10 is the host. The GPU is busy 21 s of 211 s. Peak RSS 61 GB (E10's decode holds the per-read arrays).

**What the other track is doing (read-only look at origin/gpu/host-offload, 2026-11-01):** S8 merged "three serial host passes on
the job's cores" (PR #107: Solo record files parsed in parallel, hand-off and host passes chunked; wall -21%); S8/S9 drafted the
staged read format `.1fg` v2 (docs/research-beta/staged-format-1fg2.md: per-read `cb`/`cbst`/`umi`/`seq`/`q` planes plus a
unique-sequence table, "nothing is built"), whose consumers would replace hops 1-3 and the dedup. Their S8 plan names "(2)+(5)+(8)"
(STAR emits hand-off entries from the cache without the BAM stream) and then "(3)+(4)" (packed reads into the prefill, no temp
FASTQ) as next targets; E8/E10/E11 already did the first on this branch. The collision risk is on hops 1-3: E13 below consumes the
`.1fq` that the campaign has today and defines the packed-read input of the prefill as a struct of arrays, so that a `.1fg2`
unique-sequence table can fill the same buffers later; it does not implement `.1fg2`.

**Wave 3 (started 2026-11-01 ~05:10 from gpu/fable-engine 8e54be40; one Opus session each, own branch and worktree):**

| exp | branch | target (pbmc1k_v3, T = 18) | lever |
|---|---|---|---|
| E12 prefill | gpu/fable/prefill | prefill GPU 38 -> <= 25 s; stage 58 -> <= 45 s | the m5 best exact configuration is not wired into `mapDistinct` (no best-order, maxreg 64, batch-b 131072, warp lane M: 3.07 vs 3.64 M distinct/s); pinned and overlapped uploads; the host lane overlapped with the device lanes; dedup copies |
| E13 ingest | gpu/fable/ingest | -20 s decode stage, -10 s reading reads, -100 core-s | the prefill decodes the `.1fq` itself (RunSet, the one `.1fq` decoder) into packed buffers; no FASTQ temp files; the R11 histogram to the parent from the device (E10) through the region header; STAR's residual reads from the in-memory decode |
| E14 probe | gpu/fable/probe | STAR load + CR4 probe 36 -> <= 12 s | the two probe children concurrently; the genome once (LoadAndKeep is in the fork and the pipeline has `--genome-load on` and the N21 auto-attach); the optimistic probe or the GPU probe as stretch; `cr4_probe.json` identical; this also speeds up the CPU engine |
| E15 posterior | gpu/fable/posterior | level 5 on NovaSeq-binned and 5' samples | the exact per-read CB posterior on the device (quality at the mismatch of multi-match reads); gates on a binned 3' sample and pbmc5p_v1 vs the CPU engine |
| E16 tail | gpu/fable/tail | loop 9 + stream/exit 8 + hand-off 16 = 33 -> <= 15 s | instrument the tail; STAR's loop and exit for 157 k reads; the pileup input build (5 s); the records' double crossing (CUDA IPC across the child/parent boundary, or the input build in the child) |

If every target lands, the cold single-sample run is ~120 s (from 211), of which the host track's tail (export, callers, demux,
writers 38 s) and non-host (20 s) are half. **The floor of the current process model, with the tail untouched, is ~110-120 s per
66 M-read sample on one H100 (2.0 M pairs/s per GPU of wall, 3.6x the CPU engine's wall, ~8x its core-s).** Below that:
the persistent per-GPU service (genome and device index resident across samples, which E14's shared-memory genome starts;
non-host index resident; the parent's references loaded once), and the tail itself (demux auto-K and the writers), which is the
host track's area and is now the largest remaining piece. Level 5 should become the default of `--engine gpu` once E15 makes it
general and the 5' gate passes; until then it is opt-in.

### E14 probe (done 2026-11-01 ~06:00; branch gpu/fable/probe e2b0000e, fast-forwarded into gpu/fable-engine; report docs/gpu-engine/fable/E14-probe.md)

**What the 36-41 s were** (E11's job re-read, measurement job 796612): three genome loads of 8 s each on one core (STAR copies
16.8 GB out of the page cache; the parent's prefetch meant it was never disk), two probe mappings of ~5 s each (STAR's 15 MB input
chunks cut 200 k reads into 4 chunks, so 4 of 18 threads worked), ~1 s of parameter and whitelist loading per STAR process. The
probe's unique rate and SJ.out.tab are the same under any thread count, chunk size and load mechanism.

**What was done** (`src/star/GenomePreload.cpp` new; `Genome_genomeLoad.cpp`, the probe in `src/pipeline/singlet.cpp`): the parent
reads Genome, SA and SAindex once in the background while it decodes the `.1fq` (8 parallel `pread` readers into STAR's exact array
layout on transparent hugepages: human 16.8 GB in 0.95 s, mouse 25.6 GB in 2.5 s); the STAR child, both probes and the main STAR
inherit the pages through `fork()` copy-on-write (checked by directory, sizes and mtimes); the two probes run concurrently on T/2
threads each with 2 MB input chunks. Switches: `SINGLET_GENOME_PRELOAD=0`, `SINGLET_CR4_PROBE=serial`.

| gate (a), pbmc1k_v3 T = 18 (796618, g050) and neuron1k_v3 (796715, g052) | result |
|---|---|
| `cr4_probe.json` (plain 88.21 / cr4 87.56 ON; mouse 82.24 / 82.17 ON) | byte-identical in every comparison; the probes' SJ.out.tab and mapping statistics identical |
| CPU engine vs E11's kept CPU outputs; CPU engine vs level 5; baseline vs E14 on both paths | 60 files identical, CellReads.stats same rows (323,542; 600,534 on neuron) |

| run | stage | wall | core-s |
|---|---|---|---|
| pbmc1k_v3 CPU engine | 41 -> 7 s | 436.2 -> 405.1 s (-7.1%) | 2,632 -> 2,517 |
| **pbmc1k_v3 level 5** | 37 -> 3 s | **211.9 -> 179.6 s (-15.2%)** | 680 -> 660 |
| neuron1k_v3 CPU engine | 52-54 -> 2-7 s | 519.1 -> 477.2 s (-8.1%) | |
| **neuron1k_v3 level 5** | | **274.2 -> 228.5 s (-16.7%)** | |

Peak memory unchanged (level 5 60.0 -> 59.8 GB). **Shared memory for the persistent service:** SysV works under Slurm here (limits
unlimited, charged to the job's cgroup) but is the slower mechanism: 16.1 s single-threaded onto 4 KB pages (`shmem_enabled=never`
on these nodes), mapping from it costs +18% CPU (79.9 vs 67.7 user-s on 2 M pairs), removal 2.9 s, and across jobs it leaves memory
outside Slurm's accounting, is created mode 0666, and a half-loaded segment from a crash makes the next run wait up to 100 min.
mmap of the index files: 1.16 s set-up but 4 KB pages on NFS, random reads 35 ns vs 18 ns from the private hugepage copy.
**Recommendation (adopted for II.12's service design): one long-lived process per GPU that preloads once and forks one child per
sample, i.e. this branch's mechanism one level up; never a SysV segment across jobs.** Remaining: the probes during the decode
(another 4-5 s, restructures the start sequence), the whitelist loaded once (~1 s per STAR process).

### E12 prefill (done 2026-11-01 ~07:30; branch gpu/fable/prefill d03d6c47 with E14 merged, fast-forwarded into gpu/fable-engine; report docs/gpu-engine/fable/E12-prefill.md)

| pbmc1k_v3, T = 18 | target | result |
|---|---|---|
| device phase of the prefill | 38 -> <= 25 s | **38.5 -> 16.2 s** (g050, 796641 / 796791); 16.9 / 17.4 s on g051 against 39.8 / 39.7 for 8e54be40 (796888) |
| prefill stage | 58 -> <= 45 s | **59 -> 37 s** (STAR's own clock: "cache filled ... in 37 s") |
| whole level-5 run (796888, g051, same job) | | **161.6 s** against 187.2 s for the merged engine without E12 (-25.6 s, -13.7%); core-s 674 vs 677 |

Each part measured by turning it off in the same job (796791; `SINGLET_ENGINE_GPU_CFG=old` turns off every switchable part and gives 31.0 s):

| change | saves | off switch |
|---|---|---|
| records assembly: one allocation of the four output arrays, no zero-fill, parallel copies (was a serial resize per chunk that zero-filled ~4 GB on one thread: E11's "~9 s unaccounted") | 6.0 s (records.host 8.0 -> 1.9 s) | none |
| m5 best configuration in mapDistinct, and `src/engine_gpu/stitch.cuh` now forwards to `bench/gpu/m5/stitch.cuh` (the engine had compiled a stale copy from before E5 and E7: sharing the file alone made the old kernels 17% faster) | 4.5 s (lane kernels 12.7 -> 8.2 s) | `SINGLET_ENGINE_GPU_CFG=old`, `_WARPM=0`, `_NB/_NBM/_NBB` |
| staged device -> host copies (pinned 64 MB pipeline, 16 copy threads) | 3.9 s | `_D2H=pageable` |
| dedup on the device (k_heads, max-scan, k_collide, k_readrep, copy_if) | 3.2 s | `_DEDUP=host` |
| host lane in a pool of T - 1 threads while lane B runs (the wait after the lanes is 3 ms) | 2.5 s | `_HOSTLANE=after` |
| staged uploads, the 17.6 GB index on its own thread and stream | 2.6 s | `_UPLOAD=pageable` |
| sequence cache re-created only when results are inserted (nothing is at E10 level >= 3) | ~1.3 s of the after-engine bookkeeping (4 -> 2.7 s) | none |

Identity: gate (a) cpu vs level 5 60 identical and E11's 7; gpuD vs E11's kept gpuD 60 identical; engine dumps of all six
switch combinations byte-identical (6,053,620,001 bytes, one sha256); m5 harness 1,991,926 / 1,991,926 on all 4 variants, the engine's
lanes at 3.64 M distinct/s. Surprises: the largest step was host code (the records' zero-fill), not GPU; the staged reads upload
varies 2.5-11.9 GB/s between runs because it competes with the index thread for host memory bandwidth; g051's D2H runs at 5.2-5.6
GB/s against 7.2-7.6 on g050. Remaining: two lanes on two streams (~16% of the 8.2 s of lane kernels; needs explicit streams through
runLane, thrust and the scans, and a memory split since lane B takes all free memory); the index upload earlier (holds the device
during the reads); level 5 skipping the host records (1.9 s, needs a fallback); E10's products step 1.6 s; reading reads 16-17 s is E13.

**Running total after E14 + E12 (same job 796888 on g051): 8e54be40 228.5 s -> 161.6 s (-29%); on g050 the E11 figure of 211 s
corresponds to ~150 s.** Core-s unchanged at ~675 (the savings are wall on the critical path).

### E13 ingest (done 2026-11-01 ~09:40; branch gpu/fable/ingest fe97c8b4 with E12 and E14 merged, fast-forwarded into gpu/fable-engine; report docs/gpu-engine/fable/E13-ingest.md)

`SINGLET_ENGINE_GPU_INGEST=1` (off by default; unset, the code paths are unchanged): the `.1fq` goes from disk into the prefill's
packed struct of arrays without FASTQ text.
- The parent's `.1fq` -> FASTQ + R11 decode (20 s / 90 core-s) is replaced by a head scan (`RunSet::scan_head`: the barcode-dictionary,
  UMI and length columns only, 2.7 s) that gives barcode discovery, the 5' probe sample and the per-block plan; STAR is forked with
  `/dev/null` as its read files.
- Phase A: the pipeline decodes the `.1fq` with the RunSet decoder into `PackedReads` (`include/singlet/engine_gpu_ingest.h`, with the
  `.1fg2` mapping comment) in 3.4 s on 18 threads, beside the reference load, before the STAR fork; STAR inherits the arrays through
  fork (copy-on-write) and the parent frees its copy. No second copy of the reads exists. Phase C: STAR's CR4 clip at start-up beside
  E14's probes, once per distinct sequence (31.9 M of 66.6 M: 88 thread-s instead of 168). Phase B: E10's aux words, E11's Q30 counts
  and R11 from the same pass. The prefill swaps the arrays in (waited 0.3 s); Unmapped.out and STAR's residual reads are rebuilt from
  the packed arrays. Fallbacks: `_INGEST_EARLY=0` (all three phases in the prefill), `_INGEST_CLIPDEDUP=0` (clip every read).
- R11 handshake: the prefill writes R11 into the E8 region behind the magic `SGR11H01` in `E2EHeader`; the parent's `CbCorrector`
  fetches it lazily (`set_pending`/`settle`, call_once) the first time a barcode has two or more candidates, with a head scan of the
  `.1fq` in R11 mode as the fallback.

| identity | result |
|---|---|
| gate (a) 796992, cpu vs level 5 + ingest | 60 identical; the 7 are E11's run-dependent files; mapping stats, SJ (126,996), CellReads.stats (323,542 rows), R11 identical |
| level 5 without vs with ingest | 63 identical, read files and CellReads.stats byte-identical in file order, same GPU pileup digest; the 5' probe sample size 12,000 -> 12,194 now matches the CPU engine |
| ingest variants (second run, sync, no-dedup) | 63 identical with each other |
| switch unset: gpuD, gpuE5, cpu vs E11's kept outputs | 60 identical each |
| neuron1k_v3 (T = 8, 797012) | 60 identical, CellReads.stats byte-identical; 246.6 -> 196.6 s (-20%), -156 core-s, maxRSS 82.8 -> 60.4 GB |

| pbmc1k_v3, T = 18 | without ingest | with ingest |
|---|---|---|
| level 5 on E14 head (796992, g051, 2 runs each) | 175.9 s / 666 core-s | **149.1 s / 564 core-s** (-15.2% both) |
| **level 5 on E12 head (797219, g052)** | 153.6 s / 656 | **127.9 s / 561** (-25.6 s, -95 core-s): E12 and E13 add up |
| decode stage | 20 s / 90 core-s | 2-3 s / 19-29 |
| prefill | 54 s / 305 core-s (reading reads 12 s) | 42 s / 147 (reading reads 0); 20 s on the E12 head |
| maxRSS | 61.0 GB | 45.8 GB |

The reference-load and STAR-start stages grow by 2-3 s each (they carry the decode and the clip). Remaining: the switch on by
default; the CR4 clip now sets the length of the stage before the prefill (move it to the device next); at low thread counts the
early decode competes with the reference load and the probes; the barcode pass into the pipeline; head scan and decode as one pass;
inputs ingest does not take yet (full-quality `.1fq`, duplicate counts, no barcode dictionary, barcode reads > 63 bases, a 5' prefix
clip, complex barcodes) still go through FASTQ.

**Running total after E14 + E12 + E13 (same-job comparisons): E11's level 5 -> 127.9 s / 561 core-s (-39% wall), 3.09x the CPU
engine's wall and 4.6x its core-s. The GPU prefill is 20 s, the decode 3 s, the probe 3-7 s.** The remaining big pieces are the tail
after mapping (E16, running), non-host 20 s and the host track's export/demux 38 s.

### E15 posterior (done 2026-11-01 ~10:30; branch gpu/fable/posterior 44a6e9fa with E12-E14 merged, fast-forwarded into gpu/fable-engine; report docs/gpu-engine/fable/E15-posterior.md)

**The premise was wrong, usefully.** The `.1fq` stores only R2's qualities; every decode writes 'F' for R1, so STAR sees CB qualities
70..70 on every sample and E11's constant-quality guard never fires in a pipeline run: level 5 already ran on pbmc2m, pbmc5p_v1 and
neuron1k_v3 with E11's binary. The exact per-read posterior is on the device anyway (`cbPosteriorRead` in solo.cuh, a "multi" stage
in solo_gpu.cu; the guard is gone; the per-read path engages by itself when CB qualities vary; `SINGLET_E15_PERREAD=1` forces it),
checked against a plain STAR on pbmc1k_v3's original L001 FASTQs (33.4 M reads, 5.9% of CB bases not 'F', 626,729 multi-match reads):
0 reads decided differently, 10 of 10 Solo files identical; the one-constant control differs on 20,492 reads and 5 files.
**Side result for the atlas (not a GPU matter):** the pipeline's 'F' R1 qualities change 3.3% of the multi-match CB decisions
compared with STARsolo on the FASTQ (19,476 fewer reads with a CB on pbmc1k_v3; the same 1,135 cells), on the CPU engine and every
GPU level alike. Whether the `.1fq`/`.1fg2` should carry R1 qualities is a format decision for Zach and the S9 track.

| gate (a), T = 18, CPU engine vs level 5 | out_equiv identical / timing-path / differ | Solo | CellReads.stats rows |
|---|---|---|---|
| neuron1k_v3 (796674) | 60 / 2 / 7 | 10 of 10 | 600,534 |
| pbmc5p_v1, 5' (796674) | 58 / 2 / 7 | 10 of 10 | 355,639 |
| nuclei2k_v2 (796865) | 60 / 2 / 7 (nonhost_reads.fq differs: a pre-existing bug of the non-host writer's subsampling above 20 M unmapped reads) | 10 of 10 | 411,195 |
| pbmc1k_v3 vs E11's kept CPU outputs | 60 / 1 / 8 | 10 of 10 | 323,542 |

| T = 18, wall s / core-s | CPU engine | gpuD | level 5 | level 5 vs gpuD |
|---|---|---|---|---|
| neuron1k_v3 | 520.4 / 2180 | 326.7 / 1206 | 271.3 / 801 | -17.0% / -33.6% |
| pbmc5p_v1 | 489.8 / 2164 | 229.5 / 846 | 199.2 / 579 | -13.2% / -31.6% |
| nuclei2k_v2 | 611.2 / 4712 | 459.2 / 1751 | 391.0 / 1179 | -14.9% / -32.6% |

The kept Solo inputs (1.35-3.0 GB) now sit in page-locked host memory during STAR's loop (E11 §5.3; ~0.4 s, `SINGLET_E15_KEEP_ON_DEVICE=1`
restores E11). Merges of E14, E12 and E13 were conflict-free and re-gated; the E13 merge exposed that the ingest maps no barcode
FASTQ, so under ingest the per-read test path is ignored and logged (CB qualities cannot vary there today). Final re-gate 797379
(g051): pbmc1k_v3 level 5 154.6 / 662, + ingest **128.3 / 565**; pbmc5p_v1 CPU engine 478.9 / 2090, level 5 173.7 / 577, + ingest
**153.9 / 526** (3.1x the CPU engine's wall, 4.0x fewer core-s).

**What still falls back, and it is the mapping code, not Solo: hgmm1k_v3 (human + mouse barnyard, ~5.8 G bases).** GPU mapping
runs out of device memory in lane A at every level (even at a half and a quarter of lane A's size) and the run drops to level 0;
smaller lanes (NB 32768, NBB 16384, chunk 8 M; probe 797247) make it fit, and then E10 refuses because it packs junction starts
into 32 bits. Two ports, both in engine code: a device memory budget for the lanes (also what the A100 fit needs) and 64-bit
junction starts in E10. The CPU-engine outputs of that sample are matched by the fallback (49 / 2 / 7, Solo identical).

## II.13 Wave 4 (2026-11-01 ~10:30, from gpu/fable-engine 44a6e9fa = E1-E15 merged): generality, the node's throughput, and the release gate

**Where wave 3 leaves the sample (pbmc1k_v3, T = 18, level 5 + ingest, same-job comparisons):** 211 s / 680 core-s -> 128 s / 561
core-s (3.1x the CPU engine's wall, 4.5x fewer core-s); decode 3 s, references 4-7 s, probe + start 5-8 s (now carrying the decode
and the CR4 clip), prefill 20 s, the tail after mapping ~33 s (E16 pending, target <= 15 s), non-host 20 s, export + demux 38 s.
GPU busy ~25 s per sample. The kernels are at their exact ceiling (II.11); the per-sample fixed costs that remained have been taken
down except the host track's tail and non-host. **The next lever is not the single-sample wall but the node's throughput:** at
8.5 host core-s/M and ~25 GPU-s per 66 M pairs, one H100 NVL node (56 cores, 2 H100, 768 GB) is balanced at roughly 3 samples in
flight per GPU, where the host tail of one sample overlaps the prefill of the next. The engine already has file-lock device slots
(`SINGLET_ENGINE_GPU_SLOTS`, up to 8, with the caveat that each process must keep its device memory within (device - margin) / slots
by the chunk and arena knobs). What is missing is a device memory budget in the lanes (lane B takes all free memory, and the
barnyard genome does not fit at all: E15) and a measurement of what actually serialises when samples share a GPU.

**What must be true before level 5 + ingest can be the default of `--engine gpu` (the "reliable" half of the goal):**
1. every genome fits: a device memory budget for the lanes and 64-bit junction starts in E10 (hgmm1k_v3 falls back to level 0 today;
   large genomes such as axolotl or wheat are in GEO); the same budget is what the A100 (80 and 40 GB) needs;
2. every input the campaign produces takes the ingest path or falls back cleanly (E13's list: full-quality `.1fq`, duplicate counts,
   no barcode dictionary, barcode reads > 63 bases, 5' prefix clip, complex barcodes);
3. the all-sample gate: every benchmark sample (9 under `$W/pipeline-benchmark/data/`) and the release-gate datasets, CPU engine vs
   level 5 + ingest, with the known run-dependent files only;
4. two pre-existing pipeline bugs found by the gates, for the host track: the non-host writer's subsampling above 20 M unmapped reads
   (nuclei2k_v2) and E10's level-4 fallback (E11 §5.4).

| exp | branch | what | target |
|---|---|---|---|
| E17 budget | gpu/fable/budget | a device memory budget for the lanes (chunk, lane sizes, lane B's "all free memory") derived from the budget = (device memory - margin) / slots or `SINGLET_ENGINE_GPU_MEM_GB`; 64-bit junction starts in E10; the CR4 clip on the device (E13's item) if time allows | hgmm1k_v3 at level 5 + ingest identical to the CPU engine; pbmc1k_v3 identical under budgets of 80, 40 and 24 GB with the measured peak under each; no slowdown at the full budget |
| E18 throughput | gpu/fable/throughput | k = 1, 2, 3, 4 samples in flight per H100 with the slot locks, and 2 GPUs x k on one node; aggregate read pairs/s, device and host peaks, what serialises (the slot lock holds the prefill's device phase and the pileup), the host core-s per M at each k | the node's pairs/s curve and its bottleneck; the k and thread split to recommend for the campaign; fixes only in the bench scripts and the slot/env plumbing |
| E19 release (after E16 merges) | gpu/fable/release | level 5 + ingest as the default of `--engine gpu` behind the fallbacks; the ingest's unsupported inputs catalogued per sample and the common ones taken; the all-sample gate | every benchmark sample identical (the run-dependent files only); the defaults flipped; a one-page "how the GPU path runs a sample" for the campaign |

Not in wave 4 (noted): contract-2 alignment experiments (Zach has not answered); the host track's tail (export, callers, demux,
writers: 38 s of the remaining ~110) and non-host's index load (20 s), which the persistent per-GPU process (E14's recommendation:
preload once, fork per sample) would also absorb; two lanes on two streams (~1.3 s).

### E18 throughput, partial (2026-11-01 ~12:30; branch gpu/fable/throughput 9907d324, bench scripts only; stopped twice before the full-scale runs, not merged)

The session handed back twice saying it had been stopped early, each time before submitting the pbmc1k_v3 curve; its scripts
work (debug job 797417 on pbmc2m: k = 2 identical to k = 1 in every output, E6 dumps identical) and the remaining job list is in
its hand-back: job A `POINTS="k1t28s1 k1t14s1 k1t9s1 k1t7s1 k2t14s1 k3t9s1 k4t7s1 k4t14s1 k1t14s1m43 k2t14s2m43 k3t9s2m43 k4t7s2m43"`
on one GPU with 28 cores, job B the steady-state rounds and the mixed set, job C the node with 2 GPUs and 54 cores (CPUEfctv).
What it established from the code and the k = 1 logs before stopping:

| phase holding a device slot (pbmc1k_v3, level 5 + ingest) | held |
|---|---|
| prefill (`mapDistinct`) | 16.1 s |
| hand-off pass on the device (e2ePass) | ~3.6 s |
| Solo (E11) | ~1 s |
| pileup (`pileupRun`) | 9.5 s, of which 2.2 s is GPU |
| non-host (index load from the host 4 s + classify 0.4 s) | ~4.4 s |
| **total** | **~35 s per sample, of which ~13 s is kernels** |

So with `SINGLET_ENGINE_GPU_SLOTS=1` one H100 is capped near **1.9 M pairs/s** by the lock, not by the kernels: the records'
host assembly, the pileup's host fetches and the non-host index load run under the lock. That is the direct lever for the
service: hold the slot only around device work. `SLOTS=k` does not serialise anything; it lets k processes use the device at once
and therefore needs a memory split, and today lane B and the pileup arena each take free memory minus 3 GB (90-92 GB); E18's
two-slot split `m43` = chunk 4 M, NB 274432, NBM 155648, NBB 21504, arena 38 GB; three slots cannot fit because the prefill holds
26.4 GB before its lanes (index 16.4 + reads/clip 9.5 + context 0.5) and non-host needs a fixed 24.9 GB with no knob. These
requirements were forwarded to E17 (the budget). The k curve, the mixed set and the node point are still to be measured.

### E16 tail (done 2026-11-01 ~13:30; branch gpu/fable/tail e0e1514a with E12-E15 merged, fast-forwarded into gpu/fable-engine; report docs/gpu-engine/fable/E16-tail.md)

22 cuts, all active only at level 5 (`SINGLET_E16=0` turns them all off, `SINGLET_E16_OFF=name,...` single ones, `SINGLET_E16_TRACE=0`
the timing marks); the CPU engine, gpuD and levels 1-4 run unchanged code. The tail (prefill releasing the device -> pileup results
in the accumulators) on pbmc1k_v3 at T = 18: **36.0 -> 8.8 s** (target <= 15 s):

| piece | before | after | how |
|---|---|---|---|
| hand-off pass + pileup input build | 11.4 s | 1.2 s | zero-copy, started early |
| pileupRun | 9.2 s | 4.0 s | no digest, smaller per-alignment download, uploads from page-locked memory in place |
| STAR's loop on the residual reads | 5.1 s | 0.9 s | 18 threads instead of 2, chunks built in parallel |
| BAM stream end seen at close, not at STAR's exit | 3.1 s | 0.01 s | eof |
| the prefill's tail | 4.7 s | 1.0 s | no 12 GB cache rebuilt, Unmapped.out on a thread, frees off the critical path |

| final binary a6dfaf9b, gate 797428 (g051, T = 18, same job) | wall s | core-s | tail |
|---|---|---|---|
| CPU engine | 395.2 | 2,485 | |
| default GPU path (gpuD) | 201.2 | 993 | |
| level 3 | 194.2 | 916 | |
| E11's level-5 binary | 213.4 | 681 | |
| level 5, `SINGLET_E16=0` | 153.2 | 657 | 35.6 s |
| **level 5** | **129.5** (128.1 / 130.8) | **656** | 9.5 s |
| **level 5 + ingest** | **102.8** | **553** | 9.5 s |

Level 5 + ingest: -110.6 s (-51.8%) against E11's binary, **3.84x the CPU engine's wall, 4.49x fewer core-s**. Identity: cpu vs
level 5, vs level 5 + ingest, vs level 5 with the per-read posterior: 60 / 2 / 7 (E11's run-dependent 7) on every binary;
level 5 vs `SINGLET_E16=0`: 60 / 3 / 6 with CellReads.stats byte-identical; gpuD and levels 1, 3, 4 vs E11's kept outputs 60
identical. **Bug found by the gate:** the first merged build crashed under E13's ingest (STAR's child SIGSEGV: E13 freed its packed
reads at the prefill's end while E16's background thread still read them); fixed by holding the reads until the join, and the gate
now checks each run's exit status (it had printed "GATE RUN OK" for a crashed run). E15's host copy of Solo's kept inputs costs the
tail 0.8 s (allocating the page-locked memory, not the copy); the wall does not move. Remaining in the tail: the early hand-off pass
2.1 s, pileupRun 4.0 s (2.2 GPU, 0.8 host fetches between partitions), the accumulator puts 1.7 s, the prefill's tail 1.0 s.

**Wave 3 + E16 total (same-job comparisons on g051): E11's level 5 213 s / 681 core-s -> level 5 + ingest 103 s / 553 core-s,
with the CPU engine at 395 s / 2,485 in the same job.** What is left of the 103 s: references ~5 s, probe + start ~8 s (the CR4 clip
on the host), prefill ~20 s, the tail ~9.5 s, non-host ~20 s, export + callers + demux + writers ~38 s. The host track's tail and
non-host are now 56% of the sample.

### E17 budget (done 2026-11-01 ~15:30; branch gpu/fable/budget 503c07f7 with E16 merged, fast-forwarded into gpu/fable-engine; report docs/gpu-engine/fable/E17-budget.md)

`SINGLET_ENGINE_GPU_MEM_GB` sets the device budget; the default is (device memory - 2 GB) / `SINGLET_ENGINE_GPU_SLOTS` (91.1 / 45.5 /
30.4 GB for 1 / 2 / 3 slots on the H100 NVL); 0 restores E12's behaviour. The budget sizes the chunk, the A/M/B batches, lane M's
warp-or-thread choice, lane B (no longer "free - 3 GB") and the pileup arena (budget - 3 GB); a lean mode keeps only the index and
one chunk on the device when the normal plan does not fit; every prefill allocation is tallied (including lane M's 4 GB warp stack,
which the driver reserves for the whole device) and the counted total is within 0.2 GB of the device after every lane; a sample
that cannot fit stops with a FATAL message before allocating (`SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu` makes STAR map every read instead).
E10's junction key is `start << gb | gap` with gb = 32 below 2^32 bases and 31 on the barnyard; the other 32-bit packings (Solo
model, E8 records, hand-off entries) needed no change.

| pbmc1k_v3, level 5 + ingest, gate 797462 (g051, merged binary; all identical to E11's kept CPU outputs: 60 files, Solo 10/10, 323,542 rows, 126,996 junctions) | wall s | prefill peak |
|---|---|---|
| E16 head | 103.2 / 104.5 | 90.4 GB |
| full budget (91.1 GB) | 103.0 / 103.2 | 88.6 GB |
| 80 GB | 105.1 | 77.5 GB |
| 40 GB | 108.4 | 38.0 GB |
| 24 GB (lean) | 121.4 | 23.2 GB |

Two slots peak at 43.8 GB and three at 29.4 GB in probes, so three samples can share an H100's memory. **hgmm1k_v3 (human + mouse
barnyard) runs at level 5 + ingest and matches a fresh CPU run in the same job (797468): 49 identical files, Solo 10/10, SJ.out.tab
identical at 171,869 junctions, CellReads.stats the same 524,497 rows; 88.7 s / 438 core-s against the CPU engine's 374.9 s / 1,609.**
The default GPU path runs on it too now (176.9 s, identical) where E15 saw both fall back to level 0. gpuD on pbmc1k_v3 still
matches E11's kept gpuD. Left outside the budget and documented as a fixed extra: the non-host screen (25.4 GB with its index, 29.3
GB at the classify step), which does not overlap the prefill within one sample. The device lock was not touched: measured hold
16.1 s at the full budget, 20.9 s at 40 GB, 32.3 s at 24 GB, with the records' host assembly under it (E18's lever). Fixed along the
way: a budget failure reset the device while staging buffers still held handles (SIGSEGV); the lean mode's per-chunk gather (3.7 ->
0.9 s). Not done: the CR4 clip on the device. **The A100 fit is now a budget setting: 80 GB costs +2%, 40 GB +5% on pbmc1k_v3.**

### E19 release (done 2026-11-01 ~18:30; branch gpu/fable/release dc45ef09 with E17 merged, fast-forwarded into gpu/fable-engine; report docs/gpu-engine/fable/E19-release.md; operator's page docs/gpu-engine/fable/GPU-PATH.md)

**Defaults:** `--engine gpu` with `SINGLET_GPU_PILEUP=on` now runs level 5 + ingest + E16 with no other variable (values already
set are kept; opt-outs `SINGLET_ENGINE_GPU_E2E=0`, `SINGLET_ENGINE_GPU_INGEST=0`; E16's timing marks off unless `SINGLET_E16_TRACE=1`);
every run prints a defaults line and the level it reached (": FELL BACK" when below the request); the 13 fallbacks that wrote only
to Log.out now also print `[engine_gpu] fallback: ...` on stderr; a `singlet gpu-probe` child runs before STAR so a node without a
usable GPU keeps the CPU pileup instead of exiting 9; the level-4 hole (E10 §5.4) now stops with exit 104; `--help` documents the
switches; the CPU engine is untouched.

**All-sample gate 798027 (binary ced7f553 = E19 + E17, T = 18, g052; CPU references from 797474 on g051). All 10 runs byte-identical
to the CPU engine, each at level 5 + ingest with no fallback line, hgmm1k_v3 included.** out_equiv finds only E11's run-dependent
files (4); the 10 Solo files, SJ.out.tab, the mapping statistics, the unmapped mates (same records) and CellReads.stats (same rows)
identical. The one exception is not the GPU path: nuclei2k_v2's nonhost_reads.fq (the non-host writer bug of E15 §5.7, which also
differs between two GPU runs).

| sample | pairs | CPU engine wall / core-s / RSS GB | GPU path wall / core-s / RSS GB | wall x | core-s x |
|---|---|---|---|---|---|
| pbmc2m | 2.0 M | 49.6 / 161 / 25.8 | 32.4 / 121 / 21.8 | 1.53 | 1.33 |
| pbmc10m | 10.0 M | 100.5 / 482 / 26.7 | 45.2 / 193 / 24.9 | 2.23 | 2.50 |
| pbmc1k_v3 | 66.6 M | 403.9 / 2,507 / 29.7 | 101.5 / 548 / 46.1 | 3.98 | 4.58 |
| pbmc1k_v2 | 76.9 M | 380.8 / 2,798 / 29.4 | 104.4 / 536 / 47.5 | 3.65 | 5.22 |
| pbmc1k_cite_v3 (GEX) | 45.4 M | 278.0 / 1,713 / 28.2 | 78.4 / 411 / 38.2 | 3.55 | 4.17 |
| pbmc1k_cite_v3 antibody library (mapped as RNA, identical) | 12.6 M | 129.9 / 1,255 / 32.9 | 65.0 / 274 / 29.6 | 2.00 | 4.58 |
| pbmc5p_v1 (5') | 60.0 M | 495.4 / 2,123 / 32.3 | 123.6 / 509 / 39.6 | 4.01 | 4.17 |
| neuron1k_v3 (mouse) | 92.9 M | 491.5 / 2,148 / 35.3 | 131.5 / 615 / 60.8 | 3.74 | 3.49 |
| hgmm1k_v3 (barnyard) | 63.1 M | 363.1 / 1,597 / 62.2 | 90.5 / 443 / 78.4 | 4.01 | 3.61 |
| nuclei2k_v2 | 118.7 M | 609.9 / 4,604 / 34.9 | 198.7 / 954 / 84.4 | 3.07 | 4.83 |
| **10 runs** | 548 M | **3,303 s / 19,388 core-s** | **971 s / 4,603 core-s** | **3.40** | **4.21** |

Opt-out `SINGLET_ENGINE_GPU_E2E=0` on pbmc1k_v3: 195.1 s / 965 core-s, identical to the CPU engine and to E11's kept gpuD. Device
peak 86.5-89.3 GB against the 91.1 GB budget on every sample but pbmc2m; every sample plans in normal mode. Before the E17 merge
(797474) hgmm1k_v3 fell back to STAR with the fallback line and was still identical (235.7 s). **Declines:** no benchmark sample
declines the ingest or any level; the port for each decline condition is listed in the report (full qualities, duplicate counts,
no barcode dictionary, barcode read > 63 bases, CB or UMI > 21, a 5' prefix, barcode in R2 / Complex / no whitelist, reads > 128
bases, experimental intron rules, the BAM paths). The pipeline counts antibody tags only from a CITE-typed `.1fq` with
`--feature-ref` in a CPU pass after the export, the same with either engine; the benchmark has no tag reference, so that path was
not run.

**Surprises:** g050's GPU is not visible to jobs ("No devices were found", empty /dev/nvidia-caps; jobs there exited 9 before the
probe; later jobs exclude g050: a cluster-admin item). **A trial merge of gpu/fable-engine into prod/campaign conflicts in 9 files
and 27 hunks (engine_gpu.cu alone 10): prod/campaign has 291 commits of its own, several on the GPU path (device dedup, pinned
staging, the GPU pileup's kFlank and USA labels), so the campaign merge needs a staging merge and a full re-gate with refreshed CPU
references (`REFRESH_REF=1`); the draft PR description is in E19 §9.** neuron1k_v3's GPU pileup holds the device for 28 s against
0.4-5.3 s elsewhere (141 M mitochondrial tuples; pileup.cu). provenance.json records the binary's directory in the whitelist path.

**State of the track after wave 4 (E18 pending): everything on the GPU path, byte-identical to the CPU engine on all 10 benchmark
runs, 3.4x the CPU engine's wall and 4.2x fewer core-s over the set, on by default under `--engine gpu`; 24-91 GB device budgets;
barnyard genomes.**

### E18 throughput (done 2026-11-01 ~22:00; branch gpu/fable/throughput 283105e6 with E16, E17 and E19 merged, fast-forwarded into gpu/fable-engine; report docs/gpu-engine/fable/E18-throughput.md; bench scripts only)

**The node (2 H100 NVL, 54 cores, 730 GB; job 798028 on g052, E17 head): 2 slots, 8 samples per GPU as a stream, 7 threads each,
workers pinned to a GPU: 4.39 M read pairs/s in steady state (3.24 over the span of 32 samples), 3.0-3.2x the CPU engine's
1.38-1.44 on the same node; 44 of 54 cores busy, 9.1 core-s per M pairs, host memory peak 607 GB.** Other node points: 1 slot x 6
per GPU pinned 2.74; 2 slots x 6 pinned 3.78; 2 slots x 8 unpinned 4.08.

| one GPU, 28 cores, pbmc1k_v3 streams, M pairs/s | k = 1 | k = 3 | k = 4 | k = 6 | k = 8 | one-slot limit (queue model) |
|---|---|---|---|---|---|---|
| E15 head, 1 slot | 0.44-0.49 | 1.00 | 1.16 | 1.27 | | 1.62 |
| E16 head, 1 slot | 0.52 | 1.10 | | 1.40 | 1.57 | 1.84 |
| E17 head, 1 slot | 0.52-0.54 | 1.10 | 1.16 | 1.44 | | |
| **E17 head, 2 slots** | | 1.21 | 1.36 | **1.86** | **2.17** | |

Two slots with E17's budget alone gain 10-17% at k = 3-4 and 29-38% at k = 6-8; E15's hand-made split gained 0-8% because it slowed
a lone prefill from 18.2 to 27.4 s; three slots add nothing over two. **Why one slot bends:** the samples move in a convoy; at k = 4
the lock sits idle 31% of the time while every run waits 70-100 s for it; a first-come-first-served queue model from the k = 1
profile (e18_sim.py) matches every one-slot point within 1-5%. **Lock held per sample at k = 1:** E15 head 41.1 s for 11.6 s of
kernels (prefill 23.1 of which 6.8 host work, hand-off 3.9, Solo 0.7, pileup 9.0 of which ~4.6 host, non-host 4.4; ~8.6 s could
run outside); E16 head 36.1 s (prefill 22.2, early hand-off 2.5, Solo 0.6, pileup 6.3, non-host 4.5; ~7.9 s outside: 5.7 s in the
prefill between lane B's download and e2e.build, 2.2 s in the pileup where E16's background reset keeps the lock through the host
pass). Holding the lock only around device work lifts the one-slot limit to ~2.4 M pairs/s per GPU; a persistent per-GPU process to
~3.0. **What bounds k is host memory, not the device:** 38-43 GB per pbmc1k_v3 sample (nuclei2k_v2 83 GB), the private 16.8 GB genome
copy per process the largest piece; the device stays at 86-89.5 GB at every k with the budget (per process 45 GB at 2 slots, 30 GB
lean at 3); the non-host screen's fixed 25-29 GB sits outside the budget, so 4 slots do not fit. Identity: 266 runs compared with
their sample's k = 1 run, all count outputs identical; the 44 mapDistinct dumps identical (one sha256). Findings: unpinned workers
wander between GPUs (12 of 32) and 4 of 32 lost the device in the non-host screen and screened on the CPU (outputs identical);
g050's GPU failed CUDA initialisation in 798046 (void, re-run); nuclei2k_v2's nonhost_reads.fq differs per run (the writer indexes
the screen's subsample against the whole unmapped file, whose order became run-dependent with E16: the known writer bug).
**Campaign arithmetic: ~20.6 T read pairs at 4.4 M pairs/s per node = ~54 node-days, against ~170 on the CPU engine.** Carried into
GPU-PATH.md §5 (commit on gpu/fable-engine).

## II.14 Where the ceiling is now, and wave 5 (2026-11-01 ~22:30, from gpu/fable-engine after E18)

**The single sample** is at 101.5 s / 548 core-s (pbmc1k_v3, T = 18): kernels ~13 s at their exact ceiling; ~43 s of GPU-path host
work (references 5, probe + start 8 with the CR4 clip on the host, prefill 20 of which 5.7 s host work under the lock, tail 9.5);
~58 s of the host track's tail and non-host. **The node** is at 4.39 M pairs/s with 44 of 54 cores busy at 9.1 core-s/M: it is
host-core-bound, not GPU-bound (pure kernel time would allow ~5 M pairs/s per GPU, ~10 per node). So the ceiling of this process
model on this node is set by host core-s per M: at 9.1 it is ~5.9 M pairs/s with every core busy, and each core-s/M removed is
worth ~11% of node throughput. Where the 8.2 core-s/M of a lone pbmc1k_v3 run go (E13/E19 stage tables): prefill ~147 core-s
(of which the CR4 clip 88 thread-s, E10/E11's decode, uploads), demux/ancestry/writers ~94, non-host ~45, export/QC/callers ~36,
probes ~35, STAR's residual loop + stream ~35, hand-off + pileup ~30, decode ~20 (per 66.6 M pairs). The host track's pieces
(demux, export, non-host's EM) are ~175 core-s = 32%; the GPU path's own host work is the rest, and the two largest of those are
the CR4 clip (88) and the probes (35), both of which belong on the device.

| exp | branch | what | target |
|---|---|---|---|
| E20 lockclip | gpu/fable/lockclip | the device lock held only around device work (the records' assembly, the pileup's host pass and E16's reset outside it; ~7.9 s per sample); the CR4 clip on the device (88 thread-s, ~5 s of the start stage); the CR4 probe on the device (two mapDistinct runs of the 200 k reads with the two parameter sets, ~35 core-s, ~5 s) | pbmc1k_v3 identical; host core-s/M 8.2 -> <= 6.5; one-slot k = 8 stream 1.57 -> >= 2.0 M pairs/s per GPU; node >= 5 M pairs/s re-measured with E18's run.sbatch |
| E21 pool | gpu/fable/pool | `singlet gpu-pool`: one resident process per GPU that preloads the genome, the non-host index on the device and the parent's references once, then forks one worker per sample from a queue with E18's settings (slots, pinning, threads, stream), so that per-sample host memory drops by the 16.8 GB genome copy (k up to 12-16 per GPU), the non-host index load (4 s, 25 GB per sample) and the reference loads vanish, and the campaign's orchestrator calls one command per node | identical outputs per sample; node throughput re-measured with the pool against E18's 4.39; host memory per sample in flight; a one-page operator's section in GPU-PATH.md |
| E23 mtpileup | gpu/fable/mtpileup | neuron1k_v3's GPU pileup holds the device 28 s (141 M mitochondrial tuples) against 0.4-5.3 s elsewhere: profile pileup.cu's mito path and fix the scaling | neuron1k_v3 identical; pileup lock <= 6 s; no change on the other samples |

Not taken, flagged to Zach: the host track's tail (demux auto-K + VB ~16 s and ~94 core-s, export/QC/callers 17 s / 36 core-s,
non-host's EM 7 s) is now the largest piece of both the sample's wall and the node's core budget; the non-host writer bug; the R1
quality question (E15); contract-2 (unanswered); the campaign merge (card).

### E23 mtpileup (done 2026-11-02 ~00:30; branch gpu/fable/mtpileup f73938b2, fast-forwarded into gpu/fable-engine; report docs/gpu-engine/fable/E23-mtpileup.md)

**The cause was not the mitochondrial path** (the chrM stages take 16 ms for 141 M tuples, linear). It was the junction R9 collapse
kernel `k_r9`: 25.75 s of neuron1k_v3's 26.8 s of pileup GPU time (profile 798059). `k_r9` gives each (cell, junction) group to one
thread and tests every pair, O(n^2) per group: neuron1k_v3 has a group of 14,097 UMIs and 6 above 10,000 (one highly expressed
nuclear gene), pbmc1k_v3's largest is 456. Two more one-thread-per-group kernels had the same shape at smaller scale (the SNP R9
collapse 0.35-1.42 s, `k_cr_roots` for kUmiGeneCR 0.41-0.50 s). **The scaling law: the pileup's GPU time was set by the single
largest group (n_max^2 / 2 pair tests on one thread at 48-259 ns per pair), not by the tuple count.**

**Fix** (pileup.cu, m6.cu; `SINGLET_GPU_PILEUP_E23=0` restores the old kernels): candidate pairs from radix-sorted buckets of
(group, one half of the base-5 UMI), so two UMIs one base apart meet exactly once and take `k_r9`'s own test; passing pairs joined
by a lock-free union-find that keeps the smaller index as root (the same root the serial kernel leaves); gene-assigned entries take
the first entry of their gene component from a stable sort; SNP counts summed per root with atomic adds; kUmiGeneCR replays the
CPU's sequential union per connected component over the existing edges sorted by (component, lower rank). Scratch reuses dead
arrays; peak device memory unchanged (24 GB).

| gate 798081 (T = 18, g052, old and new binary in one job vs E19's CPU references; 21 of 21 verdicts identical, all 10 samples) | lock held | pileup GPU time | wall | core-s |
|---|---|---|---|---|
| neuron1k_v3 | 28 -> 2 s | 26,757 -> 274 ms | **127.7 -> 100.2 s (-21.5%)** | 588 -> 560 |
| pbmc1k_v3 | 4.1 -> 2.1 s | 2,148 -> 237 ms | 102.1 -> 100.7 s | 551 -> 549 |

The three kernels on neuron1k_v3: 26,485 -> 3.9 ms; the pileup's GPU time now <= 0.32 s on every sample, and the lock hold is
host-side (the results fetch 0.7-0.9 s, 3.06 s on pbmc5p_v1, which was not investigated). The default path (E2E=0) still matches
E11's kept gpuD. The six mt outputs, the Solo files, SJ.out.tab, the mapping statistics, the sorted read files and CellReads.stats
are identical on every sample.

## II.15 Program stopped on the usage limit (2026-11-02)

Wave 5 stopped with E20 and E21 measured but unreported. Their results, what is left and the open decisions are in
`HANDOFF.md` beside this file. Where the two disagree, the handoff is newer.
