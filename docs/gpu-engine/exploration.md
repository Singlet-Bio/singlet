# GPU-native Singlet engine: exploration (W11)

Written 2026-10-27 by the W11 design agent, from [gpu-aligner-brief.md](../research-beta/gpu-aligner-brief.md). This is
the exploration pass Zach asked for: ideas (kept and rejected), quantitative models, a recommended architecture, a
per-stage throughput model, the riskiest assumptions, and the microbenchmarks that decide them. The design document
(`design.md`) and the build come after Zach approves a direction.

**What is measured and what is modelled.** Nothing in this document was measured on a GPU. The real inputs are:
- the CPU engine's profile ([audit-2026-10-27.md](../research-beta/audit-2026-10-27.md) §3, PR #41);
- the code of the vendored STAR 2.7.11b (`origin/prod/campaign`);
- public hardware and library figures (§11).

Everything else is a **model**, and is labelled as one. The model is executable: `python bench/gpu/throughput_model.py`
prints every number in §7. Every assumption names the microbenchmark (§9) that confirms or kills it.

---

## 0. Summary

**The bottleneck is not where a STAR port looks for it.**
- **Bandwidth is not binding.** An H100 NVL can serve roughly 20–25 G random 32-byte sectors per second (derived from
  public hash-table figures). STAR needs ~300 random sectors per read. On bandwidth alone, even a direct port could map
  ~70 M reads/s.
- **What actually binds a direct port:**
  1. **Warp divergence.** One heavy read (repeat or too-short) stalls 31 others. The modelled SIMT efficiency is 24% on
     3′ and 14% on 5′.
  2. **STAR's per-read state.** About 56 MB is allocated per `ReadAlign`, of which the window-bin map alone is 195 KB.
     That cannot live per GPU thread.
  3. **Redoing identical work.** 53% of 3′ reads repeat an earlier sequence.
- **The model reproduces the evidence.** Run through the model, a direct port reaches ~3× a CPU node per GPU. That is
  close to the only public evidence: Parabricks' GPU STAR is reportedly 8–9× on 4 GPUs, i.e. ~2× per GPU.

**Recommended architecture** (§6): a persistent engine service per GPU that keeps the index resident and streams many
samples through a sort-and-wavefront pipeline:

1. **Ingest:** CPU decodes `.1fq` (67 M reads/s per thread) into pinned buffers. 53 B per pair cross PCIe, about 1% of
   the link.
2. **Dedup:** the GPU sorts reads by sequence and maps each **distinct cDNA sequence once** per batch, which can span
   samples. Barcodes and UMIs are carried, never aligned.
3. **Seed search in wavefront rounds, with STAR's exact semantics:**
   - **Certified seeds:** a seed whose locus is predicted from an earlier seed of the same read is answered from a
     6.4 GB **U-array**. The U-array holds the minimum unique-prefix length of every genome suffix, built from STAR's
     own suffix array in seconds.
   - **Searched seeds:** the remaining seeds run a port of STAR's suffix-array search. Before the search they are
     sorted by prefix and deduplicated exactly with an LCP rule.
   - **Heavy seeds:** repeat-rich seeds go to a warp-cooperative kernel.
4. **Stitching:** STAR's window and stitch logic, ported with compact bounded state. Overflow reads take an exact slow
   path.
5. **Per-sequence annotation:** each distinct alignment is annotated once into a 64-byte record: gene, velocyto class,
   junctions, intron, SNP-panel and chrM overlaps.
6. **Everything after alignment is sort and reduce:**
   - one 64-bit key, (cell, gene, UMI), radix-sorted;
   - UMI 1-mismatch collapse;
   - junction, intron, SNP and chrM tallies;
   - barcode correction with bit-exact float replication.

   No BAM exists anywhere.
7. **On the CPU:** cell calling, doublets, ambient, demux and export.

**Modelled throughput per H100 NVL** (mapping stage, distinct reads, in M read pairs/s; §7):

| scenario | 3′ | 5′ | of which heavy reads |
|---|---|---|---|
| pessimistic (η_mem 0.25, SIMT 0.45) | 53 | 24 | 3′ 28%, 5′ 67% of GPU work |
| base (η_mem 0.35, SIMT 0.60) | 98 | 44 | |
| direct STAR port, no dedup (base) | 6.4 | 2.2 | |
| CPU node, 64 cores (measured) | 1.0–1.3 end to end, ~2.0 mapping | | |

All the other GPU stages model at ≥ 370 M pairs/s.

**The end-to-end bottleneck moves off the GPU.**
- **Host-side callers and export:** cell calling, doublets, ambient, demux and `.s1` export run at W core-seconds per
  sample, on 26 cores per GPU. Depending on W, which is unmeasured:
  - at W = 120 core-s, 66 M-pair samples are capped at 14 M pairs/s per GPU;
  - at the atlas-mean depth of 294 M pairs, the cap is 64 M pairs/s.
- **Input supply.** At 10 M pairs/s, six GPUs finish the 20.6 T-pair atlas in 4 days. The campaign is therefore bound
  by download and conversion, not by GPU compute, beyond about 2 GPUs.

**Verdict on the brief's kill gate (P1: < 5 M pairs/s kills the project).** The modelled rate clears it by 5–20× in
every scenario. The pessimistic 5′ case still models at 24 M pairs/s, which is ≥ 2× the 10 M target. Only the combined
worst case of three assumptions (A1, A7, A8 in §8) reaches the 5 M kill line, and only on 5′.

**Top 5 ideas** (§5 ranks all of them):

| # | idea | estimated gain |
|---|---|---|
| 1 | Map each distinct sequence once (within a sample, then within multi-sample batches) | 2.1× (1/d, d = 0.47 on 3′); batches add 1.1–1.7× (unmeasured) |
| 2 | No BAM: per-sequence annotation records plus GPU sort/reduce for counts, UMIs, barcodes, junctions, SNP and chrM | removes the stages that are 27–71% of CPU wall; on GPU they run at ≥ 370 M pairs/s, so < 5% of GPU time at 10–20 M/s |
| 3 | Wavefront kernels with compact per-read state, instead of a thread-per-read STAR megakernel | 2.7× (3′), 5.7× (5′) over a port |
| 4 | U-array seed certificates with locus propagation (exact) | 1.7× (3′), 1.4× (5′) less mapping work; shorter dependency chains |
| 5 | Persistent per-GPU service with multi-sample batches and pipelined stages | removes per-job reference load (58 s per sample today); enables #1's cross-sample dedup |

**Riskiest assumptions** (§8):
- **A1, random-access rate.** An H100 NVL sustains ≥ 15 G random sectors/s at a 32–60 GB footprint. TLB reach is the
  unknown.
- **A2, seed equivalence.** STAR's SAi plus binary search returns the mathematical maximal-mappable-prefix answer on every
  seed. The certificates and the LCP reuse rely on this.
- **A3, compact state.** Compact per-read state covers ≥ 99.9% of reads.
- **A4, stitcher port.** A port of the recursive stitcher can be made bit-exact: enumeration order, ties, and
  `log2`/`ceil` scoring.
- **A5, host work.** Host work W ≤ ~100 core-s per sample.
- **A6, distinct fraction.** d stays ≤ ~0.6 on 5′ and deep samples.

---

## 1. The contract, and where equivalence is checked

**Outputs** are fixed by the brief (§2): matrices, junctions and PSI, introns and multimapper rules, mtDNA, SNP allele
counts, labels, and non-host.

**The contract is "same outputs as the CPU engine at the pinned commit".** The engine is free to reach them by
different code, provided every shortcut is provably equivalent or falls back to an exact path.

**Equivalence boundaries.** Each is a place where the GPU and CPU engines can be diffed record by record:

| boundary | object | exact by | test |
|---|---|---|---|
| E0 | read → distinct-sequence id | STAR maps one cDNA read from its numeric sequence and length only (`SeqCache.h`: single-end, no chimeric, no WASP, no random multimapper order; qualities, name, barcode and read number unused until output); the key includes clip settings | full-sequence compare after the hash sort; never hash-only |
| E1 | per-read seed list (STAR's `PC` array, in call order) | the GPU produces the same (rStart, L, Dir, Nrep, loci) for every seed, then replays `storeAligns` in STAR's call order | M3/M4: 100% identical PC arrays on dumped real reads |
| E2 | per-read transcripts (windows, stitched alignments, multimapper set, primary) | port of STAR's window, stitch and extend code over identical E1 input | M5: 100% identical transcripts |
| E3 | per-distinct-sequence annotation record (DSA) | the same annotation functions as STARsolo and the pileup, applied to E2 | identical gene, junction and SNP calls per sequence |
| E4 | aggregated outputs | sort/reduce with the same rules (1MM_CR or 1MM_All UMI, `1MM_multi_Nbase_pseudocounts`, SJ filters) | `perf_equiv`, `txf_equiv.py`, `bench/public/` |

**E1 is the key decision.** PR #41's transcriptome-first (L1) failed exactness because it skipped STAR's genome-wide seed
search. A mismatched read can have competing seeds elsewhere. Every GPU shortcut here therefore reproduces STAR's
seed-search **result** exactly. Everything downstream of the seeds is a port of STAR's code. On a GPU that downstream
code is cheap for typical reads (one window, three to seven seeds), so nothing is lost by keeping it exact.

**Determinism rules** (byte-identical across runs, batch sizes and GPU counts):
1. **Per-read results are pure functions of the sequence and fixed parameters.** Batch boundaries cannot change them.
2. **Orderings come from radix sorts on fixed keys.** CUB radix sort is stable and deterministic. Compaction uses
   scan-based select (stable), never atomics that hand out slots.
3. **Integer atomics are allowed only for histograms.** Integer addition is exact and order-free. No floating-point
   atomics.
4. **Floating point that reaches an output is replicated bit-exactly.** Three places need care:
   - STARsolo's barcode posterior: `float pin = count * std::pow(10, -q/10)`. On the GPU this uses a host-computed
     `pow` table.
   - The stitcher's `int(ceil(log2(gLength) * scale ...))`. On the GPU this uses host-computed breakpoints, then a
     binary search.
   - `double` compares in `extendAlign`. These are single IEEE operations and are exact with `-fmad=false`.

   All such kernels compile with `-fmad=false` (no FMA contraction). Per-cell EM and caller floats stay on CPU, in
   canonical order.
5. **Read-order semantics are kept explicitly.** One example: the non-host cap of "the first 20 M unmapped reads"
   becomes a cap on the original read index.

---

## 2. Inputs: what is known

### 2.1 Measured on the CPU engine (real data)

| fact | value | source |
|---|---|---|
| mapping rate | ~32k pairs/s per core (31 µs core time per pair) | brief §3 |
| end to end | ~20k pairs/s per core; 1–1.3 M pairs/s per packed 64-core node | brief §3 |
| cost model | wall = 300 + 12.79·U s at 4 threads (U = M pairs); 316k core-h → mean sample ≈ 294 M pairs; atlas ≈ 20.6 T pairs | audit §3 (derived) |
| profile, 66 M PBMC, idle node | mapping 61%, disk BAM sort 17%, index + CB refresh + pileup 10%, reference load + export 7%, non-host +10% | audit §3 |
| profile at depth | sort 49% (110 M), 60% (5′, 122 M) | audit §3 |
| repeated sequences | ~53% of PBMC 1k v3 reads repeat an earlier read (d ≈ 0.47) | `src/star/SeqCache.h` |
| too-short reads | 3′: 4.1% of reads, 19% of mapping; 5′: 7.4%, **55%** | PR #41 |
| window assignment | 25–28% of mapping; 12.2 G loci for pbmc5p too-short reads alone | PR #41 |
| txome-first ceiling | gene-concordant clean reads 23–24% of mapping; perfect reads ~4% | PR #41 |
| `.1fq` decode | ~67 M reads/s single-thread (zstd + 2-bit unpack) | `docs/format-1fq.md` §11 |
| serial share | S = 358 s serial in the Amdahl fit | audit §3 |

### 2.2 STAR parameters that shape the GPU work

These are defaults in `src/star/parametersDefault`. Singlet sets only solo, clip, `outFilterScoreMin 10` and output
options.

| parameter | value | consequence |
|---|---|---|
| `seedSearchStartLmax` | 50 | a 90-nt read gets `Nstart = 2` start points (0 and 45) per direction |
| `seedMultimapNmax` | 10,000 | seeds with ≤ 10k loci are stored; their loci are walked in window assignment |
| `winAnchorMultimapNmax` | 50 | only seeds with ≤ 50 loci create windows |
| `winBinNbits`, `winAnchorDistNbins`, `winFlankNbins` | 16, 9, 4 | windows are 64 kb bins, merged within 9 bins |
| `seedPerReadNmax`, `seedPerWindowNmax`, `alignWindowsPerReadNmax` | 1,000, 50, 10,000 | STAR's static per-read arrays |
| `outFilterMultimapNmax`, `outMultimapperOrder`, `outSAMprimaryFlag` | 10, Old_2.4, OneBestScore | deterministic multimapper order (SA order of loci, window order) |
| `soloFeatures`, `soloUMIdedup`, `soloCBmatchWLtype` | Gene, 1MM_All, 1MM_multi_Nbase_pseudocounts (set by Singlet) | E4 rules |
| `gSAindexNbases`, `gSAsparseD` | 14, 1 | dense SA, 14-mer direct-address SAi |

### 2.3 Hardware and library figures (public)

| item | value | source |
|---|---|---|
| H100 NVL | 94 GB HBM3, 3.9 TB/s, PCIe Gen5 (128 GB/s bidirectional), NVLink bridge 600 GB/s, FP32 60 TF; 132 SMs and 50 MB L2 per the brief (not on the datasheet) | NVIDIA datasheet [S1] |
| DRAM latency, Hopper | 555 / 744 cycles near / far L2 miss (H800 PCIe); 554 / 728 cycles (H100 SXM5); ≈ 0.3–0.45 µs at ~1.75 GHz | Luo et al. [S2]; arXiv 2607.19922 [S3] |
| L2 | split into two partitions: far hits ~470 vs ~295 cycles near (H100 SXM5); far ≈ 2× near on H100 PCIe | [S3]; Chips and Cheese [S4] |
| random lookups | cuco `static_map` find 16.8 G/s, insert 10.9 G/s (H100 SXM, 4 B/4 B keys, 2^27 pairs, 50% load) | NVIDIA blog [S5] |
| radix sort | Onesweep 29.4 G keys/s (32-bit, A100); in CUB | Adinets & Merrill [S6] |
| PCIe pinned | 53.8 GB/s H2D on H100 HGX (bandwidthTest) | [S7] |
| kernel and graph launch | ~2.5 µs per repeat graph launch on Ampere+; ~1–5 µs per kernel | NVIDIA [S8] |

### 2.4 Prior GPU aligners (what they say about the ceiling)

**Parabricks** (proprietary; black-box figures only):
- `fq2bam` (BWA-MEM) does a 30× WGS in 10.1 min on 2× H100 SXM [S9].
- `rna_fq2bam` (STAR) has published runtimes, but only on RTX PRO 6000 and L4, with no CPU baseline [S10].
- The only comparison with CPU STAR is a forum report of "8–9×" on 4 GPUs, alongside a user who saw no gain [S11].

**Seed-and-extend aligners:**
- SOAP3-dp: 2 to tens of × over BWA [S12].
- BarraCUDA: one GPU ≈ 5 BWA cores [S13].
- nvBowtie on K80: 12.8× over Bowtie2 on 12 threads [S14].

**FM-index backward search:** 2.7–3.8 G LF-steps/s on P100 or GTX Titan [S15]. Each LF step is one random access, so
these rates match a bandwidth-bound random gather.

**Extension kernels:**
- GASAL2: up to 21× over 28 CPU cores [S16].
- AGAThA: 18.8× over CPU [S17].
- WFA-GPU: 4.3× over multithreaded WFA [S18].
- CUDASW++4.0: 5.71 TCUPS on H100 with DPX [S19].

**minimap2 ports:** chaining 2.6–5.4× (mm2-ax on A100, mm2-gb on MI210) [S20, S21].

**What these say.** End-to-end aligner ports land at **2–10× per device**. Isolated kernels (sort, hash, DP) reach
10–100×. The gap is the irregular control flow and state of the full aligner, which is the problem §3 quantifies.

---

## 3. Anatomy of one STAR read, in GPU terms

### 3.1 Seeds per read

`ReadAlign_mapOneRead.cpp` drives the seed search. For a 90-nt read, `Nstart = 90/50 + 1 = 2` and `Lstart = 45`:

| read class (assumed mix, M3 measures) | seed searches (in call order) | of which only a real search can answer |
|---|---|---|
| perfect, 40% | dir 0 from 0 (maps all 90, which sets `flagDirMap=false`); dir 0 from 45; dir 1 from 44 backward | 1 (the first) |
| 1 mismatch at x, 35% | dir 0: 0→x, restart at x (short MMP elsewhere), rest; dir 0 from 45; dir 1 symmetric, plus from 44 | ~3 (first and both restarts) |
| ≥ 2 mismatches or soft clip, 25% | ~9 | ~4.5 |

That averages **5.9 seed searches** per normal read. "Only a real search can answer" counts the seeds where neither the
locus of an earlier seed nor any certificate can predict the result. The restarts land on a mismatching base, so their
maximal mappable prefix (MMP) is a short random match elsewhere in the genome.

### 3.2 Random sectors per seed search (RS = one random 32-byte HBM sector)

The code path is `maxMappableLength2strands` → `maxMappableLength` → `findMultRange` (`SuffixArrayFuns.cpp`):

| step | accesses | RS |
|---|---|---|
| SAi probe(s): 14-mer direct address, plus the next entry for `iSA2` | 2 dependent | 2 |
| binary search in the bucket: each step reads SA[i] (33-bit packed: 1.125 RS) then the genome word at the suffix (unaligned 8–24 B: ~1.1 RS) | ~8 steps (read-weighted bucket ~2^8; GRCh38 has 5.9e9 suffixes / 2.68e8 14-mers = 22 on average) | ~18 |
| `findMultRange` × 2: interval boundaries | ~2 × 3 steps | ~14 |
| store (PC array) | | ~1 |
| **total per searched seed** | **~26 dependent steps** | **~35–37** |

**Calibration against the CPU:**
- **Per read:** 5.9 searched seeds × 37 RS, plus windows, stitch and annotation ≈ **258 RS per normal read**.
- **Heavy reads:** the too-short and repeat class costs 5.5× (3′) or 15.3× (5′) a normal read (PR #41). That gives
  ~306 RS per 3′ read on average.
- **Check against CPU time:** at ~100 ns per DRAM miss on a CPU core (with modest prefetch overlap), 258–306 misses take
  26–31 µs. That matches the measured 31 µs of core time per pair, so the RS count is the right order.

### 3.3 Why a direct port stalls: per-read state and divergence

**STAR's per-read structures, as allocated in `ReadAlign.cpp`:**

| structure | size per `ReadAlign` | typical use |
|---|---|---|
| `winBin[2][winBinN]` (uint16; GRCh38 has winBinN ≈ 48.8k) | 195 KB, memset per read (lazily) | ≤ 3 windows |
| `PC[seedPerReadNmax=1000][8]` | 64 KB | 3–10 seeds |
| `WA[10000][50][7]` | 28 MB | ≤ 20 aligns |
| `trArray` + `alignTrAll`: 2 × 10,000 `Transcript` (~1.4 KB each, `MAX_N_EXONS=20`) | ~28 MB | 1–10 transcripts |
| `stitchWindowAligns` recursion: `Transcript trA` passed **by value** at every level | ~1.4 KB per level | depth ≤ number of aligns |

**State.** A GPU has ~50–270k resident threads. Even 2 KB per thread spills to local memory, which is HBM through L1.
The direct port therefore turns cache-resident CPU state into HBM traffic.

**Divergence.** In a thread-per-read megakernel, a warp runs as long as its slowest read:

    η_div = E[cost] / E[max of 32 costs]
          ≈ (1 − h + h·k) / (P(heavy in warp)·k + (1 − P)·3)

Here h is the heavy-read share, k their relative cost, and 3 is the assumed max-of-32 spread among normal reads.

| chemistry | h | k | P(≥1 heavy in a warp of 32) | η_div |
|---|---|---|---|---|
| 3′ | 0.041 | 5.5 | 1 − 0.959^32 = 0.74 | 1.18 / 4.85 = **0.24** |
| 5′ | 0.074 | 15.3 | 1 − 0.926^32 = 0.91 | 2.06 / 14.25 = **0.14** |

**The model reproduces the Parabricks ratio.** A port runs at about 23 G RS/s × η_mem 0.35 × η_div 0.24 ≈ 2 G RS/s per
GPU. A 64-core node runs at about 2 M pairs/s × 306 RS ≈ 0.6 G RS/s. That is ~3× a node per GPU, close to the reported
8–9× on 4 GPUs. **The port is not limited by memory bandwidth.** It is limited by divergence, state and duplicated work.
Those are the three things to engineer away.

### 3.4 The rest of the read

**Windows and assignment.**
- Reads the contiguous SA range of each seed: `Nrep` × 4.1 B.
- On a GPU this is coalesced. 10,000 loci are 41 KB, about 13 ns of HBM bandwidth.
- The CPU cost is the per-locus window test (PR #41). On a GPU that test is a binary search over ≤ 8 windows held in
  registers.

**Stitch and extend.**
- Touches 100–300 B of genome around each window, plus sjdb lookups: ~16 RS.
- Compute is ~10–40k thread instructions per read, most of it in `stitchAlignToTranscript`'s junction scan.

**Instruction bound.** 132 SMs × 64 INT32 lanes × 1.785 GHz = 15 T int ops/s. At 30% SIMT and issue efficiency and
~30k instructions per distinct read, that is ~150 M distinct reads/s. Not binding, but within 2–3× of the memory bound,
so instruction count matters for heavy reads.

---

## 4. Ideas considered

Each entry gives the mechanism, its cost in bytes or RS, the exactness argument, and the estimated effect. "Gain" is on
the mapping stage, from `throughput_model.py`, unless stated otherwise.

### 4.1 Map each distinct sequence once (within a sample) — KEEP, #1

- **Mechanism.**
  - Pack R2 to 2 bits, three 64-bit words for ≤ 96 nt, after the per-sample clip.
  - Compute a 64-bit hash, then a CUB radix sort of (hash, read index) on 40 key bits.
  - Compare the full sequences of neighbours, so collisions are split exactly. That gives `seq_id` per read and a
    distinct list.
  - Barcodes and UMIs ride along as payload; they are never aligned.
- **Cost.**
  - ~190 B of contiguous traffic per read: read 24, hash 8, five passes × 12 B × 2, compare 24, permutation 8.
  - At 3.9 TB/s × 0.6, that is **12 G reads/s**, or ~0.1% of GPU time at 10 M/s.
  - Memory: 12 B × 2 per read in flight.
- **Gain.** 1/d, with d = 0.47 on PBMC 1k v3 (SeqCache): **2.1×**. Deeper samples have lower d.
- **Exactness.** E0 (§1). The key is (clip settings, original length, clipped sequence). Reads with N are split by
  `qualitySplit` exactly as STAR does, and the split pieces are part of the key.
- **Determinism.** Distinct ids are ranks in (hash, sequence) order, which is independent of the batch.

### 4.2 Dedup across samples — KEEP (batch), MEASURE (atlas table)

**(a) Multi-sample batches.** Put K samples of the same species, chemistry, clip and read length into one batch.
Sequences shared between samples are mapped once.
- Exact for free: E0 holds across samples with the same key.
- Gain 1/(1 − s), where s is the share of a sample's distinct sequences already present in the other K − 1. If s is
  0.1–0.4, the gain is **1.1–1.7×**. s is unmeasured (M2 measures it).

**(b) Persistent atlas answer table.** The table maps a sequence key to its DSA record, keyed by engine version,
reference, parameters and clip.
- **Storage:** 1e9 entries × 48 B = 48 GB, in host RAM (768 GB).
- **Lookup:** sort the batch's distinct keys, then merge-join against the table streamed through PCIe:
  48 GB / 50 GB/s ≈ 1 s per pass. One pass serves a multi-sample batch of ~500 M reads (30–60 s of GPU work), so the
  overhead is ≤ 3%.
- **Gain:** 1/(1 − h), with h the hit rate (unknown). If h = 0.5, the gain is 2×.
- **Exactness and determinism:** by construction. A table hit returns the record the same engine produced, and outputs
  are byte-identical with or without the table.
- **Cost:** 4 weeks of engineering, plus versioning and invalidation.
- **Verdict:** decide after M2/M14. If h < 0.3 it does not pay for its complexity.

### 4.3 Wavefront kernels with compact state — KEEP, the foundation

**Mechanism.** Replace "one thread runs STAR's `oneRead` for one read" with a pipeline of short kernels over SoA queues,
in the style of GPU path tracers:
1. Generate the seed queries of all active reads.
2. Certify (§4.4).
3. Prefix-sort and reuse (§4.5).
4. Search, split by class: normal buckets thread-per-seed; heavy buckets or `Nrep` > 512 warp-per-seed.
5. Replay `storeAligns` per read, in STAR's call order.
6. Generate restart seeds. Repeat until no read is active (≤ ~10 rounds for 90 nt).
7. Windows and assignment.
8. Stitch: thread-per-window, with heavy windows warp-cooperative.
9. Per-read selection: multimappers, primary, filters.

Compaction between steps uses CUB `DevicePartition`, which is stable.

**State per in-flight read** (fixed caps; global SoA unless stated):

| field | bytes |
|---|---|
| packed read | 24 |
| split pieces (≤ 4) | 16 |
| seed list (≤ 24 entries × 16 B) | 384 |
| counters | 32 |
| windows (≤ 8 × 16 B) and aligns (≤ 16 per window × 16 B), during stitch only, in shared memory | ≤ 2 KB per window |

A read that exceeds any cap is flagged and re-run by an **overflow kernel** with STAR-sized global scratch (warp per
read). M3 must show this is rare (≤ 0.1%).

**Gain** comes from removing divergence and spills: η_div goes from 0.24/0.14 to ~0.6 (assumed). Heavy reads also get a
contiguous-scan kernel (assumed 2× fewer effective RS). That is **2.7× (3′), 5.7× (5′)** over the port.

**Exactness.** Each step is a port of STAR's function, with the same inputs in the same order:
- `storeAligns` replay uses the original call sequence. Its "same rStart and L → drop" rule makes the call order
  observable: dir 0 is kept over dir 1.
- Windows: `winBin` becomes a sorted interval list with STAR's merge, left-overwrites-right and kill semantics.
- Stitching uses an explicit stack in the same enumeration order (include before exclude) and the same tie rules.
- The `log2`/`ceil` score uses host breakpoints (§1, rule 4).

**Effort:** high, about 8 engineer-weeks for the mapper core. This is where most of the equivalence risk sits (A3, A4).

### 4.4 U-array seed certificates with locus propagation — KEEP

This is the exact form of the brief's "certified fast path". It is written down in `bench/gpu/ucert.cu`.

**Definition.**
- The text is STAR's: forward G plus reverse-complement G, the strands that SA indexes, including the sjdb inserts.
- For every suffix start t, with i = ISA[t]:
  - `lcp_prev(i)` = LCP of the suffixes at SA[i−1] and SA[i], over A/C/G/T only, capped at 254;
  - **U[t] = 1 + max(lcp_prev(i), lcp_prev(i+1))**, the shortest prefix length that occurs only at t;
  - U = 255 means "do not certify".

**Certificate.** Take a seed query Q and a candidate t. Let m = lcp(Q, text[t..]). Suppose m ≥ U[t], and m = |Q| or
Q[m] ≠ text[t+m]. Then STAR's seed result is L = m, Nrep = 1, loci = {t}.

**Proof.**
1. Any suffix sharing ≥ m ≥ U[t] bases with Q shares U[t] bases with suffix t. Only t has that prefix: by the
   definition of U and the sortedness of SA, the LCP with any non-neighbour is ≤ the LCP with a neighbour.
2. Q's (m+1)-prefix fails at t and therefore everywhere.

So the answer is the mathematical MMP. STAR returns the same answer whenever its SAi and binary search are correct for Q
(assumption A2, which M4 tests on every seed).

**Locus propagation.** Seeds never contain N (`qualitySplit` cuts reads at N). Once one seed of a read is answered as a
single locus t0, by search or by the locator (§4.7), the other seeds of the read have known read offsets:
- the next seed's candidate is t0 + offset;
- a reverse-direction seed's candidate is the mirrored coordinate on the other strand.

A wrong hypothesis (indel, splice, another locus) fails m ≥ U[t] and falls back to the search. **No heuristic can change
an answer; it can only change which code path computes it.**

**Cost.**
- 1 RS for U[t]. The verify compare reuses genome words the previous seed already loaded (≤ 1 RS).
- So **~2 RS and 2 dependent steps per certified seed**, vs ~37 RS and ~26 steps searched.

**Memory.** 2 × nGenome bytes = **6.4 GB**, or 3.2 GB with 4-bit U (values 12–26, 15 meaning "no"). There is no ISA:
the certified seed carries its locus directly (tagged "direct" in the PC entry). Downstream code only ever reads
`SA[iSA]` to get the locus, and the locus is unique.

**Build.** One GPU pass over STAR's SA computes adjacent LCPs: two genome windows per pair, about 3e10 random sectors in
total, a few seconds. A second pass scatters U by text coordinate (5.9e9 random 1-byte writes, seconds). The build also
re-checks SA sortedness for free. It needs SA, G and U on the GPU at once: 24 + 1.6 + 6.4 GB.

**Coverage (modelled).** Of the 5.9 seeds per normal read, ~3.3 are certifiable:
- every non-first seed that does not start on a mismatch;
- including intronic and intergenic reads, because the U-array covers the whole text, not a hot set.

**Gain:**
- normal read 258 → 142 RS;
- per pair, 130 → 78 RS (3′), 193 → 139 RS (5′);
- that is **1.67× (3′), 1.39× (5′)**;
- dependency chains also fall from ~150 to ~70 steps per read.

**Why this differs from L1.** L1 inferred the read's alignment from a transcript-level match. That is not provable,
because a mismatched read can have competing seeds elsewhere. Here every seed STAR would search keeps exactly its
genome-wide result. Only seeds whose answer is provably unique are skipped, and the restart seeds after mismatches,
which produce the competing alignments, are always searched.

### 4.5 Prefix-sorted seed wavefronts with exact LCP reuse — KEEP

**Mechanism.** In each round, sort the unanswered seed queries by their first 64 bits (32 bases), then by full sequence.
A seed's result (L, interval) depends only on its first L + 1 bases when L < |Q|. So in a sorted run, a follower q2 with
lcp(q1, q2) ≥ L1 + 1 and |q2| ≥ L1 + 1 has **exactly** q1's answer:
- q1's (L1+1)-prefix occurs nowhere;
- q2 has that same prefix, so q2's MMP is ≤ L1;
- q2's L1-prefix equals q1's, so it occurs, at the same interval.

Leaders are elected with a segmented scan, searched, and their answers broadcast to followers.

**Side effect.** Sorted queries walk SAi and the SA in lexicographic order, so neighbouring threads share SAi lines and
the top steps of the binary search: free coherence (§4.9).

**Where reuse comes from:**
- distinct reads that start at the same position and differ only downstream (sequencing errors at depth);
- reads sharing a polyA, adapter or TSO tail (5′);
- restart seeds that land in the same repeat.

**Cost:** sort 12 B × 8 passes per unanswered seed, ~200 B contiguous; negligible.

**Gain:** if 30% of searched seeds are reused (assumption; M2's LCP proxy and M11 measure it), **1.2× (3′), 1.1× (5′)**.
**Exactness:** proved above. **Effort:** 2 weeks.

### 4.6 SA bucket sample keys (SBK) — KEEP, cheap

**Mechanism.** Store the first 64 bits (32 bases) of every 16th suffix in SA order: 5.9e9 / 16 × 8 B = **2.95 GB**. A
search finds its SAi bucket, then binary-searches the bucket's sample keys. These are contiguous: a 256-entry bucket has
16 samples in 128 B, 1–4 sectors, with no genome dereference. It then finishes with ≤ 4 genome-compare steps.
`findMultRange` is accelerated the same way.

**Cost:** ~24 RS and ~10 dependent steps per searched seed, vs 37 and 26.

**Exactness:**
- Samples are exact 32-base prefixes, so comparisons on them are exact.
- A tie at 32 bases falls through to genome compares.
- The result is the same interval as STAR's search, given A2.

**Gain:** 1.2× (3′), 1.09× (5′). **Effort:** 1.5 weeks.

### 4.7 Locator for the first seed — LATER

**Mechanism.** A hash table of sampled k-mers (k = 24–32, minimizer sampling w ≈ 16) maps to text positions. It
proposes t0 for the first seed of each read. About 5 independent probes, plus a verify, cost ~6 RS with one latency.
The U-array then certifies.

**Memory and gain.** ~6–8 GB. Gain is 1.1× (3′) and 1.04× (5′), because searched seeds after locus propagation are
mostly restarts, which a locator cannot answer.

**Exactness:** the same as §4.4; the locator only proposes. **Verdict:** after M4 shows how much first-seed search time
is left.

### 4.8 Heavy-seed kernel: repeats and too-short reads — KEEP

**Profile.** On 5′, heavy reads are **67% of the remaining GPU work** in the recommended design; on 3′, 28%. Their cost
on CPU is deep binary searches in huge 14-mer buckets (Alu, polyA, TSO) and loci loops over up to 10,000 loci per seed.

**On the GPU:**
- **Searches:** warp-cooperative binary search, with 32 lanes probing 32 split points per step. Each step cuts the range
  by 33× instead of 2×, so a 2^20 bucket takes 4 steps instead of 20. SBK and LCP reuse also apply.
- **Loci walks:** contiguous SA reads (41 KB per 10k-locus seed). Each lane tests a locus against the read's ≤ 8 windows,
  held in shared memory.
- **Repeat answer tables:** heavy seeds recur within a batch (the same polyA, adapter or Alu stretches). LCP reuse
  (§4.5) catches them. Tables precomputed per reference for the top repeat k-mers are possible, but need A2-level
  verification; they are rejected for now.

**Gain:** the model assumes heavy reads cost half of the port's RS (k × 258 / 2). Profiling-based tuning could gain
1.5–2× more on 5′. **Exactness:** the same functions, in a different parallel decomposition. `storeAligns` and window
order are preserved by sorting each lane's results by locus index before the replay.

### 4.9 Coherent query order — KEEP, free

The brief's "sort queries by k-mer" idea. Dedup already sorts reads, and §4.5 sorts seeds lexicographically. The gain
is L1/L2 reuse of SAi lines and top binary-search steps among neighbours sharing ≥ 14 bases.

**Gain is modest after dedup.** Most coherence came from identical reads, which dedup removes. Assumed ≤ 1.1–1.3× on
searched seeds and not counted in the model. It is measured as a side result of M3 (sorted vs unsorted).

### 4.10 No BAM: per-sequence annotation plus GPU sort/reduce — KEEP, #2

**Mechanism.** For each distinct sequence, the alignment results feed one **DSA record** of 64 B:

| field | bytes |
|---|---|
| flags (mapped class, NH, nM, AS, strand, chrM, too-short marker) | 8 |
| primary locus (33-bit text coordinate) and compact blocks (≤ 4 × (rStart u8, len u8, gDelta u32)) | 32 |
| STARsolo Gene id (u32) and GeneFull id | 8 |
| velocyto class (2 bits), intron and multimapper (#46) class | 4 |
| junction ids (2 × u32; overflow list) | 8 |
| offset into a side list of SNP-panel and chrM overlaps (positions + read offsets) | 4 |

Multimappers (NH > 1) keep a side list of all alignments. The DSA is a pure function of the sequence, so it is computed
once per distinct sequence.

**Per read:**
- The read's `seq_id` gathers its DSA. This is coalesced: reads are already grouped by sequence, so each DSA is read
  once and broadcast.
- From that, emit one 64-bit key: **(cell 23 bits | gene 17 | UMI 24)**.
  - v3 whitelist: 6.79 M < 2^23. GENCODE genes: < 131k. UMI: 12 nt.
  - A larger whitelist would use 128-bit keys, at 2× a negligible cost.
- Sort with CUB (8 passes × 16 B × 2 = 256 B per read).
- UMI collapse per (cell, gene) segment:
  - 1MM_All: connected components with min-label union-find, deterministic.
  - 1MM_CR / directional (pileup): STAR's and N6's tie rules.
  - Candidate pairs come from the **split-halves trick**: two 12-nt UMIs at Hamming ≤ 1 share one 6-nt half exactly.
    So sort by each half and test within the groups. This is exact, O(n log n), and needs no all-pairs pass.
- Segmented reduce to CSR.

**Junctions, SJ.out, PSI, introns, SNPs, chrM.**
- Junction tallies come from DSA ids.
- SNP and chrM tuples (position, allele, cell, strand, binned quality) are emitted only for reads whose DSA lists
  overlaps.
- These are tuples, then a sort, then a reduce. SJ filters (`outSJfilter*`) run on the small reduced table on CPU.

**Cost.**
- ~270 B contiguous plus ~2 RS per pair for counts; ~200 B for pileup tuples.
- **≥ 2.6 G pairs/s** and **≥ 11 G pairs/s** respectively (base).

**Gain.**
- On CPU, the sorted BAM, the index, the CB refresh and the pileup are **27–71% of wall** (17–60% sort plus 10–25%
  rest, audit §3).
- On GPU these stages cost < 1% of GPU time at 10–20 M/s.
- **Kept on CPU, the pileup would cap the GPU engine.** The pileup pass alone runs at about 66 M / (85 s × 4 cores) ≈
  0.19 M pairs/s per core, or ~5 M pairs/s on 26 cores. That is below target, so this idea is mandatory.

**Exactness.** E3 and E4. This is the same route as rollout §3b #1 (streaming, order-independent callers). The GPU
version gets order independence by construction, because everything is sorted on fixed keys.

### 4.11 Barcode correction on the GPU — KEEP

**Steps:**
1. **Exact lookup.** The whitelist goes into a cuco `static_map`: 6.8 M keys × 8 B / 0.5 load = 109 MB. Lookup is
   1 RS per read.
2. **Pass-1 counts.** Exact-match counts per whitelist entry are an integer-atomic histogram, which is order-free.
3. **`1MM_multi_Nbase_pseudocounts`** for the ~5–10% of non-exact barcodes (assumed):
   - 48 neighbour probes, or N-position variants;
   - the posterior is `pin = (float)(count × pow10[q])`, accumulated in STAR's candidate order with `-fmad=false`;
   - `pmax ≥ cbMinP·ptot` is tested in float;
   - `pow10[]` is computed on the host with `std::pow`, which is **bit-exact** to `SoloReadFeature_inputRecords.cpp`.

**Cost.** ~3.9 RS per read; **≥ 2 G reads/s**. Correction runs at sample end, once pass-1 counts are complete. Alignment
does not depend on the barcode, so nothing waits on it.

### 4.12 Non-host screen on the GPU — KEEP (small)

**Pipeline:**
1. Unmapped distinct reads, with multiplicities, go through k = 21, w = 11 minimizers.
2. The host Bloom filter (313 MB) removes host k-mers.
3. Query minimizers are sorted.
4. They are merge-joined with the 25 GB index, either resident in HBM (8 ms per scan at 3 TB/s) or streamed from host
   RAM once per multi-sample batch (0.5 s per pass).
5. Per-read hits and multiplicities go to the existing CPU EM.

**Exactness.**
- Same hash (`murmurhash64A`) and the same minimizer rule as `min_sketch.h`.
- The 20 M cap becomes "unmapped reads with original index rank < 20 M".
- EM stays on CPU, in canonical order.

**Cost.** < 0.5% of GPU time (≥ 370 M pairs/s bound). **Stage 2**, per-species genome mapping across samples, reuses
the mapper with a small index.

### 4.13 Ingest and host pipeline — KEEP (CPU decode)

**CPU decode, not GPU decode.** `.1fq` decodes at 67 M reads/s per thread (measured), so 1–3 threads per GPU suffice.
nvCOMP zstd on the GPU is unnecessary. There are also no published H100 zstd figures, and it would compete for SMs.
**Rejected unless host cores become the bottleneck.**

**Pipeline:**
- pinned ring of 4 × 256 MB buffers;
- `cudaMemcpyAsync` on a copy stream: 53 B per pair (R2 24, CB+UMI 7, binned quality 23), so 10 M pairs/s = 0.53 GB/s,
  about 1% of 53.8 GB/s;
- events gate the compute stream;
- one CUDA Graph per batch: ~50 kernels × 3 µs = 150 µs per batch, negligible for 8–16 M-read batches.

**Storage is the real input risk.** Each GPU needs `.1fq` at about 20 B per pair, which is 0.2 GB/s at 10 M/s. Stage to
local NVMe ahead of time. Shared NFS throughput on Clipper is unmeasured, so it is part of M8.

### 4.14 Where things run; multi-GPU — KEEP (independent GPUs)

**Placement.**
- **GPU:** everything from dedup to sparse matrices.
- **CPU:** decode, orchestration, cell calling (`called_ed`), ambient, doublets, demux, sex and ancestry labels, EM,
  and `.s1` export. Those consume small per-cell data.
- They pipeline: the GPU maps sample n + 1 while the CPU runs callers on sample n.

**HBM budget, GRCh38:**

| item | GB |
|---|---|
| STAR SA, 33-bit packed (unchanged) | 24.3 |
| SAi, 14-mer | 1.5 |
| genome, forward + reverse complement, 2-bit, plus N bitmaps | 1.6 |
| U-array (8-bit) | 6.4 |
| SBK | 2.95 |
| annotation, sjdb, SNP panel, chrM, whitelist, host Bloom | ~1.0 |
| **core index** | **~38** |
| optional locator | +6–8 |
| optional resident non-host index | +25 |
| batch working set (16 M pairs in flight, plus per-sample count tuples up to 260 M pairs × 8 B, double-buffered) | 10–14 |

The core index plus batches is **~50 GB**. Adding the resident non-host index gives ~75–80 GB, which fits in 94 GB. With
4-bit U and the non-host index streamed it is ~45 GB.

**Species placement.** Keep both species' indexes in pinned host RAM: ~38 GB each of 768 GB. Swap one into a GPU in
< 1 s (38 GB / 53.8 GB/s), and schedule batches by species.

**Multi-GPU.** The two GPUs of a node run **independently**, with samples as the unit. No cross-GPU reduction exists, so
determinism across GPU counts is trivial.

**NVLink is not needed.** Splitting the index across the pair was rejected: random accesses over NVLink cost ~µs and
the link has ~1/13 of HBM bandwidth. Peer-streaming a shared non-host index is possible, but unnecessary at 25 GB.

### 4.15 Persistent service with multi-sample batching — KEEP, #5

**What it removes.** Today each job loads the genome and reference: 58 s with export, and the fixed cost in the model is
130–300 s. A per-GPU service loads the index once and streams samples. Per-sample fixed GPU cost falls to the
sample-end steps: barcode correction, the count sort and D2H. That is under 0.1 s for 66 M pairs.

**What it enables.** Cross-sample dedup (§4.2a), and overlap between CPU callers and GPU mapping.

**Exactness.** Unaffected, because per-sample outputs come from per-sample keys.

### 4.16 Rejected ideas, with reasons

| idea | why rejected (quantified) |
|---|---|
| FM-index (backward search) instead of SA | one random access per base: ~45–90 dependent LF steps per seed vs ~26 for SA. Locate needs a sampled SA anyway, and STAR's SA order must be kept for multimapper order. k-step FM halves the steps but quadruples occ tables. Published GPU rates (2.7–3.8 G LF/s) imply ~40–80 M seeds/s, no better than the SA path. |
| minimizer/chaining aligner (minimap2-style) as the primary seeder | its seeds are not STAR's MMPs, so E1 equivalence is impossible. Kept only as a *locator* that proposes candidates (§4.7). |
| deeper SAi (16-mer direct table) | 4^16 × ~4.5 B ≈ 19 GB to save ~2 binary-search steps; SBK gets more for 3 GB |
| learned or interpolation index over SA | needs a key array of suffix prefixes (47 GB at 64 bits per suffix); SBK is the sampled version of the same idea at 1/16 the size |
| full inverse suffix array for certificates | 24 GB, and unnecessary: certified seeds carry their locus directly (§4.4) |
| transcriptome-first (L1) | not exact (PR #41); replaced by seed-level certificates |
| whole-read certificates (skip the stitcher for perfect reads) | the stitcher is cheap on GPU for 1-window reads (≤ ~20 RS); the exactness argument would have to cover STAR's scoring and filters; gain < 1.1× |
| atlas junction table as preferred anchors | changes alignments relative to STAR (not in STAR's index): violates the contract. Possible later as a "better than STAR on real truth" mode. |
| sparse SA (`gSAsparseD` > 1) to save HBM | changes STAR's results (the `iDist` loop): must use the reference index parameters |
| index in host memory (zero-copy or unified memory) | each random 32 B access becomes a PCIe transaction of ~1 µs; capacity is not the problem (≤ 50 GB fits) |
| index split across the NVLink pair | remote random access is slow; bandwidth ~1/13 of HBM; both species fit by swapping in < 1 s |
| tensor cores | nothing maps: integer, branchy compare and stitch, no dense DP. CUDASW++'s Hopper gain comes from DPX, not tensor cores, and STAR has no DP kernel worth it. |
| Myers / BitPAl bit-parallel edit distance | STAR has no edit-distance step; the mismatch test is XOR + popcount on 2-bit words (3 words per read) |
| GPU zstd decode (nvCOMP) | CPU decode is 67 M reads/s per thread; no H100 zstd figures are published; it would take SMs. Revisit only if host cores bind. |
| float atomics, or atomic slot assignment, anywhere in outputs | violates determinism; replaced by sort and scan |

---

## 5. Ranking: expected gain × probability ÷ effort

**How the columns are defined.**
- **Gain** is the multiplicative effect on the stage it targets, from §4 and the model.
- **p** is the probability that the gain is realised with exactness intact.
- **Effort** is in engineer-weeks, at Sonnet-assisted pace.
- **Score = (gain − 1) × p ÷ effort.**
- Foundations are needed regardless of score.

| rank | idea | gain | p | effort (wk) | score | note |
|---|---|---|---|---|---|---|
| 1 | 4.1 distinct-sequence dedup | 2.1× mapping | 0.95 | 1 | **1.05** | |
| 2 | 4.10 no BAM, DSA + sort/reduce | ≥ 5× on post-alignment (CPU pileup would cap at ~5 M/s) | 0.8 | 8 | **0.40** | foundation |
| 3 | 4.15 persistent service + multi-sample batches | 1.3× e2e on small samples; enables 4.2a | 0.95 | 1 | **0.29** | |
| 4 | 4.3 wavefront kernels, compact state | 2.7× (3′) to 5.7× (5′) | 0.7 | 8 | **0.15–0.41** | foundation |
| 5 | 4.2a cross-sample batch dedup | 1.1–1.7× (s unmeasured) | 0.6 | 1 | **0.24** (at 1.4×) | |
| 6 | 4.9 coherent query order | 1.1–1.3× searched seeds | 0.5 | 0.5 | 0.20 | free with 4.5 |
| 7 | 4.4 U-array certificates | 1.67× (3′), 1.39× (5′) | 0.7 | 3 | 0.16 | |
| 8 | 4.8 heavy-seed kernel (beyond the modelled 2×) | 1.5× (5′) | 0.5 | 3 | 0.08 | |
| 9 | 4.6 SBK | 1.2× (3′), 1.09× (5′) | 0.6 | 1.5 | 0.08 | |
| 10 | 4.5 LCP seed reuse | 1.2× (3′), 1.1× (5′) | 0.5 | 2 | 0.05 | |
| 11 | 4.2b atlas answer table | 1.3–2× | 0.4 | 4 | 0.05–0.10 | decide after M14 |
| 12 | 4.12 non-host on GPU | 1.1× e2e | 0.9 | 2 | 0.05 | |
| 13 | 4.7 locator | 1.1× (3′) | 0.5 | 3 | 0.02 | later |

**How to read the ranking.**
- The first five give ~15–30× over a direct port on the mapping stage (model, base).
- They are also what makes the post-alignment work cheap.
- Items 6–13 are refinements, each worth ≤ 1.7×. They matter mainly for 5′ and for headroom against the pessimistic
  scenario.

---

## 6. Recommended architecture, end to end

```
 host (per GPU: 26 cores)                         GPU (H100 NVL, index resident ~38 GB)
 ────────────────────────                         ──────────────────────────────────────────────────────────
 .1fq on local NVMe ─► decode (2-3 thr) ─► pinned ring ══PCIe 53 B/pair══► K0 unpack (R2 2-bit, CB+UMI, binned Q)
                                                                            │
                                                     K1 hash + radix sort + compare ─► seq_id per read, distinct list
                                                     (multi-sample batch: same species/chem/clip/length)
                                                                            │
                                                     K2 barcode exact lookup + pass-1 histogram (int atomics)
                                                                            │  distinct sequences only
                                                     K3 qualitySplit + clip ─► active reads (compact state)
                                                         ┌──────────────── seed rounds (≤ ~10) ───────────────┐
                                                         │ K4 emit seed queries (known starts; restarts)      │
                                                         │ K5 U-array certificate (locus propagation)         │
                                                         │ K6 prefix sort + LCP leader election               │
                                                         │ K7 SA search: thread/seed │ warp/seed (heavy)       │
                                                         │ K8 broadcast + storeAligns replay (call order)      │
                                                         └────────────────────────────────────────────────────┘
                                                     K9  windows + assignment (thread/read; warp for big Nrep)
                                                     K10 stitch + extend (thread/window; overflow kernel)
                                                     K11 multimapper select, primary, filters ─► transcripts (E2)
                                                     K12 annotate ─► DSA record per distinct sequence (E3)
                                                     K13 expand per read ─► (cell,gene,UMI) keys, junction/SNP/chrM tuples
                                                     K14 unmapped distinct ─► minimizers ─► non-host merge-join
 sample end (stream 2, overlaps next sample's mapping):
                                                     K15 1MM barcode correction (float-exact) ─► rekey
                                                     K16 radix sort, UMI 1MM collapse, segmented reduce ─► CSR
                                                     K17 junction / intron / SNP / chrM reductions
 callers + export (CPU) ◄══ D2H sparse matrices, tallies (~100 MB/sample) ══
 called_ed, ambient, doublets, demux, sex/ancestry, non-host EM, .s1 writer
```

**Data structures** (byte layouts in §4.3, §4.4, §4.6, §4.10, §4.14):
- **Reference:** STAR's SA and SAi unchanged; 2-bit genome, forward and reverse complement; U-array; SBK.
- **Per in-flight read:** ≤ 456 B of SoA state.
- **Per distinct sequence:** a 64 B DSA, plus side lists.
- **Per read after mapping:** one 64-bit count key, plus optional pileup tuples.

**Scheduling:**
- **Batches:** 8–16 M reads, from one or more samples of the same class.
- **Streams:** three per GPU (copy, map, sample-end), with CUDA Graphs per batch.
- **Wave size:** the seed rounds run over the batch's distinct reads, 4–8 M. That is far above the ~10k in-flight
  accesses Little's law needs to hide 0.4 µs at 23 G RS/s.

**Debugging.** A `--emit-sam` debug path renders SAM records from E2 for a chosen subset of reads. It replaces
`--keep-sorted-bam` for IGV.

**Ownership.**
- The engine lives in `src/gpu/` (W11).
- STAR code is ported, not edited in place. `src/star/` stays owned by W2 and W3.
- The equivalence dumper (§9, M3) is a W11 tool that links the vendored STAR objects.

---

## 7. End-to-end throughput model

Run `python bench/gpu/throughput_model.py`. The constants are in the file. RS_clean = 23 G random sectors/s; η_mem is
the fraction of RS_clean a complex kernel achieves; SIMT is η_div for the wavefront kernels.

### 7.1 Mapping stage, distinct reads (M read pairs/s per H100 NVL)

| design (cumulative) | 3′ pess. | 3′ base | 3′ opt. | 5′ pess. | 5′ base | 5′ opt. | RS/pair 3′ / 5′ |
|---|---|---|---|---|---|---|---|
| STAR port, megakernel, no dedup | 4.6 | 6.4 | 9.2 | 1.6 | 2.2 | 3.1 | 306 / 531 |
| + read dedup | 9.8 | 13.7 | 19.6 | 3.1 | 4.4 | 6.3 | 144 / 266 |
| + wavefront, compact state, heavy kernel | 19.9 | 37.2 | 61.9 | 13.4 | 25.1 | 41.8 | 130 / 193 |
| + U-array certificates | 33.4 | 62.3 | 103.8 | 18.7 | 34.8 | 58.0 | 78 / 139 |
| + LCP seed reuse | 40.0 | 74.7 | 124.5 | 20.6 | 38.5 | 64.2 | 65 / 125 |
| + SBK | 47.8 | 89.3 | 148.8 | 22.6 | 42.1 | 70.2 | 54 / 115 |
| + locator | 52.7 | 98.3 | 163.9 | 23.6 | 44.1 | 73.5 | 49 / 110 |

**Worked example: 3′, base, recommended design.**
- RS per pair: 49.1 = d 0.47 × (0.959 × 77 RS normal + 0.041 × 5.5 × 258 / 2 RS heavy).
- Effective rate: 23e9 × 0.35 × 0.60 = 4.83 G RS/s.
- Throughput: 4.83e9 / 49.1 = **98 M pairs/s**.
- Instruction bound: 15.1e12 × 0.3 / (30k × 0.47) = 321 M pairs/s, so not binding.

### 7.2 Every stage (base scenario, M pairs/s; the lowest bound is the bottleneck)

| stage | where | per-pair work | 3′ | 5′ |
|---|---|---|---|---|
| storage read of `.1fq` (~20 B per pair) | host | 0.2 GB/s per GPU at 10 M/s | NVMe ~75; NFS unmeasured | same |
| `.1fq` decode, 3 threads | host | 67 M reads/s per thread (measured) | 201 | 201 |
| H2D | PCIe | 53 B | 943 | 943 |
| dedup (hash, sort, compare) | GPU | ~190 B contiguous | 8,533 | 8,357 |
| barcode lookup + 1MM | GPU | 3.9 RS | 2,075 | 2,075 |
| **mapping (distinct reads)** | GPU | 49 / 110 RS | **98** | **44** |
| expand + count sort + UMI collapse | GPU | ~300 B + 2 RS | 2,643 | 2,643 |
| pileup tuples (junctions, SNP, chrM) | GPU | ~200 B | 11,700 | 11,700 |
| non-host (unmapped distinct) | GPU | 12–16 RS | 691 | 519 |
| callers + export, W core-s per sample, 26 cores | host | W unmeasured | see 7.3 | see 7.3 |

GPU time adds across GPU stages, because they share SMs. For 3′ base:

    1 / (1/98 + 1/8533 + 1/2075 + 1/2643 + 1/11700 + 1/691) M/s = 1 / 0.01267 ≈ 79 M pairs/s

For 5′ it is ≈ 39 M pairs/s.

**The GPU-side bottleneck is mapping**, and within mapping:
- **5′:** heavy reads are 67% of the work.
- **3′:** heavy reads are 28%, searched seeds ~30%, and certificates, windows, stitching and annotation ~42%.

### 7.3 Host bound (the likely end-to-end bottleneck)

| W (core-s per sample) | 66 M-pair sample | atlas-mean 294 M-pair sample |
|---|---|---|
| 30 | 57 | 255 |
| 60 | 29 | 127 |
| 120 | 14 | 64 |
| 240 | 7.2 | 32 |
| 480 | 3.6 | 16 |

Values are M pairs/s per GPU.

**What W might be today.** The CPU engine's "index, CB refresh, pileup" plus "reference load, export" is 85 + 58 s of
wall at 4 threads on a 66 M sample. The GPU removes the pileup, index, CB refresh and reference load. What remains is
the callers and export, unmeasured and plausibly 50–250 core-s.

**End to end, modelled per GPU (base):**
- 3′ = min(79, host).
- 5′ = min(39, host).
- The host binds once W > 26 × U / GPU-rate:
  - on atlas-mean samples (U = 294 M), once W > ~100 core-s (3′) or ~200 core-s (5′);
  - on small samples (U = 66 M), once W > ~22 core-s (3′) or ~44 core-s (5′).
- **Target ≥ 10 M pairs/s: met in every row except W ≥ 240 on 66 M samples.**

If M10 finds W large, port the heaviest callers to the GPU or parallelise them. EmptyDrops Monte Carlo, doublet kNN and
demux EM are all GPU-friendly, and Singlet already has GPU kNN/Leiden code in `src/gpu`.

**Against the baselines:**

| comparison | value |
|---|---|
| CPU node, end to end | 1–1.3 M pairs/s |
| one H100, modelled end to end (base, GPU stages summed; host not binding, i.e. W ≤ ~100 core-s at atlas depth) | 39 (5′) – 79 (3′) M pairs/s: **30–60× a node** |
| one H100, pessimistic (GPU stages summed) | 21 (5′) – 45 (3′) M pairs/s: ~17–35× a node |

### 7.4 Campaign view

**Compute time.** The atlas is ≈ 70k samples × 294 M pairs ≈ **20.6 T read pairs**. At 10 M pairs/s per GPU, 6 GPUs
(3 nodes) need **4.0 days**; at 40 M/s, 1.0 day.

**Input supply.** The raw input is 1.58 PB (≈ 77 B per pair). Six GPUs at 10 M/s would consume 4.6 GB/s of raw data,
well above the download tier's stated 100–200 MB/s per stream (`format-1fq.md` §1.1). **Beyond ~1–2 GPUs, the
campaign's wall time is set by download and conversion (rollout §3a), not by the engine.** This argues for building the
engine to ~10–20 M pairs/s with full exactness, and spending no effort beyond that on speed.

---

## 8. Riskiest assumptions

Ranked by impact × uncertainty:

| # | assumption | used where | if wrong | decided by |
|---|---|---|---|---|
| A1 | an H100 NVL sustains RS_clean ≈ 23 G random sectors/s (pessimistic 15 G) at a 32–60 GB footprint. TLB reach on random access over 40+ GB is unknown and could cost 2–3×. | every mapping number | throughput scales linearly; at 8 G/s, base 3′ drops to ~34 M/s (still > 10) | M1 |
| A2 | STAR's SAi + binary search + `findMultRange` returns the mathematical MMP and maximal interval for every seed, including SAi-absent prefixes, N-marked SAi entries, `iSA2good=false`, sjdb inserts and text ends | certificates (4.4), LCP reuse (4.5), SBK (4.6) | each shortcut must replicate the deviating cases or exclude them (lower coverage); E1 identity still testable | M4 (and M3 for the plain port) |
| A3 | compact per-read state (≤ 24 seeds, ≤ 8 windows, ≤ 16 aligns per window) covers ≥ 99.9% of distinct reads | wavefront design (4.3) | an overflow share of a few % costs ~2× on those reads; > 10% needs bigger caps (memory) | M3 dumps (distributions of nP, nW, nWA) |
| A4 | the recursive stitcher, `extendAlign`, junction scan, multimapper selection and filters can be ported bit-exactly (enumeration order, ties, `log2`/`ceil`, double compares without FMA) | E2 | equivalence fails on a tail of reads; each case must be found by differential testing | M5 |
| A5 | host-side callers and export W ≤ ~100–150 core-s per sample | e2e bound (7.3) | host becomes the bottleneck; callers must move to GPU | M10 |
| A6 | distinct fraction d ≈ 0.47 (3′) and ≤ 0.6 (5′, nuclei, deep); heavy class shares as in PR #41 | 4.1, all rows | at d = 0.8, mapping rates fall 1.7× | M2 |
| A7 | heavy reads on GPU cost ≤ 0.5 × k × normal (contiguous loci scans, warp search) | 5′ rows (heavy = 67% of work) | at 1.0 × k, base 5′ drops from 44 to ~26 M/s | M9 |
| A8 | wavefront SIMT efficiency ≈ 0.6 and η_mem ≈ 0.35 | all designs | pessimistic row (0.45 / 0.25) is still 53 / 24 M/s | M3, M5 |
| A9 | certificate coverage ≈ 3.3 of 5.9 seeds per normal read | 4.4 | coverage 1.5 → gain 1.67× drops to ~1.25× | M4 |
| A10 | Clipper GPU nodes can read `.1fq` at ≥ 0.2–0.4 GB/s per GPU sustained | e2e | stage via local NVMe | M8 |

**What could kill the project.**
- **No single speed assumption failing does it.** A1 at 1/3 of the assumed rate still gives 15 M/s (5′) and 33 M/s
  (3′) in the base scenario, and 8 / 18 M/s in the pessimistic one.
- **Three failing together reaches the kill line on 5′.** With A1 at 8 G sectors/s, A7 heavy reads at 1.0 × k, and A8
  pessimistic, the model gives ~5 M/s on 5′ and ~14 M/s on 3′.

**The real threats are equivalence (A2, A4) and engineering time**, not speed.

---

## 9. Microbenchmarks

Prioritised; each has an exact spec. All are for later execution. **No cluster jobs ran in this pass.**

**Shared conventions for every benchmark:**
- **GPU:** `--gres=gpu:nvidia_h100_nvl:1` in the `debug-gpu` or `short-gpu` partition.
- **Container:** an NVIDIA CUDA devel image pulled with apptainer.
- **Record:** the CUDA, driver and compiler versions, and the clocks (`nvidia-smi -q -d CLOCK`).
- **Repeats:** 3, reporting the median.
- **Datasets:** real ones only (brief §7):
  - `pbmc1k_v3` and the 66 / 110 / 260 M 3′ sets;
  - `pbmc5p` and the 122 M 5′;
  - mouse `nuclei2k_v2` and `neuron1k_v3`;
  - the Kang pool and the high-mtDNA sample.

**The STAR dump (needed by M3–M6).** A W11 tool links the vendored STAR objects and runs CPU STAR over a dataset's
distinct reads. For each distinct read it writes:
- the clipped sequence, and the split pieces;
- each `maxMappableLength2strands` call in order: (dir, Shift, seedLength) → (L, Nrep, indStartEnd);
- the final PC array;
- the windows (WC);
- the aligns per window (WA);
- the transcripts with scores, the multimapper set and the primary;
- the unmapped marker;
- a class label (perfect / 1 mm / ≥ 2 mm / heavy; the "heavy" class reuses PR #41's definition).

Format: a binary SoA plus a JSON header. 10 M distinct reads per dataset (~5 GB).

### M1 — random-access roofline (A1); first, about 1 GPU-hour

| | |
|---|---|
| **inputs** | none (synthetic addresses); footprints {1, 8, 16, 32, 48, 64, 80} GB |
| **kernel** | `bench/gpu/m1_gather.cu`: (a) independent hashed gathers of {8, 16, 32} B; (b) dependent pointer chase with {1, 2, 4, 8} chains per thread and occupancy {1, 4, 16} blocks per SM. Variants: `cudaMalloc` vs `cudaMallocAsync` pool; plus an L2 persistence window on the SAi first 12 levels |
| **metric** | G sectors/s; ns per dependent step; the curve vs footprint, where the TLB knee shows |
| **pass** | gather ≥ 15 G sectors/s at 32 and 48 GB, 32 B. **Kill the base scenario** if < 8 G/s at 48 GB, and redo the model at the measured value |

### M2 — distinct fraction, cross-sample sharing, seed-reuse proxy (A6); CPU only, no GPU

| | |
|---|---|
| **inputs** | R2 of every dataset above, with each sample's real clip and length; plus 3 batches of 8 same-study 3′ samples and 1 batch of 8 5′ samples (from the atlas queue) |
| **tool** | `bench/gpu/distinct_fraction.py` |
| **metric** | d per sample; d at 1/5/20/50% depth; LCP ≥ 32/50/70 neighbour shares; cross-sample `share_seen_elsewhere` and `batch_dedup_gain` |
| **pass** | d ≤ 0.6 for 3′ at ≥ 66 M; ≤ 0.7 for 5′. Batch dedup gain is reported; it is not gating (it decides 4.2b) |

### M10 — host work per sample, W (A5); CPU only, from existing logs first

| | |
|---|---|
| **inputs** | stage timers of existing CPU runs on the 8 datasets, adding timers where missing: called_ed, ambient, doublets, demux, sex and ancestry, non-host EM, `.s1` export. Also cells at 1k / 10k / 30k (audit "measure first" item 6) |
| **metric** | core-s per sample for everything that stays on CPU in §6 |
| **pass** | W ≤ 120 core-s on a 66 M sample. If not, list the callers to port, by cost |

### M3 — exact SA seed search on GPU (A2 port side, A3, A8)

| | |
|---|---|
| **inputs** | STAR dump of `pbmc1k_v3`, `pbmc5p` and `nuclei2k_v2`; GRCh38 and GRCm39 STAR indexes loaded to HBM unchanged |
| **kernel** | a port of `maxMappableLength2strands` / `maxMappableLength` / `findMultRange` (thread per seed); a warp-per-seed variant for buckets > 4096; queries in (a) dump order, (b) prefix-sorted order |
| **metric** | seeds/s by class; RS per seed (Nsight Compute `dram__sectors_read`); warp execution efficiency; **identity** of (L, Nrep, indStartEnd) per seed vs the dump |
| **pass** | 100.000% identical; ≥ 300 M normal seeds/s (≈ 60 M distinct reads/s at 5 searched seeds); report the sorted / unsorted ratio. Also report the PC, WC and WA size distributions (A3): **pass** if ≥ 99.9% of reads fit the §4.3 caps |

### M4 — U-array build and certificate coverage (A2, A9)

| | |
|---|---|
| **inputs** | the GRCh38 STAR index, with the same dumps as M3 |
| **kernels** | `build_lcp` and `scatter_u` in `bench/gpu/ucert.cu`; a certificate pass using locus propagation from the dump's first answered seed |
| **metric** | build time; SA sortedness violations (expect 0); the U distribution; the share of seeds certified per class; **identity** of certified answers vs the dump, and vs the M3 port on the same seeds |
| **pass** | 0 mismatches over all dumped seeds; certified ≥ 50% of normal-read seeds (the model assumes 56%) |

### M5 — windows, stitch and extend port (A4, A8)

| | |
|---|---|
| **inputs** | dumped PC arrays → windows and transcripts |
| **kernel** | the stitch pipeline of §4.3 (thread per window, shared-memory state; overflow kernel) |
| **metric** | reads/s by class; identity of transcripts (exons, score, nMM, junction motifs and annotations, multimapper order, primary) |
| **pass** | 100% identical on ≥ 30 M distinct reads across the 3 datasets; ≥ 100 M single-window reads/s |

### M6 — dedup, count sort and UMI collapse (A8, determinism)

| | |
|---|---|
| **inputs** | the real `.1fq` of the 66 M PBMC sample (R2, CB, UMI); gene ids from the CPU engine's per-read output |
| **kernels** | 40-bit hash sort + neighbour compare; (cell, gene, UMI) sort; split-halves UMI 1MM (1MM_All and 1MM_CR rules); segmented reduce to CSR |
| **metric** | pairs/s per step; output bytes vs the CPU matrices (`perf_equiv`); byte identity across batch sizes {4, 16, 64} M and 3 runs |
| **pass** | ≥ 1 G pairs/s through the chain; identical matrices; identical across batch sizes and runs |

### M7 — barcode correction, bit-exact

| | |
|---|---|
| **inputs** | the 66 M 3′ sample and `pbmc5p`; v3 and 5′ whitelists; STARsolo CB tags from the CPU engine |
| **kernel** | cuco `static_map` lookup, pass-1 histogram, 1MM_multi_Nbase_pseudocounts with the host `pow10` table and `-fmad=false` |
| **metric** | identity of the corrected CB per read; reads/s |
| **pass** | 100% identical; ≥ 1 G reads/s |

### M8 — host↔device pipeline with dummy kernels (A10)

| | |
|---|---|
| **inputs** | real `.1fq` files on (a) node-local NVMe, (b) the shared filesystem |
| **pipeline** | decode threads {1, 2, 3} → pinned ring (4 × 256 MB) → copy stream → dummy kernels that touch every byte and spin for a configurable time per read |
| **metric** | sustained pairs/s; GPU idle % (Nsight Systems); host cores used |
| **pass** | ≥ 50 M pairs/s with ≤ 3 cores and < 5% idle from NVMe. Report the shared-filesystem rate (it decides staging) |

### M9 — heavy-seed kernel (A7)

| | |
|---|---|
| **inputs** | the dump's heavy-class reads from `pbmc5p` (7.4% of reads) and the 3′ sets |
| **kernel** | warp-cooperative 33-way search; contiguous loci walk with shared-memory window test |
| **metric** | RS per heavy read; heavy reads/s; identity |
| **pass** | ≤ 0.5 × k × normal-read RS (the model's assumption); identical |

### M11 — LCP seed reuse and SBK (4.5, 4.6)

| | |
|---|---|
| **inputs** | the M3 dumps; an SBK built from the GRCh38 SA (every 16th suffix, 64-bit prefix) |
| **metric** | the share of searched seeds answered by reuse; RS per searched seed with and without SBK; identity |
| **pass** | reuse ≥ 20% or SBK ≥ 1.3× on RS. Otherwise drop the one that fails |

### M12 — locator (4.7); later

| | |
|---|---|
| **inputs** | GRCh38; a minimizer table (k = 28, w = 16) over all text positions with U ≤ 32 |
| **metric** | the share of reads whose first seed is located and certified; probes per read; table GB |
| **pass** | ≥ 60% located, at ≤ 8 GB |

### M13 — non-host on GPU

| | |
|---|---|
| **inputs** | unmapped reads from 3 samples (0.5 / 2.6 / 20 M unmapped, audit "measure first" item 4); the current viral, bacterial and fungal indexes |
| **metric** | time per batch (HBM-resident vs streamed index); identity of per-read hits and of the species table vs CPU `min_sketch` |
| **pass** | identical; < 1% of GPU time per sample |

### M14 — atlas answer-table hit rate (4.2b)

| | |
|---|---|
| **inputs** | 200 real 3′ human samples from the atlas queue, in campaign order |
| **metric** | h(n), the share of sample n's distinct sequences seen in samples 1..n−1, for tables capped at 1e8, 3e8 and 1e9 entries (keep the most frequent) |
| **pass** | h ≥ 0.3 at 1e9 entries to justify building 4.2b |

### Equivalence suite (gates P2, P3 and P4 in the brief)

| gate | what must hold |
|---|---|
| E1 | 100% identical seeds on ≥ 1e9 seeds across all datasets |
| E2 | ≥ 99.9% identical primary alignments on unique reads, with every difference explained (target 100%) |
| E3 / E4 | `txf_equiv.py`-style identity of matrices, Solo files, SJ tables and calls; then `bench/public/` |

---

## 10. Phases and kill criteria

The phases follow the brief's table (§6), filled in from this exploration:

| phase | content | gate | kill if |
|---|---|---|---|
| P1 (now → +2 wk) | M1, M2, M10 (cheap, decisive); M3 seed port on dumps | M1 ≥ 15 G sectors/s; d as modelled; W known; M3 identical | modelled e2e < 5 M pairs/s after plugging in M1, M2 and M10 |
| P2 | M4 U-array, M11, M9; seed rounds end to end on GPU | E1 100% on all datasets | certificates or reuse cannot be made identical (drop them; the port alone models at 37 / 25 M) |
| P3 | stitch port (M5), multimapper selection, overflow path | E2 ≥ 99.9%, every difference explained | stitch identity < 99.9% after 4 weeks |
| P4 | DSA, counts, UMI, barcodes (M6, M7), junctions, SNP, chrM, non-host (M13) | `perf_equiv` identical; capability benchmarks pass | |
| P5 | service, multi-sample batches, pipeline (M8) | ≥ 10 M pairs/s sustained e2e; byte-identical across batch sizes and reruns | < 10 M after profiling-driven fixes |
| P6 | 50-sample validation vs CPU, then the release gate | brief §6 | |

---

## 11. Sources

- [S1] NVIDIA H100 datasheet (H100 NVL column). https://www.nvidia.com/en-us/data-center/h100/
- [S2] Luo et al., Hopper microbenchmarks (H800 PCIe). arXiv 2402.13499 / 2501.12084. https://arxiv.org/abs/2501.12084
- [S3] H100 SXM5 vs A100 memory hierarchy, arXiv 2607.19922. https://arxiv.org/abs/2607.19922
- [S4] Chips and Cheese, "Nvidia's H100: Funny L2, and Tons of Bandwidth".
  https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth
- [S5] NVIDIA, "Maximizing performance with massively parallel hash maps on GPUs" (cuco on H100).
  https://developer.nvidia.com/blog/maximizing-performance-with-massively-parallel-hash-maps-on-gpus/
- [S6] Adinets & Merrill, "Onesweep: a faster least-significant-digit radix sort for GPUs". arXiv 2206.01784.
  https://arxiv.org/abs/2206.01784
- [S7] H100 HGX bandwidthTest (pinned H2D 53.77 GB/s). https://www.icc-usa.com/blog/hgx-8x-h100-benchmarks
- [S8] NVIDIA, CUDA Graphs constant-time launch.
  https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements
- [S9] NVIDIA Clara Parabricks overview (fq2bam 30× WGS on H100). https://docs.nvidia.com/clara/latest/Overview.html
- [S10] Parabricks docs (rna_fq2bam runtimes). https://docs.nvidia.com/clara/parabricks/latest/Overview.html
- [S11] NVIDIA developer forum, rna_fq2bam vs STAR (anecdotal).
  https://forums.developer.nvidia.com/t/clara-parabricks-4-0-0-1-and-4-2-0-1-star-equivalent-parameters-no-speed-up-observed-in-rna-fq2bam-compared-to-equivalent-star/277323
- [S12] Luo et al., SOAP3-dp. arXiv 1302.5507. https://arxiv.org/abs/1302.5507
- [S13] Klus et al., BarraCUDA (SC11 slides). https://www.nvidia.com/content/events/sc11/pdf/brian-lam.pdf
- [S14] nvBowtie on K80 (HPCwire).
  https://www.hpcwire.com/2015/01/28/nvidia-k80-gpu-system-speeds-bioinformatics-tool-12x/
- [S15] GPU FM-index backward search (NVBIO and 2-step FM-index rates).
  https://riuma.uma.es/bitstreams/a87e9d3a-532e-4dc5-86ab-304dffc30487/download
- [S16] Ahmed et al., GASAL2. https://pmc.ncbi.nlm.nih.gov/articles/PMC6815017/
- [S17] AGAThA. arXiv 2403.06478. https://arxiv.org/abs/2403.06478
- [S18] Aguado-Puig et al., WFA-GPU. https://pmc.ncbi.nlm.nih.gov/articles/PMC10697739/
- [S19] Schmidt et al., CUDASW++4.0. https://doi.org/10.1186/s12859-024-05965-6
- [S20] mm2-ax. https://pmc.ncbi.nlm.nih.gov/articles/PMC10018915/
- [S21] mm2-gb. https://www.biorxiv.org/content/10.1101/2024.03.23.586366
- [S22] Dobin et al. 2013, STAR. https://doi.org/10.1093/bioinformatics/bts635
- STAR 2.7.11b as vendored: `src/star/` on `origin/prod/campaign` (`SuffixArrayFuns.cpp`,
  `ReadAlign_mapOneRead.cpp`, `ReadAlign_maxMappableLength2strands.cpp`, `ReadAlign_storeAligns.cpp`,
  `ReadAlign_stitchPieces.cpp`, `ReadAlign_createExtendWindowsWithAlign.cpp`, `stitchWindowAligns.cpp`,
  `extendAlign.cpp`, `SoloReadFeature_inputRecords.cpp`, `SeqCache.h`, `TxFirst.h`, `GenomePacked.h`, `ReadAlign.cpp`,
  `parametersDefault`); `include/singlet/pileup/nonhost/min_sketch.h`; `docs/format-1fq.md`; PR #41.

**Sketches in `bench/gpu/`** (not wired into CMake):

| file | what it is |
|---|---|
| `throughput_model.py` | the model in §7 |
| `m1_gather.cu` | M1 |
| `ucert.cu` | U-array build and certificate (§4.4) |
| `distinct_fraction.py` | M2 |
