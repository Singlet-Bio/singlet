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

---

## 12. Measured results (M1, M2, M10, M3, M4, M6, M8, host timers; round 2: M5, GPU pileup, end to end, cost)

Run on Clipper on 2026-10-27, after §9 was written (§12.5 onwards: the second pass, same day). Raw data and scripts:
`bench/gpu/results/` and `bench/gpu/jobs/*.sbatch`, plus per-benchmark folders `bench/gpu/{m3m4,m6,m8,host}/` and the STAR dump
tool in `src/gpu/stardump/`. First pass: `bench/gpu/results/{m1,m2,m10}.json` (plus
`m1_raw.jsonl`, `m1_knee.jsonl`, `m10_runs.json`, one `m2_*.json` per sample) and `bench/gpu/jobs/*.sbatch`.
`bench/gpu/model_measured.py` re-runs the §7 model with the substitutions below.

| benchmark | measured | threshold | verdict |
|---|---|---|---|
| M1 gather, 32 GB / 48 GB, 32 B | 28.2 / 29.0 G sectors/s | ≥ 15 G/s; kill < 8 G/s at 48 GB | **PASS** (1.9× the gate; base scenario not killed) |
| M2 d, 3′ ≥ 66 M | 0.474 (pbmc1k_v3), 0.515 (263 M deep), 0.392 (GSM5411066, mouse), 0.298 (GSM7036403, v2), 0.369 (neuron1k_v3); **0.658 (nuclei2k_v2)** | ≤ 0.6 | **PASS for 5 of 6; FAIL for nuclei2k_v2** |
| M2 d, 5′ | 0.306 (pbmc5p_v1), 0.303 (GSM6470208, 121 M) | ≤ 0.7 | **PASS** |
| M10 W, 66–70 M sample | 201 and 143 core-s (export + non-host residual, an upper bound); export alone 122 and 51 | ≤ 120 core-s | **FAIL** as bounded (see caveats) |
| M10b W, timed (6 samples) | pooled 2.18 core-s per M pairs (1.21–2.64); demux auto-K + non-host screen = 74% | node ≥ 20 M pairs/s | **PASS** with a persistent non-host index (every sample ≥ 20.7 M pairs/s per node; §12.9) |
| M3 exact seed search | 323,688,051 of 323,688,051 seed records identical to STAR; ~1.0 G records/s | 100% identical; ≥ 300 M seeds/s | **PASS** (3.3× the rate gate) |
| M4 certificates | 0 wrong; 16–23% of calls resolved | 0 mismatches; ≥ 50% certified | **PASS on exactness, FAIL on coverage**: dropped from the plan (§12.7) |
| M6 dedup + count + UMI | 1.6–2.0 G pairs/s; matrices identical to the CPU engine; identical across batch sizes and runs | ≥ 1 G pairs/s; identical | **PASS** |
| M8 ingest | 12–15 M pairs/s per decode thread, 0.085 core-s/M (0.175 with CRC); 50 M/s needs 5–6 threads | ≥ 50 M/s with ≤ 3 cores | **FAIL on cores**, irrelevant at the 14–18 M pairs/s per GPU the host allows (§12.10) |

### 12.1 M1: random-access roofline (H100 NVL, job 789081; knee 789085)

**Setup.**
- Node g050, H100 NVL 94 GB, 132 SMs, **L2 = 60 MB** (the brief and §2.3 say 50 MB).
- Driver 610.43.02 (CUDA UMD 13.3), nvcc 12.4.131 from `nvidia/cuda:12.4.1-devel-ubuntu22.04` via `apptainer --nv`, `-arch=sm_90`.
- Clocks: SM max 1785 MHz, memory 2619 MHz. The SM clock was 345 MHz (idle) at the first `nvidia-smi`; no clock lock was applied.
- 3 repeats after a warm-up; medians below. The range over the 3 repeats is at most 9% of the median in any cell.
- The whole sweep took 18 s of GPU time (knee follow-up 10 s), so M1 used about 0.01 GPU-hours, not the ~1 GPU-hour in §9.

**Independent gathers, G accesses/s (1 access = one 32 B sector or less), `cudaMalloc`:**

| footprint GB | 1 | 8 | 16 | 32 | 48 | 64 | 72 | 74 | 76 | 78 | 80 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 B | 33.9 | 32.6 | 32.0 | 32.3 | 32.4 | 32.4 | 32.0 | 31.7 | 28.9 | 23.2 | 19.5 |
| 16 B | 31.6 | 29.2 | 29.0 | 27.9 | 28.9 | 28.9 | 28.5 | 28.3 | 23.4 | 18.2 | 14.3 |
| **32 B** | 30.8 | 29.3 | 28.2 | **28.2** | **29.0** | 29.0 | 28.7 | 28.6 | 24.0 | 18.8 | 15.1 |

- **No knee up to ~74 GB.** The rate is flat at 28–29 G/s from 8 to 74 GB, then falls gradually to 15 G/s at 80 GB. This is
  the only knee. The design's working set (index ~38 GB plus U-array, SBK and buffers, under 60 GB) is far below it. Whether the
  76–80 GB fall is TLB reach or HBM pressure (80 GB of 94 GB is allocated) was not separated.
- 29 G × 32 B = 0.93 TB/s of requested sectors, 24% of the 3.9 TB/s peak. If each random access moves a 64 B burst, the
  DRAM traffic is ~1.9 TB/s (an assumption about HBM3 burst size; not measured with a profiler).
- **`cudaMallocAsync` pool = `cudaMalloc`** (within 1%) at every footprint, apart from one 8 B outlier at 48 GB (28.1).

**Dependent pointer chase (random walk, each address depends on the loaded word), `cudaMalloc`:**
- **Latency:** 1.03 µs per dependent step at 1 GB and **1.17 µs at 32 and 48 GB** (1 chain, 1 block of 256 threads per SM,
  34k chains in flight). It is 1.74 µs at 80 GB. This includes some hash arithmetic. §2.3's 0.3–0.45 µs is the idle DRAM
  latency; the loaded step is ~2.5× that.
- **Throughput:** 28.8 G loads/s at 1 block/SM (34k chains), 31.0 at 4 blocks/SM, 31.3 at 16 (32 GB); 29.1 / 32.2 / 32.4 at
  48 GB. Extra chains per thread add ~1 G/s at most.
- So **34k independent chains in flight already reach ~90% of the dependent-chase rate**, which equals the gather rate. The
  batch design (4–8 M reads in flight) is far above this. A dependent step costs latency, not throughput.

**L2 persistence** (32 MB persisting window over a synthetic hot region taking 1 access in 4; max persisting size 37.5 MB):

| footprint GB | 1 | 8 | 16 | 32 | 48 | 64 | 80 |
|---|---|---|---|---|---|---|---|
| no window, G/s | 33.0 | 31.8 | 31.3 | 31.7 | 31.0 | 31.1 | 21.5 |
| with window, G/s | 39.6 | 37.0 | 33.5 | 35.1 | 32.1 | 31.5 | 21.5 |

- **Gain: +11% at 32 GB, +4% at 48 GB, +1% at 64 GB.** The hot region is a synthetic 32 MB block, not the real SAi. The SAi's
  top 12 levels (4^12 entries) exceed the 37.5 MB cap, so only the top ~10 levels could be pinned. Not worth engineering
  effort before the rest is done.

**Substitute in `throughput_model.py`: `RS_CLEAN = 28.2e9`** (the lower of the 32 GB and 48 GB medians, vs 23e9). The model's
η_mem (0.25 / 0.35 / 0.50) still applies on top: it is what the full kernels achieve relative to this clean gather, and M3
measures it.

**Deviations from the §9 spec.**
- The script was changed (see below), and the chase is a random walk with a step-dependent hash.
- The persistence variant uses a synthetic hot region, not the real SAi first 12 levels.
- The pool variant ran at every footprint but without a persistence window.
- The knee follow-up (66–78 GB) is a second job.

**Script changes** (`m1_gather.cu`, minimal fixes plus the specified sweeps):
- Range reduction by `__umul64hi` instead of 64-bit `%`.
- The chase successor depends on the step number: the original fill makes a functional graph whose chains fall into
  ~sqrt(N)-sector cycles that fit in L2.
- Chains are templated (1, 2, 4, 8).
- Added the footprint, occupancy, allocator and persistence sweeps, 3 timed repeats after a warm-up, and JSON output.
- It compiled and ran with no other fixes.

### 12.2 M2: distinct-sequence fraction on real reads (jobs 789069 array, 789074 array)

**Method.**
- `singlet decode --r2` streamed the cDNA read into `distinct_fraction.py --len 90`, with no FASTQ on disk.
- Reads shorter than 90 and reads with N are excluded and counted in the JSON. The shorter-than-90 share is 0–3.2%.
- No clip beyond what the `.1fq` encoder applied (R2 lengths in the `.1fq` are variable).
- LCP shares use every 16th read, capped at 2 M distinct sequences.
- Species: mouse for neuron1k_v3 and nuclei2k_v2 per the §9 spec (the benchmark runs of these two used the GRCh38 index); GSM5411066 was run
  against GRCm39. d is computed on the reads alone, so the index does not matter.

| sample | chemistry | reads used | **d** | d@1% | d@5% | d@20% | d@50% | lcp≥32 | lcp≥50 | lcp≥70 |
|---|---|---|---|---|---|---|---|---|---|---|
| pbmc1k_v3 | 3′ v3 | 65.1 M | **0.474** | 0.726 | 0.648 | 0.583 | 0.523 | 0.199 | 0.075 | 0.035 |
| GSM7656149 (deep) | 3′ v3 | 259.5 M | **0.515** | 0.793 | 0.706 | 0.629 | 0.573 | 0.263 | 0.080 | 0.040 |
| GSM5411066 (mouse) | 3′ v3 | 67.3 M | **0.392** | 0.681 | 0.582 | 0.500 | 0.443 | 0.188 | 0.061 | 0.028 |
| GSM7036403 (deep) | 3′ v2 | 109.0 M | **0.298** | 0.528 | 0.423 | 0.349 | 0.314 | 0.167 | 0.076 | 0.035 |
| neuron1k_v3 (mouse) | 3′ v3 | 90.2 M | **0.369** | 0.739 | 0.630 | 0.511 | 0.431 | 0.108 | 0.039 | 0.019 |
| nuclei2k_v2 (mouse) | 3′ v2 nuclei | 114.5 M | **0.658** | 0.901 | 0.852 | 0.778 | 0.709 | 0.228 | 0.045 | 0.020 |
| pbmc5p_v1 | 5′ v2 | 59.8 M | **0.306** | 0.590 | 0.465 | 0.387 | 0.341 | 0.157 | 0.120 | 0.041 |
| GSM6470208 | 5′ v2 | 120.9 M | **0.303** | 0.529 | 0.442 | 0.375 | 0.335 | 0.281 | 0.234 | 0.090 |

**Findings.**
- **The anchor reproduces.** pbmc1k_v3 gives 0.474 against `SeqCache.h`'s 0.47.
- **Depth.** d falls with depth (pbmc1k_v3: 0.726 at 1%, 0.523 at 50%, 0.474 at 100%). A deep sample is not worse than a
  shallow one; GSM7656149 is the highest of the human 3′ samples at 0.515.
- **Spread.** The 3′ samples run 0.30–0.52, except **nuclei2k_v2 at 0.658, which fails the ≤ 0.6 gate**. A6 already named
  nuclei as the risk. The atlas-wide d depends on the nuclei share, which was not measured here.
- **5′ is 0.30, below the model's 0.50.** Both 5′ samples agree. The cause was not investigated.
- **LCP proxy.** Neighbour shares (≥ 32 shared leading bases) are 11–28% of the sampled distinct sequences, and 4–23% at ≥ 50.
  It is an upper proxy for exact seed reuse in a prefix-sorted wavefront. The 5′ samples have the highest shares at ≥ 50
  (12% and 23%), so §4.5 should help 5′ at least as much as 3′.

**Cross-sample sharing (batch dedup, §4.2).**

| batch (same study) | libraries | share of each library's distinct sequences seen in another | union / sum of distinct | **batch dedup gain** |
|---|---|---|---|---|
| Kang 2018 (GSE96583): b1A, b1B, b1C, b2stim | 4; 163–184 M reads each; d 0.205 / 0.220 / 0.233 / 0.397 | 33.9% / 34.8% / 32.6% / 15.8% | 149.5 M / 179.6 M | **1.20×** |
| GSM6996272/3/4 | 3; shallow, 2–4 M reads each | 9.0% / 10.3% / 6.9% | 5.20 M / 5.45 M | 1.05× |

- Reported, not gating. Four deep same-study libraries share 17% of their distinct sequences; shallow samples share almost
  nothing.
- **Not done:** the spec's "3 batches of 8 same-study 3′ samples and 1 batch of 8 5′ samples from the atlas queue" were not
  staged. These two batches are what existed, so the 4.2b number is **≤ 1.2× for 4 deep same-study libraries**, not an atlas
  figure. M14 (answer-table hit rate) remains the test.

**Substitute: d(3′) = 0.474 (pbmc1k_v3), d(5′) = 0.31.** The 3′ sensitivity from d = 0.30 to 0.66 is in §12.4.

**Script change** (`distinct_fraction.py`): reads shorter than `--len` were dropped silently; they are now counted in `reads_shorter_than_len`.

### 12.3 M10: host work per sample, W (existing logs; job 789078)

**Method.**
- `m10_hostwork.py` parsed 239 `run.log` directories under `bam-stream/runs`, `pipeline-benchmark/perf`,
  `pipeline-benchmark/runs` and `deep-perf/runs`.
- Only 11 of them have `monitor.tsv` (cumulative cgroup CPU seconds every ~5 s): the unsorted-BAM benchmark's phase 1 runs
  (`bam-stream/runs/*/t4_r*`) and 4 `deep-perf` runs. **The other 228 have wall timers only.**
- Core-seconds are CPU-time deltas over the windows after STAR finishes: index, CB refresh, pileup, "non-host residual" (the
  end of the pileup up to the start of the export) and export (the last `[export] Total` seconds plus a 10 s tail).
- All monitored runs use `--threads 4` with the `singlet-campaign` binary and the `called_v2` caller.
- "Export" is everything between "Exporting feature matrices" and `[export] Total`: cell calling, ambient, doublets, demux, sex
  and ancestry, saturation, polyA, mt events, and the `.1pz` writers.

**Core-seconds per sample, measured** (the GPU removes index, CB refresh and pileup, listed last):

| sample | reads | export | non-host residual | **W upper = export + residual** | W per M pairs | removed by GPU |
|---|---|---|---|---|---|---|
| GSM4006843 (3′ v2) | 66.7 M | 121.8 | 79.1 | **200.9** | 3.0 | 185 |
| GSM5411066 (3′ v3, mouse) | 69.6 M | 50.6 | 92.8 | **143.4** | 2.1 | 146 |
| GSM7901264 (3′ v2) | 100.3 M | 58.1 | 147.6 | **205.7** | 2.1 | 177 |
| GSM7036403 (3′ v2) | 111.6 M | 202.4 | 92.7 | **295.1** | 2.6 | 294 |
| GSM6470208 (5′ v2) | 121.8 M | 90.2 | 169.6 | **259.8** | 2.1 | 610 |
| GSM7656149 (3′ v3, deep) | 263.4 M | 468.2 | 179.7 | **647.9** | 2.5 | 650 |

- **Median W (upper) = 2.3 core-s per M pairs.** At 66–70 M that is 143–201 core-s against the 120 gate: **FAIL** as bounded.
- **Export alone** is 51–122 core-s at 66–70 M (median 1.26 per M pairs, over the six samples).
- **Caveat on the residual.** It has no timer. It holds the non-host EM plus everything else between the pileup and the export
  (writing, cleanup), so it overstates the non-host cost. It is included because the non-host EM stays on the CPU in §6.
  Separating them needs an engine timer.
- **Export is nearly serial.** Export CPU time is 1.1–1.3× its wall time (122 core-s in 97 s for GSM4006843; 468 in 426 at
  263 M). Each sample therefore holds ~1.2 cores, and a 26-core GPU host could run ~20 samples at once.

**Wall timers per caller** (seconds, `--threads 4`; the six monitored samples, one run each):

| caller | range | printed as |
|---|---|---|
| cell calling (`called_v2`) | 2.1–18.1 | `[cell_calling] ... time=` |
| ambient | 1.2–5.9 | `[ambient] ... time=` |
| doublets | 1.0–4.8 | `[doublet] ... time=` |
| sex; ancestry | 0.5–1.3; 0.02 | `[sex_calling]`; `[ancestry]` |
| saturation curve | 0.7–2.7 | `[saturation_curve]` |
| **demux, auto-K selection** | **12.5–310** | `[demux] auto-K: selected K=.. in .. s` |
| export total | 31–416 | `[export] Total` |

- **Demux is the largest timed caller.** The auto-K fits are 74% of the export wall at 263 M (310 of 416 s), 50% at 111 M, 58% at 66.7 M, and 30–49% on the other three samples.
  Cell calling, ambient, doublets, sex and saturation together add 8–32 s.
  The remainder of the export (the `.1pz` writers, mt events, `cell_qc`) has no timers.
- **Callers with no timer:**
  - the non-host EM (the `[nonhost]` lines print no seconds);
  - `called_ed`: no log contains it, only `called_v2` and the older `emptydrops`;
  - demux outside the K-selection fits (VB on called cells, back-assignment, VCF writes);
  - the export writers (`pz_writer`, `splice_psi`, `sj_features`, `ase`, `vdj`, `cell_qc`, `metrics_summary`, `rrna`).
  No engine timers were added in this pass.
- **Not measured:** cells at 1k / 10k / 30k (audit "measure first" item 6).

**Substitute: W = 2.3 core-s per M pairs (upper bound; 1.26 export only)**, i.e. ~150 core-s at 66 M and ~680 at 294 M. In the §7.3
table the 66 M sample lands between the W = 120 and W = 240 rows.

### 12.4 The model, re-run with the measured values

`python bench/gpu/model_measured.py` (base scenario: η_mem 0.35, SIMT 0.60; "GPU stages summed" adds mapping, dedup, barcode,
expand, pileup-tuple and non-host as in §7.2). With the original constants it reproduces §7.1 and §7.2 (98.3 / 44.1 and 78.9 / 39.0).

| | original (RS 23 G/s, d 0.47 / 0.50) | measured (RS 28.2 G/s, d 0.474 / 0.31) |
|---|---|---|
| **mapping stage, 3′ base** | 98.3 M pairs/s | **119.6** |
| **mapping stage, 5′ base** | 44.1 | **87.3** |
| GPU stages summed, 3′ base | 78.9 | **95.4** |
| GPU stages summed, 5′ base | 39.0 | **74.9** |
| mapping, 3′ pessimistic / optimistic | 52.7 / 163.9 | 64.0 / 199.3 |
| mapping, 5′ pessimistic / optimistic | 23.6 / 73.5 | 46.7 / 145.4 |

**3′ base mapping stage against d:** d = 0.30 gives 189; 0.43 gives 132; 0.474 gives 120; 0.515 gives 110; 0.658 (nuclei2k_v2) gives 86 M
pairs/s (GPU stages summed: 70).

**End to end, per GPU (base; host = 26 cores ÷ W × pairs):**

| | GPU stages summed | host, W upper (2.3 core-s per M) | host, export only (1.26) | **end to end** |
|---|---|---|---|---|
| 3′ | 95.4 | 11.3 | 20.6 | **11.3 (upper) to 20.6 M pairs/s** |
| 5′ | 74.9 | 11.3 | 20.6 | **11.3 (upper) to 20.6 M pairs/s** |

- **The host now binds, not the GPU.** W per M pairs is nearly constant with depth (2.1–3.0), so the host rate is ~11 M pairs/s
  per GPU (26 cores) at any depth, 8× below the 3′ GPU stages.
- End to end stays above the ≥ 10 M/s target (11.3 M/s upper bound, 20.6 M/s export only). That is **9–11× (upper) or 16–21×
  (export only) a CPU node's 1.0–1.3 M pairs/s**, against the 30–60× in §7.3, which assumed the host would not bind.
- **Campaign:** at 11.3 M pairs/s per GPU, 20.6 T pairs on 6 GPUs take 3.5 days; at 20 M/s, 2.0 days.
- **Plan consequence.** The first host lever is demux (auto-K selection is 30–74% of the export wall). Cutting W to ~1.0 core-s
  per M pairs would lift the host bound to ~26 M pairs/s. The non-host EM needs a timer, since its cost is hidden in the residual.

### 12.5 Finding: the campaign's GRCh38 index is sparse (`genomeSAsparseD 2`)

`/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b/genomeParameters.txt` records
`--genomeSAsparseD 2`: the SA holds every other suffix, 3.10 G entries in 12.8 GB (§4.14 assumed a dense 24.3 GB SA). Mouse
GRCm39-2024-A is dense (5.46 G suffixes). §2.2's `gSAsparseD = 1` is therefore wrong for human, with three consequences:
- **Every human seed call runs two SA searches** (`maxMappableLength2strands` loops iDist 0 and 1: the piece and the piece
  shifted by one base), and `storeAligns` keeps the ones with the largest `maxL + iDist`. The GPU port does the same (M3).
- **Certificates need a second array.** Only one of the two sub-searches lands on a sampled suffix; the other is the MMP
  among sampled suffixes of a shifted query, which a U-array cannot certify. The **V-array** (§12.7) bounds it instead.
- **HBM is smaller than budgeted:** the human core index (SA 12.8 + SAi 1.6 + byte genome 3.2 GB) loads in 6 s from NVMe and
  occupies 17.6 GB.

The index must not be rebuilt dense for the GPU: `gSAsparseD` changes STAR's results (§4.16), and the contract is the CPU
engine's results with this index.

### 12.6 M3: exact STAR seed search on the GPU (jobs 789234 build, 789235 array)

**Method.**
- **The dump (§9).** `src/gpu/stardump/` is the vendored STAR of `prod/campaign` (28969b87) with
  `ReadAlign_maxMappableLength2strands.cpp` replaced by a copy that writes, per mapping thread, every seed call in STAR's call
  order: the read (STAR's `Read1[0]`, `Read1[1]`, `splitR`), and per iDist the inputs and the answer (`maxL`, `Nrep`, SA
  interval, `SA[ind0]`, which branch answered). It is built from a scratch copy (`bench/gpu/jobs/stardump_build.sbatch`);
  `src/star/` is untouched. The sequence cache is off (`SINGLET_STAR_SEQCACHE_MB=0`) so every read is searched.
- **Reads.** All R2 reads of the sample, deduplicated with multiplicities in job `/tmp`; every k-th sequence of the sorted
  distinct list is kept (~8–9 M distinct reads per sample), with the sample's STAR parameters from its campaign run log
  (`--outFilterScoreMin 10`; pbmc5p_v1 adds `--clip3pAdapterSeq AAGCAGTGGTATCAACGCAGAGTACATGGG --clip3pAdapterMMp 0.1`).
- **Kernel.** `bench/gpu/m3m4/seedsearch.cu` `seed_search`: a line-by-line port of `maxMappableLength2strands` (one iDist
  pass), `maxMappableLength`, `findMultRange` and the byte-genome `compareSeqToGenome`, over STAR's own `Genome`, `SA` and
  `SAindex` files loaded unchanged into HBM (packed 33-bit SA and 35-bit SAindex read with STAR's `PackedArray` rule). One
  thread per dumped record; queries in dump order and prefix-sorted order (first 32 bases in search direction, CUB/thrust sort).
  "Steps" counts SAindex probes plus genome compares, the dependent accesses of one search.
- H100 NVL (g050/g051), driver 610.43.02, nvcc 12.4.131, `-O3 -arch=sm_90`; 3 timed repeats after a warm-up, medians.

| sample | index | distinct reads (reads) | seed records | **identical to STAR** | records/s dump order | prefix-sorted | SIMT dump / sorted | steps per record |
|---|---|---|---|---|---|---|---|---|
| pbmc1k_v3 (3′ v3) | GRCh38, sparse | 7.97 M (16.6 M) | 104,010,422 | **104,010,422** | 995 M | 1,025 M | 0.40 / 0.61 | 10.5 |
| pbmc5p_v1 (5′) | GRCh38, sparse | 9.25 M (30.1 M) | 127,455,158 | **127,455,158** | 976 M | 1,109 M | 0.38 / 0.62 | 10.8 |
| nuclei2k_v2 (mouse nuclei) | GRCm39, dense | 8.97 M (13.2 M) | 92,222,471 | **92,222,471** | 1,085 M | 1,020 M | 0.39 / 0.61 | 9.9 |

- **Identity: 323,688,051 of 323,688,051 seed records identical** (maxL, Nrep, SA interval and the answering branch), on
  three real samples, two species, sparse and dense SA. A2 holds for the port side: STAR's search is reproduced exactly.
- **Throughput: ~1.0 G records/s, 3.3× the §9 pass line** (300 M normal seeds/s). By branch (sorted): SAindex-only 2.2–2.4 G/s,
  unique extension 1.7 G/s, full SA search 0.96–1.04 G/s. At ~10 dependent steps per record, each with an SA word and a
  genome window, the kernel issues roughly 20 G random accesses/s, ~70% of M1's 28.2 G sectors/s.
- **Records per distinct read:** 13.1 (pbmc1k_v3), 13.8 (pbmc5p_v1), 10.3 (nuclei2k_v2, dense: one record per call).
  **The seed stage alone therefore runs at 78–105 M distinct reads/s**, i.e. 166 M (3′, d = 0.474), 259 M (5′, d = 0.31) and
  160 M (nuclei, d = 0.658) read pairs/s, before any certificate or reuse. 47–49% of records have Nrep > 1.
- **Sorted vs unsorted:** prefix sorting lifts SIMT efficiency from 0.39 to 0.61 but the rate only by 1.03–1.14× (human) and
  0.94× (mouse): scattered result writes and read fetches eat most of the gain. §4.9's "≤ 1.1–1.3×" is confirmed at the low
  end. Not worth a sort pass by itself; it comes free if seeds are sorted for another reason.
- **Not measured here:** the PC/WC/WA size distributions (A3) need a dump of the window and stitch stage (M5); the overflow
  share is still unmeasured. The heavy-read kernel (M9) was not built: heavy reads are inside these rates, unseparated.

### 12.7 M4: U-array, V-array and certificate coverage (same jobs)

**Build** (on the GPU, from STAR's SA): pack the two-strand text to 2 bits plus a 1-bit N mask (8 ms); adjacent-suffix LCP
over A/C/G/T with a sortedness check (0.33 s for 3.10 G suffixes, 0.62 s for 5.46 G); scatter U[t] by text coordinate and
mark sampled positions (0.46 / 0.80 s); for the sparse index, the **V-array**: for every unsampled position, the longest
prefix shared with any sampled suffix, by binary search of the text suffix in STAR's SA (5.9 s for GRCh38). **SA order
violations: 0** on both indexes.

**Certificate rule.** Anchors come from earlier records of the same read that STAR answered with one locus and maxL ≥ 20
(a certified answer equals STAR's, so this is what the engine itself would have). For each record, the candidate t is the
anchor shifted by the read offset (mirrored for reverse searches). If t is sampled: certified when lcp(Q, text[t..]) = m ≥
U[t] (STAR's answer must then be maxL = m, Nrep = 1, locus t). If t is unsampled: bounded when m > V[t] (STAR's maxL must
then be ≤ V[t]). A call is **resolved** when every record of it is certified or bounded below the best certified
maxL + iDist, i.e. `storeAligns`' input is known without any SA search.

| sample | records certified | records bounded | **certified or bounded WRONG** | calls resolved (read-weighted) | perfect reads | heavy or short reads | other | certify rate |
|---|---|---|---|---|---|---|---|---|
| pbmc1k_v3 | 19.1% | 10.8% | **0** | **22.8%** | 35% of 22% of calls | 11% of 44% | 29% of 34% | 1.66 G records/s |
| pbmc5p_v1 | 17.1% | 5.2% | **0** | **16.0%** | 27% of 29% | 6% of 31% | 16% of 40% | 1.85 G/s |
| nuclei2k_v2 | 21.5% | — (dense) | **0** | **21.5%** | 62% of 13% | 11% of 64% | 28% of 23% | 2.60 G/s |

(Shares read-weighted, i.e. per original read. "Perfect": the first record maps its whole piece uniquely; "heavy or short":
any record with Nrep > 50 or no record with maxL ≥ 20.)

- **Exactness holds:** 0 wrong certificates and 0 wrong bounds over 323.7 M records, so the U and V proofs and A2 for
  certified seeds are confirmed on real data.
- **Coverage fails the §9 gate** (≥ 50% of normal-read seeds; the model assumed 3.3 of 5.9 seeds per normal read): 16–23% of
  calls. Why, from the decline counters (pbmc1k_v3): 33% of records have no anchor yet (every read's first call, both iDist
  records); U above the remaining query length on 26.6 M attempts (short seeds near read ends); the anchor's prediction fails
  (m < U) on 29.5 M (restart seeds after mismatches, splices); V unknown or not exceeded on 49.9 M.
- **Decision: drop certificates from the build plan.** The exact search already runs at ~1 G records/s (§12.6). Removing the
  certified records saves at most ~1/5 of the seed stage, which already measures at 160–260 M pairs/s, for 6.4 GB (U) +
  6.4 GB (V) of HBM and two extra code paths. Kill criterion §10 P2
  ("certificates ... cannot be made identical: drop them") is not what happened: they are identical; they are just not worth
  it. The code stays in `seedsearch.cu` as a validated tool (it also proves SA sortedness of an index in < 1 s).

### 12.8 M6: dedup, count sort and UMI collapse (Sonnet subagent; jobs 789186–789241)

Details, CPU rules with file:line references and per-run JSON: `bench/gpu/m6/README.md`, `bench/gpu/results/m6*.json`.

**CPU rules found** (prod/campaign): the pileup counts raw `UR` UMIs (STAR is not asked for `UB`); `--pipeline` forces
directional dedup: two UMIs of one (cell, gene) are linked when they differ in exactly one base (N packed as T) and
`max count ≥ 2·min count − 1`. With counts ordered this rule is symmetric, so a molecule is a **connected component**, which
makes the GPU result independent of tie order. Molecule evidence is the OR of flags and the minimum exon/intron index;
spliced → exon matrix, unspliced → intron matrix, ambiguous → exon matrix plus a per-gene count; counts clamp at 65535.

**Truth.** A dev patch (`bench/gpu/m6/tuple_dump.patch`, timers-only style, outputs unchanged) dumped every tuple at
`count_read` and the CPU's final matrices from a real engine run.

| | GSM4006843 (3′ v2, 66.7 M pairs) | pbmc1k_v3 (3′ v3, 66.6 M pairs) |
|---|---|---|
| a. read dedup, exact full-length key (≤ 128 nt), hash sort + neighbour compare | 19.2 ms, 3.46 G reads/s | 18.6 ms, 3.57 G reads/s |
| b. (cell, gene, UMI) key sort + reduce | 8.7 ms | 8.4 ms |
| c. UMI collapse (split halves + connected components) | 8.7 ms | 4.4 ms |
| d. class select + CSR | 4.3 ms | 1.6 ms |
| **chain per read pair** | **1.63 G pairs/s** | **2.01 G pairs/s** |
| matrices vs CPU (exon / intron / ambiguous entries) | **24.6 M / 6.03 M / 39 k, all identical** | **3.98 M / 3.56 M / 158 k, all identical** |
| CSR SHA-256 across batch sizes {4, 16, 64, all} M × 3 runs × 2 processes | identical, = CPU truth | identical, = CPU truth |

- **Pass on all three gates:** ≥ 1 G pairs/s (1.6–2.0), identical matrices, identical across batch sizes and runs.
- Transfers are not in the chain rate: non-overlapped H2D of packed reads and tuples (~320 ms) and D2H of the CSR (17–63 ms)
  would cut it to 0.2 G pairs/s. In the engine the tuples are produced on the GPU and only the CSR leaves, so ~60 ms per
  66 M-pair sample remains, overlapped with the next sample.
- The read-dedup distinct fraction matches M2 under M2's definition (pbmc1k_v3: 0.4739 vs 0.474).
- **Open:** stage a and stages b–d are not linked per read yet (no per-sequence annotation exists); only 3′ v2/v3 tested;
  UMI-heavy samples (more component rounds, larger split-half groups) untested.

### 12.9 Host work, timed (M10 follow-up; Sonnet subagent; jobs 789193, 789215, 789253) and the host-side plan

**Method.** A DEV build of `prod/campaign` (28969b87 + `bench/gpu/host/timers.patch`, PGO with the campaign's own
`pgo_build.sh` and training data; timers only, outputs unchanged; the campaign binary untouched) ran the six §12.3 samples with
the campaign command line (`--threads 4 --pipeline`, run screen). Every stage after STAR, every caller and writer in the export,
and every demux phase prints wall, process CPU and thread CPU. The timer totals agree with the cgroup CPU counter within 1–3%.
Details and the full table: `bench/gpu/host/README.md`, `bench/gpu/results/m10b_*.json`.

**W, the work after the pileup** (core-s; the GPU design removes STAR, sort, index, CB refresh and pileup):

| sample | M pairs | W | per M pairs | non-host index load | non-host classify | demux auto-K | demux final fit | non-host EM |
|---|---|---|---|---|---|---|---|---|
| GSM4006843 (3′ v2) | 66.7 | 174.9 | 2.62 | 33.2 | 20.5 | 47.9 | 28.5 | 4.9 |
| GSM5411066 (mouse 3′ v3) | 69.6 | 92.2 | 1.32 | 28.6 | 20.3 | 12.2 | 1.6 | 7.8 |
| GSM7901264 (3′ v2) | 100.3 | 121.2 | 1.21 | 28.0 | 40.8 | 11.9 | 1.9 | 19.2 |
| GSM7036403 (3′ v2) | 111.6 | 243.6 | 2.18 | 53.3 | 29.7 | 92.9 | 8.9 | 5.8 |
| GSM6470208 (5′ v2) | 121.8 | 271.4 | 2.23 | 46.2 | 77.2 | 49.4 | 5.0 | 24.7 |
| GSM7656149 (3′ v3, deep) | 263.4 | 694.8 | 2.64 | 42.7 | 126.0 | 337.5 | 27.2 | 17.8 |
| **pooled** | 733.4 | 1,598 | **2.18** | 0.32 /M | 0.43 /M | 0.75 /M | 0.10 /M | 0.11 /M |

(auto-K here is process CPU of the auto-K window; the README's 0.68 /M is its thread CPU.)

- **M10's bound is confirmed:** pooled W = 2.18 core-s per M pairs (M10: 2.3). The "non-host residual" is the non-host screen.
- **Where it goes:** demux auto-K (31–49% of W on the deep samples), the non-host screen (43% pooled: index load 0.32, classify
  0.43, EM 0.11, I/O 0.09), then a long tail of callers and writers at ≤ 0.08 /M each (CSC conversion, writers, cell calling,
  doublets, ambient, saturation, mt variants).
- **Demux:** all six samples are single donor. Auto-K tries K = 1..4 (it stops after 3 K without gain), so K = 2..4 are ~75% of
  auto-K. Cost follows SNP observations in the fit cells, not cells (the 263 M sample: 3,274 fit cells, 1.48 M covered SNPs,
  322 thread-s). The export calls demux with `threads = 1`, so the fits are serial and their floating-point sums run in one
  canonical order (relevant for any port: a GPU port must keep that order per accumulator).
- **Non-host:** the three indices (viral, bacterial, fungal) are re-loaded for every sample, 28–53 s; the EM hits its
  500-iteration cap on every sample.
- **Threads and contention** (replay of GSM4006843's export from its kept BAM; same cells and demux input): export core-s at
  `--threads` 1 / 4 / 8 = 117.7 / 119.3 / 116.0; **eight concurrent copies on one node: 116.6–120.8 core-s each (< 3% more)**.
  The node had no other tenants. Host work therefore scales linearly with the number of samples in flight.

**The host-side plan, with its measured capacity per GPU node.** A Clipper H100 node has 56 cores; one is kept for
orchestration. Ingest (M8, §12.10) costs i = 0.085 core-s per M pairs (0.175 with the `.1fq` CRC check). The node's host
capacity R solves R · (W + i) = 55 cores:

| step | what moves | exact because | pooled W (core-s/M) | **node capacity, pooled** (CRC off / on) | worst sample (CRC off / on) |
|---|---|---|---|---|---|
| 0 | nothing (today's CPU callers and export) | — | 2.18 | **24.3 / 23.4 M pairs/s** | 20.2 / 19.5 (263 M) |
| 1 | persistent per-node service loads the non-host indices once | same index bytes | 1.86 | **28.3 / 27.0** | 21.4 / 20.7 |
| 2 | non-host k-mer classify on the GPU (§4.12, M13) | same hash and minimizer rule, sort-merge | 1.43 | **36.3 / 34.3** | 26.4 / 25.3 |
| 3 | demux fits (auto-K + final fit) on the GPU or a bit-exact parallel CPU path | serial-order sums per accumulator kept | 0.58 | **82.7 / 72.8** | 64.3 / 58.2 |

- **The ≥ 20 M pairs/s per node target is met by step 1 on every measured sample** (worst 20.7 with CRC on, the 263 M sample),
  and on the pooled mix already by step 0 (24.3). Step 1 is the persistent service the design already needs (§4.15); it costs nothing in exactness.
- **Step 2** is the planned GPU non-host screen. Its GPU cost is unmeasured (M13), but the §7.2 model puts it at ≥ 500 M pairs/s,
  < 5% of GPU time. It adds 30% host headroom.
- **Step 3 is not needed for the target.** It is the lever if callers grow (Beta adds callers) or if a node is shared. A
  bit-exact demux port is feasible because the CPU fits are serial: one GPU thread per (SNP, K) accumulator walking its cells in
  the CPU's cell order reproduces every float sum (with `-fmad=false`); PCA power iteration and the RNG streams stay on the host.
  Cheaper exact CPU cuts exist but are unverified (e.g. the K = 1 restarts of the final fit may be bit-identical; 28.5 s on
  GSM4006843).
- **Why not just parallelise the export:** its wall is long (serial demux: 352 s at 263 M) but its core-seconds do not change
  with threads (above). Throughput needs many samples in flight per node (~45 at ~1.1 cores each), not more threads per sample.
  Memory per in-flight export is the remaining host risk: not measured (the 8-copy replay ran without trouble).
- **Not on the host any more** (vs the CPU engine): STAR, BAM sort and index, CB refresh, pileup (`stage.pileup` alone is
  100–437 core-s per sample, 1.1–3.6 /M, more than all of W).

### 12.10 M8: host→device ingest (Sonnet subagent; jobs 789223 single GPU, 789286 dual GPU)

Details: `bench/gpu/m8/README.md`, `bench/gpu/results/m8.json`. `m8_ingest.cu`: N decode threads read `.1fq` blocks, zstd-decode
and pack each pair into the GPU layout (R2 2-bit, CB+UMI 2-bit, a quality slot), fill a pinned ring of 4 × 256 MB, a copy
stream does H2D, and a dummy kernel reads every byte and spins a set time per pair. The direct decoder was checked against
the `.1fq` library's decoder on 150 M + 121.8 M reads (0 mismatches) and by host-vs-GPU checksums.

**GSM7656149 (263 M pairs), sustained M pairs/s per GPU, kernel spin 0:**

| decode threads | local NVMe, cold cache | shared filesystem, direct |
|---|---|---|
| 1 | 11.6 | 13.1 |
| 2 | 21.5 | 24.7 |
| 3 | 29.9 | 36.8 |
| 4 | 38.3 | 47.9 |
| 6 | 51.3 | 70.1 |

- **Host cost: 0.085 core-s per M pairs** (5′ GSM6470208: 0.076), of which zstd ~43 ns and packing ~20 ns per pair per thread:
  one thread sustains 12–15 M pairs/s, not the 67 M reads/s of `format-1fq.md` (that figure is decode only, without the
  per-pair packing). **The library's CRC check doubles it to 0.175 core-s per M pairs.**
- **§9 gate (≥ 50 M pairs/s with ≤ 3 cores, < 5% idle): FAIL on cores** (50 M/s needs 5–6 threads, ~4.4 cores). Idle is 0.02%
  at 6 threads against a 20 ns/pair kernel. For this goal it does not matter: at 14–18 M pairs/s per GPU (§12.11) ingest
  needs 1.2–1.5 cores per GPU (2.5–3.2 with CRC).
- **PCIe: 12.3 GB/s pinned H2D** (D2H 14.3), not 53.8; both GPUs of a node sit behind one host link and get 6.2 GB/s each when
  copying together. At 56 B per pair that is still > 100 M pairs/s per GPU.
- **Dual GPU:** two pipelines run at the single-GPU rate each (6 threads: 49.1 and 44.5 M/s from NVMe; 64.5 M/s each from the
  shared filesystem, 2.9 GB/s combined).
- **Staging is not needed for rate:** reading the shared filesystem directly is no slower than cold local NVMe (NFS server
  caching cannot be excluded: the file had been read that day).
- The atlas `.1fq` files store no base qualities (`qual_mode` 0); the quality slot was filled synthetically (real payload
  32 B/pair; within 3%). The mtDNA/SNP pileup's binned-quality input (§2) therefore does not exist in these files.

### 12.11 End to end per node, with what is now measured

| stage | where | measured rate per GPU (or node) | source |
|---|---|---|---|
| `.1fq` ingest (decode + pack + H2D) | host + PCIe | 12–15 M pairs/s per thread; 0.085 core-s/M (0.175 with CRC); H2D 12.3 GB/s | M8 |
| read dedup (exact) | GPU | 3.5 G reads/s | M6 |
| seed search, exact STAR port, no certificates | GPU | 166 M pairs/s (3′), 259 M (5′), 160 M (nuclei) | M3 (78–105 M distinct reads/s × 1/d) |
| windows, stitch, extend, multimapper selection, annotation | GPU | **unmeasured** (M5) | — |
| count sort, UMI collapse, CSR | GPU | 1.6–2.0 G pairs/s (with dedup) | M6 |
| callers, export, non-host EM | host, 55 cores | 27.0–28.3 M pairs/s per node (step 1), 34.3–36.3 (step 2) | §12.9 |

**What this says.**
- **The node is host-bound at 27–36 M pairs/s** until M5 says otherwise, with two GPUs. Each GPU therefore has to deliver only
  14–18 M pairs/s. Seeds plus dedup and counting take ~7 ns per pair (3′), so the stitch stage may use up to ~50 ns per pair,
  i.e. **~100 ns per distinct read: M5 needs ≥ 10 M distinct reads/s per GPU** to keep the node host-bound. §9's 100 M
  single-window reads/s was a target for headroom, not for this goal.
- **Target ≥ 20 M pairs/s per node: met on the host side by measurement** (§12.9, step 1, every sample ≥ 20.7). The GPU side
  is measured for every stage except M5; the earlier prototype (memory note, 2026-10-27) ran probe + stitch at 10–17 M reads/s
  per GPU with a non-STAR stitcher, so the M5 requirement is in reach but not shown.
- **Against a CPU node** (1.0–1.3 M pairs/s end to end): a GPU node at 27–36 M pairs/s is **21–36×** per node, or ~10–18× per
  GPU. The campaign (20.6 T pairs) on 3 GPU nodes: 2.2–2.9 days of compute, below the input-supply bound (§7.4).
- **The GPU is idle most of the time at this point.** Raising the host bound (step 2 then step 3, §12.9) is the way to use it;
  step 3 lifts the host to ~73–83 M pairs/s per node, where the GPUs (and M5) would bind.

### 12.12 M5: STAR's whole mapping path on the GPU, byte-identical (jobs 789536–789772)

Round 2 (2026-10-27/28). Code: `bench/gpu/m5/stitch.cuh` (the port), `bench/gpu/m5/m5.cu` (driver, lanes, comparison),
`src/gpu/stardump/{ReadAlign_oneRead_dump.cpp,m5_dump.h}` (truth). Results: `bench/gpu/results/m5_pbmc1k_v3_2m*.json`.

**Truth.** The campaign's STAR already has an exact per-sequence encoding of a read's mapping result: the sequence cache
(`ReadAlign_seqCache.cpp`, verified in the W3 work: identical outputs with it on). Its blob holds nW, nTr, the best transcript's
index and every field of every transcript that mapping writes (exons, junction motifs and shifts, scores, mismatch and gap counts,
primary flag). `stardump` now also links a copy of `ReadAlign_oneRead.cpp` that writes, for every mapped read, the sequence STAR
mapped (after clipping) and that blob (without the cache's 255-transcript limit), plus `unmapType`; and, once per process, every
alignment parameter and the genome arrays that stitching reads (chrStart/Length, chrBin, the 386,483 sjdb junctions with motifs,
shifts and strands, sjDstart/sjAstart). The gate compares the GPU's blob with STAR's byte for byte, and `unmapType`.

**The port.** One read per thread, from the read sequence to the final transcripts: `qualitySplit` → the seed loop of
`mapOneRead` → `maxMappableLength2strands` (both iDist passes of the sparse SA) → `storeAligns` → `stitchPieces`
(`createExtendWindowsWithAlign`, flanks, `assignAlignToWindow` on the FASTASSIGN path) → `stitchWindowAligns`
(`stitchAlignToTranscript`, `extendAlign`, `blocksOverlap`, the score-range recording) → `multMapSelect` → `mappedFilter` → the
blob. Each function is a line-by-line port of the vendored STAR 2.7.11b; differences are representation only (stitch.cuh header):
winBin is a per-read open-addressing map with the same read/write semantics plus a 1,024-bit prefilter; the include/exclude
recursion runs on an explicit stack of transcripts; transcript copies move only the live exon rows; across windows only the
transcripts that can still pass `multMapSelect`'s score range are kept, in order; `scoreGenomicLength`'s `log2` uses
breakpoints computed on the host with the C library's `log2`. Two STAR quirks reproduced on the way: `WlastAnchor` is a `uint`
set to −1, so the "last anchor" mark never fires; after `MARKER_TOO_MANY_ANCHORS_PER_WINDOW` STAR keeps assigning to windows with
nW = 0 (nothing observable follows).

**Lanes.** Every array has a capacity; a read that would exceed one where STAR's own limit is larger stops and goes to the next
lane, which runs the same code with larger capacities; the last lane is the host with STAR's limits (seedPerReadNmax 1000,
alignWindowsPerReadNmax 10000, seedPerWindowNmax 50, MAX_N_EXONS 20, alignTranscriptsPerWindowNmax 100). A per-read budget of
stitching steps (STAR's include/exclude recursion is exponential in the aligns of a window) moves pathological reads on.

| lane | capacities (windows / aligns per window / exons / transcripts kept) | where | reads (2 M set) |
|---|---|---|---|
| A | 16 / 16 / 8 / 8; budget 1,024 steps | GPU, staged, batches of 131k reads | all 1,991,926 |
| M | 64 / 8 / 8 / 32; budget 2,048 | GPU, staged | 524,577 (26%) |
| B | 128 / 50 / 20 / 64; budget 2,048 | GPU, staged, batches of 32k | 98,152 (4.9%) |
| host | STAR's limits, no budget | 24 host threads, concurrent with the GPU | 5,304 (0.27%) |

**Staging.** Per-phase cycle counters showed the cost was not stitching but window assignment: a multimapper piece (up to
seedMultimapNmax = 10,000 loci) makes `stitchPieces` visit every SA entry, ~2,300–3,000 per read in the wide lanes. Each lane now
runs three kernels per batch: phase 1 (thread per read: seeds, windows from anchors, flanks), the scan (warp per read, lanes over
consecutive SA entries, coalesced, marking the entries whose bin holds a window; exact because the assignment never changes
winBin, so the test gives the answer STAR gets during it), and phase 3 (thread per read: the assignments in STAR's order from the
marks, then stitching, selection, encoding). The scan went from >50% of lane time to 3–8%.

**Results** (pbmc1k_v3, every 16th distinct cDNA sequence: 1,991,926 sequences, 4,118,999 reads; median of 3 after a warm-up):

| version | identical (distinct / reads) | GPU ms | M distinct reads/s per GPU |
|---|---|---|---|
| one kernel, lanes A (16 windows) + B | 100% / 100% | 5,561 | 0.36 |
| + live-row copies, epoch winBin, bitmask | 100% / 100% | 2,558 | 0.78 |
| + staged scan, bigger batches | 100% / 100% | 2,041 | 0.98 |
| + step budget | 100% / 100% | 1,157 | 1.72 |
| + lane M (cascade A → M → B → host) | 100% / 100% | 1,014 | 1.96 |
| + batches of 262k (A, M) and 65k (B) reads (job 789812) | **100% / 100%** | **844** | **2.36** |

STAR-unmapped reads (151,095) and mapped reads (1,840,831) are both 100% identical in every version. The whole sample
(31,870,813 distinct sequences, 66.6 M reads; job 789771) maps in 17.1 s of GPU time (1.86 M distinct reads/s, 3.9 M reads/s).

**Where the time is now:** lane A runs at 6.4 M distinct reads/s (313 ms for all 2 M reads), close to the 10 M requirement; lanes
M and B (the 26% of sequences with more than 16 windows: multimappers whose anchors map up to winAnchorMultimapNmax = 50 times)
take 70% of GPU time. **M5 is below the ≥ 10 M distinct reads/s requirement (2.36 M), by 4.2×.** What would close it, in order of
expected gain: windows of a read stitched in parallel across a warp (exact with two passes: each window's maximum finalized score
is independent of recording, so a prefix max gives STAR's maxScoreMate at each window's start); compact transcripts (32-bit rows)
and per-read state in shared memory; running the M/B lanes concurrently with lane A of the next batch; a cross-sample result
cache (a sequence's result depends on the sequence only; see §12.16).

**Design inputs from the Aligner Brainstorming session** (read only, `singlet-agents/.claude/worktrees/gpu-align-vision`,
`bench/gpu-prototype/{stitch.h,probe.cu,results/RESULTS.md}`): its separate repeat/heavy-read lane became the lane cascade and
the step budget here; its warp-cooperative chaining is the next step above. Its seeding and scoring were not used: they are not
STAR's (about 90–100 genes per sample off by more than 19% in its own gene comparison).

### 12.13 GPU annotation and pileup, identical to the engine's callers (Sonnet subagent; jobs 789562, 789652, 789669–789671)

Details: `bench/gpu/pileup/README.md` (§1 is the specification ported: which records reach the counters, gene assignment,
the multimapper buffer, junctions, SNP AD/DP, chrM bases and indels, every tie-break and clamp), `bench/gpu/results/pileup*.json`.
Kernels (`pileup.cu`): annotate each distinct alignment signature against the gene model (merge-join), expand to reads, M6's
collapse for the gene tuples, and sort + segmented reduce for junctions, SNP AD/DP, chrM bases (total and reverse strand) and
chrM indels. Inputs were built from the engine's own STAR BAM in job /tmp (M5 will supply them; see the order caveat below).
Idea taken from the Aligner Brainstorming session (`gpu-align-vision/bench/gpu-pileup/`, read only): pileup as sort + merge-join +
segmented reduce with an arena allocator; every rule here is the engine's, and the gate is the engine's own accumulators.

| sample | records | GPU ms (8 stages) | reads/s | peak device memory | identical outputs |
|---|---|---|---|---|---|
| GSM4006843 (human 3′ v2) | 60.9 M | 55.3 | 1.10 G | 23.8 GB | **9 of 9** (exon, intron, ambiguous, junctions, SNP AD, SNP DP, chrM, chrM reverse, chrM indels) |
| GSM5411066 (mouse 3′ v3) | 61.0 M | 39.6 | 1.54 G | 16.7 GB | **9 of 9** |
| GSM6470208 (human 5′ v2) | 88.5 M | 63.1 | 1.40 G | 25.1 GB | **9 of 9** |

- Host work removed: the pileup stage plus BAM index and CB refresh, **3.3–7.6 core-s per M read pairs** (campaign stage timers),
  more than all of W after the pileup (§12.9). The STAR BAM sort (17–75 core-s per M pairs) disappears too.
- Two rules the gate caught: N bases count in SNP DP except where the decode's quality marks them (sample-dependent, now read per
  sample); junction ids depend on the CPU's worker order, so junctions compare by (chromosome, donor, acceptor).
- Caveats: multimapper tie rules that depend on STAR's coordinate-sort order need M5 to carry an order key; host-to-device
  transfer of the records from pageable memory (0.8–0.9 s) is 14–20× the compute and must move to pinned, overlapped buffers;
  not ported: the EM-rescue list, poly(A) records, VDJ and guides.
- Cost: 1.1 GPU-hours, 31 core-hours.

### 12.14 A whole sample through the engine with GPU mapping (jobs 789760 build, 789771, 789810)

**Method** (`bench/gpu/m5/inject/`). The engine consumes M5's results through STAR's own sequence cache, so nothing downstream
changes: `singlet-m5inject` is `prod/campaign` c4a1be38 with one addition to `SeqCache::create` (`apply_preload.py`, dev build
only): with `SINGLET_STAR_SEQCACHE_PRELOAD` it inserts (sequence, result) records before mapping. A read whose sequence is in the
cache takes its mapping state from the record (`seqCacheRestore`) instead of from STAR's mapping; with
`SINGLET_STAR_SEQCACHE_VERIFY=1` STAR maps it anyway and counts every record that differs from its own result.
`e2e.sbatch`: every distinct cDNA sequence of the sample → `m5` (lanes A/M/B/host) → records → the full campaign command line
(`--pipeline --output-format 1pz`, run screen, 24 threads) three ways; then `bench/star_accel/out_equiv.py` (the W3 harness)
on every output file. pbmc1k_v3 (3′ v3, 66.6 M read pairs; 3′ samples are not clipped, so STAR maps the raw sequence).
(GSM4006843 was planned; its staged copy had been deleted.)

| | result |
|---|---|
| distinct sequences mapped on the GPU | 31,870,813 in 17.1 s of GPU time (1.86 M distinct/s, 3.9 M reads/s); 31,763,130 records (sequences without N) |
| **verify run**: reads taking the GPU's result, each re-mapped by STAR and compared | **66,004,541 of 66,488,663 (99.27%), 0 mismatches** |
| reads not served (mapped by STAR as usual) | 113,225 with N (not cacheable), ~370 k lost to the cache's 2-way index collisions |
| GPU-fed run vs reference run, every output file (54 files; `out_equiv.py`, then an order-independent check of each file that differs) | every matrix, cell table, pileup, junction, SNP, mtDNA and QC output identical; the differences are the same as between **two CPU-only runs** of the engine (below) |

What differs, and why it is not the GPU (job 789810 ran the reference twice, `ref` and `ref2`, both CPU only):

| file | ref vs ref2 (CPU vs CPU) | ref vs GPU-fed | cause |
|---|---|---|---|
| `star_Unmapped.out.mate1/2`, `nonhost_reads.fq` | differ; same content sorted | differ; same content sorted | STAR's threads write unmapped reads in arrival order |
| `read_stats.tsv`, `star_Log*`, timing fields of `summary.json`, `provenance.json`, `ambient_contamination.json` | differ | differ | timings |
| `pileup_stats.json` `bytes_public` | 959,255,571 vs 959,255,576 | vs 959,255,669 | the summed size of public files that hold timings |
| `nonhost/nonhost_summary.json` `em_iterations` | 458 = 458 | 458 (run 1, job 789771) / **462 (run 2)** | the non-host EM reads `nonhost_reads.fq` in its (varying) order; floating-point sums in a different order converge in a different number of iterations. 0 species above threshold in every run. |

So on a whole real sample every read that took M5's result got STAR's exact mapping state, and the engine's outputs equal a
reference run's up to the run-to-run variation the CPU engine already has. **One CPU-engine finding:** the non-host EM result
depends on the order of its input, which multithreaded STAR leaves nondeterministic; sorting `nonhost_reads.fq` (or the EM's
reads) before the EM would make it deterministic (a CPU-engine change, outside W11; a follow-up). The cache preload is also a
deployable integration path: the GPU fills STAR's cache, the engine is unchanged. Engine wall time per run (24 threads): 7:17
reference, 6:56 GPU-fed: STAR's mapping is ~half cached already in the reference, and every other CPU stage still runs, so this
path proves equivalence, not cost (cost: §12.16).

### 12.15 Ranked bottlenecks per GPU node (after round 2)

Capacity of each stage for one Clipper GPU node (2 × H100 NVL, 56 cores), in M read pairs/s, from the measurements above
(pbmc1k_v3 unless stated; 2.09 reads per distinct sequence there). The node runs at the minimum.

| rank | stage | where | capacity per node | basis | status |
|---|---|---|---|---|---|
| 1 | **M5: seeds → windows → stitching → selection** | GPU | **7.8** (3.9 per GPU; whole sample, job 789771); **9.9** with the larger batches of job 789812 (2.36 M distinct/s on the 2 M set) | measured (§12.12) | **the bottleneck**: 70% of it in lanes M and B (reads with > 16 windows) |
| 2 | host callers + export (W = 2.18 core-s/M), 55 cores | host | 25 (step 0) / 28 (step 1) / 36 (step 2) / 83 (step 3) | measured (§12.9) | not binding until M5 is 3.2× faster |
| 3 | records host→device (pileup input), pageable copies | PCIe | ~70 per GPU as measured (0.8 s per 61 M records); > 100 pinned | measured (§12.13), projected | move to pinned, overlapped buffers |
| 4 | ingest: `.1fq` decode + pack + H2D | host | 0.085 core-s/M: 0.7 cores at 7.8 M/s | measured (§12.10) | fine |
| 5 | annotation + pileup + M6 collapse | GPU | > 1,000 | measured (§12.13, §12.8) | negligible |

- The node is **GPU-bound at 7.8–9.9 M pairs/s**: below the 27–36 M pairs/s host bound and the ≥ 20 M target, which therefore
  holds only as a host-side capacity until M5 improves by about 2× (to 20 M) or 2.8× (to the step-1 host bound).
- Ranked M5 levers (all exact): (a) warp-parallel windows in lanes M and B with the two-pass maxScoreMate (70% of M5 time);
  (b) compact transcripts and shared-memory state in lane A (6.4 M distinct/s alone); (c) lanes M/B overlapped with lane A of
  the next batch (tails); (d) cross-sample result reuse (§12.16), −19% of distinct sequences for a same-chemistry sample now,
  more as a campaign accumulates results.
- Host levers stay as in §12.9: step 1 (per-node non-host indices) and step 2 (M13, a Sonnet subagent's port in progress at this
  stop) are needed only once M5 passes ~25 M pairs/s per node.

### 12.17 Open items

- **M5, the stitch port** (windows, `stitchPieces`, `stitchWindowAligns`, `extendAlign`, scoring, multimapper selection,
  filters) is the remaining unmeasured GPU stage and the equivalence risk (A4). Requirement for the node goal: ≥ 10 M distinct
  reads/s per GPU (§12.11). Truth for it: extend `stardump` to dump windows, aligns and transcripts (or STAR's SAM output of
  the same distinct-read set). This also gives A3 (PC/WC/WA sizes, overflow share), still unmeasured.
- **M13** (non-host screen on the GPU) for host step 2; **M7** (bit-exact barcode correction); **M9** (heavy reads; their cost is
  inside the M3 rates, unseparated).
- **Demux** (host step 3, optional): verify whether the K = 1 final-fit restarts are bit-identical; design the serial-order GPU
  port if callers grow.
- **Memory per in-flight export** on the host (~45 samples in flight per node at step 1).
- **The atlas `.1fq` files carry no base qualities** (M8), so the CPU engine already runs the mtDNA and SNP pileup without real
  base qualities; the "binned quality" payload of §4.10 and §4.13 can be dropped (32 B per pair instead of 53–56).
- M2 on the spec's 8-sample same-study batches; M10 at 1k / 10k / 30k cells; `called_ed` timers (the campaign logs show
  `called_v2`); the 76–80 GB fall in M1 (TLB or HBM pressure; does not affect the design).
- M11 (LCP seed reuse, SBK), M12 (locator) and M14 (answer table): de-prioritised, the seed stage is no longer near the
  bottleneck (§12.6).
