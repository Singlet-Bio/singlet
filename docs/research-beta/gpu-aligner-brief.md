# Brief: design a GPU-native Singlet engine that is 10× faster per H100

Written 2026-10-27 by the orchestrating session for a design agent (workstream W11). It holds the context needed to start
without the conversation that produced it. Your job is **design first**:
- the algorithms, data structures and system architecture;
- microbenchmarks that prove or kill each idea;
- a phased build plan with kill criteria.

Building the full engine comes after Zach approves the design.

## 1. Why this exists

Singlet re-processes every public single-cell RNA-seq sample from SRA, about 70,000 samples and 1.58 PB of raw reads,
with an exact genome aligner.
- **The engine:** a vendored, modified STAR 2.7.11b plus our own pileup engine.
- **Where it runs:** today, on CPU nodes on GVSU's shared Clipper cluster.
- **The problem:** the campaign would occupy ~20 CPU nodes for months, a real burden on other Clipper users. Two or three
  GPU nodes would not be.
- **Zach's decision (2026-10-27):** design a GPU-optimised pipeline aiming at **an order of magnitude more throughput per
  H100** than a CPU node. Nothing is off limits: custom data structures, low-level CUDA, careful CPU↔GPU communication,
  cuSPARSE / CUB / nvCOMP where they help.

**Target hardware and goal (Zach, 2026-10-27, after the exploration in `docs/gpu-engine/exploration.md`):**
- **Hardware:** design for Clipper's H100 NVL nodes specifically.
- **Goal:** the most performant pipeline those nodes allow that **reproduces the CPU engine's results**. Profile carefully
  and optimise exhaustively to push H100 node performance to its maximum.
- **Why:** GPU is a critical part of having more options for running analyses, beyond the Beta campaign.
- **Plan:** microbenchmarks M1, M2 and M10 run first on Clipper, under the Alpha-first rules.

**Relation to Research Beta.** This is a research track. It must not delay the Beta Software Freeze (BSF) for the CPU
engine; see [rollout-plan.md](rollout-plan.md). If the GPU engine reaches equivalence before or during the campaign, it
can take over from that point under the same release gate. "Same outputs" is the contract, not "same code".

## 2. What the engine must produce (the contract)

Per sample, from 10x Chromium 3′ (v2/v3/v4) and 5′ (v1/v2) libraries, human (GRCh38) or mouse (GRCm39):

| Output | Today's source | Notes |
|---|---|---|
| cell × gene UMI matrix (exonic, intronic, spliced/unspliced/ambiguous) | STARsolo + pileup | UMI dedup with STAR's 1MM_CR rules |
| cell barcode correction | STARsolo `1MM_multi_Nbase_pseudocounts` | against the 10x whitelists |
| cell calling (`called_ed`), ambient, doublets | pileup callers | need per-cell counts only |
| splice junctions and PSI | STAR SJ.out + pileup | public junction table |
| intron retention / multimapper rules | pileup (#46) | NUMT-aware |
| mtDNA heteroplasmy, chrM variants | per-position pileup on chrM | needs base qualities (binned) |
| donor SNP allele counts → genotype-free demux | per-position pileup at a SNP panel | private outputs |
| sex and ancestry labels | from counts and allele tallies | public labels |
| non-host species per cell | sketch screen on unmapped reads | [rollout-plan.md](rollout-plan.md) §3e |

Requirements:
- **Exactness.** Singlet sells itself as an exact genome-wide aligner; that is its reach advantage over pseudo-aligners.
  The GPU engine must match STAR's alignments, or beat them on real-truth benchmarks. Proposed acceptance (part of your
  design to firm up):
  - ≥ 99.9% identical primary alignments on uniquely mapped reads;
  - the multimapper classification identical, or explained;
  - matrices, junctions and calls within the tolerances the benchmark harness accepts;
  - every capability benchmark in `bench/public/` passing.
- **Determinism.** Byte-identical outputs across runs, GPU counts and batch sizes. Ordering from floating-point and atomic
  operations must not leak into results: use sort-based, ordered reductions.
- **Reads, BAMs and genotypes never become public outputs.** No BAM is needed at all unless a design wants one internally.

## 3. The current CPU engine: where time goes

**Pipeline per sample:**
1. Read pairs come in as `.1fq`, Singlet's 2-bit packed, zstd-blocked read format (`docs/format-1fq.md`).
2. STAR runs with STARsolo; ~450 source files live under `src/star/` and `include/singlet/star/`.
3. STAR writes unsorted BAM bins, then sorts by coordinate on disk.
4. Index, then a CB-tag refresh.
5. The pileup engine makes one pass in coordinate order, where all modalities accumulate (`process_read_parallel`).
6. The non-host screen.
7. Export.

**Measured profile at 4 threads** (audit-2026-10-27.md §3):

| stage | 66 M pairs, idle node (845 s) | 110 M (2,196 s) | 260 M (5,097 s) |
|---|---|---|---|
| STAR mapping + unsorted bins | 515 s (61%) | 630 (29%) | 2,820 (55%) |
| disk BAM sort | ~145 (17%) | 1,080 (49%) | 990 (19%) |
| index, CB refresh, pileup | 85 (10%) | 486 (22%) | 1,287 (25%) |
| reference load, export | 58 (7%) | | |
| non-host screen | +94 s (+10%) | | |

**CPU baseline to beat.** About 32k read pairs/s per core for mapping, and ~20k pairs/s per core end to end. A 64-core
Clipper node therefore does ~1–1.3 M pairs/s end to end when packed. The measured job-packing penalty is 1.28×.

**The 10× target:** one H100 sustaining **≥ 10–13 M read pairs/s end to end**, about one 66 M-pair sample every 5–7
seconds, all outputs included. Mapping alone must therefore run well above that rate. Whole-campaign compute is ~250–450k
core-hours on CPU.

**What we already know about STAR's costs:**
- **FASTASSIGN (PR #41):** window assignment scanned every suffix-array locus of every seed, up to 10,000 per seed, and
  took 25–28% of mapping. Skipping no-op calls saved 19%.
- **Too-short reads:** 4.1% of reads but 19% of mapping on 3′, and 7.4% of reads but **55%** of mapping on 5′. The cost
  is in the seed search on repetitive sequence, before any bound exists. An early exit after the search saved nothing.
- **Transcriptome-first (L1):** gene-concordant clean reads are 23–24% of mapping time, but only perfect reads (~4%) are
  provably equivalent from per-locus data. Mismatched reads can have competing genome-wide alignments.
- **Sequence cache:** a sequence → alignment cache exists (`SeqCache.h`); 10x libraries have heavy sequence duplication.
- **Serial share:** about 40% of the 4-thread wall does not scale (358 s serial in the Amdahl fit).

## 4. The hardware

- **GPU:** NVIDIA H100 NVL.
  - Memory: 94 GB of HBM3 at ~3.9 TB/s, 50 MB L2, 132 SMs.
  - Compute: ~60 TFLOPS FP32 and fast INT32 paths.
  - Host link: PCIe Gen5 (~64 GB/s each way in theory). An NVLink bridge connects a pair of cards.
- **Clipper:** GPU nodes with H100 NVL. The clipper skill notes list 2× H100 NVL with 56 cores and 768 GB RAM; confirm
  the node count with `sinfo`. Partitions are `gpu` (5 days), `short-gpu` (4 h) and `debug-gpu` (30 min), with
  `--gres=gpu:nvidia_h100_nvl:1`. Older V100S and Quadro RTX 8000 cards are there for development only.
- **Software:** no `--fakeroot` image builds on Clipper, so pull an NVIDIA CUDA devel image with apptainer and build
  inside it. Record the CUDA, driver and compiler versions.
- **Memory budget:**
  - Human genome 3.1 Gbp, 2-bit packed: ~0.8 GB.
  - STAR's genome + suffix array + SAindex: ~25–32 GB.
  - Both human and mouse indexes fit in HBM at once, with room left for batches.
  - Host RAM (768 GB) can stage many samples.

## 5. Why a naive port is only ~10× at best, and where the room is

The CPU bottleneck is **latency-bound random memory access**: the suffix-array binary search, the SAindex lookups, and
walking up to thousands of loci per seed. An H100 has ~10× the memory bandwidth of a CPU node, but each random 4–8 byte
access wastes most of a 32-byte sector, and warps diverge on read-dependent control flow. NVIDIA's GPU STAR
(Parabricks `rna_fq2bam`) reportedly gives ~8–9× on a 4-GPU node, and sometimes nothing. Older GPU seed-and-extend
aligners report ~10×.

**Reaching 10× per GPU therefore needs algorithmic change, not a port.** Directions to evaluate. They are hypotheses for
you to measure, not decisions:

1. **Don't align a sequence twice.**
   - Within a sample, and across a batch of samples, sort or hash cDNA sequences (or their 2-bit packs) and align each
     distinct sequence once.
   - Measure the distinct-sequence fraction on real 3′ and 5′ samples; deep libraries are often 30–70% duplicate.
   - Barcodes and UMIs are carried, never aligned.
2. **Make memory access coherent.**
   - Batch millions of reads, sort the queries by their first k-mer or minimizer, and walk the index in that order, so
     neighbouring threads touch neighbouring memory (the classic GPU trick for suffix-array and FM-index search).
   - Sort-and-merge instead of random lookup. The non-host sketch already does this on CPU (`min_sketch.h`).
3. **Choose the index data structure for the GPU, not STAR's.** Candidates:
   - a direct-address k-mer table (k = 12–14) into suffix-array intervals, as STAR's SAindex does but laid out for
     coalescing;
   - a sampled or compressed suffix array, or an FM-index with GPU-friendly occurrence blocks;
   - a minimizer hash index with position lists, as in minimap2;
   - a learned or interpolation index over the suffix array;
   - a two-level index: a transcriptome (exon and junction sequences) for the common case, the genome for the rest.
   For every candidate you must show that it finds the same maximal mappable prefixes (MMPs) or seeds STAR would, or
   define exactly where it differs and why that is acceptable.
4. **A fast path for the easy majority, an exact path for the rest.**
   - Most 10x cDNA reads are short (90–100 nt), transcriptomic and nearly perfect.
   - Prove a cheap filter that certifies "this read's best alignment is X, and no competitor scores ≥ X − δ anywhere in
     the genome". For example, use k-mer uniqueness certificates and a mismatch budget: a read whose k-mers are all
     unique to one locus, with ≤ n mismatches, cannot have a competitor.
   - Send only uncertified reads (repeats, too-short reads, multimappers) to the full exact search. The L1 study showed
     that naive transcriptome-first is not exact; certificates are the missing piece.
5. **Repeats and too-short reads.** On CPU, 55% of 5′ mapping time goes to 7% of reads. Design a bounded,
   repeat-aware search: precomputed repeat classes, seed caps with exactness arguments, or a separate GPU kernel tuned for
   high-occurrence seeds.
6. **Extension and stitching.**
   - Warp-per-read or thread-per-read banded alignment.
   - Bit-parallel edit distance (Myers / BitPAl) for the mismatch budget.
   - Splice-aware chaining that uses the annotated and atlas junction database (junction table v1, 145 packs) as
     preferred anchors, with STAR's scoring (`--alignSJ*`, `--score*` parameters) reproduced exactly.
7. **Everything after alignment is sort, reduce and sparse.** That is GPU-native.
   - **Counting:** CUB radix sort by (cell, gene, UMI), then segmented reduce. UMI 1-mismatch collapse is a per-(cell,
     gene) graph step.
   - **Matrices:** build with cuSPARSE or custom CSR.
   - **Pileup:** SNP, mtDNA and introns become sort-by-position histograms; there is no BAM and no coordinate sort of
     records.
   - **Barcode correction:** whitelist lookup in a GPU hash table, with STAR's pseudocount rules.
   - **Determinism:** fixed sort keys, no unordered atomics in outputs.
8. **I/O and pipelining.**
   - Decode `.1fq` on the GPU (nvCOMP zstd plus 2-bit unpack).
   - Pinned host buffers, CUDA streams, double or triple buffering, and many samples in flight, so the GPU never waits on
     the host.
   - Measure PCIe: 10 M pairs/s × ~60 B per pair is ~0.6 GB/s, far below the link, so the risk is latency and
     synchronisation, not bandwidth.
9. **Non-host screen on the GPU.** The sketch merge-scan against a ~25 GB index is a bandwidth-friendly sorted-merge
   problem, and should cost seconds per batch. The stage-2 species alignment (§3e of the rollout plan) can reuse the same
   mapper with small per-species indexes.

## 6. What you deliver

**1. Design document,** `docs/gpu-engine/design.md`. It covers:
- the data structures, with byte layouts and their HBM footprint;
- the kernels and how they are scheduled;
- the host↔device pipeline;
- the exactness argument for every shortcut;
- a memory and throughput model per stage, which must reach ≥ 10 M pairs/s end to end or explain the gap;
- risks and alternatives.

**2. Microbenchmarks** in `bench/gpu/`, each answering one question with numbers on an H100:
- the distinct-sequence fraction and the dedup throughput on real samples;
- the seed search over candidate indexes, in queries per second, with sorted vs unsorted query order;
- the certificate coverage: what fraction of real reads the fast path certifies, and how fast;
- extension and stitching throughput;
- sort and reduce counting throughput;
- GPU `.1fq` decode throughput;
- the end-to-end PCIe pipeline with dummy kernels.

**3. Phased build plan,** each phase with a measurable gate and a kill criterion:

| Phase | Gate |
|---|---|
| P0 | Profile and baseline the CPU engine on the shared datasets. Define the metrics. |
| P1 | Design and microbenchmarks. Kill the project if the modelled end-to-end rate is < 5 M pairs/s per H100. |
| P2 | Seed search plus the certified fast path. Seeds match STAR's MMPs on real reads. |
| P3 | The full mapper. Primary alignments ≥ 99.9% identical to STAR on real 3′, 5′, mouse and deep samples. |
| P4 | Counting, barcode correction and the pileup on the GPU. Matrices and calls pass `perf_equiv` and the capability benchmarks. |
| P5 | End to end with multi-sample batching. ≥ 10 M pairs/s sustained, and deterministic. |
| P6 | A 50-sample validation against the CPU engine, then the release gate. |

**4. A handoff note,** `docs/research-beta/ws-w11-gpu.md`, written at every stop.

## 7. Real data and truth

- **Datasets:** the same 8 real datasets as [bam-stream-protocol.md](bam-stream-protocol.md) (3′ at 66 / 110 / 260 M
  pairs, 5′, mouse, multi-run, the Kang genotype pool, high-mtDNA), plus pbmc1k_v3, pbmc5p and the GRCm39 nuclei and
  neuron sets used for FASTASSIGN.
- **Equivalence reference:** the CPU engine at the pinned commit, through `bench/star_accel/txf_equiv.py` and
  `perf_equiv`.
- **Truth:** the published benchmark cohort (`bench/public/`, PR #40) with real ground truth only. Simulation may only
  supplement a real-data result, never stand alone.

## 8. Rules

- **Licences:** MIT code. You may port and modify STAR (MIT). Don't copy or link proprietary code (Parabricks); comparing
  against it as a black box is fine if a licence allows it on Clipper. Don't run 10x software; compare against 10x's
  published outputs.
- **Clipper:**
  - Use the `clipper` skill. One shared bridge: never start or restart it; if it is down, stop and tell Zach.
  - Never run `hpc/clipper/sync.sh`. The login node is only for `sbatch`, `squeue`, `sacct` and log reads.
- **Alpha first:** while Research Alpha runs, jobs:
  - use `--nice=10000` and small array throttles;
  - stay off partitions where Alpha has pending jobs;
  - stop if an Alpha job is waiting.
  GPU partitions are rarely used by Alpha, but check `squeue -u $USER` anyway, and state GPU-hours before each batch.
- **No Purdue Anvil work. No Cloudflare R2, D1 or Worker changes.**
- **Storage:** reads and alignments live only in job `/tmp`; long-term storage is ~10 TB on Clipper for everything.
- **Naming:** nothing is named "v2" or "store"; the data format is `.s1`.
- **Process:** Sonnet 5.5 subagents for gruntwork (benchmark scripts, sweeps); hard design questions come back to the
  orchestrating session. Commit handoffs so a fresh session can resume.
- **Files:** you own `docs/gpu-engine/`, `bench/gpu/` and a new `src/gpu/` tree. You don't edit `src/star/`,
  `include/singlet/pileup/` or `scripts/atlas_rerun/`; other workstreams own them.

## 9. Pointers

- `docs/research-beta/rollout-plan.md`: the plan, gates and workstreams (branch `docs/beta-rollout-plan`, PR #57).
- `docs/research-beta/audit-2026-10-27.md` §3: the CPU profile, cost model and ranked CPU speed-ups.
- PR #41: FASTASSIGN, the too-short read analysis and the transcriptome-first ceiling (`bench/aligner/txome_ceiling.py`).
- `docs/format-1fq.md`: the read format and its decode speed (~67 M reads/s single-thread on CPU).
- `include/singlet/star/` and `src/star/`: STAR as vendored, including `SuffixArrayFuns.h`, `ReadAlign*.h`,
  `SeqCache.h`, `TxFirst.h`, `SoloReadBarcode.h`.
- `include/singlet/pileup/`: the pileup engine and callers; `nonhost/min_sketch.h` is the sorted-merge sketch.
