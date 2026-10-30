# M6: read dedup, count sort and UMI collapse on one H100 NVL

Spec: `docs/gpu-engine/exploration.md` §9 M6. Result file: `bench/gpu/results/m6.json` (plus `m6_*.json` per run).
Sources: `m6.cu` (kernels, host reference, JSON), `tuple_dump.patch` (CPU-engine dev patch), `jobs/*.sbatch`.

## Result in one table (real data, H100 NVL, median of 3 after a warm-up)

| | GSM4006843 (3' v2, 66.7 M pairs) | pbmc1k_v3 (3' v3, 66.6 M pairs) |
|---|---|---|
| a. read dedup (exact, full key), ms / G reads/s | 19.2 / 3.46 | 18.6 / 3.57 |
| b. key build + sort + reduce-by-key, ms / G tuples/s | 8.7 / 6.4 | 8.4 / 6.1 |
| c. UMI collapse (split halves + components), ms / G tuples/s | 8.7 / 6.5 | 4.4 / 11.8 |
| d. class select + sort + RLE + CSR, ms | 4.3 | 1.6 |
| tuples (b-d input) | 55.79 M | 51.77 M |
| b-d chain, ms / G tuples/s | 21.6 / 2.58 | 14.5 / 3.58 |
| **chain a+b+c+d per read pair** (66.6 M pairs), ms / **G pairs/s** | 40.9 / **1.63** | 33.1 / **2.01** |
| H2D reads (packed, 2.1 GB) / tuples (1.8 GB), ms | 178 / 145 (12.3 GB/s, pinned) | 178 / 134 |
| D2H of the three CSR outputs, ms | 63 | 17 |
| distinct fraction d (full length, exact) | 0.4049 | 0.4777 |

Pass criteria: >= 1 G pairs/s through the chain: **met** (1.6 and 2.0 G pairs/s, transfers excluded). With non-overlapped
H2D (reads 178 ms + tuples 145 ms) and D2H (63 ms) the same chain is 0.2 G pairs/s: transfers dominate and must be overlapped or
avoided in the real engine (reads would be produced on the GPU there). Identical matrices: **met** (below). Identical across batch
sizes and runs: **met** (below).

"Pairs" in the chain rate are all read pairs of the sample (stage a runs on every R2 read; b-d run on the 55.8 / 51.8 M tuples,
i.e. reads that reach UMI counting). Stage a and stages b-d are not linked per read (no per-sequence annotation exists yet, see
Deviations), so the chain time is the sum of the two independent halves.

## 1. The CPU rules (prod/campaign, commit 28969b87)

Campaign engine call: `singlet --pipeline ...` (recipe `jobs/bam-stream/phase1.sbatch`). Pileup runs on the sorted BAM
(`run_parallel`, `src/pipeline/singlet.cpp:7777`) or streaming (`run`); both feed `count_read`.

**Where the UMI comes from.** STAR is asked for `NH nM AS CR UR CB` (`singlet.cpp:6980-6982`): there is **no `UB` tag**, so the
pileup uses the raw UMI `UR` (`pileup_engine.h:3044-3045` reads `UB`, falls back to `UR`). STAR's own UMI correction is not
applied. The cell is the STARsolo-corrected `CB` (1MM_multi_Nbase_pseudocounts for CB_UMI_Simple, `singlet.cpp:7012-7018`),
mapped to a barcode index by `bc_index_.lookup` (`pileup_engine.h:3011`; reads whose CB is absent or not in the discovered
barcode set are dropped).

**UMI rule: directional, 1 mismatch, on raw UMIs.** `pipeline_mode` forces `umi_dedup_directional = true`
(`singlet.cpp:3348`); `count_read` -> `DirectionalUmiStore::record_read` -> `finalize_molecules`
(`include/singlet/pileup/umi_dedup.h`):
1. `umi_pack_2bit` (`umi_dedup.h:100-112`): A=0, C=1, G=2, **T, N and any other character = 3**, first base in the low bits. So an
   N-containing UMI is identical to the same UMI with T at the N (no N filter anywhere).
2. `record_read` (`:322-337`): reads of one (barcode, gene) with the same packed UMI merge into one entry: `count++`,
   `flags |= flags`, `exon_idx = min`, `intron_idx = min` (smallest index, not first seen).
3. `finalize_molecules` (`:350-418`), per (barcode, gene) group with more than one distinct UMI: sort by (count desc, packed
   asc) (`:385-388`), then for every pair i<j: **edge iff Hamming(i, j) == 1 (masked to `umi_len` bases) and
   `count_i >= 2*count_j - 1`** (`:396-402`), merged by union-find. With count_i >= count_j this is symmetric (equal counts: only
   when count <= 1), so a molecule is exactly a **connected component of the undirected graph**; the sort order and the root
   chosen cannot change the result.
4. A component's evidence is the union over its UMIs (`:404-413`): flags OR, exon_idx min, intron_idx min.
5. Without directional mode or with a single UMI, every entry is its own molecule (`:378-382`).

**Classes** (`label_of`, `umi_dedup.h:298-303`, flags from `pileup_engine.h:2718-2750`): a read sets `kReadExonOnly` (1) if all
aligned bases are exonic, `kReadIntron` (2) if some are intronic; molecule flags are the OR. Exon-only evidence -> *spliced*
(exon matrix, row = merged-exon index); intron evidence only -> *unspliced* (intron matrix, row = intron index); both ->
*ambiguous* (exon matrix at the min exon index **and** a per-gene ambiguous count, `emit`, `:360-372`). Counts per (feature,
cell) are summed in `SparseAccumulator::to_csc` and **clamped to 65535** (`sparse_accumulator.h`, uint16).

**Which reads become tuples** (all upstream of the dedup, unchanged by M6; recorded for completeness):
- unmapped, supplementary dropped; primary NH == 1 with MAPQ < `min_mapq` (20) dropped (`:3037-3039`); no CB/UB-UR handled as above;
- strand filter per block (`:3107-3116`, 10x forward unless auto-strand flips it);
- gene = the single gene whose exons the read overlaps (exon priority); a read touching exons of two genes is *multigene* and
  not counted (`:3127-3144`); failing that, the single gene whose introns it overlaps (`:3146-3177`, `classify_read`
  `:2718-2750`); intron hits of another gene are ignored;
- NH > 1: buffered by qname until all alignments are seen; counted only if every alignment resolves to the same gene
  (`resolve_mm_parallel` `:2682-2700`, cross-worker merge `:1346-1433`), otherwise dropped to the EM-rescue list (not in this
  matrix);
- a read without UMI would be counted directly (`count_read` else-branch): **none occurred** (`direct` = 0 in both samples).
- Per-worker stores are finalized separately (`:1435-1443`); genes do not span chromosomes and cross-worker multimappers are
  routed to the store that has the (barcode, gene), so this equals one global collapse.

## 2. Truth data (`tuple_dump.patch`)

Applies to `src/../include/singlet/pileup/pileup_engine.h` of prod/campaign (`patch -p1`). With `SINGLET_M6_TUPLES=path` set it
(1) writes every tuple exactly as it enters `record_read` (cell index, gene, exon feature, intron feature, packed UMI, flags,
UMI length, "UMI had a non-ACGT char", direct flag; 32 B) to `path`, and (2) after the molecules are finalized and the worker
accumulators merged writes the CPU's exon / intron / ambiguous matrices (`to_csc` triplets) to `path.truth`. Unset, it does
nothing; no output of the engine changes. Built non-PGO (`jobs/m6_build.sbatch`, job 789186, conda env `pbench`, binary
`$W/jobs/w11/m6/bin/singlet-m6dump`); ran with the campaign flags (`jobs/m6_dump.sbatch`, jobs 789189 test, 789201 array):

| sample | reads | tuples dumped | engine wall (16 threads, non-PGO) | CPU matrices (nnz exon / intron / amb) |
|---|---|---|---|---|
| pbmc2m (test) | 2.0 M | 1.49 M | 112 s | 507 374 / 383 927 / 747 |
| GSM4006843 | 66 699 625 | 55 792 540 | 582 s | 24 604 078 / 6 031 894 / 39 356 |
| pbmc1k_v3 | 66 601 887 | 51 766 531 | 353 s | 3 979 682 / 3 557 558 / 158 074 |

The tuple files (ids only, no sequences) were stored under `$W/jobs/w11/m6/data/` (approved) and deleted after the last run.

## 3. Kernels (`m6.cu`)

- **a. read dedup.** R2 from `singlet-w3 decode --r2` is packed on the host to 2 bits (A0 C1 G2 T3) in `NW` u64 words **plus the
  exact length**; no clipping. Key = (words, length). 64-bit hash (fmix chain), CUB `SortPairs` on the top 40 bits (5 passes),
  neighbour compare of the full key, then an **exact repair**: any run of equal 40-bit keys that holds more than one sequence is
  re-ordered by (hash, words, length, index) with a small comparison sort (1.1-1.9 k elements per 66 M reads, so the repair path is
  exercised on every run). Ids are ranks in (hash, sequence, length) order; the representative is the smallest read index.
  Batches are deduplicated independently and their distinct lists merged by the same procedure into global ranks, so the
  global id of every read is independent of the batch size. Reads containing N (0.14% / 0.17%) are set aside (STAR splits them
  by `qualitySplit`; not modelled here) and counted in the JSON. Observed R2 lengths: GSM4006843 20-95 nt (98% are exactly 95),
  pbmc1k_v3 up to 91 nt; `NW=4` (128 nt) was used for the main runs. `NW=3` (the 96 nt layout of the spec) gives the same
  d and 19.5 ms (`m6_GSM4006843_nw3.json`); reads longer than 32*NW abort instead of being clipped.
- **b. count key.** Key = cell | gene | UMI with widths taken from the data (GSM4006843: 17+16+20 = 53 bits, pbmc1k_v3:
  18+16+24 = 58 bits); payload = exon(24) | intron(24) | flags(3) (flags are checked to be derivable from exon/intron presence
  on all 107 M tuples). CUB radix sort of (key, payload) on only the used bits, `ReduceByKey` into (count, flags OR, min exon,
  min intron) per exact (cell, gene, UMI). Batches pre-aggregate, then the concatenated unique lists are sorted and reduced again.
- **c. UMI collapse.** Exactly the rule of §1. Only (cell, gene) groups with >= 2 distinct UMIs are active (31.0 M / 13.1 M entries).
  Candidate pairs by the **split-halves trick**: sort active entries by (segment, low half) and by (segment, high half); inside
  a group of equal key test Hamming == 1 and the count rule (two passes: count, scan, write). No all-pairs pass. Components by
  deterministic min-label hooking (`atomicMin` on labels, 2-3 rounds); per-component flags OR / exon min / intron min by
  `atomicOr` / `atomicMin` (integer, order-free); each root emits one molecule.
- **d. CSR.** Molecules -> three key arrays (exon class incl. ambiguous, intron class, ambiguous-per-gene), select, radix sort of
  (cell | feature), run-length encode, CSR (cell-major, feature ascending, u16 clamped counts, `indptr` by binary search).

Determinism rules followed: no float arithmetic anywhere in the kernels (`-fmad=false` anyway); every output order comes from
radix sorts on fixed keys or scans; the only atomics are `atomicMin`/`atomicOr` on integers and plain flag stores of one value.
`thrust::sort` is used only for the repair set and allocates temporary memory internally (inside the timed region, ~ms).

## 4. Equivalence with the CPU engine

The GPU consumes the dumped tuples (arbitrary file order = CPU thread order) and its CSR is compared entry by entry with the CPU
matrices; additionally a literal host copy of the CPU algorithm (`host_reference`, O(n^2) union-find per group) is run on the
same tuples and compared.

| sample | class | CPU nnz | identical | only GPU | only CPU | value differs | sum of counts (GPU = CPU) |
|---|---|---|---|---|---|---|---|
| GSM4006843 | exon | 24 604 078 | 24 604 078 | 0 | 0 | 0 | 45 120 052 |
| GSM4006843 | intron | 6 031 894 | 6 031 894 | 0 | 0 | 0 | 6 408 333 |
| GSM4006843 | ambiguous | 39 356 | 39 356 | 0 | 0 | 0 | 39 714 |
| pbmc1k_v3 | exon | 3 979 682 | 3 979 682 | 0 | 0 | 0 | |
| pbmc1k_v3 | intron | 3 557 558 | 3 557 558 | 0 | 0 | 0 | |
| pbmc1k_v3 | ambiguous | 158 074 | 158 074 | 0 | 0 | 0 | |

All entries identical, for every batch size, and equal to the host-literal reference. The SHA-256 of the GPU CSR equals the
SHA-256 of the CPU truth converted to the same CSR layout. The collapse is non-trivial: 31.0 M entries lie in multi-UMI groups
and 55 315 (v2, 10-nt UMIs) / 923 397 (v3, 12-nt UMIs) Hamming-1 edges pass the count rule. Truth is taken after
`finalize_molecules` and the worker merge, before EM rescue and ambient correction (those act on later copies).

## 5. Determinism

3 timed runs + 1 warm-up per configuration inside one process, plus a second process (job 789241) for GSM4006843:

| sample | batch (M reads / tuples) | CSR SHA-256 (all 4 batch sizes and all runs identical) | read-id SHA-256 |
|---|---|---|---|
| GSM4006843 | full, 4, 16, 64 | `c83df0512ffe44ead688b5ba1f5a81f9bea8823a4bb03dd51a0e8073435d67c5` (= CPU truth) | `ecc010e6e38b1a93c104d863bbe5c725c8e880b9bed04487e759c9eeb3650150` |
| pbmc1k_v3 | full, 4, 16, 64 | `71b760d02efa2b2b4d29741f5c16c3cf6104a7f48de983dc75405651fc702069` (= CPU truth) | `d488e12c536944ede7683c2655805ddebb90001d5c5a0d0b441d572d4118f10b` |

The read-id hash is over the global distinct id of every read in read order (a function of the data alone). Note the 64 M tuple
batch is the whole sample for the tuples (55.8 M / 51.8 M) but splits the 66.6 M reads into two batches. Batch-local distinct
fractions (d_local) are the cost of batching: GSM4006843 0.584 / 0.496 / 0.415 at 4 / 16 / 64 M reads against 0.405 globally;
pbmc1k_v3 0.650 / 0.578 / 0.488 against 0.478.

## 6. Distinct fraction and the M2 definition

M2 (`distinct_fraction.py --len 90`): truncate to 90 nt, drop reads shorter than 90 and reads with N, BLAKE2b digests.
`m6 --m2len 90` applies that definition: pbmc1k_v3 **0.4739** on 65 114 751 reads (M2: 0.474, 65.1 M). The headline d here is
different by definition: it keeps every read length (shorter reads are distinct keys by length) and compares full sequences:
0.4777 (pbmc1k_v3) and 0.4049 (GSM4006843). For comparison, M2 measured 0.298 on the deeper 109 M v2 sample GSM7036403; the cause of the difference to this 66.7 M v2 sample was not investigated.

## 7. Per-step detail, versions, deviations

- Environment: node g05x, NVIDIA H100 NVL 94 GB, 132 SMs, driver 610.43.02 (CUDA 13.3 driver API, runtime 12.4), nvcc 12.4.131 from
  `cuda-12.4.1-devel-ubuntu22.04` via apptainer, `-arch=sm_90 -fmad=false`, CUB/thrust from the toolkit. SM clock was idle
  (345 MHz) at start, max 1785 MHz; no clock lock. Peak device memory 13.7 GB (arena).
- Timing: CUDA events around each phase, medians of 3 after one warm-up; phase gaps include the small host synchronisations (a few
  4-byte D2H reads per stage). H2D from pinned memory, D2H and SHA-256 excluded from the chain times and reported separately.
- GPU time used: about 0.2 GPU-hours (6 jobs, 15 min wall). Engine runs were on CPU nodes (short-cpu), no GPU-hours.

Deviations and limits:
1. The CPU engine ran in separate CPU jobs and the tuple files were staged under `$W` (approved), not in the GPU job.
2. Stage a and stages b-d are not linked per read: the per-sequence gene annotation (DSA) does not exist yet, so a's ids do not feed
   b's keys. The tuples come from the CPU engine's per-read output as the spec says; the chain time is a + (b+c+d).
3. Host work is outside the numbers: FASTQ parse and 2-bit packing of 66.6 M reads took 75-86 s single-threaded (CPU) here; in
   the GPU design the decode feeds the device directly. H2D/D2H (above) are not overlapped.
4. Reads with N are excluded from stage a (no `qualitySplit` pieces).
5. Radix sort of the read hash uses 40 bits as in §4.1 plus the exact repair, not a hash-only dedup.
6. Spec batch sizes {4, 16, 64} M tuples: 64 M exceeds the sample's tuple count, so it equals the unbatched run for b-d.
7. The CC iteration count is tiny (2-3) because raw 10/12-nt UMIs rarely form chains; deep-error samples or UMI-poor protocols
   (short UMIs, large groups) would raise it and the split-half group sizes (not measured here).
8. Only 3' v2 and v3 samples were run; UMI length must be constant, the key must fit 64 bits (cell+gene+UMI), and feature indices
   must be < 2^24 - 1 (checked at load, the program aborts otherwise).
