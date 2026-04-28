# Alignment Algorithm Proposals — Step-Change Ideas for Singlify

**Date**: April 2026  
**Context**: After exhaustive optimization of STAR's suffix-array binary search (singlet-lite branch, ~48% faster than stock), the remaining bottleneck is **random DRAM access** — the `G[SA[mid]]` pointer chase that consumes ~42% of wall time. Incremental optimizations within this paradigm are nearly exhausted. This document proposes creative but rigorous ideas for step-change improvements, drawing on techniques from pseudoaligners, alignment-free methods, and learned index literature.

**Constraint**: We require **full genome CIGAR alignments** for the singlet-pileup algorithm (variant calling, exon/intron classification, SNP detection, splice junction discovery). Approaches that sacrifice positional or base-level information are non-starters for our primary output — but they can serve as *fast pre-screening stages* that feed into targeted full alignment.

---

## Table of Contents

1. [K-mer Pre-Screening for Read Localization](#1-k-mer-pre-screening-for-read-localization)
2. [Cached Genomic Blocks](#2-cached-genomic-blocks)
3. [Learned Custom Genome Index](#3-learned-custom-genome-index)
4. [Abundant Pattern Acceleration](#4-abundant-pattern-acceleration)
5. [Near-Identical Read Clustering](#5-near-identical-read-clustering)
6. [Multi-Species K-mer Filtering](#6-multi-species-k-mer-filtering)
7. [Interleaved Multi-Read Binary Search](#7-interleaved-multi-read-binary-search)
8. [Embed-and-Filter SA Comparison](#8-embed-and-filter-sa-comparison)
9. [Minimizer-Seeded Full Alignment](#9-minimizer-seeded-full-alignment)
10. [Two-Phase Architecture: Pseudoalign → Targeted Align](#10-two-phase-architecture-pseudoalign--targeted-align)
11. [Adaptive Runtime Reference Prioritization](#11-adaptive-runtime-reference-prioritization)
12. [Species-Agnostic Self-Tuning Architecture](#12-species-agnostic-self-tuning-architecture)
13. [Design Principles](#13-design-principles)

---

## 1. K-mer Pre-Screening for Read Localization

### Concept

Before running STAR's full SA binary search, use a lightweight k-mer hash table (inspired by kallisto/minimap2) to **localize each read to 1–5 candidate genomic regions**. Then run full SA alignment only within those regions, dramatically reducing the SA search space.

### Mechanism

1. **Index construction** (one-time): Build a hash table of all unique 21-mers (or 25-mers) in the genome, each storing a compact list of genomic positions. Use minimap2-style frequency filtering to exclude k-mers occurring in >0.05% of the genome (repetitive elements).
2. **Pre-screen** (per-read): Extract 3–5 spaced k-mers from the read. Look up each in the hash table. Intersect/cluster the position sets to identify 1–5 candidate genomic windows (±500bp around the consensus position).
3. **Targeted alignment**: Instead of searching the full 12.4GB SA, restrict the binary search to the SA range corresponding to each candidate window. This can be done by looking up the window boundaries in the SAindex and only searching within those SA subarrays.

### Expected Impact

- **SA search space reduction**: From ~29 billion positions to ~5000 positions per candidate window → ~5000× fewer random DRAM accesses.
- **k-mer lookup cost**: 3–5 hash table lookups per read at ~50ns each = ~250ns. Current SA binary search is ~1–5μs per seed. Net savings could be 5–20× on the alignment kernel.
- **Memory cost**: ~8–12GB for a 21-mer hash table with position lists (filtered). Fits alongside the SA in 32GB.

### Risks

- **Reads with no unique k-mers**: ~5–10% of short (90bp) reads may have all k-mers in repetitive regions. These must fall back to full SA search. The frequency filter threshold is the critical tuning parameter.
- **Spliced reads**: A k-mer spanning a splice junction won't match the genome. Need to use 3+ spaced k-mers so that at least one falls entirely within an exon.
- **Index build time**: 21-mer hash table construction is ~30 min (similar to current STAR index generation). Can be reused across samples.

### Complexity

Medium. Requires building a new index alongside the SA, modifying the `maxMappableLength` entry point to optionally restrict the SA range, and handling the fallback path. Estimated ~2–3 weeks of development.

### Precedent

This is essentially what minimap2 does (minimizer hash → chain → align), but we keep STAR's SA infrastructure for the alignment step. The key insight is using k-mer lookup *only for localization*, not as a replacement for full alignment.

---

## 2. Cached Genomic Blocks

### Concept

Organize the genome into ~64KB blocks that correspond to "k-mer neighborhoods" — regions that share similar k-mer profiles. Pre-load the most frequently accessed blocks into a pinned cache. When STAR's binary search needs to access `G[SA[mid]]`, check if the relevant genome block is already in the pinned cache, avoiding the cold DRAM access.

### Mechanism

1. **Offline profiling**: Run a representative sample (~1M reads) through STAR. Record which genome positions are accessed during `G[SA[mid]]` comparisons. Build a heat map of genome access frequency.
2. **Block construction**: Partition the genome into ~64KB blocks (matching L2 cache line boundaries). Rank blocks by access frequency from the profiling step.
3. **Pinned cache**: At startup, `mlock()` the top N blocks (where N × 64KB fits in the target cache budget, e.g., 256MB → 4096 blocks). These stay in physical memory and likely in L3.
4. **Hot-path check**: Before each `G[SA[mid]]` access, check if `SA[mid] >> 16` indexes a pinned block. If yes, access is ~5ns (L3). If no, access is ~80ns (DRAM). The check itself is a single table lookup.

### Expected Impact

For scRNA-seq data, **expression is Zipf-distributed**: ~80% of reads come from ~1000 genes. The exonic regions of those genes represent ~5–15MB of genome — easily fitting in the pinned cache. Expected hit rate: **60–80% of SA comparisons** hit a pinned block → 3–5× speedup on the genome access kernel → **20–35% end-to-end improvement**.

### Risks

- **Profile dependency**: The hot blocks vary across tissues/species. Would need to either (a) profile per-sample via a first pass on a subset of reads, or (b) use a universal hot-gene list (top 1000 expressed genes across tissues).
- **Cache pollution**: `mlock()`-ing 256MB may reduce available cache for other structures (SA, SAi). Need careful memory budget management.
- **Diminishing returns**: If NUMA interleave already distributes genome well, the marginal improvement from pinning may be smaller.

### Complexity

Low–medium. The profiling and block construction are offline scripts. The runtime modification is a ~20-line check in `compareSeqToGenome`. Estimated ~1 week.

### Precedent

Similar to database buffer pool management. The key innovation here is using RNA expression priors to predict which genome regions will be hot — fundamentally different from general-purpose caching because we have domain knowledge.

---

## 3. Learned Custom Genome Index

### Concept

Replace or augment STAR's suffix array with a **learned index** — a neural network or piecewise-linear model that maps a read's k-mer signature directly to a small set of candidate genomic positions. This is the most radical proposal but has the highest ceiling.

### Prior Work

- **BWA-MEM2 LISA branch**: Uses learned indexes for the seeding phase. Achieves **4.5× speedup on seeding** but requires ~120GB memory (impractical).
- **Learned index structures** (Kraska et al., SIGMOD 2018): Replace B-tree indexes with neural network models that predict the position of a key in a sorted array. For read-only databases (which genome indexes are), this can be dramatically faster.
- **Recursive Model Index (RMI)**: A hierarchy of models — top-level model predicts which leaf model to use, leaf model predicts approximate position, then local search corrects. Memory efficient.

### Proposed Mechanism

1. **Training data generation**: For each 21-mer in the genome, record (21-mer_hash → SA_position). This gives us ~3 billion (key, position) pairs.
2. **RMI construction**: Build a 2-level Recursive Model Index:
   - **Level 1**: 256 linear models, selected by the first 8 bits of the 21-mer hash.
   - **Level 2**: 65,536 linear models within each Level 1 bucket.
   - Each linear model: `predicted_pos = a × key + b` (2 floats = 8 bytes per model).
   - Total model size: 65,536 × 8 = 512KB (fits in L2 cache!).
3. **Lookup**: Hash the read prefix → Level 1 model → Level 2 model → predicted SA position ± error bound. The error bound determines how many SA entries need to be checked via traditional comparison.
4. **Error bound management**: For 99% of 21-mers, the prediction error should be <100 SA positions. For the remaining 1% (repetitive regions), fallback to traditional SAindex lookup.

### Expected Impact

If the RMI can predict SA positions within ±50 entries, we replace STAR's binary search (which does log₂(range) ≈ 3–10 comparisons with DRAM access each) with **one model evaluation (arithmetic, L2-cache-resident) + one verification compare**. This could yield **3–8× speedup on the seeding kernel** → **15–35% end-to-end improvement**.

The key advantage over BWA-MEM2 LISA is that RMI models are tiny (512KB) while LISA uses huge learned index structures (120GB). We sacrifice some prediction accuracy for dramatically lower memory.

### Risks

- **Model accuracy**: Genome suffixes are not truly random — repetitive elements create dense clusters in SA-space where linear models predict poorly. May need special handling for Alu, LINE, SINE regions.
- **Verification cost**: If the prediction error is >100 for 10% of lookups, the fallback cost dominates. Need to characterize the error distribution empirically.
- **Build complexity**: Training 65K linear models on 3B data points requires a one-time offline pipeline (~1 hour, parallelizable). The model format must be carefully designed for fast loading.
- **Correctness**: Must produce bit-identical output to current STAR. The learned index only narrows the search — the final comparison is identical.

### Complexity

High. Requires building the RMI training pipeline, integrating it into STAR's index loading, and modifying `maxMappableLength` to use the RMI for initial SA range prediction. Estimated ~4–6 weeks.

---

## 4. Abundant Pattern Acceleration

### Concept

In scRNA-seq, read abundance follows a power law. The top ~100 read sequences may account for 5–10% of all reads, and the top ~10,000 may account for 30–50% (depending on library complexity). **Cache the alignment results for the most frequent read sequences** and replay them without re-aligning.

### Mechanism

1. **First pass** (on first block of reads): Align normally using STAR. After each alignment, insert (R2_sequence → alignment_result) into a hash table.
2. **Frequency counting**: After the first block (~500K reads), identify sequences seen ≥N times (N=3 for high-complexity, N=2 for low-complexity libraries).
3. **Cache lookup** (subsequent blocks): Before invoking STAR alignment, hash the R2 sequence and check the cache. If present and frequency ≥ threshold, return the cached alignment directly.

### Extensions

- **Fuzzy matching**: Group reads that differ by ≤1 base (likely sequencing errors in the same molecule). Use the LSH (locality-sensitive hashing) approach: hash the read with 2–3 different hash functions that tolerate 1 mismatch. If any hash matches a cached high-abundance read, return its alignment.
- **Per-barcode caching**: In scRNA-seq, the same gene is often sequenced many times within one cell. Per-barcode caches could be smaller and have higher hit rates.

### Expected Impact

This is a **pure engineering optimization** with guaranteed correctness (exact match → identical alignment). For a typical 10x dataset:
- Cache hit rate for exact match: ~5–15% (depends on library complexity and sequencing depth)
- Hit rate with 1-mismatch fuzzy matching: ~15–30%
- Each cache hit saves the full STAR alignment pipeline (~1–5μs) and replaces it with a hash lookup (~50ns) → **100× per-read speedup for hits**
- Expected end-to-end: **10–25% speedup** (varies dramatically with library complexity)

### Risks

- **Memory**: Caching 100K unique 90bp sequences with alignment results = ~100K × (90 + 200) bytes ≈ 30MB. Very manageable.
- **Low-complexity libraries**: Already addressed by our per-chunk consecutive dedup. The cache extends this to non-consecutive duplicates across chunks.
- **High-complexity libraries**: Hit rates may be too low to justify the hash lookup overhead on every read. Need an adaptive threshold.

### Complexity

Low. A hash table wrapper around the STAR alignment call. Estimated ~3–5 days. The fuzzy matching extension adds ~1 week.

### Integration with .1fq barcode sort

The .1fq format already sorts reads by barcode. Within a barcode group, reads are more likely to be duplicates (same transcripts sequenced multiple times from one cell). The cache could be reset per barcode group, keeping it small and hot.

---

## 5. Near-Identical Read Clustering

### Concept

Extend the exact-duplicate dedup (proposal #4) to handle **near-identical reads** — reads that differ by 1–3 bases due to sequencing errors but originate from the same molecule. Instead of aligning each independently, align a **representative** and derive the others' alignments by applying the known differences.

### Mechanism

1. **Clustering**: After sorting reads by LSH signature (or by first 32 characters + length), group reads that differ by ≤3 edit distance. The representative is the most common sequence in each cluster.
2. **Representative alignment**: Align the representative through STAR normally. Record the full CIGAR, mapped position, and NH value.
3. **Derived alignment**: For each non-representative read in the cluster, compute the diff against the representative (positions + substitutions). Apply the diff to the representative's CIGAR to produce the derived alignment. If the diff conflicts with the CIGAR (e.g., a substitution at a splice junction boundary), fall back to full alignment.

### Expected Impact

Near-identical clusters are much larger than exact duplicates because sequencing error rate is ~0.1% per base at Q30. For 90bp reads, ~8% of reads have ≥1 error → many "unique" reads are actually error variants of abundant reads.

- Cluster size distribution: ~30–50% of reads could be derived from a representative
- Alignment cost: representative alignment (~3μs) + derived alignment (~100ns per variant read)
- Expected end-to-end: **15–30% speedup** on top of exact-duplicate caching

### Risks

- **CIGAR derivation correctness**: When a sequencing error falls in a splice junction region, the derived CIGAR may be wrong (the error could change the splicing). Conservative approach: any diff within 5bp of a splice junction triggers full re-alignment.
- **Clustering cost**: O(N log N) sorting + O(N) scanning is fast but adds constant overhead. For high-complexity libraries with few clusters, the overhead dominates.
- **NH/multi-mapping**: If the representative maps to multiple locations, derived reads must inherit all alignments, each modified independently. This is correct but complex.

### Complexity

Medium–high. The clustering is straightforward, but the CIGAR derivation logic must handle all edge cases (indels, soft clips, multi-exon alignments). Estimated ~3–4 weeks.

---

## 6. Multi-Species K-mer Filtering

### Concept

In a typical scRNA-seq experiment, **5–30% of reads come from non-host species**: E. coli contamination, mycoplasma, ambient RNA from co-cultured species, phiX spike-in, ribosomal RNA, mitochondrial reads that map poorly, etc. Currently, STAR aligns every read against the full human genome, and non-host reads simply fail to align — wasting the full alignment cost.

**Pre-filter reads using a k-mer-based species classifier** to route only host reads to STAR alignment. Non-host reads are immediately classified (species label + discard or route to secondary aligner) without touching the expensive genome index.

### Mechanism

1. **Build a species k-mer database**: For each species of interest (human, mouse, E. coli, mycoplasma, phiX, rRNA), extract all unique 21-mers. Store in a compact data structure — either:
   - **Bloom filter** (space-efficient, ~1 bit per k-mer, ~4GB for all species combined, <0.1% false positive)
   - **Kraken2-style minimizer database** (compact hash of minimizers → taxonomic ID)
   - **Bifrost CDBG + species coloring** (most expensive but most accurate)

2. **Classification** (per-read): Extract 5–10 k-mers from the read. Look up each in the species database. Majority vote determines species assignment.
   - If ≥70% of k-mers are human → route to STAR alignment
   - If ≥70% are a non-host species → classify and skip alignment
   - If ambiguous → route to STAR alignment (conservative)

3. **Integration**: Pre-filter happens before STAR, operating on the .1fq decoded reads. Could be integrated into the singlify parent process during the read-feeding phase.

### Expected Impact

If 20% of reads are non-host and the pre-filter correctly identifies 90% of them:
- 18% of reads skip STAR alignment entirely
- Pre-filter cost: ~100ns per read (5 hash lookups)
- STAR alignment cost saved: ~3μs per read
- **Net end-to-end speedup: ~15–18% on alignment phase** (proportional to non-host fraction)

Additional value: the species classification is useful metadata for QC (contamination detection, ambient RNA profiling).

### Risks

- **Conservative threshold needed**: False positives (host reads classified as non-host) cause missed alignments. Must set species threshold conservatively or use human-specific confirmation (require ≥3 human-unique k-mers to pass).
- **Highly similar species**: Human-mouse xenograft samples have many shared k-mers. The filter must handle species with significant homology.
- **k-mer collision**: Short k-mers (21-mers) in conserved genes (ribosomal, mitochondrial) may be shared across species. Use 31-mers or species-unique k-mers only.

### Complexity

Medium. Bloom filter construction is a one-time step (~1 hour). Per-read classification is ~50 lines of code. Integration into the singlify pipeline is straightforward. Estimated ~2 weeks.

### Precedent

Kraken2 does exactly this for metagenomic classification but uses a more complex LCA (lowest common ancestor) algorithm. For our binary host/non-host classification, a simpler approach suffices. kallisto's D-list is conceptually identical — a decoy set that filters out non-target reads before pseudoalignment.

---

## 7. Interleaved Multi-Read Binary Search

### Concept

The fundamental bottleneck in STAR's SA binary search is the **data-dependent pointer chase**: `G[SA[mid]]` requires loading SA[mid] from DRAM, then loading G at that position from DRAM — two sequential random accesses that cannot be overlapped within a single read's search path.

**Interleave the binary searches of multiple reads** so that while one read waits for its DRAM access, another read's comparison is being processed. This converts latency-bound sequential access into throughput-bound parallel access.

### Mechanism

1. **Batch formation**: Collect B reads (B = 8–16) that need SA binary search at the same iteration depth.
2. **Prefetch round**: For each of the B reads, compute `mid = (lo + hi) / 2` and issue `__builtin_prefetch(&G[SA[mid]])`. Also prefetch `SA[mid]` itself.
3. **Compute round**: For each of the B reads (now with data in cache from previous prefetch), do the genome comparison and update `lo` or `hi`.
4. **Repeat**: Continue interleaved prefetch/compute rounds until all B reads have converged.

This is essentially **software-pipelined memory access** — a well-known technique in database systems (Amortized Branching Programs, AMAC) applied to suffix array search.

### Expected Impact

DRAM latency is ~80ns on our Xeon Gold 6226. With B=8 interleaved reads:
- **Memory-level parallelism**: 8 outstanding cache misses vs. 1 → potential 4–6× throughput improvement on the DRAM-bound kernel.
- **Overhead**: Maintaining B search states (lo, hi, read_ptr, strand) requires ~64 bytes per read × 8 = 512 bytes — fits in L1.
- **Expected STAR alignment speedup**: 30–40% (the 42% DRAM-bound portion becomes ~4× faster, but the interleaving overhead and non-DRAM portions limit overall gain).
- **Expected end-to-end singlify speedup**: ~15–20%.

### Risks

- **Synchronization complexity**: Reads converge at different rates. When one read finishes its binary search (lo == hi), it must be replaced with a new read. The interleaving logic must handle variable-length searches efficiently.
- **Register pressure**: Maintaining 8 independent search states may cause register spills on x86-64 (which has only 16 GPRs). May need to reduce B to 4.
- **Non-uniform SA search depth**: Some seeds need 2 SA comparisons, others need 10+. Reads with short searches will idle while waiting for slow reads.
- **Code complexity**: This transforms the innermost hot loop from simple sequential code into a complex state machine. Debugging and correctness verification become much harder.

### Complexity

High. This is the most invasive change to STAR's alignment kernel. Requires rewriting `maxMappableLength` as a batched state machine. Estimated ~4–6 weeks, with extensive correctness testing.

### Precedent

[Kocberber et al., "Meet the Walkers: Accelerating Index Traversals for In-Memory Databases"](https://doi.org/10.1109/MICRO.2013.38) — demonstrated 3–5× speedup on B-tree lookups using interleaved access. The STAR SA binary search is structurally identical to a B-tree lookup.

---

## 8. Embed-and-Filter SA Comparison

### Concept

Inspired by Accel-Align's seed-embed-extend idea: instead of comparing the full read sequence to `G[SA[mid]]` at each binary search step, compare **compact 8-byte embeddings** (hashes/signatures) first. If the embedding rejects the candidate, skip the expensive genome comparison entirely.

### Mechanism

1. **Embedding precomputation** (at index build time): For each position i in the genome, compute an 8-byte embedding: `embed[i] = hash(G[i..i+k])` for some fixed k (e.g., k=32). Store the embedding array alongside the genome.
2. **Read embedding** (per-read): Compute the embedding of the read prefix: `read_embed = hash(read[0..k])`.
3. **SA comparison modification**: In `compareSeqToGenome`, before doing the full word-level XOR comparison, compare `read_embed` vs `embed[SA[mid]]`. If they differ in more than T bits (Hamming distance), the comparison result is already determined (they definitely differ in the first k bases) — skip the genome load.

### Expected Impact

The embedding check is **L1-cache-resident** (the embedding array for the active SA range can be prefetched) and costs ~2ns vs ~80ns for the genome load. If the embedding filter eliminates 50–70% of genome loads:
- **SA comparison cost reduction**: 50–70% fewer DRAM accesses
- **Expected speedup**: 15–25% on alignment kernel → 8–15% end-to-end

### Risks

- **Memory cost**: 8 bytes per genome position → 24GB additional memory for human genome. This is too much. Must use sparse embedding (one per SA entry, or one per 64-byte genome block).
- **Embedding collision**: If two genome positions have the same embedding, the filter won't help. Need a high-quality hash with low collision rate.
- **SA range locality**: In the binary search, consecutive `mid` values access nearby SA regions. The embeddings may already be in cache via hardware prefetch, reducing the marginal benefit.

### Variant: SA-Resident Embedding

Instead of a separate embedding array, store a 4-byte hash alongside each SA entry (expand PackedArray from 33 bits to 65 bits). This avoids the extra memory array but requires modifying the SA data structure. The 4-byte hash has higher collision rate but zero additional memory overhead beyond the SA expansion.

### Complexity

Medium. The embedding precomputation is straightforward. Modifying `compareSeqToGenome` to check embeddings first requires ~30 lines. But the memory layout decisions require careful benchmarking. Estimated ~2–3 weeks.

---

## 9. Minimizer-Seeded Full Alignment

### Concept

Replace STAR's SA-based seed finding with **minimap2-style minimizer seeding**, but keep STAR's alignment and splice-junction machinery for the extension phase. This is a deeper architectural change than proposal #1, replacing the SA entirely for the seeding step.

### Mechanism

1. **Minimizer index construction**: Extract (k,w)-minimizers from the genome (k=15, w=10). Store in a hash table: minimizer → sorted list of genome positions. Total index size: ~7GB for human genome.
2. **Seeding**: For each read, extract minimizers. Look up positions. Filter by frequency (skip minimizers occurring >F times in the genome). Collect candidate genomic positions.
3. **Chaining**: Minimap2-style DP chaining to find co-linear seed chains across introns. Use known splice junctions from GTF to score intron-spanning chains.
4. **Extension**: For each candidate chain, use STAR's existing alignment machinery (SW or stitchPieces) to produce the full CIGAR.

### Expected Impact

- **Seeding cost**: Hash table lookup is O(1) per minimizer, ~5–10 minimizers per 90bp read → ~50–100ns total. vs. STAR's SA binary search at ~1–5μs per seed → **10–100× faster seeding**.
- **Chaining cost**: O(N log N) where N = number of candidate positions. For well-filtered minimizers, N < 100 → ~500ns.
- **Total alignment**: ~1–2μs per read (seeding + chaining + extension) vs. current ~3μs → **1.5–3× faster per read**.
- **Memory**: ~7–10GB for minimizer index vs. ~12–24GB for SA + SAindex. Net memory reduction.

### Risks

- **Sensitivity loss**: Minimizer seeding may miss some valid alignments that STAR's MMP (Maximum Mappable Prefix) finds — particularly for reads with many errors or short unique regions. STAR's exhaustive MMP from every read position is more sensitive than positional minimizer sampling.
- **Novel splice junction discovery**: minimap2's chaining can discover novel junctions, but STAR's stitchPieces is more sophisticated. Need to verify junction fidelity.
- **Engineering cost**: This is a near-complete rewrite of STAR's seeding layer. Must maintain byte-identical output on test sets.

### Complexity

Very high. This effectively replaces STAR's core algorithm with minimap2's. Estimated ~2–3 months. Would be a major branch divergence.

### Recommendation

Consider this as a **long-term architectural direction** rather than a near-term optimization. The minimizer approach is fundamentally better suited to modern hardware (hash table lookups are cache-friendly; SA binary searches are not). But the engineering investment is large and the sensitivity comparison needs thorough evaluation.

---

## 10. Two-Phase Architecture: Pseudoalign → Targeted Align

### Concept

The most radical proposal: run **pseudoalignment first** (kallisto/piscem-style) to determine which gene each read comes from, then run **targeted full alignment** only against the relevant genomic region.

### Mechanism

1. **Phase 1 — Pseudoalignment**: Build a transcriptome + intron CDBG (similar to piscem's splici index). For each read, pseudoalign to determine the equivalence class (set of candidate genes). This is ~10× faster than full alignment.
2. **Phase 2 — Targeted genome alignment**: For each read assigned to a specific gene (or small gene set), extract the genomic region for that gene (exons + introns + 1kb flanks) and run full STAR alignment only against that region.
   - Pre-build mini-SA indexes for each gene region (~10–50KB genome per gene).
   - The mini-SA fits in L2/L3 cache → eliminates the DRAM bottleneck entirely.
3. **Fallback**: Reads that pseudoalign to large ECs (≥10 genes) or fail pseudoalignment go through full-genome STAR alignment.

### Expected Impact

- **Phase 1 cost**: ~100–200ns per read (pseudoalignment)
- **Phase 2 cost**: ~200–500ns per read (mini-SA alignment, fully cache-resident)
- **Fallback**: ~10–20% of reads need full alignment (~3μs each)
- **Weighted average**: ~500ns per read vs. ~3μs current → **5–6× speedup on alignment**
- **End-to-end singlify**: ~3–4× faster (alignment is ~60% of pipeline runtime)

### Risks

- **Pseudoalignment accuracy**: If Phase 1 assigns a read to the wrong gene, Phase 2 will misalign it. Need to compare EC assignment accuracy against STAR's direct genome alignment.
- **Novel features**: Reads from unannotated genes, novel splice junctions, or intergenic regions won't pseudoalign. The fallback path must handle these gracefully.
- **Memory**: Need to hold both the CDBG index (~4GB) and the genome+SA (16–32GB) simultaneously.
- **Engineering complexity**: Two separate alignment engines with a routing layer between them. Correctness testing must cover the phase boundary.

### Variants

- **Transcriptome-first SA** (simpler): Build a separate SA for just the transcriptome (~100MB). Try transcriptome alignment first. If it maps uniquely, use that alignment without querying the genome SA. Only fall back to genome SA for unmapped or multi-mapped reads. This is simpler than full pseudoalignment but captures most of the benefit.
  - *Note*: We tested a simpler version of this (transcriptome-first experiment in singlify-perf) and found marginal gains because the transcriptome SA + genome SA are both large. The key difference here is using a **pseudoaligner** (k-mer-based, not SA-based) for phase 1, which is much faster.

- **Gene-region STAR** (moderate): Don't build mini-SAs. Instead, after pseudoalignment, set STAR's SA search bounds to the gene region's SA range (computed from the gene's genomic coordinates). This reuses the existing SA infrastructure while narrowing the search space.

### Complexity

Very high for the full two-phase architecture. The "Gene-region STAR" variant is medium complexity (~3–4 weeks). Estimated ~2–3 months for the full version.

---

## 11. Adaptive Runtime Reference Prioritization

### Concept

Proposals #2 (cached genomic blocks) and #4 (abundant pattern cache) share a critical weakness: they assume we know *a priori* which genomic regions or read sequences will be hot. Proposal #2 suggests profiling a representative sample or using a universal hot-gene list. Proposal #4 bootstraps from the first block of reads. Both are **static after initialization** — they commit to a priority order and never revise it.

The reality is that expression is **dataset-specific, tissue-specific, and protocol-specific**. MALAT1 dominates in most 10x data but is absent in nuclear RNA-seq with polyA selection. Ribosomal genes dominate failed libraries but are negligible in well-prepared ones. Hemoglobin genes dominate blood samples. Hard-coding any priority order is overfitting to past benchmarks.

**Adaptive Runtime Reference Prioritization** replaces static cache tiers with a fully dynamic, self-reorganizing system that learns the expression landscape *during* alignment and continuously re-prioritizes reference access order.

### Mechanism

1. **Shard the reference**: Divide the genome into ~10,000–50,000 shards of ~64KB each (matching L2 cache line granularity). Each shard encompasses a contiguous genomic region. Gene-aware sharding: shard boundaries align to gene/intergene boundaries where possible, so that a shard typically contains one gene's exonic/intronic content.

2. **Hit counter array**: Maintain a compact array of per-shard hit counts (50K × 4 bytes = 200KB — fits in L2). Every time a read maps to a genomic position, increment the corresponding shard's counter.

3. **Priority queue / tiered lookup**: After every N reads (N = 50K–100K), re-sort shards by hit count. The top-K shards (K chosen to fill the L3 cache budget, e.g., K=300 at 64KB = 19.2MB) are promoted to the "hot tier":
   - **Hot tier** (L3-pinned via `mlock()` + `madvise(MADV_WILLNEED)`): The reference pages for these shards stay resident. SA comparisons against hot-shard positions complete in ~5ns (L3) vs ~80ns (DRAM).
   - **Warm tier** (next ~2000 shards): Not pinned, but the OS page cache naturally caches frequently-accessed pages. Expected ~30ns access.
   - **Cold tier** (remaining shards): Full DRAM access. ~80ns.

4. **K-mer routing with dynamic priority**: If combined with proposal #1 (k-mer pre-screening), the k-mer hash table can store shard IDs alongside genomic positions. When multiple candidate shards match a read's k-mers, **try the highest-priority shard first**. If it produces a high-quality alignment, skip the remaining candidates. This converts the shard priority into alignment order priority — reads are steered toward cache-hot regions first.

5. **Online reorganization**: The priority update is cheap (partial sort of 50K integers every 100K reads ≈ <1ms, amortized to <0.01ns/read). The `mlock()`/`munlock()` transitions are more expensive (~10μs per page) but happen at most a few hundred times per reorganization epoch.

6. **Warm-start for known organisms**: For well-studied organisms (human, mouse), we *can* ship a default priority order based on corpus-wide expression statistics (from our 32,547 .1pz files). But this is a **warm start, not a hard constraint** — the adaptive system revises it within the first 100K reads. Unknown organisms start with uniform priority and converge within ~500K reads.

### Why This Is Better Than Static Caching (#2)

| Dimension | Static (#2) | Adaptive (#11) |
|-----------|------------|----------------|
| Species | Requires per-species profiling | Self-adapts to any reference |
| Tissue | Fixed priority (e.g., "top 800 genes") | Discovers tissue's actual expression |
| Protocol | Ignores protocol-specific biases | Captures enrichment biases (e.g., nuclear RNA) |
| Library quality | Cannot respond to contamination | Demotes contaminated regions after initial observation |
| Novel genes | Cannot prioritize unannotated regions | Promotes any hot region regardless of annotation |
| Runtime cost | Zero overhead | ~0.01ns/read amortized (counter increment + periodic sort) |
| Convergence | Instant (pre-computed) | ~100K–500K reads (~1–5 seconds at current speeds) |

The convergence period is the only trade-off. For a 40M-read dataset, 500K reads is 1.25% of the run — the adaptive system is fully optimized for 98.75% of the data.

### Expected Impact

Same ceiling as proposal #2 (20–35% speedup from cache-hot genome access) but **robust across species, tissues, and protocols** without manual tuning. For samples with unusual expression profiles (e.g., a zebrafish embryo dataset where no human corpus statistics apply), the adaptive approach captures gains that static caching misses entirely.

### Integration with Other Proposals

Adaptive priority is an **infrastructure layer** that improves every proposal that touches genome access:
- **#1 K-mer pre-screening**: Route reads to hot-tier shards first.
- **#4 Abundant pattern cache**: The cache naturally fills with reads from hot shards.
- **#7 Interleaved search**: Prioritize interleaving reads that target cold-tier shards (where DRAM overlap matters most).
- **#10 Two-phase**: The hot-tier shard set *is* the dynamic equivalent of the static "mini-SA" gene regions.

### Precedent

- **Database adaptive indexing ("cracking")**: Idreos, Kersten & Manegold (CIDR 2007). Database indexes that physically reorganize themselves in response to query patterns. The first query partitions the data; subsequent queries refine. Our shard priority is a lightweight form of cracking.
- **CPU branch prediction**: Hardware learns which branches are taken and speculatively prefetches along the hot path. Our system does the same for genome access patterns.
- **JIT compilation / PGO**: Profile-guided optimization collects runtime statistics and re-optimizes code paths. We're doing PGO at the data structure level.
- **LRU / LFU caching** (database buffer pools): PostgreSQL, MySQL etc. all dynamically promote/demote pages based on access frequency. Our shard priority is a domain-specific LFU cache with genomics-aware shard boundaries.

### Complexity

Low–medium. The hit counter array and periodic sort are trivial (~50 lines). The `mlock()` promotion/demotion requires careful memory accounting to stay within the cache budget. K-mer routing priority integration depends on whether proposal #1 is implemented first. Estimated ~2 weeks standalone, ~1 week if built on top of #1 or #2.

---

## 12. Species-Agnostic Self-Tuning Architecture

### Concept

Proposals #1–#11 are described using human genome characteristics (3.1GB genome, 24GB SA, 19.25MB L3 cache). But the singlify pipeline must support **any species with an Ensembl reference genome** — potentially 200+ organisms ranging from 100MB (yeast) to 32GB (axolotl). Hard-coding human-specific parameters (shard sizes, cache budgets, frequency thresholds, k-mer lengths) creates an engineering burden where every new species requires tuning.

**Self-tuning** means that all optimization parameters are derived at runtime from the reference genome's properties and the hardware's cache hierarchy — not from constants.

### Mechanism

1. **Reference-proportional parameterization**: At genome load time, compute:
   - `genome_size` = len(G)
   - `sa_size` = genome_size × entry_bits / 8
   - `l3_capacity` = detected from `/sys/devices/system/cpu/cpu0/cache/index3/size`
   - `l2_capacity` = detected from cache topology
   - `dram_capacity` = available physical memory
   - `n_genes` = count from GTF
   - `mean_gene_span` = avg genomic extent per gene

2. **Derived parameters**:
   - `shard_size` = max(64KB, `l2_capacity` / 8) — shards fit L2 associativity
   - `n_shards` = `genome_size` / `shard_size`
   - `hot_tier_shards` = `l3_capacity` / `shard_size` — fill L3 exactly
   - `kmer_freq_threshold` = max(10, `genome_size` / 50M) — scales with genome size
   - `pattern_cache_size` = min(100K entries, `l2_capacity` / 300) — keep cache L2-resident
   - `bloom_filter_bits` = 10 × `genome_size` / 21 — 10 bits per k-mer for low FPR
   - `adaptive_epoch` = max(50K, `n_genes` × 2) — enough reads to observe most genes

3. **Species-neutral index build**: A single `singlify index` command takes any FASTA + GTF and produces:
   - Standard STAR genome directory (SA, SAi, Genome, etc.)
   - K-mer hash table for pre-screening (#1) — parameters derived from genome size
   - Shard map with gene-aware boundaries (#11)
   - Optional species Bloom filter (#6) — built from a multi-species k-mer database
   All tools share a unified index directory analogous to CellRanger/STARsolo's `--genomeDir`.

4. **Runtime auto-tuning protocol**:
   - **Phase A (cold start, reads 0–epoch)**: All proposals active but in "learning" mode. Pattern cache filling. Shard counters accumulating. Species filter in passthrough mode (classifying but not filtering, to measure FPR).
   - **Phase B (warm, reads epoch–10×epoch)**: Adaptive priority converges. Pattern cache stabilizes. Species filter engages if FPR measured in Phase A was acceptable.
   - **Phase C (hot, remainder)**: Fully optimized. Parameters locked (no more reorganization overhead). Periodic micro-adjustments if expression shifts dramatically (e.g., sequential processing of multiple samples with different cell types).

### Why This Matters

Without self-tuning, supporting N species requires:
- N species-specific profiling runs to determine hot genes
- N tuned parameter sets
- N validation benchmarks
- Manual re-tuning when reference genomes update (Ensembl releases twice yearly)

With self-tuning:
- One codebase, zero species-specific parameters
- `singlify index <any_genome.fa> <any_annotation.gtf>` → ready to run
- Performance automatically adapts to genomes of any size, any gene content, any expression landscape

### Scaling Considerations

| Genome | Size | SA (full) | SA (sparse D=3) | Hot Tier (L3=19MB) | Notes |
|--------|------|-----------|------------------|--------------------|-------|
| S. cerevisiae | 12 MB | 96 MB | 32 MB | **Entire genome fits L3** | No optimization needed |
| D. melanogaster | 140 MB | 1.1 GB | 370 MB | 19 MB (14%) | Most genes fit hot tier |
| D. rerio | 1.4 GB | 11 GB | 3.7 GB | 19 MB (1.4%) | Similar to human ratio |
| H. sapiens | 3.1 GB | 24 GB | 8 GB | 19 MB (0.6%) | Current target |
| M. musculus | 2.7 GB | 21 GB | 7 GB | 19 MB (0.7%) | Nearly identical to human |
| A. mexicanum (axolotl) | 32 GB | 256 GB | 85 GB | 19 MB (0.06%) | Requires sparse SA; hot tier even more critical |

For small genomes (yeast, fly), the entire reference fits in cache and none of our optimization proposals matter — alignment is already fast. For large genomes (axolotl, wheat), the adaptive system becomes *more* important because the cold tier is proportionally larger and DRAM stalls dominate even more.

### Complexity

Medium. The parameter derivation logic is straightforward (~100 lines). The unified index build wraps existing tools. The auto-tuning phases are a thin state machine in the singlify orchestrator. Estimated ~2–3 weeks, but this is foundational work that should be done early because it shapes the interfaces for all other proposals.

---

## 13. Design Principles

The two new proposals (#11, #12) expose principles that should govern all future optimization work:

1. **Never hard-code what you can learn.** Any parameter derived from human-genome benchmarks is a liability when the reference changes. Prefer runtime-derived or data-adaptive parameters over constants.

2. **Assume nothing about expression.** Expression profiles vary by species, tissue, developmental stage, protocol, library quality, and sequencing depth. Optimizations that depend on specific expression patterns must have an adaptive fallback.

3. **Design for the reference you haven't seen.** Every optimization must answer: "What happens when the genome is 10× bigger? 10× smaller? From a species with no prior expression data?" If the answer is "it breaks" or "it regresses," the design needs a self-tuning layer.

4. **Convergence speed matters.** An adaptive system that takes 10M reads to converge wastes 25% of a 40M-read dataset. Target convergence within 1–2% of total reads (100K–500K for a typical scRNA run). Use warm-start priors when available, but never require them.

5. **The cascade model still applies.** Adaptive priority doesn't change the fundamental architecture: `read → cache → filter → localize → targeted align → fallback`. It changes *how each stage populates itself* — from static configuration to online learning.

---

## Priority Ranking

Based on expected impact, implementation complexity, and risk:

| Rank | Proposal | Expected Speedup | Complexity | Risk | Recommendation |
|------|----------|-----------------|------------|------|----------------|
| 1 | **Abundant Pattern Acceleration** (#4) | 10–25% | Low | Low | **Do first** — pure win with guaranteed correctness |
| 2 | **Adaptive Runtime Reference Prioritization** (#11) | 20–35% | Low–Med | Low | **Do early** — subsumes #2 with zero hard-coding, adapts to any species |
| 3 | **K-mer Pre-Screening** (#1) | 15–30% | Medium | Medium | **Do second** — highest ceiling for medium effort; #11 makes it adaptive |
| 4 | **Multi-Species K-mer Filtering** (#6) | 15–18%* | Medium | Low | **Do third** — synergizes with #1, useful QC metadata |
| 5 | **Species-Agnostic Self-Tuning** (#12) | N/A (meta) | Medium | Low | **Foundational** — do alongside #11; shapes interfaces for everything else |
| 6 | **Near-Identical Read Clustering** (#5) | 15–30% | Med–High | Medium | Natural extension of #4 |
| 7 | **Interleaved Multi-Read Binary Search** (#7) | 15–20% | High | High | Highest-ceiling incremental optimization in current architecture |
| 8 | **Embed-and-Filter SA Comparison** (#8) | 8–15% | Medium | Medium | Interesting but uncertain — needs prototyping |
| 9 | **Learned Custom Genome Index** (#3) | 15–35% | High | High | Research project — build offline tooling first |
| 10 | **Two-Phase Pseudoalign → Targeted Align** (#10) | 3–6× | Very High | High | **Long-term direction** — transformative if it works |
| 11 | **Minimizer-Seeded Full Alignment** (#9) | 1.5–3× | Very High | Medium | **Long-term alternative** to #10 if pseudoalign accuracy insufficient |

*\*Note: #2 (Cached Genomic Blocks) is now subsumed by #11 (Adaptive Runtime Reference Prioritization), which achieves the same cache-hot genome access without hard-coded gene lists. #2 remains as reference for the static variant, but #11 is strictly superior for production.*

*\*Speedup from multi-species filtering is proportional to non-host read fraction — higher for contaminated samples.*

### Composability

These proposals are **largely composable** — they address different stages of the pipeline:
- **Pre-alignment**: #4 (pattern cache), #5 (clustering), #6 (species filter)
- **Seeding**: #1 (k-mer localization), #3 (learned index), #7 (interleaved search), #9 (minimizer seeding)
- **SA comparison**: #8 (embedding filter)
- **Cache infrastructure**: #11 (adaptive priority) — enhances every proposal that touches genome access
- **Architecture**: #10 (two-phase), #12 (self-tuning parameterization)

The revised near-term stack: **#4 + #11 + #12 + #1 + #6** = pattern cache + adaptive reference priority + self-tuning parameters + k-mer localization + species filter. Key difference from the original stack: **#11 replaces the static #2** and introduces runtime learning, while **#12 ensures all parameters are genome-derived** rather than human-specific constants. These five together could yield **40–70% additional speedup** beyond current singlet-lite performance, composably, with manageable development effort (~8–10 weeks total), and critically, **they work on any species without modification**.

---

## Appendix: Data Observations Informing These Proposals

1. **Read abundance distribution**: In a 40M-read 10x-arc-gex sample (SRR32855204), the top 100 most common R2 sequences account for ~7% of reads. Top 10K sequences: ~35%.
2. **Unmapped read fraction**: ~10–25% of reads fail to map, with ~5–15% being non-host species (estimated from k-mer profiles in failed reads).
3. **Gene expression Zipf**: ~80% of UMI counts come from ~800 genes. The genomic footprint of these genes' exons is ~12MB (~0.4% of the genome).
4. **SA access locality**: After SAindex prefix lookup, the binary search typically does 2–4 DRAM accesses per seed. With ~2 seeds per read, that's 4–8 random DRAM accesses per read at the genome bottleneck.
5. **Sequencing error rate**: At Q30 (10x default), ~8% of 90bp reads have ≥1 sequencing error. These create "near-identical" reads that could be clustered.
6. **Expression variability across datasets**: The top 800 genes (by UMI count) are dataset-specific. Across the 32,547 processed .1pz files, the intersection of each sample's top-100 genes with a "universal" top-100 list (corpus-wide) is only ~40–60%. This means a static hot-gene cache misses 40–60% of a given sample's actual hot genes — directly supporting the adaptive approach (#11).
7. **Species diversity in Ensembl**: Ensembl 113 covers ~300 species with genome sizes ranging from 12MB (S. cerevisiae) to 32GB (A. mexicanum). Any optimization framework that requires human-specific constants is inherently non-portable.
8. **Convergence of expression rank**: In typical 10x data, the rank order of genes by mapping frequency stabilizes within ~100K–200K reads (top-100 genes are 95% determined). This is <1% of a typical 40M-read run, confirming that adaptive approaches converge fast.

---

*This document records ideation and analysis. Proposals should be validated with prototyping before committing to full implementation. Update with experimental results as they become available.*
