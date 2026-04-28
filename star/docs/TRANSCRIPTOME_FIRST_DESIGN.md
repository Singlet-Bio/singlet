# Transcriptome-First STAR Fast Path: Design Document

## Status: Implementation in progress

## Problem Statement

STAR's suffix array (SA) binary search is the dominant cost (~65% of wall time) for
scRNA-seq alignment. The SA is 12 GB and lives in DRAM. Each probe costs ~300 cycles
due to cache misses. Most scRNA-seq reads come from a small set of abundant genes
(top 500 genes account for ~39% of all UMIs across 200 human samples).

## Key Data (from 200-sample .1pz survey)

| Top N genes | % UMIs covered | Exonic bases (merged) | Filtered SA size (est.) |
|-------------|----------------|-----------------------|------------------------|
| 100         | 24.6%          | ~550 KB               | ~3 MB                  |
| 200         | 30.7%          | ~1.7 MB               | ~9 MB                  |
| 500         | 38.6%          | ~2.2 MB               | ~12 MB                 |
| 1,000       | 46.0%          | ~5 MB                 | ~27 MB                 |
| 2,000       | 55.0%          | ~11 MB                | ~60 MB                 |

## Failed Approach: Two-Pass External (Dead End)

Built a STAR index from concatenated exonic sequences of top 500 genes (2.2 MB genome,
18 MB SA). Ran alignment as a separate STAR process.

**Result**: 2× SLOWER than full genome (79.3s vs 39.8s wall, 1341s vs 200s CPU).

**Root cause**: Non-matching reads (68%) trigger partial-match explosions. In 2.2 MB
of sequence, short seeds (5-15 bp) match many locations. STAR's stitching pipeline
exhaustively tries all combinations → combinatorial blowup. The overhead of processing
unmapped reads dominates.

**Lesson**: The transcriptome-first approach MUST be internal to STAR's alignment loop,
not a separate process. Non-matching reads must be detected and SHORT-CIRCUITED — not
put through the full stitching pipeline.

## Chosen Approach: Filtered Genome SA

### Core Idea

Create a **subset of the full genome SA** containing only entries that point to exonic
positions of top N genes. This filtered SA:
- Stores **genome coordinates** (not transcriptome coordinates) → no coordinate conversion
- Uses the **same genome G buffer** as the main SA → zero additional genome memory
- Is **sorted by suffix content** (it's a subset of a sorted array) → binary search works
- Is **small enough to stay in L3 cache** (~12 MB for top 500 genes)

### Architecture

```
                     ┌─────────────────────────┐
                     │   Read: seed L-mer       │
                     └──────────┬──────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Filtered SAi_tx lookup │ ~1–2 ns (L2 hit)
                    │  (L-mer → SA_tx range)  │
                    └───────────┬────────────┘
                                │
                    ┌───YES──── │ ────NO─────┐
                    │  L-mer    │   L-mer     │
                    │  exists   │   absent    │
                    │           │             │
          ┌─────────▼───────┐  │  ┌──────────▼───────────┐
          │  Binary search  │  │  │  Full genome SA       │
          │  SA_tx (12 MB)  │  │  │  (12 GB, DRAM-bound)  │
          │  ~20 iter × 40c │  │  │  ~32 iter × 300c      │
          │  = ~800 cycles  │  │  │  = ~9600 cycles        │
          └────────┬────────┘  │  └──────────┬───────────┘
                   │           │             │
          ┌────────▼─────┐    │   ┌──────────▼──────┐
          │ Unique match │    │   │ Normal alignment│
          │ → done       │    │   │ pipeline        │
          └──────────────┘    │   └─────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │ Fall through to      │
                   │ full genome SA       │
                   │ (same as normal)     │
                   └─────────────────────┘
```

### Expected Performance

- **30-39% of reads**: SA search in L3 cache (10× faster per read)
- **61-70% of reads**: Zero overhead (SAi_tx L-mer miss is ~2 ns, <0.5% of read alignment time)
- **Net SA speedup**: ~25-30% (SA is 65% of wall → ~16-20% wall time improvement)
- **On our 30s benchmark**: saves ~5-6 seconds → 24-25s total

### Implementation Plan

#### Phase 1: SA Filter Tool (`filter_genome_sa.py`)

Python tool that:
1. Reads `exonInfo.tab` + `geneInfo.tab` from STAR genome directory
2. Reads gene abundance ranking (our survey TSV)
3. Builds exon bitmask for top N genes
4. Reads the full SA (12 GB packed array) via memory-mapping
5. Filters: keeps only entries whose genome position falls in a top-gene exon
6. Writes the filtered SA in STAR's PackedArray format
7. Builds SAi_tx (L-mer index over filtered SA) with reduced gSAindexNbases (9-10)

Output files (stored alongside main genome index):
- `SA_tx` — filtered SA (packed array, ~12 MB)
- `SAindex_tx` — filtered SAi (~500 KB with Nbases=9)
- `transcriptomeFirstInfo.tab` — gene list + metadata

#### Phase 2: STAR Integration

**Genome.h** — Add fields:
```cpp
PackedArray SA_tx;          // Filtered transcriptome SA
PackedArray SAi_tx;         // Filtered SAi
uint nSA_tx;                // Number of filtered SA entries
uint *genomeSAindexStart_tx;
uint pGe_gSAindexNbases_tx; // Reduced L-mer depth for filtered SA
bool hasTxSA;               // Flag: transcriptome SA loaded
```

**Genome_genomeLoad.cpp** — Load filtered SA:
```cpp
// After loading main SA, try loading filtered SA
ifstream SA_txIn((pGe.gDir+("/SA_tx")).c_str());
if (SA_txIn.good()) {
    // Load SA_tx, SAi_tx with correct bit widths
    hasTxSA = true;
} else {
    hasTxSA = false;
}
```

**ReadAlign_mapOneRead.cpp** — Add fast path before genome SA search:
```cpp
// Inside the seed search loop, before maxMappableLength2strands():
if (mapGen.hasTxSA) {
    uint Lmax_tx = min(mapGen.pGe_gSAindexNbases_tx, seedLength);
    uint ind1_tx = computeLmerIndex(Read1, pieceStart, Lmax_tx, dirR);
    
    uint iSA1_tx = mapGen.SAi_tx[mapGen.genomeSAindexStart_tx[Lmax_tx-1] + ind1_tx];
    if ((iSA1_tx & mapGen.SAiMarkAbsentMaskC) == 0) {
        // L-mer exists in transcriptome → binary search filtered SA
        uint iSA2_tx = /* compute upper bound */;
        uint maxL_tx = Lmax_tx;
        uint Nrep_tx = maxMappableLength_TX(mapGen, Read1, pieceStart,
                                             seedLength, iSA1_tx, iSA2_tx,
                                             dirR, maxL_tx, indStartEnd);
        if (Nrep_tx <= 10 && maxL_tx >= seedLength) {
            // Good transcriptome match — store alignment and skip genome SA
            storeAligns(iDir, pieceStart, Nrep_tx, maxL_tx, indStartEnd, iFrag);
            L = maxL_tx;
            goto next_seed;
        }
    }
    // Fall through to genome SA search
}
maxMappableLength2strands(Shift, seedLength, iDir, 0, mapGen.nSA-1, L, splitR[2][ip]);
```

**SuffixArrayFuns.cpp** — Add `maxMappableLength_TX()`:
Same as `maxMappableLength()` but operates on `mapGen.SA_tx` instead of `mapGen.SA`,
while using `mapGen.G` for genome sequence comparison (shared genome buffer).

#### Phase 3: Testing

1. **Correctness**: Run with `--quantMode GeneCounts`, compare gene counts with/without transcriptome-first
2. **Performance**: A/B benchmark on 5M reads, 20 threads, warm cache
3. **Edge cases**: Multi-mapped reads, reads spanning exon-exon junctions of non-top genes

## Open Questions

1. **How many genes?** 200-500 is the sweet spot for L3 (12-20 MB SA).
   Fewer genes = smaller SA (better cache) but fewer reads benefit.
   More genes = more reads benefit but SA spills from L3.

2. **SAi_tx L-mer depth?** Lower = smaller SAi but more binary search iterations.
   For 2.2 MB transcriptome, gSAindexNbases=9 is optimal (STAR recommends log2(n)/2-1).

3. **Multi-mapping threshold?** If filtered SA shows >1 match, should we still skip genome SA?
   Conservative: only skip for unique matches (Nrep=1).
   Aggressive: skip for Nrep ≤ 10 (reduces genome SA queries).

4. **Is full seed length required?** For short reads (91 bp R2 in 10x), seeds often span
   near-full read length. Requiring maxL >= seedLength is strict but safe.

## Files Created This Session

- `scripts/survey_transcript_abundance.py` — 200-sample .1pz abundance survey
- `scripts/gene_abundance_human.tsv` — 134K gene ranking by total UMIs
- `scripts/build_transcriptome_fasta.py` — TOP-N gene exon extractor
- `experiments/top500_transcriptome.fa` — 498 genes, 2.2 MB exonic sequence
- `experiments/star_mini_idx/` — Mini STAR index (proof of concept, DO NOT USE for alignment)
- `experiments/bench_txome_first.sh` — Two-pass benchmark (demonstrated dead end)

---

## Future Direction: Interleaved Multi-Read SA Binary Search

> **Status**: Designed, NOT yet implemented. Blocked by active TxFirst work in
> `ReadAlign_maxMappableLength2strands.cpp`. Must coordinate with singlify-perf agent.
> Implement ONLY after TxFirst is committed and benchmarked.

### Prior Benchmark Evidence (from PERFORMANCE_SUMMARY.md)

Before designing a new implementation, note what has already been tried and rejected:

| Approach | Result | Reason |
|---|---|---|
| `SA_SPECULATIVE_PREFETCH` | **+0.14s worse** | Instruction overhead; post-SAi range = 2–4 entries, already in L3 |
| `SA_BATCH_FINDRANGE` | Noise | 2–4 entry range → boundary search is 1 iteration |
| `SA_GENOME_PREFETCH` | Noise | Data-dependent pointer chase; prefetch too close to access |
| `SA_NEXT_SEED_PREFETCH` | Noise | — |
| `GENOME_PACKED` | **27% slower** | Bit manipulation overhead exceeds memory savings |

**Key finding**: Post-SAi SA binary search has only 2–4 entries. SA entries are already
in L3 from `SA_BOUNDARY_PREFETCH + OPT_SORT_CHUNK_BY_PREFIX`. The bottleneck is
`G[SA[mid]]` — ~300 cycle DRAM access into the 3.1 GB uncompressed genome. This is
a data-dependent pointer chase: `iSA → SA[iSA] → SAstr → G[SAstr]` — two serial memory
hops, each requiring the previous to resolve before the address is known.

**Why within-search prefetch is noise**: With 2–4 SA iterations per read, issuing a
genome prefetch after learning SAstr gives ~20 cycles of lead time before the access.
Against a 300-cycle DRAM miss, this is 6% overlap — unmeasurable.

### Why Cross-Read Interleaving IS Different

With B reads processed in round-robin lockstep:
- Skip to read[1] while read[0]'s G[SAstr_0] is in-flight (~300 cycles)
- Skip to read[2] while read[1]'s G[SAstr_1] is in-flight
- By the time we return to read[0], 300+ cycles have elapsed

For B = 8 and 2–4 SA iterations per read: 8 × (2–4) × ~40 cycles/iteration = 640–1280
cycles available to overlap the 300-cycle DRAM load. This is sufficient.

**Expected gain if implemented correctly**: ~15–25% on the 42% of wall time attributed
to genome loads → **6–10% overall wall time reduction**.

### Motivation

`maxMappableLength()` in `SuffixArrayFuns.cpp` is the inner loop of STAR's suffix array
binary search. Each call to `compareSeqToGenome` issues a random DRAM load for the genome
window at `G[SAstr + L]` (~300 cycle stall). The existing `SA_SPECULATIVE_PREFETCH` option
prefetches both possible *next midpoints* of the binary search within one read, hiding
some of this latency. `OPT_SORT_CHUNK_BY_PREFIX` sorts reads by 14-mer prefix before
mapping, improving SA locality and reducing the miss rate from ~80% to ~30% — but does
not address the genome hop.

The remaining gap: during the ~300-cycle stall waiting for *one read's* genome load to
arrive, no useful work is done. If N reads' genome loads were all in flight simultaneously,
DRAM bandwidth could be fully utilized — each read's stall hides behind another's compute.

### Approach

Restructure `maxMappableLength` as a step function that advances the binary search by one
iteration and returns control (CONTINUE | DONE | EXACT_MATCH). A batch driver in
`mapChunk` maintains `B = 8` (or 16) reads simultaneously:

```cpp
struct SASearchState {
    uint i1, i2, L, L1, L2;
    // ... all binary search tracking fields from maxMappableLength ...
    bool done;
};

// In mapChunk, replace serial oneRead() loop with:
SASearchState states[B];
// Initialize B reads
while (any_active(states)) {
    for (int r = 0; r < B; r++) {
        if (states[r].done) continue;
        // Issue genome prefetch for states[r]'s next midpoint
        __builtin_prefetch(G + SA[median(states[r].i1, states[r].i2)], 0, 0);
    }
    for (int r = 0; r < B; r++) {
        if (states[r].done) continue;
        sa_search_step(mapGen, states[r], read_data[r]);  // one binary search iteration
    }
}
```

By the time `sa_search_step` is called for read `r`, the prefetch issued B reads earlier
(~B × comparison_cycles ≈ 8 × 40 = 320 cycles ago) has had time to load from DRAM.

### Expected Gain

- Benchmark context: `STAR_production_v3` on 5M reads, 8 threads, warm SA cache.
- SA binary search: ~2–4 iterations × 300 cycles DRAM stall = 600–1200 cycles per read.
- With B=8 interleave: stall effectively ≈ max(300, 8×40) = 320 cycles per iteration
  per read when B_reads × comparison_time >= DRAM_latency.
- **Estimated wall time savings: 6–10%** (42% of wall = genome loads × 15–25% hide rate).
- Most pronounced with warm SA cache (OPT_SORT_CHUNK_BY_PREFIX already active) because
  the SA entries are L3 hits, so the compute-per-step is fast enough to overlap DRAM.

### Implementation Notes

1. **`sa_search_step()` signature**:
   ```cpp
   // Returns: 0=continue, 1=done(no exact match), 2=exact match
   int sa_search_step(Genome& mapGen, SASearchState& st, char** s, uint S, uint N, bool dirR);
   ```
2. **State size**: ~10 uint32 fields per read × 8 reads = ~320 bytes — fits in L1 cache.
3. **Compile guard**: `#ifdef SA_INTERLEAVED_SEARCH` — off by default, benchmark A/B.
4. **Do NOT use SA_SPECULATIVE_PREFETCH simultaneously** — the interleaved batch already
   issues prefetches for all B reads' midpoints before any comparison. SA_SPECULATIVE_PREFETCH
   is redundant and adds instruction overhead.
5. **OPT_SORT_CHUNK_BY_PREFIX must stay on** — orthogonal and complementary. Sort first
   to improve SA locality, then interleave to hide remaining stalls.
6. **Boundary case**: Final `findMultRange` calls per read can stay serial — they are
   O(1 iteration) with 2–4 SA entries and not the dominant cost.

### Coordination Constraint (IMPORTANT)

The singlify-perf agent has uncommitted changes in:
- `source/ReadAlign_maxMappableLength2strands.cpp` — TxFirst fast path (+116 lines)
- `source/Genome.h`, `source/Genome_genomeLoad.cpp` — TxFirst SA_tx fields  
- `source/ReadAlign_stitchPieces.cpp` — TXFIRST_SA_FLAG dispatch
- `source/TxFirst.h` (new file)

**Do NOT touch these files** until TxFirst is committed and benchmarked.

The interleaved search targets different files:
- `source/SuffixArrayFuns.cpp` — add `SASearchState`, `sa_search_step()`
- `source/SuffixArrayFuns.h` — declarations
- `source/ReadAlignChunk_mapChunk.cpp` — batch driver (SAFE: not touched by TxFirst)

### Files to Modify

- `source/SuffixArrayFuns.cpp` — Add `sa_search_step()` + move `SASearchState` struct here
- `source/SuffixArrayFuns.h` — Declare `SASearchState`, `sa_search_step()`
- `source/ReadAlignChunk_mapChunk.cpp` — Replace main `oneRead()` loop with B-read batch
- `source/Makefile` — Add `-DSA_INTERLEAVED_SEARCH` to `STAR_production_v4` target

### Benchmark Protocol

```bash
# On c006 or c004, warm SA, 3 runs each:
STAR_production_v3  # baseline (SA_BOUNDARY_PREFETCH + SA_LAZY_WINBIN + PGO+LTO)
STAR_production_v4  # + SA_INTERLEAVED_SEARCH B=8
STAR_production_v4b # + SA_INTERLEAVED_SEARCH B=16 (check if B=8 is optimal)
# Report median ± IQR wall time at 5M reads, 8 threads, warm SA
# Note: TxFirst should be off for this benchmark to isolate the SA interleave gain
```
