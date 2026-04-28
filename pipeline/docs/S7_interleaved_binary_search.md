# S7: Interleaved Multi-Read Binary Search — Design Document

**Author**: singlify-perf agent  
**Date**: 2026-04-10  
**Status**: Design complete, implementation not started

## 1. Problem Statement

42% of STAR alignment wall time is spent on random DRAM accesses during the
suffix array binary search (`G[SA[mid]]` in `compareSeqToGenome()`). Each
comparison involves:
1. Load `SA[i]` — random access into 12.4 GB suffix array
2. Load `G[SA[i]]` — random access into 3.1 GB genome
3. Compare read characters against genome characters (cheap, in-register)

A single binary search takes ~20-25 iterations. Each iteration has ~200ns of
DRAM latency for the two random loads. Total: ~4-5 μs per seed × 1-3 seeds
per read = ~3-15 μs per read.

**Key insight**: Within a single binary search, we cannot overlap the loads
because step 2 depends on the result of step 1 (data-dependent pointer chase).
But across independent reads, the loads are independent and can be overlapped.

## 2. Proposed Solution

Batch B=4-8 reads through the binary search simultaneously. While read_k waits
for its `G[SA[mid]]` load, we process the comparison result from read_{k-1}.
When all reads in the batch have finished their current iteration, we advance
to the next iteration.

```
Iteration i:        Read_0          Read_1          Read_2          Read_3
              ┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
  prefetch    │ SA[mid_0]    ││ SA[mid_1]    ││ SA[mid_2]    ││ SA[mid_3]    │
  compare     │ G[SA[mid_0]] ││ G[SA[mid_1]] ││ G[SA[mid_2]] ││ G[SA[mid_3]] │
              └──────────────┘└──────────────┘└──────────────┘└──────────────┘
```

The CPU initiates memory loads for all B reads, then cycles through comparisons.
By the time we return to read_0's next iteration, its prefetched data should be
in cache.

## 3. Architecture

### 3.1 New Function: `maxMappableLength_batched()`

```cpp
// Process B reads through binary search simultaneously.
// Each read has independent SA range [i1,i2] and identity length L.
// Returns when all B reads have completed their binary search.

struct SeedSearchState {
    uint i1, i2, i3;           // SA range and midpoint
    uint L, L1, L2, L3;       // identity lengths
    uint L1a, L1b, L2a, L2b;  // boundary tracking
    uint i1a, i1b, i2a, i2b;
    char** s;                  // read sequence
    uint S;                    // read start offset
    uint N;                    // max seed length
    bool dirR;                 // direction
    bool active;               // still searching
    uint* indStartEnd;         // output: SA range
};

void maxMappableLength_batched(
    Genome& mapGen,
    SeedSearchState* states,       // array of B states
    int B,                         // batch size
    uint* outNmapped               // output: num mappings per read
);
```

### 3.2 Integration Point

In `ReadAlign_mapOneRead.cpp`, the inner loop calls `maxMappableLength2strands()`
for each seed. The batched version would:

1. Collect B seeds from B different reads (requires cross-read coordination)
2. Initialize SeedSearchState for each
3. Call `maxMappableLength_batched()`
4. Distribute results back to reads

### 3.3 Thread-Level Integration

STAR processes reads in parallel across threads. Each thread handles one read
at a time. To batch across reads, we need either:

**Option A: Intra-thread batching** — Each thread buffers B reads, then processes
all B seeds in a batch. Requires refactoring ReadAlign to support deferred
processing.

**Option B: Cooperative batching** — Multiple threads contribute seeds to a
shared batch. More complex, requires synchronization.

**Recommendation: Option A** — simpler, no synchronization overhead.

## 4. Ceiling Estimate (Amdahl's Law)

- Total STAR wall time (5M reads, 16T): ~24s
- Binary search is ~42% of wall time: ~10s
- If interleaving achieves 3× MLP (effective bandwidth): reduces to ~3.3s
- Net saving: ~6.7s from 24s = **28% of STAR time**
- As fraction of full pipeline (115s): ~6% improvement
- **Estimated pipeline time**: 115s → ~108s

This is meaningful but not transformative. The implementation complexity is high.

## 5. Implementation Risk

| Risk | Severity | Mitigation |
|------|----------|------------|
| Batch overhead exceeds savings for small B | Medium | Profile at B=2,4,8; stop if B=4 is slower |
| Cross-read state management complexity | High | Keep SeedSearchState as simple POD struct |
| Register pressure from B parallel states | Medium | B=4 uses ~64 registers; B=8 may spill |
| Thread interaction with OpenMP | Low | Option A is per-thread, no sync needed |
| Correctness regression | High | Bit-identical SJ.out.tab required |

## 6. Alternative: Lock-Step SA Binary Search

Instead of full interleaving, a simpler approach:

1. Compute SA index (SAi lookup) for B seeds simultaneously
2. Prefetch all B SA[i1], SA[i2] simultaneously
3. Run individual binary searches for each seed sequentially

This doesn't overlap the binary search iterations but does overlap the initial
SAi→SA loads, which are the first (and often largest) DRAM accesses.

**Estimated gain**: 10-15% of binary search time (only first 2 loads overlap).
**Implementation**: Much simpler — just batch the SAi lookups.

## 7. Decision

Given that all milestones are met and the ceiling is ~6% of pipeline time,
S7 implementation should be deferred until either:
- A step-change approach (S9: pseudoalign→targeted, or S10: minimizer seeding) is ruled out
- The 120s target needs to be pushed to 90s

The design is documented here for future implementation.
