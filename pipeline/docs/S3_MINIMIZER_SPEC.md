# S3 Minimizer Pre-Screening: Architecture Specification

**Status**: Design (not yet implemented)  
**Target speedup**: ~1.6× (38% wall reduction); Amdahl ceiling assuming SA probes = 42% of wall  
**Correctness invariant**: SA binary search remains ground truth; minimizer is a hint only. No false negatives permitted.

---

## 1. Problem Statement

STAR's `maxMappableLength2strands` narrows the SA search to `[iSA1, iSA2]` using a precomputed 14-mer index (`gSAindexNbases=14`). The subsequent `maxMappableLength` binary search then does ~log₂(range_width) × gSAsparseD probes, each requiring an uncached DRAM load from the 12 GB SA file. For GRCh38 (gSAsparseD=2), this is typically 20–35 probes per seed at ~100 ns/probe.

**Objective**: Narrow `[iSA1, iSA2]` further using a compact secondary index keyed on minimizers (k=21, w=10), reducing probe count from ~30 to ~3–5 for 91–98% of reads.

---

## 2. Genome Parameters (GRCh38)

| Parameter | Value |
|---|---|
| `gSAindexNbases` | 14 |
| `gSAsparseD` | 2 |
| SA file size | 12 GB (~30-bit packed entries, ~3.2B entries) |
| Genome size | ~3.2 GB |
| Node RAM (c001) | 192 GB (SA=12 GB + Genome=3.2 GB = 15.2 GB loaded) |

---

## 3. Data Structure

### 3.1 Minimizer Index (MinimizerSAIndex)

A flat sorted array of fixed-size records, stored as `minimizerIndex.bin` alongside the SA:

```
struct MinimizerRecord {      // 16 bytes/entry
    uint64_t  hash;           // canonical minimizer hash (MurmurHash3 of canonical k-mer)
    uint32_t  SA_lo;          // lower bound in SA (inclusive)
    uint32_t  SA_hi;          // upper bound in SA (inclusive)
};
```

Array sorted by `hash` → O(log N) binary search at query time.

**Build parameters**:
- k = 21 (minimizer length)
- w = 10 (window size; select minimum k-mer hash over w consecutive k-mers)
- Multi-mapping threshold T = 100: if any minimizer maps to >T SA positions spanning a range >T, mark as AMBIGUOUS and omit from index
- Ambiguous minimizers fall back to [iSA1, iSA2] unchanged (zero speedup, no error)

**Size estimate for GRCh38**:
- Distinct minimizers with ≤T copies: ~300M × 16 bytes = **4.8 GB**
- With T=100 filtering removes ~5–10% of genome: **~4.3 GB** in practice
- Acceptable: adds 4.3 GB to 15.2 GB existing = 19.5 GB total, well within 192 GB

For GRCm39 (genome ~2.7 GB): ~3.6 GB minimizer index.

### 3.2 Runtime Data Structure

```cpp
struct MinimizerSAIndex {
    MinimizerRecord* records;   // mmap'd from minimizerIndex.bin
    size_t           nRecords;
    uint32_t         k;         // 21
    uint32_t         w;         // 10
    uint32_t         T;         // 100 (ambiguous threshold)
};
```

Add to `Genome` class:
```cpp
MinimizerSAIndex* minimizerIdx;   // nullptr if file not present
```

---

## 4. Integration Point

**File**: `STAR/source/ReadAlign_maxMappableLength2strands.cpp`  
**Location**: After `iSA2` is computed (line ~63), before the `#ifdef SA_BOUNDARY_PREFETCH` block (line ~67), inside the `for (uint iDist=...)` loop.

```cpp
// --- S3 MINIMIZER NARROWING ---
// Inject after: iSA2 is finalized (line ~63)
// Inject before: #ifdef SA_BOUNDARY_PREFETCH (line ~67)
if (mapGen.minimizerIdx != nullptr && pieceLength >= (uint)(mapGen.minimizerIdx->w + mapGen.minimizerIdx->k - 1)) {
    MinimizerQueryResult mq = mapGen.minimizerIdx->query(
        Read1[0] + pieceStart, pieceLength, dirR);
    if (mq.valid) {
        // mq.SA_lo/SA_hi guaranteed ⊇ true SA range (built from full genome scan)
        // Clamp to intersection with 14-mer bounds → always safe
        uint clo = max(iSA1 & mapGen.SAiMarkNmask, mq.SA_lo);
        uint chi = min(iSA2, mq.SA_hi);
        if (clo <= chi) {
            iSA1 = clo;  // replaces iSA1 for subsequent maxMappableLength call
            iSA2 = chi;
        }
        // else: minimizer range outside 14-mer bounds → ignore, use [iSA1, iSA2]
    }
    // on mq.valid==false: unchanged fallback
}
// --- END S3 ---
```

### 4.1 Query Function Signature

```cpp
struct MinimizerQueryResult { bool valid; uint32_t SA_lo; uint32_t SA_hi; };
MinimizerQueryResult MinimizerSAIndex::query(const char* seq, uint len, bool revComp) const;
```

Algorithm:
1. Extract the single minimizer from bases `[0, w+k-2]` of the read piece (forward or RC)
2. Compute canonical hash (min of forward and RC hashes)
3. Binary search `records[]` by hash
4. If found → return `{true, records[i].SA_lo, records[i].SA_hi}`
5. If not found → return `{false, 0, 0}`

---

## 5. Index Build Procedure

**Tool**: `singlify/src/build_minimizer_index.cpp` (new file, standalone binary)  
**Inputs**: genome SA file (`SA`), genome sequence (`Genome`), genome parameters (`genomeParameters.txt`)  
**Output**: `minimizerIndex.bin` in the genome directory

```
Algorithm:
  1. Load SA (mmap) and Genome (mmap)
  2. Allocate hash map: phmap::flat_hash_map<uint64_t, pair<uint32_t,uint32_t>>
     (key=minimizer_hash, value={SA_lo, SA_hi})
  3. For each SA index i ∈ [0, nSA):
     a. genome_pos = SA[i]                   // unpack from PackedArray
     b. if genome_pos + k + w - 1 > nGenome: skip (near end)
     c. minimizer = compute_minimizer(Genome + genome_pos, k, w)
     d. if hash in map: expand range: map[hash] = {min(lo,i), max(hi,i)}
        else: map[hash] = {i, i}
  4. Filter: remove entries where (SA_hi - SA_lo) > T
  5. Sort by hash → write flat array to minimizerIndex.bin
  6. Write header: magic(8B) + k(4B) + w(4B) + T(4B) + nRecords(8B)
```

**Build time estimate**: O(nSA) scan ≈ 3.2B iterations → ~30–60 seconds at step 3 with hash map insertions. Acceptable as one-time offline build.

**Canonical k-mer**: use 2-bit encoding; canonical = min(forward_hash, revcomp_hash). Use `MurmurHash3_x64_64` for distribution uniformity.

---

## 6. Correctness Argument

**Invariant**: For any genome position `g` at SA index `i`, the minimizer of `Genome[g..g+k+w-2]` hashes to a record whose `[SA_lo, SA_hi]` contains `i`.

**Proof**: By build construction (step 3d above), every SA index `i` that appears for minimizer M extends `[SA_lo, SA_hi]` to include `i`. The stored range is the UNION of all SA positions for M; the true range for a query can only be a SUBSET.

**Consequence at query time**: If the read's true alignment falls at SA index `i*`, and the minimizer extracted from the read matches M with stored range `[lo, hi]`, then `lo ≤ i* ≤ hi`. The clamped `[max(iSA1,lo), min(iSA2,hi)]` still contains `i*`. ✓

**When invariant can fail** (impossible):
- Minimizer lookup returns a range that doesn't cover the true position → only possible if build was incorrect (would be a build bug, not a design flaw)

**When minimizer misses** (falls back safely):
- Read contains N bases (no canonical minimizer) → `valid=false` → unchanged
- Minimizer is ambiguous (filtered during build) → not in index → `valid=false` → unchanged
- Read is too short (`pieceLength < w+k-1`) → skip → unchanged

---

## 7. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| False negative (missed true alignment) | Impossible by construction | Correctness break | Build verification test: scan all SA entries, confirm each hash maps to range containing that SA index |
| Index build bug producing wrong ranges | Low | Silent error | After build, verify 1M random SA positions |
| Memory pressure: +4.3 GB on 192 GB node | None | — | Node has 192 GB; total with index = ~19.5 GB |
| Minimizer miss rate >9% | Low for unique reads | <9% speedup loss | Measure hit rate on 5M-read benchmark before shipping |
| Build time overhead | ~60s per genome | One-time cost | Run at genome-index time, store alongside SA |
| RC minimizer extraction bug | Medium | Wrong clamping | Unit test: check forward and RC queries on 1000 known SA entries |
| SA range clamping when `clo > chi` | Expected ~2–5% | No-op fallback | Code already handles: just skip narrowing |

---

## 8. File Layout

```
<genome_dir>/
    SA                      (existing, 12 GB)
    Genome                  (existing, 3.2 GB)
    minimizerIndex.bin      (new, ~4.3 GB)
    minimizerIndex.bin.md5  (new, build verification)
```

Load at genome-load time in `Genome::genomeLoad()`:
- Check for `minimizerIndex.bin`; if present, mmap it
- Log: `minimizerIndex: loaded N records (k=21, w=10, T=100)`
- If absent, set `minimizerIdx = nullptr` → silent fallback to current behavior

---

## 9. Implementation Dispatch Checklist

1. [ ] `build_minimizer_index.cpp`: standalone binary, K=21/W=10/T=100 defaults, CLI: `--genome-dir`
2. [ ] `MinimizerSAIndex.h`: struct, mmap loader, `query()` method
3. [ ] `Genome.h`: add `MinimizerSAIndex* minimizerIdx`
4. [ ] `Genome::genomeLoad()`: attempt mmap of `minimizerIndex.bin`
5. [ ] `ReadAlign_maxMappableLength2strands.cpp`: inject narrowing block (Section 4)
6. [ ] Correctness test: `diff SJ.out.tab` vs stock baseline on correctness_test set
7. [ ] Benchmark: STAR 5M-read benchmark before/after (target: wall ≤ baseline × 0.70)
8. [ ] Build the index: run `build_minimizer_index --genome-dir GRCh38-2024-A/star_2.7.11b`
