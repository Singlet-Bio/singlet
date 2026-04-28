# STAR singlet-lite

Singlify uses an optimized fork of [STAR](https://github.com/alexdobin/STAR) v2.7.11b
for alignment. The `singlet-lite` branch contains eight performance optimizations
that together yield **~48% wall-clock improvement** (2.1×) for STARsolo single-cell
alignment vs stock STAR 2.7.11b (warm genome cache, 8 threads, 5M reads).

> **Note**: As of singlify v0.2.0, the singlet-lite source is compiled directly into
> the `singlify` binary (see `../src/star/`). There is no separate STAR binary to
> build or install. The patch file `singlet-lite.patch` is kept here for reference;
> it documents exactly which changes were made relative to stock STAR 2.7.11b.

**Production build**: `singlify` binary (all source optimizations + GCC 13 PGO+LTO via CMake).
**Reference binary**: `STAR_stock_baseline` in `../src/star/` — stock STAR 2.7.11b for correctness diffs.
**Correctness**: `SJ.out.tab` byte-identical to stock STAR 2.7.11b on 500K-read test set.

## Optimizations

### 1. LUT + Hash Barcode Lookup (`b05ed2d`)
Replace binary search over the cell barcode whitelist with a Fibonacci-hashed open-addressing
table. Exact match: O(1) amortized vs O(log n). 1-mismatch scan: 3×16 hash probes vs 3×16
binary searches. Also adds a 256-entry LUT for nucleotide→number conversion in `SequenceFuns.cpp`.
- **Impact**: ~3% wall-clock

### 2. Per-Chunk 14-mer Prefix Sort (`42c2d4a`)
Sort reads within each STAR chunk buffer by the leading 14-mer of the biological read (R2).
Adjacent reads then probe the same suffix array region, achieving near-100% L2/L3 cache hit
rate on SA binary search (vs ~80% miss rate with random order).
- **Impact**: ~15% wall-clock (p = 0.003, d = 1.24)

### 3. Per-Chunk R2 Consecutive Dedup (`866cb44`)
After 14-mer sorting, consecutive reads with identical R2 sequences skip the full alignment
pipeline. The cached Transcript array (~384 bytes/thread) is reused; only `mappedFilter()` runs.
~15-17% per-chunk hit rate.
- **Impact**: ~5% wall-clock (p = 0.006, d = 1.85)

### 4. `-march=native` + NUMA Memory Interleave (`b04ea1a`)
Enables AVX2/BMI2 instruction use on modern Xeon. NUMA `set_mempolicy(MPOL_INTERLEAVE)`
distributes the 3GB genome and 8GB SA arrays across NUMA sockets, halving per-socket
memory bandwidth demand on 2-socket nodes.
- **Impact**: ~16% combined (6% AVX2 + 10% NUMA)

### 5. SA_BOUNDARY_PREFETCH (`1918ca4`)
Before calling `maxMappableLength`, prefetch the SA PackedArray entries for both possible
next midpoints (`SA[i1]`, `SA[i2]`, `SA[mid]`). Overlaps DRAM latency for the next binary
search step with current genome comparison. Compile-time flag: `-DSA_BOUNDARY_PREFETCH`.
- **Impact**: ~1.5% wall-clock

### 6. SA_LAZY_WINBIN (`4b5b2be`)
Replaces the per-read `memset` of the `winBin` arrays (~94KB per read) with a lazy
dirty-range reset. Tracks up to 256 modified ranges per read; only resets those ranges
on the next read rather than zeroing the entire array. Falls back to full memset on
overflow. Compile-time flag: `-DSA_LAZY_WINBIN`.
- **Impact**: ~7% wall-clock (warm cache, c006)

### 7. PGO + LTO (`0e7dece`)
Profile-guided optimization collects branch/call frequency data from a 500K-read training
run, then GCC 13 LTO enables cross-translation-unit inlining across the tight SA binary
search path (`findMultRange` → `compareSeqToGenome`). LTO is required for PGO to help —
LTO alone regresses due to I-cache pressure.
- **Impact**: ~5% wall-clock on top of all source optimizations

## Building

The singlet-lite optimizations are compiled directly into `singlify` via the `star_objects` CMake OBJECT library. To build:

```bash
cd /path/to/singlify
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) singlify
```

If you need to apply the patch to a standalone STAR checkout for comparison or debugging:

```bash
# Apply patch to stock STAR 2.7.11b
git clone https://github.com/alexdobin/STAR.git
cd STAR && git checkout 2.7.11b
git apply /path/to/singlify/star/singlet-lite.patch

# Standard build (no PGO):
source /opt/rh/gcc-toolset-13/enable
cd source
make clean && make -j$(nproc) STAR \
  CXXFLAGSextra="-march=native -DSA_BOUNDARY_PREFETCH -DSA_LAZY_WINBIN"
cp STAR STAR_singlet_lite

# PGO+LTO build (recommended for production):
GENOME=/path/to/star_index; R1=reads_R1.fastq.gz; R2=reads_R2.fastq.gz; WL=whitelist.txt

# Phase 1: instrumented build
make clean && make -j$(nproc) STAR \
  CXXFLAGSextra="-march=native -DSA_BOUNDARY_PREFETCH -DSA_LAZY_WINBIN -fprofile-generate=../pgo_profile"

# Phase 2: training run (500K reads)
mkdir -p /tmp/pgo_train
./STAR --runThreadN 8 --genomeDir "$GENOME" \
  --readFilesIn "$R2" "$R1" --readFilesCommand zcat \
  --soloType CB_UMI_Simple --soloCBwhitelist "$WL" \
  --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 \
  --outSAMtype BAM Unsorted --outBAMcompression 0 \
  --readMapNumber 500000 --outFileNamePrefix /tmp/pgo_train/

# Phase 3: PGO+LTO production build
make clean && make -j$(nproc) STAR \
  CXXFLAGSextra="-march=native -DSA_BOUNDARY_PREFETCH -DSA_LAZY_WINBIN \
    -fprofile-use=../pgo_profile -fprofile-correction -Wno-missing-profile -flto -w" \
  LDFLAGSextra="-fprofile-use=../pgo_profile -flto -fopenmp"
cp STAR STAR_pgo_lto_sa
```

## Modified Files

| File | Changes |
|------|---------|
| `ParametersSolo.h` | Hash table members + `buildCBhashTable()` / `hashLookupCB()` declarations |
| `ParametersSolo.cpp` | Hash table build + lookup implementation |
| `SoloReadBarcode_getCBandUMI.cpp` | Replace `binarySearchExact` with `hashLookupCB` |
| `SequenceFuns.cpp` | 256-entry nucleotide LUT, vectorized complement |
| `readLoad.cpp` | Fast path for .1fq numeric-char input |
| `ReadAlign.h` | `OPT_DEDUP_R2` define + dedup cache members; `SA_LAZY_WINBIN` dirty-range members |
| `ReadAlign.cpp` | Dedup cache allocation + `dedupResetCache()`; lazy winBin init |
| `ReadAlignChunk_mapChunk.cpp` | `sortChunkByPrefix()` + 14-mer sort invocation |
| `ReadAlign_oneRead.cpp` | Consecutive R2 dedup logic |
| `ReadAlign_stitchPieces.cpp` | `SA_LAZY_WINBIN` lazy reset with dirty-range tracking |
| `ReadAlign_createExtendWindowsWithAlign.cpp` | `SA_LAZY_WINBIN` `trackDirty` calls |
| `ReadAlign_mapOneRead.cpp` | `SA_NEXT_SEED_PREFETCH` (disabled by default) |
| `ReadAlign_maxMappableLength2strands.cpp` | `SA_BOUNDARY_PREFETCH` prefetch calls |
| `SuffixArrayFuns.cpp` | `SA_BOUNDARY_PREFETCH`, `SA_SPECULATIVE_PREFETCH`, `SA_BATCH_FINDRANGE` |
| `STAR.cpp` | NUMA `set_mempolicy(MPOL_INTERLEAVE)` |
| `Genome.h` | `SA_2BIT_GENOME` packed genome struct (disabled by default) |
| `Genome_genomeLoad.cpp` | `SA_2BIT_GENOME` build path (disabled by default) |
| `GenomePacked.h` | 2-bit packed genome helpers (disabled by default) |
| `Makefile` | `CXXFLAGSextra` passthrough for `-DSA_*` flags |

All compile-time optional features are guarded by `#ifdef` and off by default.
Recommended flags for production: `-DSA_BOUNDARY_PREFETCH -DSA_LAZY_WINBIN`.
