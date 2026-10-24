# m_w1: pz2 codec (W1, codec engineer)

Owner: W1. Paths: `include/singlet/pz2/`, `tests/cpp/test_pz2_*.cpp`, `bench/pz2/`, this file, the codec sections of `docs/format-v2.md`
(4.1, 4.2, 4.5, 5.1, 5.2 + a clarification in 4.6). Started 2026-10-22. **Status: codec 1 (scalar, AVX2, AVX-512), codec 2, encoder,
decoder, random access, digest (portable + SHA-NI), tests, experiments and spec are done. Open items are in section 7.**

## 0. Status board

| step | what | state |
|---|---|---|
| 1 | `pz2.h` public API, header / chunk index / ByteSource / ISA, `test_pz2_api` | DONE, stable since the first hour (additions only, listed in section 2) |
| 2 | codec 1 production (scalar, AVX2, AVX-512, runtime dispatch, no `-march`), codec 2, encoder (`auto`), digest | DONE |
| 3 | fixed-width chunk index, `plan_columns`, `decode_columns`, `column_counts`, partial sources (`NeedRange`) | DONE |
| 4 | tests: `test_pz2_api`, `test_pz2_edge`, `test_pz2_roundtrip` (both corpora) | DONE (counts in section 3) |
| 5 | experiments (a) timing, (b) chunk width, (c) 2^21 widening, (d) probability bits | DONE; (a) clean run at load 12 to 14% (run C, section 4.1) after two busier runs |
| 6 | spec 4.1 / 4.2 / 4.5 / 5.1 / 5.2 + independent Python decoder written from the text (`bench/pz2/ref_decode.py`) | DONE, decoder reproduces C++ output |

## 1. Build and use

```
bash bench/pz2/build.sh tests/cpp/test_pz2_edge.cpp        # -> bench/pz2/bin/test_pz2_edge.exe  (Rtools GCC 13, C++17, -O2, NO -march)
bash bench/pz2/build.sh bench/pz2/bench_pz2.cpp bench_pz2 -O3
```
Header-only; `#include <singlet/pz2/pz2.h>`; link zstd (+ pthreads). Namespace `singlet::pz2`. CMake: target `Singlet::pz2`
(INTERFACE, in the root `CMakeLists.txt`); tests `test_pz2_api`, `test_pz2_edge`, `test_pz2_roundtrip`, `test_pz2_multi_tu` (two TUs include the header-only codec and link) in `tests/cpp/CMakeLists.txt`
(`PZ2_TEST_MAX_MB=8` in the environment limits the corpus run to files up to 8 MB; the full run takes about an hour).

No `-march` flag is needed or wanted: the SIMD kernels carry per-function target attributes (`#pragma GCC target` / clang attribute
push regions in `detail/simd.h`) and are selected at run time (cpuid + XGETBV). `PZ2_ISA=scalar|avx2|avx512` forces a path (clamped to
what the CPU supports; unknown values ignored); `DecodeOptions::isa` overrides per call; `isa_report()` prints the selection.
`-DPZ2_SCALAR_ONLY` builds without any x86 code (tested: 664 edge checks pass). MSVC: cpuid via `__cpuidex`/`_xgetbv`, kernels are
intrinsics-only, **an MSVC build has not been tried (no MSVC on this laptop)**. NEON: not written (plan in `notes/w3_fast.md` section 3).

## 2. Public API (stable)

Everything is in `singlet::pz2`; full doc comments are in `pz2.h`.

```cpp
// input
struct CSC { uint32_t m, n; uint64_t nnz;
             const uint32_t* indptr;    // n+1 entries (nnz < 2^32)  -- OR --
             const uint32_t* counts;    // n entries (any nnz)
             const uint32_t* indices;   // nnz, strictly ascending per column, < m (int32 bits are fine)
             const void* data;  DType data_type;   // U8 | U16 | U32 | F32 (float values)
           };
struct EncodeOptions { CodecId codec /*Auto|Rans|Generic*/; uint32_t chunk_cols /*0=auto*/; uint32_t chunk_nnz_target /*0=49152*/;
                       uint8_t classes_per_octave=4, lanes=32, prob_bits=10, value_prob_bits=10; int ctx_kind=-1;
                       ValueType value_type /*Auto|U8|U16|U32|F32*/; NamesSpec row_names, col_names; KvMap kv;
                       bool compute_digest=true; bool has_digest; Digest16 digest; int threads=1; int zstd_level=19; };
struct NamesSpec { NamesKind kind /*Inline|Pack|PackSubset|Cells|None*/; PackId pack_id /*array<u8,8>*/; vector<uint32_t> pack_index;
                   vector<uint8_t> blob /*NUL-terminated logical names*/; uint64_t count; };
std::vector<uint8_t> encode_matrix(const CSC&, const EncodeOptions& = {}, EncodeStats* = nullptr);

// reading
Header      read_header(const uint8_t* p, size_t n);                    // 256-byte header, CRC checked
ChunkIndex  read_chunk_index(const Header&, const uint8_t* p, size_t n);// n_chunks+1 records of 24 bytes
class ByteSource { ByteSource(p,n) /*whole member*/; add(member_offset,p,n) /*fetched range*/; get(off,len) /*throws Error NeedRange{off,len}*/; has(off,len); };
class Matrix {                                                           // opened from the PREFIX only
  static Matrix open(const ByteSource&);          // needs bytes [0, Header::prefix_len()): header + chunk index + tables + rows
  const Header& header(); const ChunkIndex& chunk_index();
  const vector<uint32_t>& row_perm();             // stored id -> canonical id (empty = identity: codec 2)
  const vector<uint32_t>& row_inv();              // canonical -> stored, lazy
  NamesSpec row_names(src), col_names(src); KvMap kv(src);
  ColumnPlan plan_columns(col0, col1);            // chunk run + ONE member byte range to fetch + max_nnz (output capacity bound)
  void     decode(src, Output&, DecodeOptions);                         // whole matrix
  uint64_t decode_columns(src, col0, col1, Output&, DecodeOptions);     // returns exact nnz of the range
  vector<uint32_t> column_counts(src, col0, col1);
};
void decode(const uint8_t* member, size_t n, Output&, DecodeOptions = {});   // one-shot
struct Output { void* indptr; IntType indptr_type /*U32|I32|U64|I64*/; size_t indptr_capacity;
                void* indices; IntType indices_type /*U32|I32*/; size_t indices_capacity;
                void* data; DType data_type /*U8|U16|U32|F32|I32|F64*/; size_t data_capacity; };   // caller-owned, capacities checked
struct DecodeOptions { Mode mode /*Native|Canonical*/; int threads=1; Isa isa=Auto; };
Digest16 canonical_digest(const CSC&, ValueType, row_names_blob, col_names_blob, KvMap);   // docs/format-v2.md 4.6
Isa selected_isa(); Isa best_isa(); bool isa_supported(Isa); Isa resolve_isa(Isa); std::string isa_report(); const char* isa_name(Isa);
```
Errors are `pz2::Error` (a `std::runtime_error`) with `code` (`Errc`): `BadArgument, BadMagic, Unsupported, Truncated, Corrupt,
NeedRange (need_off/need_len = the member range to fetch), OutputTooSmall, TypeMismatch, NotImplemented`.

Changes to the API since the first publication (all additive): `NamesKind::None` (matrix without names; opt-in, the default `Inline`
requires a blob); `EncodeOptions::chunk_nnz_target`; `Matrix::open` now parses tables and the row classes (the first version only
parsed header + chunk index); explicit zero values are first-class in codec 1.

### How a remote client reads cells (intended call sequence)
1. Range-fetch bytes `[0, 64 KB)` of the member (its offset inside the `.s2` comes from the manifest), `read_header` -> if
   `Header::prefix_len()` is larger, fetch the rest. `ByteSource src; src.add(0, buf, len)`.
2. `Matrix mx = Matrix::open(src)`; `ColumnPlan p = mx.plan_columns(c0, c1)`; fetch `[p.byte_off, p.byte_off + p.byte_len)`;
   `src.add(p.byte_off, buf2, p.byte_len)`.
3. Allocate `indptr` (c1-c0+1), `indices`/`data` (`p.max_nnz`); `mx.decode_columns(src, c0, c1, out, {Mode::Native})`.
4. Native mode: ids are in stored (class) order; map through `mx.row_perm()` (or permute the name list once). Canonical mode returns the
   legacy matrix. For scattered cells call `plan_columns` per chunk run (columns of different chunks need different ranges).

### Decisions made while writing the API (differences from the brief)
* **Header is 256 bytes, not 160**: the field list of `docs/format-v2.md` 4.1 summed to 180; I needed codec parameters, total length,
  names kinds and a header CRC-32. Exact layout is in the spec now.
* **`value_type` is the legacy `vt_code`** for 1..3; no separate legacy field. F32 = 4 (see below).
* **Physical order** `header | chunk_index | tables | rows | chunks | cols | kv | extra`, packed, so the first four parts (the prefix) are one
  range. `rows` holds only the row-class stream; inline/subset **row names live in `extra`** (they can be huge for `sj_counts`).
* **Names kinds** `Inline, Pack, PackSubset, Cells, None`. The codec never sees pack contents. The digest needs the *logical* names, so
  pass `NamesSpec::blob` even for Pack/Cells (or a precomputed `digest`, or `compute_digest=false`).
* **f32** is stored as u32 bit patterns; codec 1 recodes matrices whose values are exactly integral and < 2^24 as integers (`flags` bit 0)
  so they cost the same as u32 counts; other f32 matrices go to codec 2 (raw bit patterns, zstd). `-0.0` and NaN are bit-exact.
* **Value 0**: flag means `value != 1`, residual `0` for value 0 and `value - 1` otherwise (a bijection). Costs nothing on matrices without
  zeros and lets `splice_psi`-like matrices (many 0/1) use codec 1.

## 3. Tests (all pass; counts are `CHECK`s)

| test | what | result |
|---|---|---|
| `test_pz2_api` | header round trip + CRC, truncation, chunk index, partial `ByteSource`, `plan_columns`, ISA forcing | 16 passed |
| `test_pz2_edge` | SHA-256 vectors (portable and SHA-NI, 400 random messages), 3 golden canonical digests computed by the independent Python implementation, n=0 / m=0 / nnz=0 / 1x1 / N<lanes / N=lanes+1, m=2^21-1 sparse (gap escapes), one dense column of 2^21-1 rows with huge values (residual escapes), values to 2^32-1, zeros, m=2^21 fallback, option grid (lanes 16/32/64 x pb 8/11/12 x kinds, cpo 0/1/2/8, chunk_cols 1..100000), thread-independent encoding, f32 integral / fractional / -0 / NaN, all output types, names kinds, kv, digest sensitivity, partial sources, 400 random corruptions, truncations, bad arguments; every case decoded on scalar + AVX2 + AVX-512, native and canonical | 1654 passed |
| `test_pz2_edge` (`-DPZ2_SCALAR_ONLY`) | same on the scalar-only build | 664 passed |
| `test_pz2_multi_tu` | header-only codec included from two translation units and linked (no ODR / duplicate symbols), encode + decode in the second | passes |
| `test_pz2_roundtrip` | **every** legacy `.1pz` of `bench/compression/corpus/*/` and `corpus_w4_corpus/*/` (see below) | see below |
| `bench/pz2/ref_decode.py` | independent decoder written from the spec text only: tiny, empty, zeros, f32, escapes (m=2^21-1) synthetics and a real 107k-nnz matrix match, stored digests reproduced | 6 / 6 |

`test_pz2_roundtrip` per matrix: encode (names inline, kv, digest, value_type = legacy vt_code), then decode on every supported ISA x
{native, canonical} (+ 3 threads on the best ISA) and compare indptr / indices / data with the legacy arrays byte for byte (native: per-column
set equality through `row_perm` and ascending stored ids), names, kv, stored digest, native output identical across ISAs, and 4 random
column ranges (canonical + native) decoded from a *partial* source (prefix + the planned chunk run only), `column_counts`.
Coverage includes the 8 barnyard matrices (579,921 and 696,825 rows: **codec 1 now codes them**, Track B could not), the 0x0 stubs (151-byte
legacy members), `splice_psi` matrices with explicit zeros, `sj_counts`, Visium, plate-based and inDrop/MARS/SPLiT protocols.

**Full corpus result (final code, 2026-10-22): 174 matrices, 6,192 checks passed, 0 failed** (`results/m_w1/roundtrip_full.log`; about 70 minutes
on the loaded laptop; `results/m_w1/roundtrip_partial_1.log` is an earlier partial run of the first version: 45 matrices, 0 failures).

## 4. Numbers

Machine: i7-1185G7 (4 cores / 8 threads, AVX-512, SHA-NI), Windows, GCC 13 `-O3`, no `-march`. **The machine was shared and loaded
(load 71 to 100%, two to three other agents' `python3.11` jobs at 90% each) during every run below EXCEPT run C of 4.1 (load 10 to 14%); all other
latency numbers (4.2) and the A/B of 4.3 are loaded-machine numbers. My dependent-add calibration read 0.9 to 2.6 GHz; the CIM clock field only says the nominal 2995 MHz.**

### 4.1 (a) Decode throughput, whole matrix, payload only (names `None`), min of 7 reps (Mnnz/s = nnz / min time)

Three runs under `results/.timing_lock`. "@3GHz" = measured x 3.0 / calibrated core GHz (the study's convention; the calibration chain and the decode may
see different clocks, so it is an optimistic bound). Core GHz is my dependent-add calibration, min of before/after (the CIM clock field only ever says the
nominal 2995 MHz). `bench/pz2/timing_final.sh` reproduces run C (it waits for the lock, about 2.5 minutes of measurement).

**Run C, the quiet window: 11:14:08 to 11:15:22, load 12% before, 14% after, 10% at the end** (below the 30% bar of the timing protocol; nothing of mine
ran concurrently; calibrated clock 1.7 to 2.6 GHz; `results/m_w1/timing_final.tsv`). Final code. Columns: native 1 thread (with @3GHz and calibrated clock),
native 2 and 4 threads, canonical 1 thread (with @3GHz and clock), canonical 2 and 4 threads:

| file (exon) | ISA | native 1 thr | native 2 | native 4 | canonical 1 thr | canonical 2 | canonical 4 |
|---|---|---|---|---|---|---|---|
| tcell_5p (10.53M nnz, 8.0 MB) | avx512 | 298.0 (422.6@3GHz; core 2.12 GHz) | 525.3 | 1083.9 | 70.7 (85.4@3GHz; core 2.48 GHz) | 133.0 | 211.5 |
|  | avx2 | 171.0 (255.8@3GHz; core 2.01 GHz) |  |  | 47.8 (76.2@3GHz; core 1.88 GHz) |  |  |
|  | scalar | 48.4 (38.5@3GHz; core 3.77 GHz) |  |  | 19.7 (29.2@3GHz; core 2.02 GHz) |  |  |
| brain_sn_v3_deep (6.35M nnz, 6.5 MB) | avx512 | 258.1 (377.8@3GHz; core 2.05 GHz) | 460.8 | 660.4 | 56.4 (68.7@3GHz; core 2.46 GHz) | 114.8 | 180.8 |
|  | avx2 | 161.0 (250.2@3GHz; core 1.93 GHz) |  |  | 42.0 (64.2@3GHz; core 1.96 GHz) |  |  |
|  | scalar | 25.9 (45.9@3GHz; core 1.69 GHz) |  |  | 17.9 (29.3@3GHz; core 1.84 GHz) |  |  |
| pbmc_v2_typical (2.57M nnz, 1.9 MB) | avx512 | 298.8 (400.4@3GHz; core 2.24 GHz) | 557.6 | 639.7 | 62.0 (82.4@3GHz; core 2.26 GHz) | 118.9 | 213.3 |
|  | avx2 | 252.0 (290.7@3GHz; core 2.60 GHz) |  |  | 71.8 (86.1@3GHz; core 2.50 GHz) |  |  |
|  | scalar | 30.2 (40.7@3GHz; core 2.22 GHz) |  |  | 21.5 (28.5@3GHz; core 2.27 GHz) |  |  |

Headlines of run C: **AVX-512 native, one thread: 298 / 258 / 299 Mnnz/s measured (tcell_5p / brain_sn_v3_deep / pbmc_v2_typical), i.e. the 200 Mnnz/s single-thread
target of the proposal is met on measured wall clock**, 378 to 423 on the @3GHz estimate. Four threads: 1,084 / 660 / 640 Mnnz/s (the 600 Mnnz/s four-thread target is
met on all three). AVX2: 161 to 252 (0.57 to 0.84 of AVX-512). Scalar: 26 to 48. Canonical AVX-512: 56 to 71 Mnnz/s on one thread (4.2 to 5.3 times slower than native),
181 to 213 on four. Against today's production reader (3.9 to 6.8 Mnnz/s on these files, full decode including names) native decode is 40 to 75 times faster at
one thread, which includes that the reader's number contains name parsing.

Whole-matrix decode speed does not depend visibly on the chunk width (tcell_5p, AVX-512 native, one thread, run C conditions): 295 / 297 / 329 / 292 / 259 Mnnz/s at
128 / 256 / 512 / 1024 / 4096 columns per chunk (`results/m_w1/chunk_decode_speed.txt`); only 4096 is slightly lower.

**Run B, a busier window** (10:50:30 to 10:52:01, load 100% at the start and 29% at the end, calibrated clock 1.2 to 2.0 GHz, my own compile and test jobs ran
during it; `results/m_w1/run1/timing_final_overlapped_with_builds.tsv`): AVX-512 native 1 thread 179 / 231 / 269, 4 threads 572 / 645 / 592; AVX2 77 / 105 / 140; scalar 12 / 16 / 18;
canonical 43 / 41 / 49. **Run A, the busiest** (10:01 to 10:03, load 71% before / 93% after, clock 0.9 to 1.4 GHz, `results/m_w1/timing_idle.tsv`, first canonical
implementation with `std::sort`): AVX-512 native 130 / 88 / 127, 4 threads 535 / 179 / 117; AVX2 57 / 50 / 56; scalar 16 / 11 / 9; canonical 24 / 15 / 20.
The spread between the runs is the machine, not the code: the same binary gives 88 to 130 under load and 258 to 299 on a quiet machine.

### 4.1b Canonical mode (bitmap sort)
Native -> canonical adds a permutation gather (2 ns/nnz in the loaded profile) and the per-column sort. `bench/pz2/canon_prof.cpp` on tcell_5p (10.5M nnz, mean 152 nnz/column, loaded
machine): gather only 532 Mnnz/s; **bitmap scatter/scan (shipped) 32.0 Mnnz/s; `std::sort` of packed keys 11.6 Mnnz/s** (2.8 times faster). Whole-decode canonical,
AVX-512: 56 to 71 Mnnz/s on one thread and 181 to 213 on four in the quiet run C, against 298 / 258 / 299 native; i.e. canonical costs 4 to 5 times native. Use native + `row_perm`
wherever the consumer can; canonical is the verification / legacy path.

### 4.2 (b) Chunk width (`results/m_w1/chunks_loaded.tsv` and `chunks_quiet.tsv`; sizes exact; the "one column" latency is from the quiet run, load 13 to 26%; the K-cell decode times were not updated and are loaded-machine)

Payload bytes (names `None`) relative to 1024-column chunks, mean chunk bytes, and the cost of random cell access (in-memory decode of the
touched chunks; "fetch" = prefix + the touched chunks, i.e. what a range-reading client downloads):

| file | cols/chunk | bytes vs 1024 | mean chunk | one column (median) | K=10 cells: chunks / fetch | K=100: chunks / fetch | K=1000: chunks / fetch |
|---|---|---|---|---|---|---|---|
| tcell_5p (n 69,249; 152 nnz/col; prefix 109 KB) | 128 | +1.01% | 14.7 KB | 11 us | 10 / 251 KB | 89 / 1.2 MB | 452 / 6.9 MB |
| | 256 | +0.43% | 29.3 KB | 25 us | 10 / 377 KB | 84 / 2.1 MB | 262 / 7.8 MB |
| | 512 | +0.14% | 58.2 KB | 39 us | 9 / 635 KB | 71 / 3.1 MB | 136 / 7.8 MB |
| | 1024 | 0 | 116 KB | 80 us | 7 / 655 KB | 51 / 5.7 MB | 68 / 7.8 MB |
| | 4096 | -0.11% | 465 KB | 290 us | 6 / 2.3 MB | 17 / 7.8 MB | 17 / 7.8 MB |
| tumor_3p (n 25,067; 10 nnz/col) | 128 | +8.06% | 1.6 KB | 11 us | 10 / 60 KB | 81 / 157 KB | 194 / 354 KB |
| | 256 | +3.45% | 3.1 KB | 17 us | 10 / 75 KB | 68 / 229 KB | 98 / 340 KB |
| | 1024 | 0 | 11.9 KB | 52 us | 8 / 157 KB | 25 / 329 KB | 25 / 329 KB |
| | 4096 | -0.87% | 42 KB | 202 us | 6 / 325 KB | 7 / 326 KB | 7 / 326 KB |
| pbmc_v2_shallow intron (n 96,446; 12 nnz/col) | 128 | +6.48% | 2.5 KB | 15 us | 10 / 112 KB | 92 / 318 KB | 549 / 1.4 MB |
| | 256 | +2.76% | 4.8 KB | 17 us | 9 / 116 KB | 88 / 507 KB | 350 / 1.7 MB |
| | 1024 | 0 | 18.5 KB | 57 us | 9 / 207 KB | 60 / 1.1 MB | 95 / 1.8 MB |
| | 4096 | -0.70% | 72.7 KB | 218 us | 8 / 554 KB | 24 / 1.8 MB | 24 / 1.8 MB |

Findings: (1) overhead of narrow chunks is about 190 bytes fixed per chunk (32 lane states + counts + index record), negligible for dense
samples (0.4% at 256 columns) and large for sparse ones (3 to 8%); (2) 1000 *scattered* cells touch almost every chunk at any width, so chunk
width only helps small cell sets (tens to a hundred cells: 256 columns fetch about half to a third of what 1024 does) and clustered/contiguous
selections already cost one range regardless; (3) decoding the one chunk that holds a wanted cell takes 11 us (128 columns) to 290 us (4096 columns) on the quiet machine, i.e. irrelevant against a network round trip
(100 to 275 ms measured in the proposal); (4) the `rows` section is most of the prefix (tcell: 100 KB of 109 KB), so the prefix is the dominant
cost for 1 to 10 cells and should be cached per sample object by clients. (5) whole-matrix decode speed shows no visible dependence on chunk width from 128 to 4096 columns (run C: 295 / 297 / 329 / 292 / 259 Mnnz/s at 128 / 256 / 512 / 1024 / 4096).

**Recommended default: a target of about 48,000 non-zeros per chunk** (`chunk_nnz_target = 49152`, `chunk_cols = 0`): a power of two in
[128, 4096] columns. It gives 256 columns for typical dense 10x data (30 KB chunks, +0.4% bytes), 512 for deep single-nucleus data,
4096 for sparse matrices (where it is 0.7 to 0.9% *smaller* than 1024), and keeps every file within +0.5% of its 1024-column size.
Measured across the 8 files of the sweep, "t48k" is within -0.9% to +0.5% of 1024-column bytes; the old 96k target was within -0.9% to +0.14% with
chunks 2x larger; 24k costs up to +1.0% for dense files. Decision D4 candidates 256 / 1024: the policy reproduces 256 for the dense case.

### 4.3 (c) Cost of the 2^21 row-id widening on normal files
* Size: payload (tables + chunk index + chunks) 5.751 / 6.030 / 8.058 bits per nnz (pbmc_v2_typical, tcell_5p, brain exon) against `w3_fast`'s
  5.749 / 6.028 / 8.056 at the same chunk widths: **+0.002 bits/nnz (+0.03%)**, entirely the 24-byte index records. The row-class stream is
  smaller than `w3_fast`'s per-row count stream (0.174 vs 0.246, 0.071 vs 0.096, 0.156 vs 0.204 bits/nnz: classes instead of exact counts and zstd-19).
* Speed, quiet machine (`results/m_w1/widen_ab_quiet.tsv`, load 8% before and 16% after, under the lock, 11:16): nine interleaved A/B pairs, single thread, AVX-512, min of 7,
  pz2 native vs `w3_fast` class-order on the same files and chunk widths (tcell 512, pbmc 512, brain 1024 columns), Mnnz/s (pz2 / w3_fast): tcell 453 / 415, 439 / 392, 344 / 289;
  pbmc 459 / 395, 366 / 278, 383 / 274; brain 428 / 361, 359 / 327, 335 / 290. **pz2 is 5 to 35% faster in all nine pairs** (calibrated core clock 2.6 to 4.7 GHz, i.e. turbo while
  the machine was quiet; the clock drifted down over the nine pairs for both programs). So widening costs no measurable speed (the extra B2/C3 phases are one mask test per vector);
  the reason pz2 is faster than `w3_fast` was not investigated (the kernel was restructured: no lambdas, one flush path, and the cache of this binary was built without `-march`, both compiled with GCC 13 -O3). Earlier, on the loaded machine (`results/m_w1/widen_ab.tsv`, load 100%): 86 to 100 vs 82 to 92 Mnnz/s, also no loss.
* What widening needed: two-part coding of uniform fields wider than 16 bits (gap extras up to 18 bits, value residuals up to 29 bits, column counts
  up to 19 bits), residual context table 84 instead of 80 entries. Barnyard (697k rows) now codes with the same 0.82 to 0.85 ratios as other matrices.

### 4.4 (d) Probability bits, lanes, classes per octave, contexts (`results/m_w1/params_sizes.tsv`, 1024-column chunks, bits/nnz incl. rows)

| setting | tcell_5p | pbmc_v2_typical | brain_sn_v3_deep | tumor_3p |
|---|---|---|---|---|
| pb 10 / pbv 10 (default) | 6.092 | 5.918 | 8.214 | 10.991 |
| pb 11 / 11 | -0.11% | -0.02% | -0.12% | +0.11% |
| pb 12 / 12 | -0.15% | +0.03% | -0.12% | +0.28% |
| lanes 16 | -0.05% | -0.05% | -0.10% | -0.48% |
| lanes 64 | +0.13% | +0.12% | +0.19% | +0.97% |
| cpo 1 / 2 / 8 / 0 (exact rank) | +1.51 / +0.49 / -0.07 / +0.07 | +1.37 / +0.54 / +0.03 / +0.14 | +0.11 / -0.07 / +0.09 / +0.29 | -0.65 / -0.14 / +0.11 / +0.45 |
| ctx kind 0 / 2 / 4 | 0 / +0.97 / +5.75 | 0 / +1.40 / +4.65 | 0 / +1.00 / +9.58 | +0.90 / 0 / +2.03 |

Defaults stay: 10 bits (11 or 12 gain at most 0.15% and enlarge the gather tables from 2 MB to 4 or 8 MB), 32 lanes (the AVX kernels are tuned for
32; 16 is up to 0.5% smaller on sparse files but slower), 4 classes per octave, context kind by estimate (kind 0 without estimating from 2M nnz;
the estimate picks kind 2 for tumor_3p).

### 4.5 Other measurements
* Size, the 14 corpus matrices of the study (`results/m_w1/sizes_corpus14.tsv`, default options, names `None`, i.e. payload + row classes + tables +
  index + header): **114,090,620 B legacy -> 44,111,744 B = 0.3866 (7.41 bits/nnz)**; per file 0.047 (tumor_3p intron) to 0.584 (tcell_5p exon). This equals
  `w3_total_v1`'s 46.44 MB minus its 2.32 MB of barcodes. Members with names stored inline at zstd-3 are 0.80 to 0.87 of the legacy `.1pz` (names dominate).
* Encode: 0.5 s single thread for 680k nnz, about 1.5 to 6 s for 10M nnz with 4 threads; inline names at zstd-19 dominate small matrices
  (85k barcodes + 104k row names take 15 s at level 19, 0.1 s at level 3): production matrices reference packs / cells instead.
* Digest: canonical digest of tcell_5p (84 MB hashed) 235 ms with SHA-NI (358 MB/s on the loaded laptop), 1744 ms portable.
* Corruption: 400 random 1 to 3 bit flips in a member never crashed a decoder and were detected or produced different output in all 400.

## 5. Known limits
* Codec 1: `m < 2^21` (the escape-free alphabet reaches gaps of 2^21); a chunk holds fewer than 2^28 non-zeros (writer: 2^27); `lanes` 16/32/64.
* Codec 2 has no such limits except `chunk` stream < 4 GiB and `m, n < 2^32`.
* `indptr` input form needs `nnz < 2^32`; `counts` form has no limit (decode into 64-bit `indptr`).
* The AVX2 ilog2 uses the exact float-exponent trick (valid for column counts < 2^24, always true for `m < 2^21`).
* Decoders read up to 1 KiB beyond a chunk blob (vector refills); `decode_*` copies the last chunk of a source into a padded buffer when needed.

## 6. Files
`include/singlet/pz2/pz2.h` (API) and `detail/`: `common.h` (CRC, LEB, zstd framing), `isa.h` (cpuid, selection), `simd.h` (target regions),
`container.h` (header, index, names parts), `values.h`, `sha256.h`, `rans1.h` (codec 1 core), `rans1_enc.h`, `rans1_dec.h` (setup + scalar reference),
`rans1_avx512.h`, `rans1_avx2.h`, `codec2.h`, `api.h` (encode / Matrix / decode / digest).
`tests/cpp/test_pz2_{api,edge,roundtrip}.cpp`. `bench/pz2/`: `build.sh`, `bench_pz2.cpp` (time / chunks / sizes), `ref_decode.py`, `pz2tool.cpp`,
`canon_prof.cpp`, `digest_prof.cpp`, `prof_enc.cpp`, `quick.cpp`, `timing_final.sh` (waits for the timing lock, runs the grid). Results: `bench/compression/results/m_w1/`
(`cache/` is git-ignored: encoded members, regenerated on demand; `ref/` holds the members the Python reference decoder was checked on).

## 7. Not done / how to resume
* **Latency numbers of 4.2 and the A/B of 4.3 were taken on the loaded machine** (only the throughput grid of 4.1 has a quiet run). The chunk sweep is dominated
  by sizes and chunk counts, which are exact; rerun `bench_pz2 chunks` on a quiet machine for clean latencies. Throughput by hand: `bash bench/pz2/build.sh bench/pz2/bench_pz2.cpp bench_pz2 -O3`, take
  `mkdir bench/compression/results/.timing_lock`, then `bench/pz2/bin/bench_pz2.exe time --reps 5 --threads 1,2,4 bench/compression/corpus/{tcell_5p,brain_sn_v3_deep,pbmc_v2_typical}/exon_counts.1pz`
  (encoded members are cached under `results/m_w1/cache`); the 200 Mnnz/s single-thread question is decided by that run.
* NEON / AVX2-only ARM ports; an MSVC build; a SHA-256 for ARM crypto extensions.
* Canonical mode is 4 to 6x native; a SIMD merge of the class runs or a fused permute-and-sort in the AVX-512 kernel are the next steps if needed.
* The `rows` section (row classes, 0.07 to 0.25 bits/nnz, most of the prefix) could use the adaptive Elias-gamma coder of `w3_total` (about 10% smaller).
* `gene` consumer mode (per-gene aggregation, spec 4.3) is not implemented; it needs the per-member stored-id -> gene map from the packs.
