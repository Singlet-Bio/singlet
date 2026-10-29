# w3_fast: sort-free fast decode of the b_rans16-class matrix payload

Status: PARTIAL, stopped at the coordinator's hard stop (2026-10-22 07:40 EDT). Owner: the `w3_fast` agent. Files:
`kernels/w3_fast.h` (registration `cs::register_w3_fast`, dispatch, threads), `w3_fast_core.h`, `w3_fast_enc.h`,
`w3_fast_dec.h` (format, encoder, scalar reference), `w3_fast_avx512.h`; `harness/w3_fast_bench.cpp`;
`tools/w3_fast_build.sh`; `results/w3_fast_*`. Build: `bash tools/w3_fast_build.sh harness/w3_fast_bench.cpp w3_fast_bench`.

## 0. Bottom line (five numbers)

1. **Size: no cost.** Class-order layout (4 classes per octave), same lock-step static rANS model as Track B: mean payload
   **8.226 bits/nnz over the 14 files** (b_rans16: 8.23), every file within +-0.015 of b_rans16. Row-count side info is the same
   LEB+zstd-3 stream as Track B (0.07-3.0 bits/nnz, not in "payload").
2. **Correctness: all 14 corpus files bit-exact**, scalar and AVX-512 decoders, in class-order mode (checked against an
   independent reference built from the input matrix) and in canonical mode (`decode(encode(M)) == M` on indptr, indices,
   data, row names, column names, key-values). `results/w3_fast_verify_14files.log`: 28/28 OK, 0 FAIL.
3. **Speed, single thread, class order (no sort), AVX-512, int32 indices + uint32 values written straight into the final arrays:**
   69-89 Mnnz/s measured on a machine at 96-100% CPU load (core clock 1.3-1.4 GHz by calibration), best window 117 Mnnz/s on
   tcell_5p; 3 GHz-calibrated estimate 159-207 (240 in the best window). **Target 200 Mnnz/s is met on the estimate for the
   favourable file (pbmc_v2_typical 207, tcell_5p 240 in the quiet window), missed on the others (159-165) and never met on
   measured wall clock.** Against b_rans16 *native* measured in the same session the kernel is not faster (+33% / -10% / -15%);
   against b_rans16 *canonical* it is 1.5-2.1x faster because it needs no sort. The sort-free layout therefore delivers the
   "native" speed of Track B to a consumer that wants a CSC, but the kernel itself is not yet better than Track B's.
4. **Threads (tcell_5p exon, chunk-parallel, shared CPU):** 1/2/4/8 threads = 117 / 231 / 539 / 281 Mnnz/s in one quiet window
   (est@3GHz 240 / 400 / 819 / 710); 80 -> 220-250 (2.6-3.1x) under 100% load. 600 Mnnz/s at 4 threads is met on the
   3 GHz estimate (819), not measured (539 max). 8 threads (SMT) is worse than 4.
5. **Canonical (original row order, exact) is not fixed:** 40-44 Mnnz/s measured (est 93-104), the same as b_rans16 canonical
   (43-52 same session), because it reuses Track B's vector merge sorter. Layout B (sort-free ascending ids) was not built.

## 1. Stage table

All numbers are min of 3 reps, payload decode only (open ~3 ms for table build excluded from the Mnnz/s; names excluded),
output = `int32 indptr[n+1]`, `int32 indices[nnz]`, `uint32 data[nnz]` (AVX-512 kernel writes uint32 values; narrowing to
uint16/uint8/float64 is a post-loop, not fused). "est" = measured x 3.0 / calibrated core GHz (Track B's method, a 1-cycle add
chain before/after; the machine was never idle: Windows reported 96-100% CPU load during every timed run, up to 8 other
processes). `ref_1pz_v1_L3` = today's production C++ reader through `bin/bench.exe`, including name parsing.

| file (exon) | codec / layout | mode | payload bits/nnz | Mnnz/s measured | est@3GHz | ISA | thr |
|---|---|---|---|---|---|---|---|
| pbmc_v2_typical 2.57 M | w3f L32 cpo4 | class-order | 5.749 | 88.7 | 207 | AVX-512 | 1 |
| | w3f L32 cpo4 | canonical (Track B sorter) | 5.749 | 44.4 | 104 | AVX-512 | 1 |
| | b_rans16 (same session) | native / canonical | 5.74 | 66.6 / 42.6 | | AVX-512 | 1 |
| | ref_1pz_v1_L3 | full decode | 6.85 (today) | 3.9 | | scalar | 1 |
| brain_sn_v3_deep 6.35 M | w3f L32 cpo4 | class-order | 8.056 | 69.3 | 159 | AVX-512 | 1 |
| | w3f L32 cpo4 | canonical | 8.056 | 44.2 | 102 | AVX-512 | 1 |
| | b_rans16 (same session) | native / canonical | 8.06 | 76.9 / 44.9 | | AVX-512 | 1 |
| | ref_1pz_v1_L3 | full decode | 9.44 (today) | 5.4 | | scalar | 1 |
| tcell_5p 10.53 M | w3f L32 cpo4 | class-order | 6.028 | 76.8 (117 quiet window) | 165 (240) | AVX-512 | 1 |
| | w3f L32 cpo4 | canonical | 6.028 | 43.2 | 93 | AVX-512 | 1 |
| | b_rans16 (same session) | native / canonical | 6.02 | 90.4 / 51.5 | | AVX-512 | 1 |
| | ref_1pz_v1_L3 | full decode | 6.96 (today) | 6.8 | | scalar | 1 |
| tumor_3p 0.245 M | w3f scalar vs AVX-512 | class-order | 9.73 | 13.1 / 46.8 (first kernel) | | | 1 |

Raw rows: `results/w3_fast_time1.tsv` (scatter kernel, L32 vs L64), `w3_fast_time3.tsv` (final flush kernel, L32 and L16),
`w3_fast_trackb_yardstick.tsv`, `w3_fast_ref1pz.tsv`, `w3_fast_threads*.tsv`, `w3_fast_chunk.tsv`, `w3_fast_sizes14.tsv`.

Payload bits/nnz, all 14 files (`results/w3_fast_sizes14.tsv`; side = row-count stream, not in payload):

| file | exon payload / side | intron payload / side | b_rans16 exon / intron |
|---|---|---|---|
| brain_sn_v3_deep | 8.056 / 0.204 | 7.262 / 0.188 | 8.06 / 7.26 |
| brain_sn_v3_intronic | 9.381 / 0.591 | 7.507 / 0.148 | 9.39 / 7.51 |
| pbmc_v2_shallow | 7.051 / 0.127 | 11.998 / 0.607 | 7.05 / 12.01 |
| pbmc_v2_typical | 5.749 / 0.246 | 8.172 / 0.588 | 5.74 / 8.18 |
| pbmc_v3_lowgenes | 5.559 / 0.071 | 7.060 / 0.205 | 5.56 / 7.07 |
| tcell_5p | 6.028 / 0.096 | 9.743 / 0.653 | 6.02 / 9.75 |
| tumor_3p | 9.730 / 1.617 | 11.872 / 3.040 | 9.75 / 11.92 |

### Knobs measured (partial; everything else in the brief's knob table was not run)

| knob | setting | payload bits/nnz (pbmc_v2_typical exon / brain exon) | speed |
|---|---|---|---|
| classes per octave | 0 (exact rank) / 8 / 4 / 2 / 1 | 5.735 / 8.052; 5.744 / 8.053; 5.749 / 8.056; 5.791 / 8.064; 5.853 / 8.098 | no systematic effect |
| lanes | 16 / 32 / 64 | 5.743 / 5.749 / 5.762 | flush kernel, pbmc / brain / tcell: L16 70 / 72 / 63, L32 89 / 69 / 77; scatter kernel: L32 64 / 66 / 64, L64 56 / 54 / 61 (64 slower) |
| contexts | kind 0 (pos40 x depth12, 1.9 MB table) vs kind 4 (pos20, 80 KB) | 5.749 vs 6.024 | 70.2 vs 71.3 Mnnz/s: table size does not matter (same finding as Track B) |
| chunk columns (tcell_5p) | 128 / 256 / 512 (auto) | 6.075 / 6.044 / 6.028 | 105 / 84 / 77-117 (noise larger than effect) |
| output path | scatter vs 16-step transposed flush | same | pbmc 64-70 -> 89; scatter probe: removing both scatters gave 69 -> 89, i.e. scatters were 22% of the time |

Column-range decode: chunks are self-contained (column counts inside the chunk blob; chunk table gives byte size and nnz), so
a cell subset needs: header + three table streams (zstd-19, ~12-14 KB, table expansion ~3 ms = the "open" column) + the chunk
table + the chunks that overlap the range. Decoding one chunk of 512 columns x 170 nnz (87 K nnz) takes ~1 ms, so table
build, not decoding, dominates a small range query; the tables could be built lazily per context. Not benchmarked beyond that.

## 2. Format W3F1 (what the code does; enough to write a spec)

Integers little-endian. `LEB` = unsigned LEB128. `STREAM` = `u64 raw_len, u64 zstd_len, zstd bytes` (Track A/B framing).

```
file    : HDR | NAMES | ROWCOUNTS | PAYLOAD
HDR     : 'W' u8(1) | u32 m | u32 n | u64 nnz | u8 vt (1/2/3 = max value fits u8/u16/u32) | u8 L (16|32|64) | u8 cpo
          | u8 pb | u8 pbv | u8 sh | u8 cap | u8 ks | u8 kcap | u8 nk | u32 chunk_cols
NAMES   : STREAM(rownames, NUL joined, ORIGINAL order, zstd-3) STREAM(colnames) u32 nkv STREAM(kv)    [Track B layout]
ROWCOUNTS: STREAM(LEB(nnz of each original row), zstd-3)         (class order is derived from it, see below)
PAYLOAD : TAB_CNT | TAB_GAP | TAB_RES | u32 nchunks | { LEB blob_bytes, LEB chunk_nnz } x nchunks | blob x nchunks | 64 zero bytes
TAB_x   : STREAM(zstd-19 of: LEB nused, per used ctx: LEB(ctx - prev) LEB nsym, per sym: LEB(sym - prev) LEB freq)
          TAB_CNT: 1 ctx, 124 symbols, 12 probability bits.  TAB_GAP: nk*(cap+1) ctx x 248 symbols (cls*2+flag), pb bits.
          TAB_RES: 80 ctx x 124 symbols, pbv bits.
blob    : LEB nwc | LEB nw | u32 cstate | u16 cwords[nwc] | LEB pp0 (for each lane, ascending, that starts inside a column)
          | u32 X[L] | u16 words[nw]
```
* **Row order (class order).** key(count) = 0 if count == 0, else 1 + lg*cpo + floor((count/2^lg - 1)*cpo), lg = floor(log2 count)
  (cpo = 0: key = count, i.e. exact rank). Row ids are renumbered by (key descending, original id ascending). The decoder
  re-derives this from ROWCOUNTS with a counting sort; `perm[class id] = original id`. A consumer that takes native order
  receives CSC `indices` in this permuted space plus `perm` (or simply the row names written in class order).
* **Elements, contexts.** Per column (k nonzeros, class-order ids r_0 < r_1 < ...): PP = previous id + 1 (0 at column start),
  gap g = r - PP, class c = Track B `g_cls` (g < 8 direct, 4 sub-buckets per octave, nb extra bits, `nb <= 16`, so m <= 2^19),
  flag = value > 1. Symbol = 2c + flag, coded with context `ctx = kc*(cap+1) + pc`, `pc = min(posb(PP) >> sh, cap)`,
  `posb(p) = floor(2*log2(p+2)) - 2` (computed as `(bits(float(p+2)) >> 22) - 256`), `kc = min(floor(log2(k+1)) >> ks, kcap)`.
  Flagged elements then code `value-2` (class + extra bits) with context `(posb(rank) >> 1)*4 + dk`, `dk = min(11*floor(log2(k+1)) >> 5, 3)`.
* **rANS.** 32-bit state per lane, renormalisation by 16 bits (`x < 2^16 -> x = x<<16 | word`), table entry
  `sym<<24 | freq<<12 | bias` with `x' = freq*(x >> pb) + bias`, `bias = (x & (2^pb-1)) - cum`. Extra bits are uniform symbols
  (`ex = x & (2^nb-1); x >>= nb`, refill). Encoder: reverse order, `x_max = freq << (32-pb)`.
* **Chunk lanes.** A chunk is `chunk_cols` consecutive columns (auto: ~96 K nnz, power of two in [128, 4096]). Its N elements
  are cut into L contiguous slices of S = N/L (+1 for the first N mod L lanes), NOT on column boundaries. Lane l starting
  inside a column stores `pp0` = previous id + 1 (LEB, lane order). Column counts are coded with TAB_CNT as a single-state rANS
  (symbol class, then extra bits) in the blob. Per step s the global decode order is: phase A for all active lanes (gap/flag
  symbol, refill), B (extra bits, refill; `rank = PP + base[c] + ex`), C1 (flagged lanes: residual symbol, refill), C2
  (flagged: residual extra bits, refill), lanes ascending in every phase; all refills read one shared u16 stream (the stream
  is stored in decode order; the encoder runs steps/phases/lanes backwards and reverses the words). Final partial step
  (N mod L lanes) is masked.
* **Why the AVX-512 refill is branch free:** `vpexpandw` of 32 words under the mask of lanes that need a word, then advance
  the pointer by popcount (same as Track B). Needs 64 readable bytes past the last blob (the 64 zero bytes).

## 3. Decoder structure, dispatch, portability

* `decode_chunk_scalar` is the format definition (portable C++, integer only). `decode_chunk_avx512<OT>` keeps NV = L/16
  zmm vectors in flight, writes ids and values into a [16 steps][L] scratch and flushes it with a 16x16 transpose into each
  lane's contiguous output slice (masked at the ends), so there is no scatter. Both give identical bytes (28/28 checks).
* Runtime dispatch (`wf::best_isa()`): `__builtin_cpu_supports` for avx512f/bw/vl/dq/vbmi2/bmi2, else avx2+bmi2, else scalar;
  the AVX-512 path needs the same checks Track B used. Lane width L is a template parameter (16/32/64) and a header field.
* **AVX2 (not written).** Needs: 4-8 ymm vectors of 8 lanes; `vpgatherdd` (exists); refill via `movemask` of the 8-bit refill
  mask, a 256-entry LUT of `vpermd` indices, `vpmovzxwd` of a 16-byte load, blend (about 11 uops per 8 lanes vs 12 per 16);
  `vpsllvd/vpsrlvd` exist; no `vplzcntd` (the float-convert trick already used for `posb` works on ymm; the depth class
  needs a short loop or the same trick); masked residual gather via `vpgatherdd` with a mask vector; the 8x8 transpose
  flush. Expected 50-70% of the AVX-512 rate on this core (ports 0/1/5 for 256-bit vs 0/5 for 512-bit).
* **NEON / ARM (not written).** 4 lanes per q-register, 8 registers for 32 lanes: no gather (4 scalar table loads inserted into
  lanes; the tables are 4 B entries so this is the dominant cost), no expand (refill via `vqtbl1q_u8` with a 16-entry LUT
  indexed by the 4-bit mask), `vclzq_u32` native, `vmulq_u32`, per-lane variable shifts `vshlq_u32` (negative count for right),
  `vminq_u32/vmaxq_s32`, 4x4 transposes (`vtrn`) for the flush. SVE2 would give gather and compact. Apple M-series should
  reach roughly the AVX2 rate; no float operations are required.

## 4. What failed / did not help, and why the target is missed

* 64 lanes (4 zmm in flight) were slower than 32 (scatter kernel: 56 / 54 / 61 vs 64 / 66 / 64 Mnnz/s): the loop is not bound by the dependency chain across
  steps; with 4 vectors the compiler spills the per-vector state. 16 lanes were close to 32 on dense files.
* Smaller tables (kind 4, 80 KB instead of 1.9 MB) gave the same speed: not cache-bound (confirms Track B).
* Scatter output cost 22% of the time (probe: remove both scatters -> 69 to 89 Mnnz/s); the transposed flush recovered most
  of it (64-70 -> 89 on pbmc_v2_typical) but the remainder of the loop still costs about 14-15 core cycles per nnz at the
  calibrated 1.3 GHz, far above the ~4 cycles/nnz that a uop count suggests (about 130 vector uops per 16 elements). I did not find
  where the rest goes; candidates are the gather latency (L2 misses, 2-4 independent gathers in flight), the serial `wp`
  pointer chain through 4+ refills per step, and SMT contention (every run shared the core with a busy sibling thread).
  A proper per-phase timestamp profile like Track B's `TB_PROFILE` was not done.
* Canonical order was not improved: it reuses Track B's `CanonVec`, so it costs what b_rans16 canonical costs. A merge of the
  per-class sorted runs (class order keeps ids ascending inside a class) should beat a full sort, but was not written.
* The kernel emits uint32 values and a 4 B id; uint16 / uint8 / float64 outputs (scipy, R `dgCMatrix`) are a narrowing loop
  after the chunk, not fused (extra ~0.2 cycle/nnz for the loop, extra memory traffic for float64).

## 5. Not finished (how to resume)

1. Layout B (ascending original ids without a sort), AVX2 kernel, fused uint16/uint8/float64 stores: not started.
2. Idle-machine timing: all numbers above were taken at 96-100% CPU load. Re-run `bin/w3_fast_bench.exe time --lanes 32 --reps 5
   --isa avx512 --cache results/w3_fast_cache corpus/<f>/exon_counts.1pz` under `results/.timing_lock` when nothing else runs.
3. Knob grid (pb 11/12, pbv, other ctx kinds, 4096-column chunks) and a TB_PROFILE-style phase profile of `chunk_avx512_u32`.
4. Per-vector prefetch of the table rows, a branch-free column-event scheme (precomputed event step per lane) and fusing the
   residual phases are the first things to try for the remaining 2x.
5. Register the codecs in a harness: `#include "w3_fast.h"` and call `cs::register_w3_fast(v)` (names `w3f`, `w3f_scalar`,
   `w3f_avx512`; canonical decode); `registry.h` was not edited.
