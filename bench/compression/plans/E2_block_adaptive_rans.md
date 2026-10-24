# E2: The fast frontier coder: block-adaptive model + interleaved rANS

> **Status 2026-10-22: partial (agent stopped on the API limit).** A bit-exact codec `e2_b1024` exists and matches
> b_rans16's size within 0.1 bits/nnz on two large files in ascending row-id order; the fast decoder and the
> block-size sweep were not done. See `notes/e2.md` for how to resume.

Context: `notes/theory_fable.md` sections 4 and 8. The data say a shipped model is worth < 0.5%; adaptation is what
matters. Per-symbol adaptation forces a serial binary arithmetic coder (~1-20 Mnnz/s: Track C 1 Mnnz/s, Idea 1
target 20). Per-block adaptation (model frozen within a block of ~1024 columns, updated deterministically between
blocks from the decoded block) keeps nearly all of the adaptive gain while permitting static-table rANS with 4-8
interleaved states, SIMD-friendly, 200+ Mnnz/s. This plan builds that coder and measures the adaptation regret of
block-static versus per-symbol.

## Design

Streams per block (1024 columns, called cells first):
1. depths N_c: varint, zstd or rANS with adaptive table.
2. presence gaps in rank order under the survival CDF of the block's per-feature × depth-bucket model
   (prefix sums of −log2(1 − p_f(d)) in 16-bit fixed point per bucket). Encoding a gap = coding the quantile
   interval [S(prev+1), S(j)] ∪ p_j. To make this rANS-able: code the gap as (e, mantissa) where e = exponent bucket
   of the survival-mass difference (adaptive 24-symbol table per depth bucket) and the mantissa as raw bits; cost
   penalty versus exact arithmetic coding ≈ 0.02-0.05 bits/nnz (measure it).
3. counts given presence: symbol = min(count−1, 15) with escape; adaptive table per (feature-frequency class ×
   depth bucket), rANS.
4. interval split given gene: adaptive per-gene table (block-static), rANS; or fold intervals into the feature
   space (then step 2 does it) and compare.
Between blocks: feature counts and depth histograms are updated from the decoded block (integers), tables rebuilt
with shrinkage toward the file-level running estimate; the first block uses a flat prior (or, if it pays, a tiny
shipped prior: ~100 KB per reference build, test both).

Decoder structure: per block, rebuild tables (O(m) once per block: ~1 ms), then decode with 8 interleaved rANS
states over the gap/count streams, binary-search the survival prefix table for each gap (two-level table: 4096
coarse entries + fine; expect ~2 cache misses per nonzero). Scalar first; AVX-512 for the interleaved rANS and the
coarse search later.

## Measurements

Per corpus file: payload bits/nnz for block sizes {256, 1024, 4096, whole file} versus per-symbol adaptive (Track
C's c_r2 / Idea 1's coder as reference), and versus the E1 L1 cross-entropy (the model's own ceiling); decode
Mnnz/s single-threaded for each; the (e, mantissa) approximation penalty; the cost of block boundaries
(first-block flat prior) for small files (tumor_3p).

## Deliverables

- `kernels/e2*.h`, `harness/e2_bench.cpp` (copy of harness/bench.cpp with its own all_codecs), results in
  `results/e2_*.tsv`, write-up `notes/e2.md` with the frontier table (bits/nnz vs Mnnz/s) and the regret curve.

## Anticipated result

Block-1024 adaptation within 0.1-0.2 bits/nnz of per-symbol adaptation; decode 100-300 Mnnz/s scalar for the
gap/count streams; total file decode for brain_sn_v3_deep exon ≈ 50-100 ms single-threaded. If the (e, mantissa)
penalty is > 0.1 bits/nnz, switch the gap coder to a binary range coder for the exponent and keep rANS for counts.

## What already exists (read before writing code)

- `kernels/idea1.h`, `idea1_codec.h`, `idea1_fast.h` (Idea 1): survival-function gap coding under a per-feature ×
  depth model with rANS (64-bit state), a 4-lane interleaved decoder with AVX-512 tree search over the log-survival
  prefix table, and the finding that the scalar decoder is DRAM-bound on the 2.5 MB prefix table.
- `kernels/track_b_rans.h`, `track_b_lane.h`, `track_b_avx512.h` (Track B): 32-lane lock-step static rANS with
  pb=10 probability tables that stay in L2, 62-74 Mnnz/s payload decode in native (rank) order; the canonicalisation
  to sorted row ids costs 3-4x (`track_b_canon.h`, `track_b_sort.h`): **the bottleneck is producing canonical CSC,
  not entropy decoding**. E2 must either (a) emit rank-order columns and sort with a SIMD-friendly method, or
  (b) define the model's feature order so that decode order *is* canonical order (the survival coder is
  order-free, so use ascending row id and keep the prefix table in that order), which removes the canonicalisation
  pass entirely. Option (b) is the one to build first.
- `kernels/idea2.h`, `idea2_core.h` (Idea 2): the rank-1 Poisson margins prior (feature counts z_f + per-cell nnz
  fit by integer alternating projection) that matches the cross-validated per-feature × depth optimum with no
  table transmitted; the order-statistic tail coder; the ambient multiset coder for empties.
- `kernels/track_c_core.h` (Track C): Q40 fixed-point exp/log tables and the run-skipping binary coder (the
  exact-arithmetic reference for the regret measurement).
Reuse freely (all in-tree); keep E2's files under `kernels/e2*.h`, `harness/e2_*.cpp`.

## Depends on

Nothing now; the inputs above are on disk.
