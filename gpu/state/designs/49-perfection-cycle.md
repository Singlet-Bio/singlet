---
feature: perfection_cycle_49
roadmap_id: 49
module: fixes across include/singlet-gpu/{graph,integrate,de,reduce,fate}/ + tests/integration_e2e_correctness.cpp
status: design
tolerance: all cycle-8/11/14/35/43 followups resolved; end-to-end integration test passes against scanpy ground truth
target_perf: no new perf claims; verify existing ones hold
ooc_plan: reuse cycle 16 streaming driver
---

## Why this exists

Cycle 48 lit-scout returned **NO STRONG CANDIDATE**: the "first-GPU implementation" pipeline has exhausted obvious targets in the current literature (2023-2026). Recent scouts rejected 7 probe categories (allelic XCI, MT clonal, splice junction DE, cell cycle, ATAC motif beyond chromVAR, ADT norm, doublets) as either already-GPU-native, <10× speedup, or not-singlify-compatible.

Rather than ship a weak 23rd/24th first-GPU feature, this cycle hardens what we have: fixes accumulated 🔴 followups and builds the end-to-end integration test on GSM4037629 (11,560 cells, full artifact suite).

**This is the FIRST non-feature cycle in singlet-gpu's history.** It exists because the orchestrator has judgment to recognize that "ship another weak feature" is worse than "strengthen the 22 strong features already shipped."

## What gets fixed

### 🔴 Cycle 8 followup: knn.h device radix sort
Replace host-side `std::partial_sort` + H2D round-trip with `cub::DeviceRadixSort::SortPairs` for per-query top-k selection. Affects `graph/knn.h` Exact backend.
- LOC: ~150 additional.
- Defers compile validation to first GPU dispatch (design + code only).

### 🔴 Cycle 11 followup: wilcoxon deterministic device segmented scan
Replace host-mediated sequential accumulation in the opt-in `cfg.deterministic=true` path with `cub::DeviceSegmentedReduce::Sum`. Affects `de/wilcoxon.h`.
- LOC: ~80.

### 🔴 Cycle 14 followup: harmony device-side segmented reduce
Replace ~2.8 GB/iter PCIe download of R + Z with `cub::DeviceSegmentedReduce::Sum` keyed by `(cluster_id * n_batches + batch_id)`. Affects `integrate/harmony.h`.
- LOC: ~120.

### 🔴 Cycle 35 followup: knn wrapper field-access style
sed fix: `compute_knn` wrapper uses method-style `.rows()` / `.cols()` on `SparseMatrixGPU` but factornet uses direct field access `.rows`, `.cols`, `.col_ptr.data()`, `.row_indices.data()`, `.values.data()`. Cycle 35 kernel-dev caught this by calling `compute_exact` directly. Fix the wrapper.
- LOC: ~10.

### 🔴 Cycle 40 followup: DAESC stratified ASE wire-up
Wire the per-cell-type masked gather kernel. `d_cell_type` is uploaded, strat arrays pre-allocated, but the per-type gather is missing (output zero-filled). Affects `ase/daesc.h`.
- LOC: ~150.

### 🟡 Cycle 43 followup: cellrank2 Arnoldi device Hessenberg
Replace host-resident H + per-step scalar D2H (10×30×8 = 2.4 KB total) with device-side triangular solve via cuSOLVER for Givens rotations on a small k×k Hessenberg. Eliminates the one flagged cudaMemcpy in cycle 43.
- LOC: ~100.

### Integration test — NEW
`tests/integration_e2e_correctness.cpp`: end-to-end pipeline on GSM4037629 (11,560 cells):
  1. Load `exon_counts.1pz` via `io/pz_device_loader.h` (cycle 0).
  2. Log-normalize via `preprocess/lognorm.h` (cycle 2).
  3. HVG top 2000 via `preprocess/hvg.h` (cycle 3).
  4. Randomized SVD (k=30) via `reduce/svd/randomized.h` (cycle 4, factornet).
  5. kNN (k=15) via `graph/knn.h` **compute_knn wrapper** (validates cycle 35 followup fix).
  6. Leiden clustering via `graph/leiden.h` (cycle 7).
  7. UMAP embedding via `embed/umap.h` (cycle 8).
  8. Wilcoxon DE between top-2 clusters via `de/wilcoxon.h` (cycle 9).
  9. Marker scoring via `anno/marker_score.h` (cycle 10).

Compare at each stage vs scanpy reference subprocess. Tolerances per original design docs.

## GPU implementation strategy

Six kernel-dev dispatches + one validator dispatch in parallel:
- kernel-dev 1: knn.h device radix + wrapper field-access fix (cycles 8 + 35 combined).
- kernel-dev 2: wilcoxon.h deterministic scan (cycle 11).
- kernel-dev 3: harmony.h device centroid (cycle 14).
- kernel-dev 4: daesc.h stratified wire-up (cycle 40).
- kernel-dev 5: cellrank2.h Arnoldi device Hessenberg (cycle 43).
- validator 1: integration_e2e_correctness.cpp (NEW test).

Realistically, we can't dispatch 6 kernel-devs in one message (max 3). Plan: cycles 49a + 49b + 49c sub-cycles, each with 3 dispatches.

## Cycle 49a dispatches (this turn)
1. kernel-dev: combined knn.h fix (cycle 8 device radix + cycle 35 wrapper field access).
2. kernel-dev: harmony.h device centroid (cycle 14 — highest-PCIe impact).
3. validator: integration_e2e_correctness.cpp (exercises all of cycles 0-10 together).

## Cycle 49b dispatches (next turn after 49a closes)
4. kernel-dev: wilcoxon.h deterministic scan (cycle 11).
5. kernel-dev: daesc.h stratified wire-up (cycle 40).
6. kernel-dev: cellrank2.h Arnoldi device Hessenberg (cycle 43).

## Correctness test spec

The new integration test is the primary correctness harness. Each fix is verified by the test's regression clause (cycle 35 fix validates via test step 5 calling the wrapper; cycle 14 fix validates via a supplementary harmony integration subtest).

Per-fix tolerance:
- Cycle 8 knn: top-k ranking ARI ≥ 0.99 vs scanpy (effectively identical).
- Cycle 11 wilcoxon: deterministic path bit-identical across seeds.
- Cycle 14 harmony: per-cell embedding Frobenius norm ≤ 1e-5 vs existing (non-PCIe-broken) golden.
- Cycle 35 wrapper: compile success + knn ranking match to `compute_exact` direct call.
- Cycle 40 daesc strat: per-type beta Spearman ρ ≥ 0.95 vs pooled baseline.
- Cycle 43 cellrank2: zero flagged cudaMemcpy in Arnoldi loop; absorption prob Spearman ρ ≥ 0.99 vs current.

## Risks

1. **No GPU compilation on login node** → all fixes stay design-only until first GPU dispatch. This is the same constraint as every prior cycle.
2. **Cycle 14 Harmony is the biggest** change — most risk of breaking existing behavior. Include a "before/after PCIe byte count" assertion in the harness.
3. **Integration test size** — could blow past 1500 LOC test budget. Split per-stage if pressure.
4. **Cycle 49a/b split** means cycle 49 is a two-cycle super-cycle. Document clearly.

## What this is NOT

- NOT a new first-GPU feature. We are not shipping a 23rd first-GPU this cycle.
- NOT a refactor. Every change is scoped to a pre-existing 🔴 followup.
- NOT a documentation pass. Production code changes, measurable fixes, one new test.
