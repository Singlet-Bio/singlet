---
feature: csi_gep
roadmap_id: 28
module: include/singlet-gpu/reduce/nmf/csi_gep.h + python/singlet_gpu/reduce/csi_gep.py + r/R/csi_gep.R
status: design
tolerance: per-program top-30 gene Jaccard ≥ 0.85 vs CSI-GEP Python reference; selected k matches reference within ±2
target_perf: 100k cells × 30k genes × 100 NMF runs × k=5..30 ≤30 min on A100 (CSI-GEP Python is ~2-4h)
ooc_plan: subsample-and-fit pattern is naturally chunked; per-bootstrap fit uses cycle 5 NMF adapter
---

## Why this exists

Cycle 27 (Cell2fate) is in flight as the first Phase B feature. Cycle 28 (CSI-GEP) is the second. Per the cycle 26 lit-scout, CSI-GEP is the third top priority for post-wrapper-sprint work. It's distinct from cycle 5 NMF in three ways:

1. **Consensus across many runs**: 100 independent NMF fits per rank k.
2. **Bootstrap subsampling**: each fit uses a random subsample of cells, not the full matrix.
3. **Jaccard-based program filtering**: a "robust" gene expression program (GEP) is one that recurs across subsample fits with top-30 gene Jaccard > 0.7.
4. **Auto-rank selection**: k is chosen at the reproducibility inflection point, not user-specified.

Cycle 5 NMF is the deterministic-fast path. CSI-GEP is the high-confidence-discovery path. They coexist.

Per cycle 27 lit-scout, the reference implementation is `geeleherlab/CSI-GEP` using torchNMF (PyTorch backend). Our advantage: factornet's GPU NMF is faster than torchNMF, and we can reuse the cycle 5 adapter as the inner loop.

## Algorithm

```
Inputs:  raw count matrix X (m × n), k_range = [5, 6, ..., 50], n_runs = 100, subsample_frac = 0.8
Outputs: chosen_k, robust_programs (k × m), program_usage (n × k), reproducibility_curve

For k in k_range:
    For run = 1..n_runs:
        cells_subsample = random_choice(n_cells, n_cells * subsample_frac, seed=run)
        X_sub = X[:, cells_subsample]
        (W_run, H_run) = nmf_fit(X_sub, rank=k)        # cycle 5 adapter call
        # store W_run as one of n_runs candidate program matrices
    # consensus: cluster the n_runs * k columns of all W_run via spherical k-means → k consensus programs
    consensus[k] = spherical_kmeans(concat(W_run for run), k_clusters=k)
    # reproducibility: Jaccard of top-30 genes per consensus program across runs
    reproducibility[k] = mean_jaccard_top30(consensus[k], W_run for run)

# Auto-select k at the reproducibility inflection
chosen_k = argmax_k reproducibility[k] - reproducibility[k+5]   # elbow detection

# Robust programs at chosen_k
robust_programs = consensus[chosen_k]
program_usage = nnls_solve(X, robust_programs)   # one per cell
```

## GPU implementation strategy

The outer loop is `len(k_range) * n_runs = 50 * 100 = 5000 NMF fits`. Each fit is a cycle-5 NMF adapter call with `factornet::nmf::nmf_fit_gpu`. Embarrassingly parallel across runs but factornet's GPU NMF is single-stream — we run them sequentially.

To accelerate:

1. **Reuse the device-resident X**: load X once, never re-upload. Subsampling is index-based (just pass a `device_indices` array to factornet).
2. **Batch consensus k-means** on the device: stack all `n_runs * k` candidate program vectors into one big matrix and run spherical k-means once per k. Use cuML or a custom kernel.
3. **Jaccard top-30 on device**: for each pair of consensus programs vs run programs, compute top-30 gene sets via `cub::DeviceRadixSort` and intersect via bitmap → popcount.
4. **Auto-k selection**: small array (k_range size), trivial host-side.

### Subsampling via device indices

factornet's `nmf_fit_gpu` takes the full CSC arrays. To subsample, we either:
(a) Build a sub-CSC on device (gather columns) — costly per run but only O(nnz/runs) per call.
(b) Pass a `cell_mask` to factornet — would require a factornet API extension.

Option (a) is simpler and only ~1 ms per gather kernel call. Use cub's `DeviceSelect::Flagged`.

## Numerical stability

- fp32 throughout. Spherical k-means uses fp32 cosine distance; consensus is robust to small fp32 errors.
- Reproducibility metric (Jaccard on integer gene IDs) is exact.
- k selection via finite-difference of reproducibility — small integer comparisons, no floating-point sensitivity.

## Memory layout

- Input X: device CSC (cycle 2 loader, kept in memory for all 5000 fits).
- Per-run W: `m × k × 4` bytes per fit. For m=30k, k=30: ~3.6 MB. We accumulate `n_runs * k_max` columns: 100 × 50 × 30k × 4 = 600 MB. Tile if needed.
- Per-run H: discarded after each fit (only W matters for consensus).
- Output: chosen_k consensus W (m × k_chosen) + program_usage (n × k_chosen).

## Streams

One stream, caller-provided. The 5000 fits run sequentially.

## Out-of-core

Subsampling is the OOC story — each fit operates on `subsample_frac * n_cells`. For very large n, smaller subsamples + more runs + averaging.

## Determinism

cuRAND Philox seeded per run (`run_seed = base_seed + run_idx`). NMF fit determinism is inherited from factornet.

## Correctness test spec

Test: `tests/reduce_nmf_csi_gep_correctness.cpp`.

Reference: CSI-GEP Python via subprocess.

Test cases:
1. **`CsiGep_TinySynthetic_VsPython`**: synthetic 200 × 100 with planted 5 programs. Compare top-30 gene sets per program: Jaccard ≥ 0.85.
2. **`CsiGep_GSM4037629_RealData`**: load exon_counts.1pz. Run CSI-GEP with k_range=[5,10,15,20], n_runs=10 (reduced for test speed). Confirm finite results, chosen_k in range, no NaN.
3. **`CsiGep_AutoKSelection_PicksKnownK`**: synthetic with 8 planted programs. Confirm chosen_k ∈ [6, 10] (allow ±2).
4. **`CsiGep_ReproducibilityCurve_Monotone`**: confirm reproducibility curve is non-decreasing then plateaus.
5. **`CsiGep_Determinism_BitIdentical`**: bit-identical with fixed seed.

Tolerances per design doc.

## Target performance

| Scale | Cells | Genes | k_range | n_runs | Wall (target) |
|---|---|---|---|---|---|
| tiny | 200 | 100 | [3,8] | 10 | <1s |
| 10k | 11,560 | 30k | [5,30] | 50 | <2 min |
| 100k | ~120k | 30k | [5,40] | 100 | <30 min |
| 1M | ~1M | 30k | [5,30] | 50 | <2h |

## Implementation notes

- Header path: `include/singlet-gpu/reduce/nmf/csi_gep.h` (~600 LOC).
- Python wrapper: `python/singlet_gpu/reduce/csi_gep.py` (~250 LOC) — adds `csi_gep(adata, k_range, n_runs, ...)` to the existing reduce module.
- R wrapper: `r/R/csi_gep.R` (~150 LOC) — adds `run_csi_gep(sce, k_range, n_runs, ...)`.
- C++ binding: `_core.csi_gep` in cycle 28.5 binding extension OR added to cycle 27 binding work.
- Reuses cycle 5 NMF adapter (`singlet_gpu::reduce::nmf::fit`) as the inner loop.
- Build flag: `FACTORNET_HAS_GPU=1`. cuRAND + cub + cuML (for spherical k-means consensus).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (consensus bootstrap NMF; reuses factornet::nmf::fit_gpu for inner loop)` first comment.

## Risks

1. **5000 fit calls** — factornet's GPU NMF startup overhead per call may add up. Profile early.
2. **Spherical k-means consensus** — cuML has it; if not, write a simple kernel.
3. **CSI-GEP Python install** is heavy (torchNMF + dependencies). Make the reference test optional.
4. **Reproducibility plateau detection** is a heuristic; tune the elbow gap threshold (default 5).
