---
feature: annotation
roadmap_id: 10
module: include/singlet-gpu/anno/marker_score.h, include/singlet-gpu/anno/reference_map.h
status: design
tolerance: marker_score: rank Spearman ρ ≥ 0.95 vs DecoupleR Python on the same input; reference_map: classification accuracy ≥ 0.90 vs CellTypist sklearn on a held-out portion of a labeled reference
target_perf: 1M cells × 100 gene sets marker_score ≤500ms on A100; 1M cells × 50-class CellTypist projection ≤200ms
ooc_plan: marker_score streams via PzDataLoader chunks (per-cell scores accumulate independently); reference_map fits on a precomputed embedding (cycle 4 PCA) so it's bounded by `n × n_classes × n_pcs`
---

## Algorithm

`anno/marker_score.h` and `anno/reference_map.h` cover the two cell-type annotation paradigms per lit-scout: marker-based unsupervised scoring (DecoupleR family) and reference-based supervised projection (CellTypist family).

### `anno/marker_score.h`

Implements four scoring methods, all native singlet-gpu (no external GPL conflict):

1. **`MarkerMethod::Mlm`** (multivariate linear model — DecoupleR's `mlm`): solve a least-squares problem per cell, mapping its expression profile onto the gene-set design matrix. Per lit-scout: 6.9× speedup vs CPU. Returns activity scores per (cell, gene set).
2. **`MarkerMethod::Ulm`** (univariate linear model — DecoupleR's `ulm`): per gene set, fit a univariate regression of expression vs the binary membership vector. Slower per cell but more robust to gene-set overlap.
3. **`MarkerMethod::Wsum`** (weighted sum — DecoupleR's `wsum`): simple weighted average of expression for member genes. Fastest. 37× speedup per lit-scout.
4. **`MarkerMethod::UCell`** (UCell rank-based scoring): rank-based AUC estimator per cell. Memory-efficient alternative to AUCell (which lit-scout flagged as OOM at 100k cells). Uses approximate ranking via the cycle 11 histogram-binned trick.

Input: `core::DeviceCSC` (genes × cells, log-normalized) + a `GeneSetDB` struct (per-set: name + member gene indices + optional weights).

Output: `core::DeviceMemory<float> scores(n_cells * n_gene_sets)` (column-major, gene_sets × cells).

### Mlm kernel (most complex of the four)

For each cell `j`, solve: `min_a ||X_j − G·a||²` where `G[m × n_sets]` is the gene-set design matrix (1 if gene g ∈ set s, 0 otherwise) and `a` is the per-set activity score. Closed form: `a = (G^T G)^{-1} G^T X_j`.

- `(G^T G)^{-1}` is `n_sets × n_sets`, computed ONCE on host (or via cuSOLVER fp64 since gene sets typically n_sets < 200).
- Per cell: `G^T X_j` is a sparse-dense product (`n_sets`-vector). For each cell column in CSC, walk nonzeros, accumulate into the sets each gene belongs to.
- Multiply by the cached inverse → scores.
- Total: O(nnz · max_sets_per_gene) per cell, parallelized across cells.

### Wsum / Ulm / UCell

Simpler reductions; one block per (cell, gene_set) or one warp per cell + gene set.

### `anno/reference_map.h`

CellTypist-style: load pre-trained logistic regression coefficients from a labeled reference atlas, project query cells, return per-cell class probabilities + argmax label.

Input:
- A query embedding (`core::DeviceDense`, `n_cells × n_pcs`, from cycle 4 PCA).
- A pre-trained model: `class_weights[n_classes × n_pcs]`, `class_intercepts[n_classes]`, `class_names[n_classes]`. Loaded from a `.npz` or HDF5.

Output: `core::DeviceMemory<float> probabilities(n_cells * n_classes)`, `core::DeviceMemory<int> labels(n_cells)` (argmax).

### Logistic regression projection kernel

```
logits[cell, class] = embedding[cell, :] · class_weights[class, :] + class_intercepts[class]
softmax(logits) over classes
labels[cell] = argmax(probabilities[cell, :])
```

Single GEMM via cuBLAS (`embedding × class_weights^T`), then per-row softmax kernel, then per-row argmax via `cub::DeviceReduce::ArgMax`.

For cluster-level annotation: aggregate per-cluster mean probabilities, then argmax per cluster. Wraps the per-cell call.

### Pre-trained models

Initial supported models (loaded from `.npz`):
- `Immune_All_Low.pkl` (CellTypist, 99 classes)
- `Immune_All_High.pkl` (32 classes)
- `Healthy_Mouse_Brain.pkl` (47 classes)
- User-supplied via `load_celltypist_model(path)`.

The CellTypist Python `.pkl` files are pickled sklearn LogisticRegression objects. We provide a Python script (`tools/extract_celltypist_to_npz.py`) that converts them to `.npz` for loading. The C++ side never reads pickle.

## Numerical stability

- fp32 throughout marker scoring. fp64 only for `(G^T G)^{-1}` (small, ≤200×200, cuSOLVER fp64 affordable).
- Softmax uses the max-subtraction trick: `softmax(z) = exp(z - max(z)) / Σ exp(z - max(z))`. fp32 stable.
- Cell with all zeros: marker score is 0 (or NaN-handled to 0).
- Class with zero training support: weight row is all zeros; its logit is `intercept` only.

## Memory layout

- Marker scoring workspace: `n_cells × n_sets × 4` bytes for the output. For 1M cells × 100 sets: 400 MB. Manageable.
- Reference mapping: `n_cells × n_classes × 4` bytes for probabilities. For 1M × 50: 200 MB.
- Pre-trained model: `n_classes × n_pcs × 4` bytes. For 50 × 50: 10 KB. Trivial.

## Streams

One stream, caller-provided. Marker scoring chains GEMM + softmax + argmax on the same stream.

## Out-of-core

Marker scoring is per-cell — trivially streams via `PzDataLoader`. Per chunk: compute scores, write to a host-mmap output. Final step concatenates.

Reference mapping requires a pre-computed embedding (cycle 4 PCA), which is itself bounded by the in-memory PCA fallback in cycle 7. For >2M cells, the streaming PCA workaround applies (subsample → fit → project remaining cells) — that's the `reference_map` cycle 7 dependency.

## Determinism

All kernels are deterministic. cub::DeviceReduce::ArgMax has stable tie-breaking (returns the lowest index on a tie).

## Correctness test spec

Tests:
- `tests/anno_marker_score_correctness.cpp`
- `tests/anno_reference_map_correctness.cpp`

Reference: DecoupleR Python (`pip install decoupler-py`) and CellTypist Python (`pip install celltypist`) in subprocesses.

Test cases:
1. **`MarkerScore_Mlm_VsDecoupleR`**: tiny synthetic + tiny gene-set DB. Run our `mlm_score` and DecoupleR's `dc.run_mlm(...)`. Compare scores element-wise: rel_err ≤ 1e-4 + Spearman ρ ≥ 0.95 per cell.
2. **`MarkerScore_Wsum_VsDecoupleR`**: same shape, `wsum`.
3. **`MarkerScore_Ulm_VsDecoupleR`**: same shape, `ulm`.
4. **`MarkerScore_UCell_VsRPackage`**: tiny synthetic. Compare our UCell scores to the R UCell package via Rscript subprocess. Spearman ρ ≥ 0.90.
5. **`ReferenceMap_CellTypistImmuneLow_AccuracyOnHeldout`**: load a small labeled reference (e.g., 200 PBMC cells with known immune labels), train a CellTypist model in Python (subprocess), extract to `.npz`, load via `load_celltypist_model`, project, compare to ground truth labels. Accuracy ≥ 0.90.
6. **`ReferenceMap_ClusterLevelAggregation`**: confirm the per-cluster aggregation produces sensible labels (e.g., majority-vote agreement with the per-cell labels).
7. **`Determinism_BitIdentical`**: run twice, bit-identical.

Tolerances:
- Marker score Spearman ρ ≥ 0.95
- Reference map accuracy ≥ 0.90
- Determinism: bit-identical

## Target performance

| Scale | Cells | Method | Target wall | SOTA |
|---|---|---|---|---|
| 10k | 11,560 | wsum (100 sets) | <5ms | ~30ms (DecoupleR CPU) |
| 100k | ~120k | mlm (100 sets) | <50ms | ~500ms |
| 1M | ~1M | mlm (100 sets) | <500ms | ~5s (rapids-singlecell GPU) |
| 1M | ~1M | celltypist project (50 classes, 50 PCs) | <200ms | n/a |

## Implementation notes

- Headers: `anno/marker_score.h` (~400 LOC) + `anno/reference_map.h` (~200 LOC) + `anno/types.h` (~50 LOC) for shared structs.
- API:
  ```cpp
  namespace singlet_gpu::anno {
      enum class MarkerMethod { Mlm, Ulm, Wsum, UCell };
      struct GeneSetDB {
          std::vector<std::string> set_names;
          std::vector<std::vector<int>> member_gene_indices;  // host-side
          std::vector<std::vector<float>> weights;             // optional
      };
      struct MarkerScoreConfig {
          MarkerMethod method = MarkerMethod::Mlm;
          int min_n_genes_per_set = 5;
      };
      struct MarkerScoreResult {
          singlet_gpu::core::DeviceMemory<float> scores;  // n_sets × n_cells
          int n_sets;
          int n_cells;
      };
      MarkerScoreResult marker_score(
          const singlet_gpu::core::DeviceCSC& mat,
          const GeneSetDB& gene_sets,
          const MarkerScoreConfig& cfg = {},
          cudaStream_t stream = nullptr);

      struct CelltypistModel {
          singlet_gpu::core::DeviceMemory<float> weights;     // n_classes × n_pcs
          singlet_gpu::core::DeviceMemory<float> intercepts;  // n_classes
          std::vector<std::string> class_names;
          int n_classes;
          int n_pcs;
      };
      CelltypistModel load_celltypist_model(const std::string& npz_path);
      struct RefMapResult {
          singlet_gpu::core::DeviceMemory<float> probabilities;  // n_classes × n_cells
          singlet_gpu::core::DeviceMemory<int>   labels;          // n_cells
      };
      RefMapResult project_to_reference(
          const singlet_gpu::core::DeviceDense& embedding,
          const CelltypistModel& model,
          cudaStream_t stream = nullptr);
  }
  ```
- `tools/extract_celltypist_to_npz.py` — Python helper to convert CellTypist `.pkl` to `.npz`.
- Build flag: `FACTORNET_HAS_GPU=1`.
- Dependencies: cycle 1 (core), cycle 4 (PCA for the embedding input to reference_map).

## Risks

1. **Mlm `(G^T G)^{-1}` ill-conditioned** — overlapping gene sets cause near-singular Gram. Add a Tikhonov regularizer `(G^T G + λI)^{-1}` with `λ = 1e-6 * trace`.
2. **Pre-trained model loading** — `.npz` format is well-defined but our C++ NPZ reader is non-trivial. Reuse the cycle 9/10 NPY reader pattern; .npz is a zip of .npy files.
3. **CellTypist is not GPU**: our `project_to_reference` is just a GEMM + softmax. The training of the model is still in Python; we only do inference.
4. **`UCell` ranking** — using the cycle 11 histogram-binned approach for ranking introduces small approximation error. Document.
