---
feature: gsea
roadmap_id: 11
module: include/singlet-gpu/gsea/fgsea.h, include/singlet-gpu/gsea/aucell.h
status: design
tolerance: ES rank Spearman ρ ≥ 0.99 vs fgsea R; p-value rank Spearman ρ ≥ 0.95
target_perf: 2860 MSigDB pathways × 22k genes preranked GSEA in ≤2s on A100 (fgsea CPU is ~50s; rsfgsea Rust is ~5s); per-cell AUCell on 1M cells × 100 sets in ≤1s
ooc_plan: GSEA per-cluster per-pathway is independent; trivially parallelized across pathways. AUCell per-cell streams via PzDataLoader chunks
---

## Why this matters

This is one of the **few singlet-gpu features where we are not wrapping an existing library** — per lit-scout, no native GPU GSEA exists. rapids-singlecell explicitly does NOT implement pathway enrichment; users currently fall back to fgsea CPU (~1 min for 2860 pathways) or rsfgsea Rust (~5s). Our target is ≤2s on a single A100, which would be 25× faster than rsfgsea and the first true GPU implementation in the field.

## Algorithm

`gsea/fgsea.h` implements **preranked GSEA** with the adaptive multilevel permutation scheme from fgsea (Korotkevich et al. 2019), translated to GPU. `gsea/aucell.h` implements **AUCell-style per-cell scoring** with the histogram-binned ranking trick from cycle 11.

### Preranked GSEA (`gsea/fgsea.h`)

Input: a per-gene preranked statistic vector (e.g., the Wilcoxon z-scores from cycle 11, or log2 fold changes), plus a `GeneSetDB` (the cycle 12 type, reused).

Output: per-pathway enrichment score (ES), normalized ES (NES), p-value, FDR-adjusted q-value.

Algorithm:

1. **Sort genes by statistic** (cub::DeviceRadixSort). Output: sorted_indices[m], sorted_stats[m].
2. **Per-pathway ES computation** (one warp per pathway):
   - Walk the sorted gene list. Maintain a running sum that increments by `|stat[g]| / Σ_member |stat|` if `g ∈ pathway`, decrements by `1 / (m - n_member)` otherwise.
   - The ES is the maximum absolute value of the running sum.
   - Cost: O(m) per pathway. Per-pathway in parallel.
3. **Permutation null** — adaptive multilevel:
   - Start with N=1000 random permutations of the gene list (block-level cuRAND).
   - For each permutation, recompute the ES per pathway.
   - Adaptive stopping per pathway: if the p-value can be bounded above 0.05 with high confidence after N permutations, stop. Otherwise increase to 10000.
   - Use cuRAND `XORWOW` or `Philox4x32` for parallel deterministic random.
4. **Normalize ES** to NES: `NES = ES / mean(|ES_perm|)`.
5. **p-value** from the empirical permutation distribution.
6. **BH adjustment** for q-value (cycle 11 pattern reused).

Adaptive scheme details:
- Each pathway gets its own permutation budget. Cheap pathways stop at 1000; expensive (extreme) pathways go to 10000.
- The fgsea trick: pre-compute the cumulative sum of `|stat|` and use binary search per permutation to find the ES — `O(m log m)` per perm instead of `O(m^2)`.
- On GPU: each warp handles one (perm, pathway) pair. Many warps run in parallel.

### AUCell (`gsea/aucell.h`)

Per-cell pathway scoring via the AUC under the recovery curve.

Algorithm:
1. **Per-cell rank genes**: use the cycle 11 histogram-binned ranking trick (B=4096) to get approximate ranks per gene per cell. The top-K genes per cell (K=500 typical) are the only ones that contribute.
2. **Per-cell per-pathway recovery curve**: walk the sorted top-K, count cumulative member-gene hits. Compute AUC = `Σ_k member_in_top_k / K`.
3. Output: `scores[n_cells × n_pathways]`.

The histogram-binned ranking is what makes this scale — exact AUCell needs the full per-cell rank vector (m × n cells = OOM at 1M × 30k). Histogram-binned approximation is O(B) per cell per pathway.

## Numerical stability

- fp32 throughout. ES, NES, p-values all in fp32.
- BH adjustment stable to floating-point; cycle 11 pattern.
- cuRAND fp32 uniform → permutation indices via Fisher-Yates on device.
- Permutation determinism: `cfg.seed` forwarded to cuRAND. Same seed → same permutations → same p-values.

## Memory layout

- Input: `core::DeviceMemory<float> stats(m)` + `GeneSetDB`.
- GSEA workspace:
  - sorted_stats[m] + sorted_indices[m] = `8m` bytes.
  - cumsum_abs[m] = `4m` bytes.
  - For each (perm, pathway) tile: temporary running-sum scratch in shared memory.
  - Output: `n_pathways × 4` bytes for ES, NES, p, q. Tiny.
- AUCell workspace:
  - histograms[n_cells × B] = `4 * n_cells * B` bytes. For 1M cells × 4096 bins: 16 GB. **OOM** — must tile by cells.
  - Tile by 64k cells: 1 GB per tile. Acceptable.
  - Output: `4 * n_cells * n_pathways` bytes.

## Streams

One stream, caller-provided. The permutation kernel is the dominant cost; many cuRAND seeds run in parallel.

## Out-of-core

GSEA: per-pathway parallel, no streaming needed (the input is just a stats vector of size m).

AUCell: per-cell streaming via `PzDataLoader`. Per chunk: compute per-cell histograms + AUC scores, write to host-mmap output. Final concat.

## Determinism

- GSEA: deterministic given fixed seed (cuRAND XORWOW with seed).
- AUCell: deterministic by construction (histogram-binned ranking is deterministic).

## Correctness test spec

Tests:
- `tests/gsea_fgsea_correctness.cpp`
- `tests/gsea_aucell_correctness.cpp`

References:
- fgsea R package via subprocess (`Rscript -e 'library(fgsea); ...'`).
- AUCell R package or DecoupleR Python (`dc.run_aucell`).

Test cases (fgsea):
1. **`Fgsea_TinyPlanted_VsR`**: synthetic stats vector (m=200) with planted up- and down-regulated gene sets. Run our `fgsea` and R fgsea on the same input. Compare ES per pathway: rel_err ≤ 1e-4. Spearman ρ on ranked p-values ≥ 0.95.
2. **`Fgsea_RealMSigDB_TopHallmarks`**: real preranked stats from cycle 11 Wilcoxon output (mocked synthetically), real MSigDB Hallmarks gene sets (50 sets, ~200 genes each). Compare top-10 enriched pathways by NES — Jaccard ≥ 0.90 vs R fgsea.
3. **`Fgsea_AdaptivePermutation_StopsEarly`**: confirm that pathways with high effect sizes converge in <1000 perms while extreme pathways go to 10000.
4. **`Fgsea_Determinism_BitIdentical`**: run twice with same seed.
5. **`Fgsea_BHCorrection_NotAllZero`**: confirm q-values are not trivially zero.

Test cases (AUCell):
1. **`AUCell_TinyPlanted_VsR`**: tiny cell × gene matrix with planted high-expression cells per gene set. Compare to R AUCell.
2. **`AUCell_RealData_RanksConsistent`**: confirm cells with high marker expression score higher.
3. **`AUCell_HistogramApprox_Error`**: compare histogram-binned ranks (B=4096) to exact ranks on a small dataset; confirm approximation error <1%.
4. **`AUCell_Determinism_BitIdentical`**.

Tolerances:
- ES rel_err ≤ 1e-4; p-value rank Spearman ρ ≥ 0.95
- Top-N enriched pathway Jaccard ≥ 0.90
- AUCell vs exact: rel_err ≤ 1%
- Determinism: bit-identical

## Target performance

| Scale | Pathways | Genes | Method | Target wall | SOTA |
|---|---|---|---|---|---|
| 50 (Hallmarks) | 50 | 22k | fgsea | <100ms | fgsea R ~5s |
| 2860 (MSigDB) | 2860 | 22k | fgsea | <2s | fgsea R ~50s; rsfgsea ~5s |
| 100k cells × 100 sets | 100 | 30k | aucell | <100ms | DecoupleR CPU ~30s |
| 1M cells × 100 sets | 100 | 30k | aucell | <1s | OOM on most CPU implementations |

**This is the first true GPU GSEA implementation in the field.** Document loudly.

## Implementation notes

- Headers: `gsea/fgsea.h` (~500 LOC) + `gsea/aucell.h` (~300 LOC) + `gsea/types.h` (~50 LOC).
- API:
  ```cpp
  namespace singlet_gpu::gsea {
      struct FgseaConfig {
          int min_perm = 1000;
          int max_perm = 10000;
          float adaptive_target_pvalue = 0.05f;
          int min_set_size = 15;
          int max_set_size = 500;
          uint64_t seed = 0;
      };
      struct PathwayResult {
          std::string name;
          float es;
          float nes;
          float p_value;
          float q_value;
          int n_member_genes;
          int n_perms_used;
      };
      struct FgseaResult {
          std::vector<PathwayResult> pathways;
      };
      FgseaResult fgsea(
          const singlet_gpu::core::DeviceMemory<float>& stats,
          const singlet_gpu::anno::GeneSetDB& gene_sets,
          const FgseaConfig& cfg = {},
          cudaStream_t stream = nullptr);

      struct AUCellConfig {
          int top_k_genes = 500;     // top-K per cell to consider
          int n_bins = 4096;
          int cell_tile = 65536;
          uint64_t seed = 0;
      };
      struct AUCellResult {
          singlet_gpu::core::DeviceMemory<float> scores;  // n_pathways × n_cells
          int n_pathways;
          int n_cells;
      };
      AUCellResult aucell(
          const singlet_gpu::core::DeviceCSC& mat,
          const singlet_gpu::anno::GeneSetDB& gene_sets,
          const AUCellConfig& cfg = {},
          cudaStream_t stream = nullptr);
  }
  ```
- Build flag: `FACTORNET_HAS_GPU=1`. Adds cuRAND dependency (`CUDA::curand`).
- Dependencies: cycle 1 (core), cycle 11 (DE for the input stats vector), cycle 12 (`anno::GeneSetDB`).

## Risks

1. **Adaptive permutation logic complexity**: getting the early-stopping right matters for both correctness and runtime. Reference fgsea's R implementation closely.
2. **AUCell histogram approximation**: 1% error is tolerable for ranking but might shuffle borderline cells across thresholds. Document.
3. **Permutation seed reproducibility**: cuRAND state is per-thread; the same global seed must produce the same permutations across architectures. Use Philox4x32 (counter-based) for cross-architecture determinism if XORWOW is unstable.
4. **Memory tiling for AUCell**: 1M cells × 4096 bins = 16 GB. Tiling by 64k cells = 1 GB. Auto-tune.
