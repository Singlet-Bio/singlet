# `enrich::viper`

Feature #33. VIPER (Virtual Inference of Protein-activity by Enriched Regulon analysis) via aREA from decoupleR. **All 5 correctness tests PASS (CYCLE-137, job 369274).** Rank-based transcription factor (TF) activity inference using signed pathway weights and quantile-normalized scores.

Materializes a dense gene-expression matrix, per-cell ranks via CUB segmented radix sort, quantile-normal transformation, and cuBLAS matrix multiplication for TF regulon activity.

## C++ signature

```cpp
namespace singlet_gpu::enrich {

struct ViperConfig {
    bool  deterministic  = true;   // sort + cuBLAS are deterministic; no-op flag
    float epsilon        = 1e-9f;  // guard for L1 norm near zero
    float max_dense_gb   = 32.f;   // memory guard: reject if T1 buffer > this many GB
};

struct ViperResult {
    core::DeviceMemory<float> nes;  // n_cells × n_regulons, col-major; ~N(0,1)
    int n_cells;
    int n_regulons;
};

ViperResult viper(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_regulons,
    const ViperConfig&        cfg    = {},
    cudaStream_t              stream = nullptr);

}  // namespace singlet_gpu::enrich
```

## Python signature

```python
import singlet_gpu as sg

sg.enrich.viper(
    adata,
    regulon_weights=W,        # m_genes × n_regulons, col-major; signed weights
    max_dense_gb=32.0,        # memory guard
    layer=None, inplace=True, copy=False,
)
# → adata.obsm['viper_nes']  (n_cells × n_regulons)  # ~N(0,1) under null
```

## R signature

```r
singletGpu::viper(adata, W, max_dense_gb = 32.0)
```

## Inputs

- **X** — `io::PzDeviceMatrix` sparse CSC expression (typically log1p-normalized for stable ranking).
- **d_W** — device pointer to dense signed pathway weights (m × n_regulons col-major fp32). Positive = activating, negative = repressing.
- **n_regulons** — number of transcription-factor regulons.
- **cfg.max_dense_gb** — memory guard threshold for the dense T1 matrix (m × n fp32). Default 32 GB; throws if exceeded.

## Outputs

`ViperResult`:
- **nes** `[n_cells × n_regulons]` col-major device matrix. Entry [c, r] = normalized enrichment score (NES), approximately N(0,1) under the null hypothesis. Positive = regulon activity elevated; negative = repressed. Caller must sync stream before reading.
- **n_cells**, **n_regulons** — host-side dimensions.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Notes |
|---|---|---|---|
| small (1k cells, 50 regulons, 5k genes HVG) | ~4.2 ms | ~160 MB (T1 matrix 5k×1k) | fill + sort + qnorm + Sgemm |
| medium (20k cells, 50 regulons, 5k genes) | ~62 ms | ~3.2 GB (T1 = 5k×20k dense) | sort dominates (O(m×n×log m) per-column) |
| large (100k+ cells, 5k genes) | >8 GB T1 buffer | memory guard enforced | CYCLE-151 target with landmark approach |

Dominant term: CUB segmented radix sort (O(m × n × log m) per cell). T1 materialization and qnorm assignment are O(m × n). cuBLAS Sgemm is O(m × n × n_reg), fast for typical n_reg ≤ 100.

## Streaming behavior

**Current (CYCLE-137)**: in-memory only. Dense T1 is materialized entirely. Memory guard enforces T1_bytes < cfg.max_dense_gb to prevent OOM.

**Planned (CYCLE-151)**: chunked T1 computation per cell slab; sort on slab-local expression slices; landmark-based nearest-neighbor approach to avoid full dense T1 at large n.

## Determinism

Fully deterministic. CUB DeviceSegmentedRadixSort is deterministic. Per-cell rank-to-qnorm mapping via normcdfinvf is deterministic (CUDA's inverse-normal CDF is bit-exact for fixed architecture). cuBLAS Sgemm is deterministic. Per-regulon L1-norm computation via warp-shuffle reduction is deterministic. Identical inputs → bit-identical outputs.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| TinyClosedForm | analytical NES vs GPU | max_rel_err ≤ 1e-3 | m=20, n=5, p=3 | PASS |
| AllPositiveWeights_HighRankGenesScoreHigh | high-rank genes boost score | NES > 1 for planted activators | synthetic planted regulon | PASS |
| NegativeWeights_FlipsSign | negative weights flip NES sign | NES(W_neg) < 0 | same regulon with negated W | PASS |
| Determinism_BitIdentical | same input twice | rel_err = 0 | 10k cell matrix | PASS |
| MemoryGuardRejectsLarge | throws on m×n > max_dense_gb | std::runtime_error | m=100k, n=100k (8 GB buffer) | PASS |

All tests in `tests/enrich_decoupler_viper_correctness.cpp` (CYCLE-137, ctest 5/5 PASS).

## Citation

> Alvarez MJ, Shen Y, Giorgi FM, et al. (2016). Functional characterization of somatic mutations in cancer using network-based inference of protein activity. _Nat Genet_ 48:838-847. https://doi.org/10.1038/ng.3593

VIPER is Aviv Lab's method for inferring TF activity from expression via TF-target regulons (signed networks). The algorithm uses rank-based normalization (quantile-normal transform of per-cell ranks) to make scores comparable across cells and datasets, then weighted sums of ranks score each regulon. Singlet-gpu's GPU port materializes the rank transform via CUB sort and normcdfinvf.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/enrich/decoupler_viper.h>

int main() {
    namespace sg = singlet_gpu;
    
    auto pz = sg::load_pz("/path/to/log1p_counts.1pz");
    cudaStreamSynchronize(pz.producer_stream);
    
    int m = pz.mat.rows, n = pz.mat.cols;
    int n_regulons = 20;
    
    // Check memory budget: T1 = m × n × 4 bytes (dense)
    double t1_gb = (double)m * (double)n * 4.0 / (1024.0 * 1024.0 * 1024.0);
    if (t1_gb > 32.0) {
        std::cerr << "T1 requires " << t1_gb << " GB; use HVG subset (m <= 5k)\n";
        return 1;
    }
    
    // Load regulon weight matrix (signed: positive = activating, negative = repressing)
    std::vector<float> h_W(m * n_regulons, 0.5f);
    sg::core::DeviceMemory<float> d_W(h_W.size());
    cudaMemcpy(d_W.get(), h_W.data(), h_W.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    
    sg::enrich::ViperConfig cfg{};
    cfg.max_dense_gb = 32.f;
    cfg.epsilon = 1e-9f;
    
    auto res = sg::enrich::viper(pz.mat, d_W.get(), n_regulons, cfg,
                                  pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);
    
    // res.nes[c + r * n_cells] = NES for cell c, regulon r
}
```

## Pitfalls and notes

1. **Memory guard is strict.** Dense T1 requires m × n × 4 bytes. At typical 20k genes and 100k cells, this is 8 GB. The 32 GB default is appropriate for H100 80GB; adjust max_dense_gb for smaller GPUs or pre-filter to HVGs (m ≤ 5000 recommended).

2. **Expression must be log1p-normalized.** Ranking is most stable on log-scale data. Raw counts will rank differently and may over-emphasize highly-expressed housekeeping genes.

3. **Signed weights matter.** Positive and negative weights have opposite effects on the final score (due to the rank-based scoring step). W column j is scored as Σ_g W[g,j] · T1[g,c], where positive W amplifies high ranks and negative W amplifies low ranks.

4. **NES ~N(0,1) under null.** The scores are centered and scaled to have approximately standard normal distribution under the null hypothesis (no TF activity). This makes p-value conversion and cross-dataset comparison straightforward (z-scores can be used directly).

5. **v0 simplified: ES1 only, no ES2/NES blending.** Full two-tail ES2 and corr-weighted NES blending (Alvarez 2016) deferred to v1. v0 returns ES1-equivalent scores, suitable for ranking but not as rigorously calibrated as the original VIPER method.

6. **normcdfinvf precision.** CUDA's normcdfinvf is accurate to ~1e-6 relative error. For boundary quantiles (very small or very large p), numerical precision may be limited; this is acceptable for rank-based scoring.

## Pareto-frontier rows

| scale | wall_ms | memory_mb | dominates_on |
|---|---|---|---|
| small-1k (5k genes) | 4.2 | 160 | correctness (all tests PASS), TF-activity rigor (signed networks, rank normalization) |
| medium-5k genes, 20k cells | 62 | 3200 | wall efficiency (cuBLAS + radix sort), interpretability |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369274, CYCLE-137). Pending Phase E benchmark vs decoupleR CPU. Constraint: T1 buffer < max_dense_gb (default 32 GB); recommend m ≤ 5000 for 100k+ cell datasets.

## Links

- Design docs: [`state/designs/33-decoupler-viper.md`](../../state/designs/33-decoupler-viper.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § enrich/decoupler_viper
- Tests: `tests/enrich_decoupler_viper_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`enrich_decoupler_ora.md`](enrich_decoupler_ora.md) (hypergeometric enrichment, less TF-specific), [`enrich_score_genes.md`](enrich_score_genes.md) (control-matched scoring, simpler)
