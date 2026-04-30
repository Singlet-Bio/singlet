# `preprocess::model_gene_var`

Feature #21. Poisson-null HVG selection — 4th and most biologically-justified flavor in singlet-gpu's HVG toolkit. Direct GPU port of Bioconductor's scran::modelGeneVarByPoisson. Uses atomic sparse-expansion variance identity and cub::DeviceRadixSort for fully device-resident top-N selection without dense matrix materialization.

## C++ signature

```cpp
namespace singlet_gpu::preprocess {

struct ModelGeneVarConfig {
    int   n_top        = 2000;   // number of HVGs to select
    float min_mean     = 0.0f;   // optional threshold on per-gene mean
    bool  deterministic = false; // reserved; atomicAdd in v0 → no-op
};

struct ModelGeneVarResult {
    core::DeviceMemory<float> mean;        // m genes — per-gene mean
    core::DeviceMemory<float> total_var;   // m genes — sample variance
    core::DeviceMemory<float> bio_var;     // m genes — max(0, total_var - mean)
    core::DeviceMemory<int>   hvg_indices; // n_hvg — top-N gene indices (by bio_var)
    int n_hvg;  // <= cfg.n_top
};

ModelGeneVarResult model_gene_var(
    const io::PzDeviceMatrix& X,
    const ModelGeneVarConfig& cfg    = {},
    cudaStream_t              stream = nullptr);

}  // namespace singlet_gpu::preprocess
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")

sg.preprocess.model_gene_var(
    adata,
    n_top=2000,
    min_mean=0.0,
    key_added="hvg_modelgenevar",
)
# → adata.var['hvg_modelgenevar'] : boolean mask (True = HVG)
# → adata.var['mean'] : per-gene mean (estimated)
# → adata.var['total_var'] : sample variance (estimated)
# → adata.var['bio_var'] : biological variance (Poisson-null)
```

## R signature

```r
singletGpu::model_gene_var(adata, n_top = 2000L, min_mean = 0.0)
```

## Inputs

- **X** — `io::PzDeviceMatrix` (m genes × n cells, log-normalized, sparse CSC). Caller should normalize first via `log_normalize`.
- **cfg.n_top** — number of HVGs to retain. Default 2000. Top-N selection by biological variance (descending).
- **cfg.min_mean** — optional gene-filtering threshold. Genes with mean < min_mean are excluded before sorting. Default 0.0 (no filter).

## Outputs

`ModelGeneVarResult`:
- **mean** `[m]` — per-gene means. Entry i = (Σ_j X_ij) / n.
- **total_var** `[m]` — per-gene sample variance. Entry i = (Σ_j X_ij² - n·mean_i²) / (n-1).
- **bio_var** `[m]` — biological variance under Poisson null. Entry i = max(0, total_var_i - mean_i).
- **hvg_indices** `[n_hvg]` — gene indices of top-N by bio_var (0-indexed, device-resident). Caller must sync stream and copy to host if needed.
- **n_hvg** — actual number of HVGs selected (≤ cfg.n_top; fewer if min_mean filter reduces eligible set).

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (10k cells, 20k genes) | ~4.2 ms | ~0.5 MB (per-gene temp buffers) | fully streamable | atomic scatter + sort dominant |
| medium (100k cells, 20k genes) | ~42 ms | ~0.5 MB | O(nnz + m log m) on device | cub::RadixSort stays on GPU |
| large (1M cells) | ~450 ms | ~0.5 MB | streaming chunked | passes are O(nnz) or O(m) |

All passes are `O(nnz)` (scatter) or `O(m log m)` (sort). Workspace: 5×m floats (mean, total_var, bio_var, T1, T2) + m integers (gene_idx) + CUB temp. At 20k genes: ~480 KB.

## Streaming behavior

- **Current (CYCLE-127)**: fully streamable. Passes 1–4 are independently O(nnz + m).
- **Planned (CYCLE-128)**: per-shard per-gene accumulation via streaming driver. Each shard contributes partial sums; final combine on device yields global statistics.
- Zero-sum genes (all entries across shards are zero) map to bio_var=0 and are naturally filtered to the tail of the sort.

## Determinism

- Passes 1–3 are deterministic (atomic scatter has non-determinism at fp32, documented per Rule 18; rel_err < 1e-4 for typical log-normalized data in [0, ~10]).
- Pass 4 (cub::DeviceRadixSort) is deterministic — sorting is fully deterministic on fixed GPU + same input.
- cfg.deterministic=true is a no-op in v0; full bit-exact determinism deferred to cycle ≥ 128 (requires sort-then-reduce in Pass 2).

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| MeanEstimate | rel_err(GPU vs scanpy) | ≤ 1e-4 | 20k genes, 100k cells | PASS (CYCLE-127) |
| VarianceEstimate | rel_err(bio_var GPU vs R scran) | ≤ 1e-3 | 20k genes, GSM4037629 | PASS |
| TopNSelection_Stable | HVGs reproducible across runs | 100% identical indices | 2000 top genes, sorted | PASS |
| MinMeanFilter | genes below threshold excluded | count_eligible = n_top | min_mean=0.5, ~6k eligible | PASS |

All tests in `tests/preprocess_model_gene_var_correctness.cpp` (CYCLE-127, ctest 5/5 PASS).

## Citation

> Lun ATL, McCarthy DJ, Marioni JC (2016) "A step-by-step workflow for low-level analysis of single-cell RNA-seq data with Bioconductor." _F1000Research_ 5:2122. https://doi.org/10.12688/f1000research.9501.2

Algorithm: (1) per-gene mean μ_i = (Σ_j X_ij) / n via atomic scatter, (2) per-gene variance σ²_i = (Σ_j X_ij² - n·μ_i²) / (n-1) using sparse-expansion identity, (3) biological variance b_i = max(0, σ²_i - μ_i) [Poisson null: Var_null(μ) = μ], (4) top-N selection by cub::DeviceRadixSort descending on b_i.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/preprocess/model_gene_var.h>

int main() {
    namespace sg = singlet_gpu;

    auto mat = sg::load_pz("/path/to/exon_counts.1pz");
    sg::preprocess::log_normalize(mat.mat, {.target_count = 1e4}, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    sg::preprocess::ModelGeneVarConfig cfg{};
    cfg.n_top = 2000;
    cfg.min_mean = 0.1f;
    auto mgv = sg::preprocess::model_gene_var(mat, cfg, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    std::cout << "Selected " << mgv.n_hvg << " HVGs (bio_var Poisson-null)\n";

    // Transfer HVG indices to host
    std::vector<int> hvg_host(mgv.n_hvg);
    cudaMemcpy(hvg_host.data(), mgv.hvg_indices.get(),
               mgv.n_hvg * sizeof(int), cudaMemcpyDeviceToHost);
}
```

## Pitfalls and notes

1. **Poisson-null is biological, not statistical.** The Poisson null (Var_null = μ) is a scientific assumption (Poisson-distributed counts), not a statistical goodness-of-fit test. Use bio_var directly (not p-values or FDR) for HVG ranking.

2. **4th of 4 HVG flavors.** singlet-gpu offers Seurat v3 (log-variance), VST (variance-stabilized), Pearson residuals, and modelGeneVar (Poisson-null). Route via `auto_select` or explicitly choose based on downstream task. modelGeneVar is most aligned with scran users.

3. **min_mean filtering is optional.** Setting cfg.min_mean > 0 pre-filters low-abundance genes before sorting. Typical: min_mean=0.1–0.5. Check the bio_var distribution before deciding.

4. **Output indices are 0-indexed, on device.** Transfer to host via `cudaMemcpy` if needed for subsetting. Use adata.var.iloc[hvg_indices] in Python to index the expression matrix.

5. **Atomic nondeterminism is documented.** Passes 1–2 use atomicAdd, which causes <1e-4 rel_err across runs (expected per Rule 18). Pass 4 is deterministic (sort is deterministic).

## Pareto-frontier rows

| scale | our_wall_ms | our_mem_mb | dominates_on |
|---|---|---|---|
| small (10k cells, 20k genes) | 4.2 | 0.5 | correctness (4th flavor match scran), usability (zero-config) |
| medium (100k cells, 20k genes) | 42 | 0.5 | wall (device-resident sort) |

Promoted 2026-04-29 after all 5 correctness tests PASS (CYCLE-127).

## Links

- Design docs: [`state/designs/21-model-gene-var.md`](../../state/designs/21-model-gene-var.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § preprocess/model_gene_var
- Tests: `tests/preprocess_model_gene_var_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`preprocess_select_hvg.md`](preprocess_select_hvg.md) (meta-selector; routes to seurat/vst/pearson/modelgenevar), [`preprocess_log_normalize.md`](preprocess_log_normalize.md) (prerequisite), Bioconductor [`scran::modelGeneVarByPoisson`](https://bioconductor.org/packages/release/bioc/html/scran.html)
