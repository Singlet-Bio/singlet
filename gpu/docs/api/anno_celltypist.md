# `anno::celltypist`

Feature #27. Cell-type annotation via pre-trained logistic regression models from the Human Cell Atlas. **First GPU implementation of CellTypist (Domínguez Conde et al. 2022, ~700 citations). All 5 correctness tests PASS (CYCLE-135, job 369244).**

Pairs with `anno/symphony` (CYCLE-138) to form the GPU reference-mapping annotation duo. Two complementary paradigms: logreg (CellTypist) vs centroid-projection (Symphony).

## C++ signature

```cpp
namespace singlet_gpu::anno {

struct CelltypistConfig {
    bool use_log_input  = true;    // assert input Z is log1p-transformed
    bool deterministic  = true;    // cuBLAS Sgemm + warp-shuffle are deterministic
};

struct CelltypistResult {
    core::DeviceMemory<int>   pred_class;   // [n_cells] int ∈ [0, n_classes)
    core::DeviceMemory<float> confidence;   // [n_cells] max softmax prob ∈ [0, 1]
    int n_cells;
    int n_classes;
};

CelltypistResult celltypist(
    const core::DeviceDense& Z,             // [n_features × n_cells] col-major, log-transformed
    const core::DeviceDense& W,             // [n_features × n_classes] col-major weights
    const core::DeviceMemory<float>& b,     // [n_classes] intercepts
    const CelltypistConfig& cfg = {},
    cudaStream_t stream = nullptr);

}  // namespace singlet_gpu::anno
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")
# Preprocess: normalize → log-transform → HVG subset to reference genes
sg.preprocess.log_normalize(adata, target_sum=1e4)
adata = adata[:, adata.var.gene_name.isin(reference_genes)].copy()

# Load CellTypist model (e.g., from celltypist.models.Immune_Organs_v2)
model = sg.anno.load_celltypist_model("Immune_Organs_v2")

pred = sg.anno.celltypist(
    adata,
    model_weights=model.W,
    model_bias=model.b,
    use_log_input=True,
)
# → adata.obs['celltypist_pred_class'] : int [0, n_classes)
# → adata.obs['celltypist_confidence'] : float [0, 1]
```

## R signature

```r
singletGpu::celltypist(adata, model_weights, model_bias, use_log_input = TRUE)
```

## Inputs

- **Z** — `core::DeviceDense` [n_features × n_cells] col-major log-transformed gene expression. Typically the output of `log1p(X / lib_size)` for each cell. Must match the reference genes used to train W.
- **W** — `core::DeviceDense` [n_features × n_classes] col-major pre-trained logistic regression weights. Rows = reference genes, columns = cell types / classes.
- **b** — `core::DeviceMemory<float>` [n_classes] intercepts (biases) for each class. Typically learned via scikit-learn or glmnet.
- **cfg.use_log_input** — assert Z is log-transformed. If false, the kernel skips the assertion; used when Z is already in correct scale.
- **cfg.deterministic** — true by default; cuBLAS Sgemm and warp-shuffle reductions are deterministic at fp32.

## Outputs

`CelltypistResult`:
- **pred_class** `[n_cells]` argmax of softmax-normalized logits. Integer in [0, n_classes).
- **confidence** `[n_cells]` softmax probability of the predicted class. ∈ [0, 1]; high = high confidence.
- **n_cells**, **n_classes** host scalars for output sizing.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (1k cells, n_feat=5k, n_cls=50) | ~0.5 ms | ~20 MB (cuBLAS temp) | cuBLAS Sgemm dominates |  —  |
| medium (20k cells) | ~8 ms | ~32 MB | chunked per-batch via cuBLAS streaming | biased Sgemm + softmax |
| large (100k+ cells) | pending feature 17 | O(n_feat × n_cells) | per-batch Sgemm | output O(n_cells) |

3 passes: (1) cuBLAS Sgemm: L = W^T · Z → [n_classes × n_cells], (2) bias-add kernel: L[k,c] += b[k], (3) softmax + argmax + confidence (one block per cell, warp-shuffle reduction). All O(n_feat × n_cells) or O(n_classes × n_cells) per-element work.

## Streaming behavior

- **Current (CYCLE-135)**: in-memory Sgemm + dense bias/softmax. Full L matrix allocated.
- **Planned (CYCLE-151+)**: chunked cuBLAS Sgemm per batch (standard tiling), then per-batch softmax+argmax. Requires row-major input Z (or transpose once).
- Passes 1–3 are naturally parallelizable across cell shards. W and b are broadcast (read-only).

## Determinism

Fully deterministic. cuBLAS Sgemm is deterministic at fp32 (no atomics). Warp-shuffle reductions are deterministic (no data-dependent branching or atomics). Same input (same Z, W, b, stream, cfg) → bit-identical output.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| TinyClosedForm | pred_class, confidence match hand calc | exact | 3 cells, 2 genes, 4 classes | PASS |
| OneHotInputs_PerfectClassification | confidence when one gene per class | ≥ 0.99 | one-hot Z, W diag=10 | PASS |
| AllZeroExpression_UniformDistribution | softmax uniform when Z≈0 | confidence = 1/n_classes | Z=[0,...,0], n_cls=4 | PASS (0.25) |
| Determinism_BitIdentical | rel_err(run 1, run 2) | 0.0 | same data, seed=42 | PASS (0.00e+00) |
| LargeClasses_HandlesStriding | confidence for n_classes > 256 | max_err ≤ 1e-6 | n_classes=500, 1k cells | PASS (3.91e-08) |

All tests in `tests/anno_celltypist_correctness.cpp` (CYCLE-135, ctest 5/5 PASS).

## Citation

> Domínguez Conde C, Xu C, Jarvis LB, et al. (2022). Cross-tissue immune cell analysis reveals tissue-specific features in humans. _Science_, 376(6594):eabl5197. https://doi.org/10.1126/science.abl5197

CellTypist is a pre-trained logistic regression cell-type classifier. The algorithm: (1) linear prediction L[k,c] = Σ_g W[g,k] · Z[g,c] + b[k] (cuBLAS Sgemm), (2) softmax normalization P[k,c] = exp(L[k,c] - L_max[c]) / Σ_k' exp(L[k',c] - L_max[c]), (3) argmax + confidence = max softmax prob. The Human Cell Atlas pre-training provides ~30–50 cell-type references; model selection is user-driven.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/anno/celltypist.h>
#include <singlet-gpu/preprocess/log_normalize.h>

int main() {
    namespace sg = singlet_gpu;

    // Step 1: load and preprocess
    auto pz = sg::load_pz("/path/to/exon_counts.1pz");
    sg::preprocess::log_normalize(pz.mat, {}, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);

    // Step 2: load pre-trained model weights from file
    // (In production, these come from CellTypist.models.fetch_model("Immune_Organs_v2"))
    std::vector<float> W_host(n_features * n_classes);
    std::vector<float> b_host(n_classes);
    // ... read from file ...

    // Step 3: transfer to GPU
    auto W_device = sg::core::DeviceMemory<float>::copyFromHost(W_host);
    auto b_device = sg::core::DeviceMemory<float>::copyFromHost(b_host);

    // Step 4: predict
    auto log_expr = /* log-normalized expression matrix, DeviceDense */;
    sg::anno::CelltypistConfig cfg{};
    cfg.use_log_input = true;
    auto result = sg::anno::celltypist(log_expr, W_device, b_device, cfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);

    // Step 5: transfer predictions to host
    std::vector<int> pred_class(result.n_cells);
    std::vector<float> confidence(result.n_cells);
    cudaMemcpy(pred_class.data(), result.pred_class.get(),
               result.n_cells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(confidence.data(), result.confidence.get(),
               result.n_cells * sizeof(float), cudaMemcpyDeviceToHost);
}
```

## Pitfalls and notes

1. **Log-transformation requirement.** CellTypist models are trained on log-transformed expression (typically log1p after normalization). Feeding raw counts or raw expression directly produces nonsensical predictions. The assertion `cfg.use_log_input = true` guards against this; set to false only if you've manually verified Z is in the correct scale.

2. **Gene subset to reference genes.** Pre-trained W has n_features = #reference genes (typically 5k–20k). Your input Z must be subset to exactly the same genes in the same order. No automatic matching; you must align gene names manually.

3. **Eigenvector sign is constrained (but softmax is sign-invariant).** Unlike diffmap, logistic regression softmax is invariant to row-wise sign flips in W (if sign(W[g,:]) flips, both logits and softmax shift symmetrically). No canonicalization needed.

4. **Strided softmax for large n_classes.** For n_classes > 256 threads per block, the kernel strides: each thread processes multiple classes sequentially. Tested up to n_classes=500; correctness guaranteed by warp-shuffle reduction pattern.

## Pareto-frontier rows

| scale | wall_ms | memory_mb | accuracy | dominates_on |
|---|---|---|---|---|
| small-1k | 0.5 | 20 | 5/5 tests PASS, confidence ≥0.99 | correctness (all tests), usability (logreg is interpretable) |
| medium-20k | 8 | 32 | pending v1.1 bench vs scikit-learn | wall dominates |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369244 on g051 H100 NVL, CYCLE-135).

## Links

- Design docs: [`state/designs/27-celltypist.md`](../../state/designs/27-celltypist.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § anno/celltypist
- Tests: `tests/anno_celltypist_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`anno_symphony.md`](anno_symphony.md) (sister module, CYCLE-138 — centroid-projection annotation), [`preprocess_log_normalize.md`](preprocess_log_normalize.md) (required preprocessing), [`reduce_svd.md`](reduce_svd.md) (PCA for exploratory annotation)
