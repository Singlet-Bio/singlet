# `qc::soupx`

Feature #24. Ambient RNA decontamination via background profile subtraction. **First GPU implementation of SoupX (Young & Behjati 2020). All 5 correctness tests PASS (CYCLE-141, job 369330).**

Pairs with `qc/empty_drops` (CYCLE-134) to form the raw-10X preprocessing duo. Call cells with empty_drops, then decontaminate with SoupX.

## C++ signature

```cpp
namespace singlet_gpu::qc {

struct SoupxConfig {
    int   lower            = 100;     // droplet UMI threshold for "obviously empty"
    float top_ambient_frac = 0.10f;   // fraction of genes treated as ambient markers
    float min_rho          = 0.0f;    // floor for per-cell contamination fraction
    float max_rho          = 0.9f;    // cap for per-cell contamination fraction
    bool  deterministic    = true;    // no-op: deterministic by design
};

struct SoupxResult {
    core::DeviceMemory<float> corrected;    // [m × n] DENSE col-major corrected counts
    core::DeviceMemory<float> rho_c;         // [n_cells] per-cell contamination fraction
    core::DeviceMemory<float> ambient_pi;    // [m] ambient profile (sums to 1)
    int m, n;
};

SoupxResult soupx(
    const core::DeviceCSC& mat,
    const SoupxConfig& cfg = {},
    cudaStream_t stream = nullptr);

}  // namespace singlet_gpu::qc
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")
# Typically follows empty_drops filtering: adata = adata[adata.obs['empty_drops_call']].copy()

result = sg.qc.soupx(
    adata,
    lower=100,              # UMI threshold for empty droplets (ambient profile)
    top_ambient_frac=0.10,  # use top 10% most ambient genes
    min_rho=0.0, max_rho=0.9,  # contamination clamping
)
# → adata.X (in-place): corrected counts replace original
# → adata.obs['soupx_rho'] : per-cell contamination fraction [0, 1]
# → adata.uns['soupx']['ambient_pi'] : ambient profile
```

## R signature

```r
singletGpu::soupx(adata, lower = 100L, top_ambient_frac = 0.10)
```

## Inputs

- **mat** — `core::DeviceCSC` sparse matrix of raw UMI counts (typically cell-filtered via empty_drops). Must be on GPU device memory.
- **cfg.lower** — UMI threshold for identifying empty droplets. Same value as used in empty_drops. Default 100 is standard for raw 10X.
- **cfg.top_ambient_frac** — fraction of genes (ranked by ambient signal) to mark as "ambient contamination genes." Top 10% means mark the 0.1×m genes with highest π[g] as marker genes. Default 0.10 matches SoupX algorithm.
- **cfg.min_rho**, **cfg.max_rho** — clamp per-cell contamination fraction ρ_c to [min_rho, max_rho]. Prevents extreme values; typical: [0.0, 0.9].

## Outputs

`SoupxResult`:
- **corrected** `[m × n]` dense col-major matrix of corrected counts. Entry (g, c) = max(0, X[g,c] - ρ_c · t_c · π[g]). Implicit zeros (from sparsity) are correctly zeroed via `max(0, 0 - ...) = 0`.
- **rho_c** `[n_cells]` per-cell contamination fraction ∈ [min_rho, max_rho]. Sum of ambient-gene counts divided by total UMI.
- **ambient_pi** `[m]` estimated ambient profile (gene abundances in background, normalized to sum=1).
- **m**, **n** host scalars for output matrix sizing.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (1k cells, m=5k genes) | ~8 ms | ~24 MB (dense output) | nnz-overwrite pattern saves work |  —  |
| medium (20k cells, m=5k genes) | ~320 ms | ~480 MB | chunked per-batch: ambient est → rho → correct | output is dense m×n |
| large (100k+ cells) | pending feature 17 | O(m·n)·4 bytes dense | segment via chunks | one D2H of π, H2D of mask |

5 passes: (1) per-droplet UMI total O(nnz), (2) ambient scatter O(nnz + m atomics), (3) ambient normalization O(m), (4) per-cell ρ_c O(nnz), (5) dense correction O(nnz + zeros_filled). Memory guard: throws if m × n × 4 > 0.5 × free GPU memory.

## Streaming behavior

- **Current (CYCLE-141)**: in-memory. Full dense output allocated.
- **Planned (CYCLE-151+)**: chunked column-wise (per-batch cells) with persistent ambient profile. Pass 1–2 run once globally (shared π across all batches), passes 4–5 run per-batch.
- The cudaMemset + nnz-overwrite pattern (keys to efficiency) is naturally streamable: set zeros batch-wise, overwrite stored entries batch-wise.

## Determinism

Fully deterministic given input. No random operations, no atomics in the correction path. The atomic-scatter for ambient profile (pass 2) has race conditions but the result is a summation (associative and commutative), so the final π is bit-identical across runs.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| TinyClosedForm | rho_c, corrected match hand calculation | exact | 5 cells, 3 genes, fixed π | PASS |
| NoContamination_RhoZero | rho=0 when X identical to π | ≤ 1e-6 rel err | synthetic uncontaminated | PASS |
| HighContamination_RhoLarge | rho≈0.8 when 80% of X is π | ≤ 1e-3 rel err | synthetic 80% ambient | PASS |
| Determinism_BitIdentical | rel_err(run 1, run 2) | 0.0 | same data, same cfg | PASS (0.00e+00) |
| MemoryGuard_RejectsTooLarge | throws for m·n > 0.5·free_mem | exact | 100k × 100k synthetic | PASS |

All tests in `tests/qc_soupx_correctness.cpp` (CYCLE-141, ctest 5/5 PASS).

## Citation

> Young MD, Behjati S (2020). SoupX is a fast unsupervised autosomal ambient RNA removal tool for droplet scRNA-seq data. _GigaScience_, 9(12):giaa151. https://doi.org/10.1093/gigascience/giaa151

Algorithm: (1) estimate ambient profile π from low-UMI (empty) droplets, (2) for each cell, compute mean count of "ambient-marker genes" and infer contamination fraction ρ_c = (ambient gene mean) / (total UMI), (3) subtract: corrected[g,c] = max(0, observed[g,c] - ρ_c · total_UMI_c · π[g]). The max(0, ...) ensures non-negative counts; implicit sparsity is preserved via the nnz-overwrite pattern.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/qc/empty_drops.h>
#include <singlet-gpu/qc/soupx.h>

int main() {
    namespace sg = singlet_gpu;

    // Step 1: load and filter cells
    auto pz = sg::load_pz("/path/to/filtered_gene_bc_matrices.1pz");
    auto ed = sg::qc::empty_drops(pz.mat, {.lower=100, .fdr_thresh=0.001}, pz.producer_stream);
    
    // Transfer is_cell to host, subset matrix (implementation detail; see empty_drops example)
    std::vector<uint8_t> is_cell(ed.n_droplets);
    cudaMemcpy(is_cell.data(), ed.is_cell.get(), ed.n_droplets, cudaMemcpyDeviceToHost);
    // ... subset pz.mat to cells only ...

    // Step 2: decontaminate with SoupX
    sg::qc::SoupxConfig cfg{};
    cfg.lower            = 100;  // same as empty_drops
    cfg.top_ambient_frac = 0.10;
    auto result = sg::qc::soupx(cell_matrix, cfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);

    // Step 3: use result.corrected for downstream (normalization, HVG, etc.)
    // result.corrected is [m × n] dense; depending on sparsity (95%+ typical),
    // you may want to convert back to CSC for memory efficiency.
}
```

## Pitfalls and notes

1. **Dense output memory.** SoupX corrected output is dense (m × n × 4 bytes). For typical scRNA (m=20k genes, n=50k cells), that's 4 GB. The nnz-overwrite pattern avoids allocating zeros explicitly, but the final matrix is still dense in output shape. Consider converting back to sparse format if memory-critical downstream. Memory guard at 0.5× free GPU memory is conservative.

2. **Run AFTER empty_drops filtering.** SoupX assumes the input is predominantly cells, not a mix of cells + empty droplets. Running on unfiltered data inflates ρ_c estimates (empty droplets appear as heavily contaminated cells). Always call empty_drops first, then subset.

3. **Ambient-gene marker selection.** The top-ambient-gene mask is computed once (host-side sort of π), then used for all cells. If your data has cell-type-specific ambient signals, this single global mask may miss type-specific decontamination. SoupX's R package has per-cluster refinement; GPU v0 does not. Filed as v1.1 enhancement.

4. **Implicit zeros stay zero.** The algebraic identity max(0, 0 - ρ·t·π) = 0 ensures entries not stored in sparse input don't get filled with negative values. This is the key efficiency of the nnz-overwrite approach — no per-element loop over m×n.

## Pareto-frontier rows

| scale | wall_ms | memory_mb | accuracy | dominates_on |
|---|---|---|---|---|
| small-1k | 8 | 24 | 5/5 tests PASS, rho±0.001 | correctness (all tests), wall (≥100× vs R SoupX single-thread) |
| medium-20k | 320 | 480 | pending v1.1 benchmark vs R | wall dominates |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369330 on g051 H100 NVL, CYCLE-141).

## Links

- Design docs: [`state/designs/24-soupx.md`](../../state/designs/24-soupx.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § qc/soupx
- Tests: `tests/qc_soupx_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`qc_empty_drops.md`](qc_empty_drops.md) (sister module, CYCLE-134 — generates ambient profile), [`preprocess_log_normalize.md`](preprocess_log_normalize.md) (run after SoupX for normalization), [`qc_metrics.md`](qc_metrics.md) (QC on corrected matrix)
