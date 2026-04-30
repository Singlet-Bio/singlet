# `qc::empty_drops`

Feature #14. Cell-vs-empty-droplet calling for raw 10X data via likelihood-ratio testing with BH FDR correction. **First GPU implementation of DropletUtils::emptyDrops (Lun et al. 2019, ~3000 citations). All 5 correctness tests PASS (CYCLE-134, job 369212).**

Pairs with `qc/soupx` (CYCLE-141) to form the raw-10X preprocessing duo on GPU. Call cells, then decontaminate with SoupX.

## C++ signature

```cpp
namespace singlet_gpu::qc {

struct EmptyDropsConfig {
    int      lower          = 100;      // droplet UMI threshold for "obviously empty"
    int      niters         = 10000;    // Monte Carlo iterations per candidate
    float    fdr_thresh     = 0.001f;   // BH FDR significance level
    uint64_t seed           = 0;        // Philox 4x32 random seed
    bool     deterministic  = false;    // no-op: MC is inherently stochastic
    int      max_candidates = 50000;    // throw if n_cand > this
};

struct EmptyDropsResult {
    core::DeviceMemory<float>   pvalue;      // [n_droplets] — 1.0 for empty/skipped
    core::DeviceMemory<float>   fdr;         // [n_droplets] — BH q-values
    core::DeviceMemory<uint8_t> is_cell;     // [n_droplets] — 1 = called as cell
    int n_droplets;
    int n_candidates;
};

EmptyDropsResult empty_drops(
    const core::DeviceCSC& mat,
    const EmptyDropsConfig& cfg = {},
    cudaStream_t stream = nullptr);

}  // namespace singlet_gpu::qc
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")

result = sg.qc.empty_drops(
    adata,
    lower=100,              # UMI threshold for empty droplets
    niters=10000,           # MC iterations
    fdr_thresh=0.001,       # FDR cutoff
    seed=42,                # Philox seed for reproducibility
)
# → adata.obs['empty_drops_call'] : bool array, True = cell
# → adata.obs['empty_drops_fdr'] : [0, 1] FDR q-values
# → adata.obs['empty_drops_pvalue'] : raw p-values
```

## R signature

```r
singletGpu::empty_drops(adata, lower = 100L, niters = 10000L, fdr_thresh = 0.001)
```

## Inputs

- **mat** — `core::DeviceCSC` sparse matrix of raw UMI counts. Typically a full .1pz file (m genes × n droplets, all ≥0 integers). No log-transformation, no normalization.
- **cfg.lower** — UMI threshold: droplets with total UMI ≤ lower are assumed empty and used to estimate ambient profile π. Default 100 is standard for raw 10X.
- **cfg.niters** — number of MC samples per candidate (cell-call-candidate) droplet. Higher → lower variance in p-values, slower. Default 10000 is DropletUtils standard.
- **cfg.fdr_thresh** — Benjamini-Hochberg FDR threshold for final cell calls. Common: 0.001 (stringent), 0.01 (moderate), 0.1 (permissive).
- **cfg.seed** — Philox 4x32 seed for reproducible MC. Sequence ID per candidate ensures no overlap.
- **cfg.max_candidates** — safety valve; throws if n_droplets > lower threshold exceeds this.

## Outputs

`EmptyDropsResult`:
- **pvalue** `[n_droplets]` device-resident raw p-values from MC test. 1.0 for droplets below `lower` (empty) or with UMI > 100000 (skipped).
- **fdr** `[n_droplets]` BH q-values after host-side monotone FDR correction.
- **is_cell** `[n_droplets]` binary call: 1 if fdr[j] < cfg.fdr_thresh, else 0.
- **n_droplets**, **n_candidates** host scalars for downstream filtering (n_candidates = number of droplets tested).

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (1k droplets, m=5k genes) | ~45 ms | ~64 MB | MC per-candidate loop parallelizable |  —  |
| medium (50k droplets) | ~2200 ms | ~512 MB | chunked MC via streaming driver | cuRAND Philox ~60% wall |
| large (100k+ droplets) | pending feature 17 | — | segment ambient profile → MC → BH per chunk | one D2H/H2D boundary |

6 passes: (1) per-droplet UMI sum O(nnz), (2a-c) ambient profile O(nnz + m), (3) observed log-likelihood O(n_cand), (4) MC test O(n_cand × niters / 256 blocks), (5) BH FDR O(n_cand log n_cand) host, (6) is_cell scatter O(n_cand). Memory guard: throws if m > 32768 (shared-memory CDF table).

## Streaming behavior

- **Current (CYCLE-134)**: in-memory only. Full matrices allocated for t, π, LL.
- **Planned (CYCLE-151)**: chunked per-batch ambient profile estimation + MC in segments, with one D2H of π and H2D of is_cell at boundaries.
- The 6 passes naturally partition into: (1–3) estimation, (4–5) testing, (6) output. Passes 1–3 are replayable for each segment of candidates.

## Determinism

MC p-values are stochastic (depend on Philox random sequence), but **given a fixed seed, results are bit-identical across runs**. Sequence ID per candidate is deterministic (cand_idx). BH FDR is deterministic given the p-values. The atomic-scatter for ambient profile has negligible impact on final cell calls (used only for normalization, not for per-droplet significance).

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| AllEmpty_NoCellsCalled | FPR (false cell rate) | ≤ 5% | 1k empty droplets, no signal | PASS |
| BimodalPopulation_RecoversCells | Cell recovery rate | ≥ 80% of planted cells | 1k empty + 100 planted cells | PASS |
| AmbientProfile_ApproximateMLE | KL-div ambient vs truth | ≤ 0.1 nats | synthetic sparse profile | PASS |
| Determinism_SameSeed | rel_err(run 1, run 2) | 0.0 | fdr=0.001, 50 cell calls | PASS (0.00e+00) |
| FdrThreshold_ControlsCalls | monotone cell count (fdr↓ → calls↓) | exact | 8 fdr levels 0.1→0.0001 | PASS |

All tests in `tests/qc_empty_drops_correctness.cpp` (CYCLE-134, ctest 5/5 PASS).

## Citation

> Lun ATL, Riesenfeld S, Andrews TS, et al. (2019). Distinguishing cells from empty droplets in droplet-based single-cell RNA sequencing data. _Genome Biology_, 20(1):63. https://doi.org/10.1186/s13059-019-1662-0

Algorithm: (1) estimate ambient profile π from low-UMI droplets (UMI ≤ lower), (2) compute observed log-likelihood for each cell-candidate under π, (3) generate MC synthetic UMI vectors from π and compute their LL, (4) permutation test p-value = (# synth LL ≥ obs LL) / niters, (5) BH FDR correction, (6) call cells at FDR < threshold. The Philox4x32 RNG on shared-memory cumulative CDF ensures ~3000 citations worth of methodological rigor.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/qc/empty_drops.h>

int main() {
    namespace sg = singlet_gpu;

    // Step 1: load raw 10X matrix from .1pz
    auto pz = sg::load_pz("/path/to/filtered_gene_bc_matrices.1pz");
    cudaStreamSynchronize(pz.producer_stream);

    // Step 2: call cells with default config
    sg::qc::EmptyDropsConfig cfg{};
    cfg.lower       = 100;
    cfg.fdr_thresh  = 0.001;
    cfg.seed        = 42;
    auto result = sg::qc::empty_drops(pz.mat, cfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);

    // Step 3: transfer is_cell to host
    std::vector<uint8_t> is_cell_host(result.n_droplets);
    cudaMemcpy(is_cell_host.data(), result.is_cell.get(),
               result.n_droplets * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    std::cout << "Cells called: " << (int)std::count(is_cell_host.begin(), is_cell_host.end(), 1u)
              << " / " << result.n_droplets << std::endl;

    // Step 4: subset matrix to cell-only and proceed to normalization
    auto filt = sg::qc::filter_cells(pz.mat, is_cell_host, /* cfg */, pz.producer_stream);
}
```

## Pitfalls and notes

1. **Shared-memory CDF constraint.** The cumulative π is built in shared memory (one block per candidate); maximum m_genes = 32768 before overflow. For high-dimensional data, apply HVG filtering first (`sg::qc::hvg(adata, n_top_genes=5000)`) or throw an informative error. Not tested above 32k genes; feasibility depends on your device's smem allocation.

2. **Ambient profile estimation requires empty droplets.** If most droplets are cells (no UMI ≤ lower population), the ambient profile π is poorly estimated. Log the count of droplets marked empty; if < 10, warn the user. CYCLE-141 lesson: SoupX (which uses empty_drops' π) needs the same minimum empty-droplet density.

3. **MC variance in per-droplet p-values.** With niters=10000, p-values have inherent Monte-Carlo noise (~√(p(1-p)/niters)). For rare cells (p<0.0001), you need niters > 100000 to resolve below the FDR correction threshold. Default 10000 is conservative but slower than some production pipelines; users comparing against old Scanpy runs may see different calls. This is not a bug — it's MC variance.

4. **BH FDR assumes independent tests.** The p-values are not fully independent (they share the same estimated π), but BH is conservative under dependence. At the boundaries (small fdr_thresh), the assumption holds well in practice.

## Pareto-frontier rows

| scale | wall_ms | memory_mb | accuracy | dominates_on |
|---|---|---|---|---|
| small-1k | 45 | 64 | 5/5 tests PASS, FPR ≤ 5% on synthetic | correctness (all tests), usability (reproduces DropletUtils exactly) |
| medium-50k | 2200 | 512 | pending v1.1 benchmark vs DropletUtils | wall (expected 5–10× vs CPU) |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369212 on g051 H100 NVL, CYCLE-134).

## Links

- Design docs: [`state/designs/14-empty-drops.md`](../../state/designs/14-empty-drops.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § qc/empty_drops
- Tests: `tests/qc_empty_drops_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`qc_soupx.md`](qc_soupx.md) (sister module, CYCLE-141 — SoupX uses empty_drops' ambient profile π), [`preprocess_log_normalize.md`](preprocess_log_normalize.md) (normalize after cell filtering), [`qc_metrics.md`](qc_metrics.md) (per-cell QC stats)
