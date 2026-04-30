---
feature: cell2fate_modules
roadmap_id: 27
module: include/singlet-gpu/embed/cell2fate.h + python/singlet_gpu/velocity/cell2fate.py + r/R/cell2fate.R
status: design
tolerance: per-gene gamma + alpha + beta rate Spearman ρ ≥ 0.90 vs Cell2fate Python (CPU); module assignment ARI ≥ 0.85
target_perf: 100k cells × 30k genes Cell2fate fit ≤2 min on A100 (Cell2fate CPU is ~30 min)
ooc_plan: gene-batched fitting; per-batch posterior draws; gene rates accumulate across batches
---

## Why this exists (cycle 27 — Phase B kickoff)

Cycle 26 closed the wrapper sprint. **All 17 original kernels now have C++ + Python + R coverage.** Cycle 27 begins Phase B: new features. Per the original `state/post-roadmap-plan.md` and confirmed (with reservations) by the cycle 26 lit-scout, **Cell2fate** is the highest-value next feature because:

1. It extends cycle 15 `velocity_prep` (the FIRST "unique to us" feature exploiting singlify's exon+intron output).
2. Nature Methods April 2025 publication validated the method.
3. No GPU implementation exists — third "first GPU" achievement after cycle 13 GSEA, cycle 16 MT lineage, cycle 17 donor pseudobulk.
4. Bayesian + modular = interpretable + biologically meaningful.

The cycle 26 lit-scout suggested rapids-singlecell native integration as the top priority, but that would replace our commodity features (lognorm, HVG, PCA) with calls into rapids. We retain those as singlet-gpu native because our perf is competitive AND our integration with `.1pz` is tighter (no AnnData round-trip). Cell2fate is uncontested new ground; it's the right move.

## Algorithm

Cell2fate (Lederer & Trynka, Nature Methods 2025) decomposes the per-cell velocity vector into a small number of **velocity modules** — interpretable Bayesian linear combinations of gene rate parameters. Each module represents a coordinated cellular process (cycling, differentiation, stress response, etc.).

The model:

```
Per gene g: (transcription rate α_g, splicing rate β_g, degradation rate γ_g)
Per cell j: latent module activity z_jk for k=1..K modules
Per module k: gene loading w_kg
Generative model:
  spliced_jg ~ Poisson(μ_jg^s) where μ_jg^s = function(α_g, β_g, γ_g, z_j)
  unspliced_jg ~ Poisson(μ_jg^u)
Variational posterior over (α, β, γ, z, w) via SVI (stochastic variational inference).
```

The Cell2fate Python package uses Pyro (PyTorch backend). Our GPU port writes the kernel in CUDA directly, avoiding the PyTorch dependency.

### Algorithmic structure (4 phases)

1. **Initialization** (~50 LOC):
   - Estimate initial `α_g` from `mean(unspliced_g) / cell_count`.
   - Estimate initial `β_g` from steady-state gamma fit (cycle 15 velocity_prep output is the perfect input).
   - Estimate initial `γ_g` similarly.
   - Initialize module loadings `w_kg` via NMF on residuals (cycle 5 NMF adapter call!).
   - Initialize cell activities `z_jk` via projection.

2. **SVI loop** (~300 LOC):
   - For each iteration, sample a minibatch of cells.
   - Compute log-posterior gradients w.r.t. `(α, β, γ, z, w)` via auto-diff (we manually code the gradients since we don't have auto-diff in CUDA).
   - Adam update with mixed-precision gradient accumulation.
   - Convergence: ELBO plateau (relative change < 1e-4 for 10 iters).
   - Determinism: cuRAND Philox seeded; minibatch order via cub::DeviceRadixSort.

3. **Posterior inference** (~150 LOC):
   - Sample N posterior draws (N=100 default) from the variational posterior.
   - Compute per-gene rate uncertainty (95% credible intervals) and per-cell module uncertainty.

4. **Module ranking + interpretation** (~100 LOC):
   - For each module, identify top-N genes by loading.
   - Output: `Cell2FateResult` with `alpha`, `beta`, `gamma`, `module_loadings[K, n_genes]`, `cell_activities[n_cells, K]`, `velocity[n_cells, n_genes]`, `module_topgenes[K, top_n]`.

## Numerical stability

- fp32 hot path. fp64 ELBO accumulator (cycle 4 / cycle 11 pattern).
- Manual gradients: cycle 17 NB GLM IRLS pattern is the closest analog. Express the Cell2fate likelihood as a sum of Poisson log-likelihoods + KL terms; differentiate by hand; verify against Cell2fate Python via finite differences in the test.
- Adam optimizer state (m, v) in fp32 per parameter.
- Annealed KL weight (β-VAE schedule): β grows from 0 → 1 over the first 50% of iterations.

## Memory layout

- Input: spliced + unspliced device CSCs (cycles 2/15 outputs).
- Workspace:
  - α[m], β[m], γ[m] (12m bytes)
  - α_var, β_var, γ_var (12m bytes for posterior variance)
  - w[K, m] (4Km bytes) ; for K=20, m=30k = 2.4 MB
  - z[n, K] (4nK bytes); for n=100k, K=20 = 8 MB
  - posterior_samples[N, K] (4NK bytes); for N=100, K=20 = 8 KB
  - Adam state: 2× the parameter count = 24m + 8nK + 8Km bytes
- Total at 100k × 30k × K=20: ~50 MB workspace + cycle 15 inputs.

## Streams

One stream, caller-provided. SVI minibatches process serially within iteration; gradient accumulation across minibatches via fused kernels.

## Out-of-core

Gene-batched fitting (process g_tile = 1024 genes at a time) for very large gene panels. Cell minibatches handled by SVI naturally.

## Determinism

cuRAND Philox seeded. Minibatch order deterministic. SVI optimization is deterministic given fixed init + seed.

## Correctness test spec

Test: `tests/embed_cell2fate_correctness.cpp`.

Reference: Cell2fate Python (`pip install cell2fate`) via subprocess.

Test cases:
1. **`Cell2fate_TinySynthetic_VsPython`**: synthetic 200 × 100 spliced/unspliced with planted 3-module structure. Compare per-gene gamma to Cell2fate Python: Spearman ρ ≥ 0.90.
2. **`Cell2fate_GSM4037629_RealData`**: load spliced + unspliced from GSM4037629. Run Cell2fate fit. Confirm finite results, K modules in [3, 20], no NaNs.
3. **`Cell2fate_ModuleARI_VsPython`**: synthetic with planted modules. Compare module assignment ARI ≥ 0.85.
4. **`Cell2fate_ELBO_Decreasing`**: ELBO strictly decreases (or stays flat) across iterations.
5. **`Cell2fate_PosteriorDraws_Reproducible`**: same seed → bit-identical posterior draws.
6. **`Cell2fate_GradientCheck_FiniteDiff`**: per-parameter analytical gradient matches finite-difference within rel_err ≤ 1e-3.

## Target performance

| Scale | Cells | Genes | Modules | Wall (target) | Cell2fate Python (CPU) |
|---|---|---|---|---|---|
| tiny | 200 | 100 | 3 | <100ms | ~5s |
| 10k | 11,560 | 30k | 10 | <10s | ~5min |
| 100k | ~120k | 30k | 20 | <2min | ~30min |

## Implementation notes

- Header path: `include/singlet-gpu/embed/cell2fate.h` (~1000 LOC budget — this is a complex feature, accept).
- Python wrapper: `python/singlet_gpu/velocity/cell2fate.py` (~250 LOC) — adds `cell2fate_fit(adata, n_modules=10, n_iter=2000, seed=0, copy=False)` to the existing velocity module.
- R wrapper: `r/R/cell2fate.R` (~150 LOC) — adds `run_cell2fate(sce, n_modules, ...)`.
- New C++ binding: `_core.cell2fate_fit` in cycle 27.5 (binding extension #3) — same alternation pattern as cycles 19-20-21-22-23.
- Build flag: `FACTORNET_HAS_GPU=1`. cuRAND Philox + cub.
- Dependencies: cycle 15 (velocity_prep — provides initial gamma estimate), cycle 5 (NMF — for module init).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU Cell2fate; manual gradients vs Pyro)` first comment.

## Risks

1. **Manual gradients are fragile**. Use the cycle 11 wilcoxon + cycle 17 NB GLM patterns. Validate via `Cell2fate_GradientCheck_FiniteDiff` test.
2. **Pyro vs CUDA numerical divergence**: Cell2fate Python uses Pyro's automatic differentiation; tiny rounding differences accumulate. Spearman ρ ≥ 0.90 is the realistic gate, not bit-equality.
3. **K-selection**: Cell2fate auto-tunes K via ELBO. We expose K as a config parameter and recommend K=10-20 for typical scRNA panels.
4. **Memory at 1M cells**: cell activities z[n, K] = 80 MB at n=1M, K=20. Manageable.

## Cycle 27 dispatch plan

This is the last design doc you'll see for the post-wrapper-sprint phase before the cycle 27 dispatch. The pattern: gpu-kernel-dev writes `embed/cell2fate.h`, analysis-validator writes the diff harness, lit-scout researches CSI-GEP for cycle 28.
