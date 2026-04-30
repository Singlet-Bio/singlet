---
feature: 13
module: preprocess/velocity_prep.h, velocity/kinetic_fit.h, velocity/embedding.h
cycle: 67 or later
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy
extends: 13-velocity-prep.md
status: draft
exploit: singlify-unique (intron_counts.1pz + splice_junctions.1pz)
---

# Feature 13 — RNA velocity Phase E (unique-to-us singlify exploit)

Feature 13 is the library's **signature unique-to-us kernel**. scVelo / velocyto / dynamo all require a separate splicing-aware alignment pass (usually velocyto or STARsolo `--soloFeatures GeneFull`) to produce spliced/unspliced count matrices. **Singlify already produces `intron_counts.1pz` and `splice_junctions.1pz` natively for every sample.** No other downstream library can read these files. This is a moat feature: load time = milliseconds (Cycle 56's loader), kinetic fit = closed-form on device, no external dependencies.

## 1. Success metrics

On GSM4037629 (has intron + exon + SJ outputs from singlify):

- **End-to-end wall (velocity prep → kinetic fit → velocity embedding)**: ≤ `scVelo` wall × 0.01 (100× headline).
- **Kinetic parameter correlation vs scVelo dynamic model**: Pearson α/β/γ ≥ 0.95.
- **Velocity arrow direction consistency**: cosine similarity ≥ 0.90 between our embedded velocity vectors and scVelo's on the same UMAP.
- **Peak dev mem**: ≤ scVelo × 0.5 (no CPU intermediate formats needed — direct from .1pz).

Frontier gate: dominance on wall AND α/β/γ correlation ≥ 0.95 AND arrow cosine ≥ 0.90. Rule 31 autonomy within 10%.

## 2. SOTA baselines

| Baseline | Path | Notes |
|---|---|---|
| **scVelo (dynamic model)** | Python / numba | Primary reference; iterative EM to fit α, β, γ per gene |
| **scVelo (stochastic model)** | Python / numba | Simpler steady-state variant |
| **velocyto** | Python / Cython | Original velocity paper reference |
| **dynamo** | Python | Extended velocity + vector field reconstruction |
| **UniTVelo** | Python | Unified tissue velocity with continuous dynamics |

## 3. Bench configurations

Scales:
- **small**: GSM4037629 (20,866 cells, intron + exon + SJ available)
- **medium**: 5-sample concat ~100k cells
- **large**: 1M synthetic (stress)

Configurations × every scale:

1. `ours_steady_state_closed_form` — Rule 30 closed-form (§4a)
2. `ours_dynamic_batched_em` — batched EM fit for per-gene α, β, γ (§4b)
3. **`ours_fused_sj_refined`** — Rule 30 novel using splice_junctions.1pz for sub-exon kinetics (§4c)
4. `ours_auto` — Rule 31 path
5. `scvelo_dynamic`
6. `scvelo_stochastic`
7. `velocyto`
8. `dynamo`

Metrics: wall p50 (split into prep / kinetic fit / embedding phases), peak dev mem, α/β/γ Pearson, velocity cosine similarity vs scVelo dynamic.

## 4. Novel pursuit (Rule 30)

### 4a. Closed-form steady-state solution (skip scVelo's EM loop)

**scVelo's dynamic model** fits per-gene kinetic parameters α (transcription), β (splicing), γ (degradation) via expectation-maximization over a gene-specific latent time. Each gene's EM takes 50+ iterations.

**The closed-form observation**: at steady state, for each cell, the spliced (S) and unspliced (U) counts satisfy:
```
dU/dt = α − β U
dS/dt = β U − γ S
```
Setting derivatives to zero: `U_ss = α/β`, `S_ss = α/γ`. This gives `γ/β = U_ss / S_ss` per gene at steady state, which is scVelo's core ratio. But scVelo computes this iteratively because real data is not at steady state — cells are mid-trajectory.

**Our closed-form approach**: for each gene, fit `γ = slope(U, S)` via **weighted total least squares** on the (U, S) cloud, with weights proportional to the per-cell distance from the spliced/unspliced origin. This is a **single eigendecomposition of the 2×2 weighted covariance matrix** — closed form in O(n_cells) per gene.

For relative velocity (cell-level velocity vectors), we then solve per-cell per-gene:
```
velocity_g(cell) = (U_{g,cell} − γ̂_g × S_{g,cell}) / S_{g,cell} × reference_rate
```
which is a single fused element-wise op across all gene × cell pairs.

**Expected wall**: per-gene fit = O(n_cells) via fused reductions; total kinetic fit = O(n_genes × n_cells). For 2000 velocity genes × 20k cells that's 40M ops — sub-millisecond on V100S. scVelo's EM takes ~5 min on the same data. **50000× headline** on the kinetic fit phase alone.

**Gate**: γ correlation ≥ 0.95 vs scVelo dynamic on a reference cell-cycle dataset.

### 4b. Batched EM for the dynamic-model opt-in

For users who want scVelo's full dynamic model (e.g. for low-steady-state tissues), ship a fully-batched EM kernel that fits all genes simultaneously:
- Per-gene state vector: (α, β, γ, t_i for each cell assigned to this gene's trajectory)
- Shared EM iteration across all genes via tensor-shaped updates
- cuBLAS batched matrix ops for the E-step regression
- Convergence per gene independently; genes that converge early get masked out of further iterations

Expected wall: ~5 sec for 2000 genes × 20k cells at 50 iterations vs scVelo's 5 min. ~60× headline.

Gate: α/β/γ Pearson ≥ 0.97 (tighter than 4a since this path is the full model).

### 4c. Splice-junction-refined kinetics (UNIQUE to us)

**The deepest moat**: singlify's `splice_junctions.1pz` provides per-cell counts of reads crossing each individual splice junction. No other scRNA pipeline produces this at per-cell resolution.

Use case: instead of per-gene α, β, γ kinetics, fit **per-junction kinetics**. A gene with 10 exons has 9 splice junctions; each junction has its own β_j. A subset of junctions will be in use at a given trajectory point — exploit this to:

1. Detect **alternative splicing modulation** during trajectory (e.g. isoform switches in differentiation).
2. Fit **local kinetics** on gene subregions where the steady-state assumption is more reliable.
3. Derive a **junction-level velocity** that captures sub-gene dynamics scVelo cannot see.

This is published as "AS velocity" in a few 2023–2024 papers but with custom analysis pipelines — because they had to build their own SJ-per-cell tables. We get them for free from singlify.

**Algorithm sketch**:
1. Load `splice_junctions.1pz` → junction × cell count matrix.
2. For each gene, extract its junctions.
3. Per junction: fit `γ_j = slope(J_j, S_g)` where `S_g` is the parent gene's spliced count.
4. Per cell: velocity = `mean_j (J_j − γ_j × S_g) / S_g`, with junction-confidence weighting.

**Expected gate**: novel so there is no direct baseline. Validate by showing improved velocity arrow consistency on known differentiation trajectories (e.g. a HSPC → granulocyte trajectory where AS is known to drive early lineage decisions).

**Caveat**: only useful when `splice_junctions.1pz` is present. Detect absence and fall back to 4a. Document in `state/novel-attempts.md` as the first "singlify-unique novel contribution" with published-paper-comparable results if the gate passes.

## 5. Autonomy pass (Rule 31)

No-args `velocity::compute(path)` returns velocity vectors with auto-selected model.

| Config | Auto strategy |
|---|---|
| `model` | Auto: `closed_form_steady_state` if per-gene CV(U/S ratio) < 0.3 (well-behaved steady state); else `dynamic_batched_em`. |
| `n_top_genes` | Auto: 2000 velocity genes filtered by (min expression + spliced-unspliced correlation > 0.3). |
| `min_ratio` | Auto: 0.05 (scVelo default). |
| `velocity_genes_filter` | Auto: auto-detected from spliced/unspliced correlation + minimum count thresholds. |
| `use_splice_junctions` | Auto: true if `splice_junctions.1pz` is detected alongside the input. |
| `embedding_grid_size` | Auto: sqrt(n_cells) for the velocity-embedding vector field grid. |
| `stream` | Auto: factornet GPUContext stream. |
| `mode` | Auto: `deterministic` (closed-form is inherently deterministic). |

## 6. OOC streaming contract

Velocity is gene-parallel and cell-parallel — streaming is natural:
- Load chunks of exon + intron + SJ matrices concurrently.
- Kinetic fit is per-gene across all cells — accumulate per-gene stats chunk-wise.
- Velocity vector construction is element-wise.
- Only the final embedding (vector field) needs global residency; can chunk with Procrustes alignment.

Document; implement with feature 16.

## 7. Determinism contract

Closed-form path is deterministic. Batched-EM path is deterministic if the initial guess is fixed (seed-controlled). Velocity embedding is deterministic up to the kNN tie-break.

## 8. Phase E dispatch spec

Dispatch `gpu-kernel-dev` with:
- Read 13-velocity-prep.md + this doc + singlify's `pz_writer.h` for intron/SJ layouts.
- Implement the 4a closed-form steady-state fit.
- Implement the 4b batched-EM dynamic model.
- Implement the 4c splice-junction-refined path (novel contribution).
- Write correctness tests against scVelo on a reference dataset.
- Dispatch `gpu-bench` in parallel for the 8-config bench table.
- Write to benchmark-registry.md, pareto-frontier.md, novel-attempts.md.

## 9. Open questions

- **Reference dataset with gold-standard velocity**: use the scVelo-provided pancreas dataset if available — it's the canonical benchmark. Otherwise use GSM4037629 cell cycle genes as a sanity check.
- **Splice junction → gene mapping**: singlify's `splice_junctions.1pz` stores per-junction counts but needs a GTF-derived junction→gene mapping. Compute once at load time from the GTF (already present in singlify's reference tree).
- **Velocity embedding method**: scVelo uses a transition-matrix-based embedding. We should port their approach (straightforward cuda kernel) rather than invent a new one for the Phase E cycle.
- **Does the closed-form γ fit match scVelo's iterative γ fit exactly on steady-state data?** Theoretically yes (both converge to the weighted regression slope). Verify empirically.

## 10. Links

- Original: `state/designs/13-velocity-prep.md`
- scVelo: Bergen et al. 2020 "Generalizing RNA velocity to transient cell states"
- velocyto: La Manno et al. 2018 "RNA velocity of single cells"
- dynamo: Qiu et al. 2022 "Mapping transcriptomic vector fields"
- singlify intron output format: `singlify/include/singlet-pileup/pz_writer.h`, `singlify/include/singlet-pileup/pileup_writer.h`
- CLAUDE.md line referring to feature 13 as the "why this library exists" moment
