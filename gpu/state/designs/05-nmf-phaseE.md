---
feature: 5
module: reduce/nmf/*.h
cycle: 61 or later (after Cycle 60 SVD)
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy
extends: 05-nmf-adapters.md
status: draft
depends_on: Cycle 55b (all 13 NMF tests runtime-correct)
---

# Feature 5 — NMF adapters Phase E frontier push

Feature 5 comprises four thin adapters over factornet's GPU NMF backend (`fit_gpu`, `FactorGraph`, `speckled_cv`, `fit_streaming_spz`). **All 13 NMF correctness tests pass as of Cycle 55b**, including the NMF_Graph multi-modal smoke test fixed for the `FactorGraph` constructor + NMF_LossFlavor_NB monotonicity expectation removed.

## 1. Success metrics

On GSM4037629 (HVG top-2000 × 20866 cells) NMF with rank k ∈ {10, 20, 50}:

- **Wall p50**: ≤ `RcppML::nmf()` (R, GPU-naive) × 0.01 (100× headline) AND ≤ `sklearn.decomposition.NMF` × 0.02.
- **Peak dev mem**: ≤ factornet GPU fit_gpu × 1.05 (adapter overhead budget).
- **Loss at convergence**: within 1e-4 relative of factornet CPU NMF (the Cycle 1 correctness reference).
- **Rank selection (Rule 30 novel)**: auto-picked k from Marchenko-Pastur edge within 1 rank of the speckled-CV optimum on 3 reference datasets.

Frontier promotion gate: dominance on wall AND memory, correctness match, Rule 31 autotune delta ≤ 10%.

## 2. SOTA baselines

| Baseline | Language | Notes |
|---|---|---|
| **factornet CPU NMF** | C++ | Correctness reference; wall floor for CPU |
| **factornet GPU NMF (`fit_gpu`)** | CUDA | OUR backend — our adapter wall cannot be slower than this by more than ~1% overhead |
| **RcppML** | R / C++ | R community standard; Zach DeBruine's own lib — we must be strictly better |
| **sklearn `decomposition.NMF`** | Python / scikit-learn | Most widely used NMF in Python |
| **cNMF** (Kotliar et al. 2019) | Python | Consensus-NMF for scRNA; useful for rank-stability comparison |
| **RAPIDS cuml NMF** | cupy | GPU baseline if installed on g001 |

## 3. Bench configurations

Scales:
- **small**: GSM4037629 HVG-2000 × 20k cells, k ∈ {10, 20, 50}
- **medium**: 5-sample concat × 100k cells, k=20
- **large**: 1M synthetic × k=50 (streaming)

Configurations × every scale × every k:

1. `ours_nmf_fit_manual` — explicit factornet adapter (`reduce::nmf::fit_gpu`)
2. `ours_nmf_fit_auto` — Rule 31 autonomy (auto k + auto loss + auto regularization)
3. **`ours_nmf_mp_rank_select`** — Rule 30 §4a (Marchenko-Pastur rank selection)
4. **`ours_factorgraph_hierarchical`** — Rule 30 §4b (hierarchical warm-start)
5. `ours_speckled_cv` — factornet adapter for cross-validated rank selection (the "manual" gold standard for rank selection, against which §4a is compared)
6. `factornet_cpu_nmf` — correctness reference + CPU wall floor
7. `factornet_gpu_nmf` — our adapter's underlying backend; wall floor
8. `rcppml_r` — R subprocess (only at small scale — will be slow)
9. `sklearn_nmf` — Python CPU
10. `cnmf_python` — consensus NMF if installed
11. `cuml_nmf` — if rapids env available (pending ENV-RAPIDS-G001)

Metrics: wall p50, peak dev/host mem, final reconstruction loss, rank selection accuracy (for §4a), hierarchy fit quality (for §4b), throughput.

## 4. Novel pursuit (Rule 30)

### 4a. Marchenko-Pastur rank selection (closed-form — replaces speckled cross-validation)

**The speckled-CV bottleneck**: factornet's `speckled_cv` runs NMF multiple times at increasing ranks with held-out entries, computing out-of-sample reconstruction error to pick the optimal rank. At 20 candidate ranks × 5-fold CV × full NMF per combination, this is ~100 NMF fits. Dominates rank-selection wall by 99%+.

**The Marchenko-Pastur observation**: the singular value spectrum of an `m×n` matrix with iid entries has a known distribution in the bulk, with an edge at `σ_+ = σ × (1 + √(n/m))` for aspect ratio `n/m`. Singular values above the edge are **signal** (non-random structure); values below are noise.

For a real scRNA count matrix, the M-P bulk is estimated from a **randomized reference distribution** built by shuffling each row independently. Then:

```
1. Compute top-K singular values of A (via factornet randomized SVD — single call, ~1 sec).
2. Compute top-K singular values of shuffle(A) — the M-P null reference (~1 sec).
3. The optimal NMF rank = number of A's singular values that exceed the M-P edge of shuffle(A).
```

**Expected wall**: ~2 sec for rank selection vs speckled_cv's ~20 min. **600× headline on rank selection alone.**

**Gate**: within 1 rank of speckled_cv's optimum on 3 reference datasets. Since speckled_cv has its own CV-fold noise (±1 rank is typical), matching within 1 rank is the "indistinguishable from the gold standard" bar.

### 4b. Hierarchical FactorGraph warm-start from parent rank

**The observation**: NMF at rank k+1 typically contains the NMF at rank k as a substructure plus one new factor. factornet's `FactorGraph` already supports hierarchical multi-modal NMF. Add warm-start: when fitting rank k+1, initialize the first k factors from the rank-k fit + randomize the (k+1)th factor.

**Expected benefit**: convergence for rank k+1 in ~30% of the iterations vs cold-start, because the first k factors are already near-optimal. At a rank-sweep over `k ∈ {5, 10, 20, 50, 100}`, sequential warm-start takes `cold_k=5 + 5 × 0.3 × cold_k` ≈ `2.5 × cold_k=5` instead of `5 × cold_k_avg`. **~2× wall reduction on rank sweeps.**

**Gate**: final loss within 1e-4 relative of cold-start at every rank. Wall ≤ cold-start × 0.6.

### 4c. Fast convergence test via gradient norm

Standard NMF stopping criterion: relative change in reconstruction loss below threshold. Requires computing the full loss every iteration (cost: ~10% of an iteration).

**Alternative**: check `||gradient||_F² < ε` directly — the gradient norm is a single reduction across the W/H matrices, much cheaper than recomputing the full loss. Published as standard in optimization literature but not typically used in NMF packages.

**Expected wall**: ~5% reduction on total NMF wall. Incremental but free.

**Gate**: final loss identical to loss-based convergence within fp32 ULP on 3 reference datasets.

## 5. Autonomy pass (Rule 31)

| Config field | Auto |
|---|---|
| `k` (rank) | Auto: Marchenko-Pastur edge detector (§4a), with `speckled_cv` fallback if MP fails on degenerate inputs |
| `loss` | Auto: `mse` for fp32 normalized data, `nb` for raw counts (detect via value distribution) |
| `max_iter` | Auto: 500, with early stop on §4c gradient criterion |
| `tol` | Auto: 1e-5 gradient norm (§4c) |
| `regularization_l1_W` | Auto: 0 unless sparsity hint supplied |
| `regularization_l1_H` | Auto: 0 |
| `init` | Auto: `nndsvd` for cold-start, `warm_start` from parent rank for hierarchical sweeps (§4b) |
| `random_seed` | Auto: 42 |

No-args `reduce::nmf::fit(matrix)` returns `NmfResult{W, H, final_loss, selected_rank, convergence_trajectory}` with every dial auto.

## 6. OOC streaming contract

Factornet already provides `fit_streaming_spz` for out-of-core NMF on `.spz`. Our adapter:

1. Wrap the streaming driver (feature 16) to feed chunks into the factornet streaming fit.
2. Provide an adapter class `reduce::nmf::StreamingStage` that hooks into feature 16's `StreamedPipeline`.
3. Maintain running W + H accumulators + chunk-local convergence per factornet's API.

Factornet handles the algorithmic streaming; we're just the I/O adapter. Ship when feature 16 lands.

## 7. Determinism contract

- Cold-start NMF is deterministic with fixed seed + deterministic reductions (segmented scan).
- Warm-start is deterministic once the parent is fixed.
- Marchenko-Pastur rank selection is seeded via the row-shuffle seed.
- Speckled-CV is seeded via fold-assignment seed.

## 8. Phase E dispatch spec

Dispatch two workers in parallel:

**Worker A (gpu-kernel-dev)**: implement the MP rank-selection kernel (§4a, ~50 LOC — just two randomized SVD calls + an element-wise comparison), implement the hierarchical warm-start path in `FactorGraph` adapter (§4b, ~30 LOC), implement the gradient-norm convergence test (§4c, ~20 LOC — a cub reduction).

**Worker B (gpu-bench)**: run the 11-config × 3-scale × 3-rank bench. Compute rank-selection accuracy on 3 reference datasets for §4a. Measure warm-start vs cold-start wall reduction on a rank sweep for §4b. Write to benchmark-registry / pareto-frontier / novel-attempts.

## 9. Open questions

- **Marchenko-Pastur on biological data**: the MP null assumes iid entries, which real scRNA count matrices violate (gene-wise heterogeneity, cell-wise library-size differences). The shuffle-reference builds an empirical null that respects these structures — but does it give the same optimal rank as speckled_cv? Empirical validation on 3 reference datasets is the gate.
- **factornet hierarchical warm-start**: does `FactorGraph::fit()` support a warm-start initialization, or do we need to patch the interface? Verify by reading `factornet/graph/factor_graph.cuh` (via code-reader if needed — one file only).
- **NMF k=1 and k=2 edge cases**: MP edge is tight at low ranks; may produce k=0 (no structure) which is uninformative. Default to min k=5 unless overridden.
- **sklearn NMF correctness match**: sklearn uses multiplicative update by default; factornet uses HALS. Final loss may differ up to ~5% due to different convergence basins. Document.

## 10. Links

- Original: `state/designs/05-nmf-adapters.md`
- factornet NMF: `/mnt/home/debruinz/factornet/include/factornet/nmf/{fit_gpu,graph,speckled_cv,streaming}.cuh`
- RcppML: https://github.com/zdebruine/RcppML (user's own lib — strict "must-beat" baseline)
- Marchenko-Pastur distribution: Marchenko & Pastur 1967
- cNMF: Kotliar et al. 2019
- Halko 2011 randomized SVD (shared with 04-svd-phaseE)
