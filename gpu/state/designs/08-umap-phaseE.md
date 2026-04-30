---
feature: 8
module: embed/umap.h
cycle: 63 or later
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy pass
extends: 08-umap.md
status: draft
depends_on: feature 6 (kNN graph on device), feature 4 (PCA for init)
---

# Feature 8 — UMAP Phase E frontier push

UMAP is the dominant visualization wall time in every scRNA pipeline. A 3× UMAP win is visible to every user on every dataset. Per Mandate v2, UMAP is a Rule-30-rich target: the literature uses adaptive-SGD cross-entropy optimization of a loss whose structure admits much better algorithms.

## 1. Success metrics

On GSM4037629 PCA-50 → kNN(k=15) → UMAP(n_components=2, n_epochs=auto):

- **Wall p50**: ≤ `rapids-singlecell umap_gpu` × 0.5.
- **Peak dev mem**: ≤ rapids-singlecell.
- **Correctness**: **trustworthiness ≥ 0.95** vs ground-truth brute-force embedding on the same graph, **continuity ≥ 0.95**, **k-neighbor preservation ≥ 0.90**, **ARI on Leiden partitions of the embedding-derived kNN graph ≥ 0.90** vs the input kNN partition (topology preservation smoke test).

Note: pixel-identity is NOT a valid UMAP correctness metric — every UMAP implementation produces a different rotation/scale. The metrics above are the only defensible correctness checks.

Frontier promotion gate: wall dominance by ≥30%, correctness match on ALL four topology metrics, Rule 31 autonomy within 10% of manual.

## 2. SOTA baselines

| Baseline | Path | Notes |
|---|---|---|
| **cuml UMAP** | rapids-singlecell / CUDA | Primary GPU baseline; used in rapids-singlecell |
| **umap-learn** | Python / numba | Original McInnes implementation, CPU |
| **Seurat `RunUMAP`** | R wraps uwot | R gold standard |
| **pynndescent + umap-learn** | CPU | scanpy default for small data |
| **tSNE-CUDA (for trust comparison)** | CUDA | Different algorithm but same task; sanity-check trustworthiness |

## 3. Bench configurations

Scales:
- **tiny**: 500 × 50, k=15 (smoke)
- **small**: GSM4037629 PCA-50 kNN(15) (20,866 cells) (primary comparison)
- **medium**: 5-sample concat ~100k cells (scale check)
- **large**: 1M synthetic (scale limit)

Configurations:

1. `ours_adam_sgd` — classic McInnes SGD + Adam (reference port, the baseline we need to beat with novel variants)
2. **`ours_riemannian`** — Rule 30 stochastic Riemannian optimization (§4a)
3. **`ours_fixed_graph_batched`** — Rule 30 fixed-graph exploitation (§4b)
4. **`ours_fused_everything`** — the full novel stack combining 4a + 4b + 4c
5. `ours_auto` — Rule 31 autonomy path
6. `cuml_umap`
7. `umap_learn`
8. `seurat_umap` (R subprocess, small scale only)

Metrics: wall p50, peak dev mem, trustworthiness, continuity, k-neighbor preservation, ARI vs input Leiden partition, throughput (cells/sec).

## 4. Novel pursuit (Rule 30)

### Observation

McInnes UMAP optimizes a cross-entropy loss on a **fixed weighted adjacency graph** using per-edge stochastic gradient descent with a linearly decaying learning rate. This has three properties the literature under-exploits:

1. The graph is fixed. Every "epoch" visits the same edges. So the sampling schedule is deterministic and cacheable.
2. The loss is a sum of per-edge terms on a 2D manifold (the embedding). 2D Riemannian metric gradients are closed-form and stable.
3. The negative sampling is O(n) per epoch but produces extremely noisy gradients because random negatives are usually far from the query point anyway. Most negative gradients contribute ~0 to the loss.

### 4a. Stochastic Riemannian cross-entropy optimization

Replace Adam-SGD with **natural gradient descent on the 2D Euclidean manifold** (trivial in 2D: the Fisher metric is the identity up to a local scale factor that can be absorbed into the learning rate). The key insight: instead of per-edge update with decaying η, use a **momentum-corrected average gradient** over a mini-batch of edges with provably faster convergence.

Provable convergence rate: `O(1/t)` for convex 2D cross-entropy, vs `O(1/√t)` for vanilla SGD. In practice, this means the typical 500-epoch UMAP converges to the same objective value in **~150 epochs**.

**Gate**: ≤ 40% of `ours_adam_sgd` epoch count for matching trustworthiness. Wall time follows linearly.

### 4b. Fixed-graph exploitation via persistent kernels

Since the graph is fixed, load it into shared memory ONCE per block and do all epoch iterations from smem without re-fetching edge weights from global memory. Only the 2D coordinates update.

This is a **persistent-kernel** pattern: launch once, loop over all epochs inside the kernel, cooperate across blocks via grid-scope synchronization (cooperative groups). Eliminates ~N_epoch × N_edge global memory reads of edge weights.

**Gate**: ≥ 2× throughput vs cuml UMAP at the same epoch count.

**Risk**: cooperative groups requires occupancy that fits the whole graph in smem + register space. For >50k cells the graph won't fit; fall back to block-local persistence with global-memory sync barriers.

### 4c. Importance-sampled negatives

Replace uniform random negatives with **importance sampling from the kNN graph's 2-hop complement** (cells that are NOT in the direct kNN but share a neighbor). These are the informative negatives — random points in a 2D plane contribute almost no gradient signal because they're already maximally far from the positive cluster.

Computed once during graph setup: for each cell, store the top-10 "hard negatives" (2-hop neighbors with low direct similarity). Use these during the negative sampling phase instead of random points.

**Gate**: ≥ 30% faster convergence at equal trustworthiness. Equal peak device memory (hard negatives fit in `n × 10 × uint32` = 800 KB for 20k cells).

### 4d. Initial embedding via PCA top-2 (instead of spectral)

cuml UMAP and umap-learn default to **spectral initialization** (Laplacian eigenmaps). This is expensive (it runs a full Lanczos eigensolver) and typically accounts for ~15% of UMAP wall time.

Alternative: initialize from the PCA top-2 components of the input embedding. Convergence is slightly slower (2–5 more epochs) but initialization is essentially free (the PCA is already computed upstream). Net wall: **save ~10% of total UMAP time**.

Ship both options; auto-pick PCA for `n_cells ≥ 5000` (where spectral becomes expensive) and spectral for `n_cells < 5000` (where the PCA-init convergence penalty dominates).

### Combined "fused everything" stack

`ours_fused_everything` applies 4a + 4b + 4c + 4d simultaneously. Expected wall on GSM4037629: `cuml_umap × 0.3`, correctness match on all four topology metrics.

If the combined stack hits the gate: it becomes the primary UMAP path. If only a subset of variants pass gates, compose the winning subset. Log all failed variants in novel-attempts.md.

## 5. Autonomy pass (Rule 31)

No-args `umap::embed(graph)` returns a 2D embedding with every dial auto.

| Config field | Auto-tune strategy |
|---|---|
| `n_components` | Auto: 2 (only useful value for visualization). Opt-in 3 for 3D plots. |
| `n_neighbors` | Auto: from kNN graph (already supplied); fallback 15 if the user passed raw embedding. |
| `min_dist` | Auto: 0.1 for `n_cells ≤ 10k`, 0.3 for larger (larger datasets need more spread). |
| `spread` | Auto: 1.0 (McInnes default, rarely needs changing). |
| `n_epochs` | Auto: Cauchy-schedule — run until the coordinate change per epoch drops below 1e-3 × dataset radius. Bounded at 500. |
| `learning_rate` | Auto: 1.0 for natural gradient (§4a), 1.0 with cosine decay for Adam-SGD. |
| `init` | Auto: PCA top-2 for `n_cells ≥ 5000`, spectral otherwise (§4d). |
| `negative_sample_rate` | Auto: 5 for Adam-SGD, 3 for importance-sampled (§4c — fewer negatives needed because each is informative). |
| `random_seed` | Auto: `42` unless user pins. |
| `metric` | Auto: euclidean (only useful metric post-kNN). |

## 6. OOC streaming contract

Large UMAP (>10M cells) needs:
- Graph streaming from disk via `.1pz` adjacency format.
- Block-wise coordinate updates where each block owns a subset of cells and communicates boundaries via push-pull pattern.
- Typically done as "batched UMAP" in rapids-singlecell: chunk the cells, run UMAP per chunk, align chunks via Procrustes. Approximate but scales.

Document; defer until streaming driver lands.

## 7. Determinism contract

UMAP is inherently non-deterministic via:
- Random negative sampling order (thread launch order).
- SGD update order on shared edges.

Opt-in deterministic path: fixed negative sample indices (precomputed from seed), segmented-scan edge updates. ~2× wall cost. Seed always exposed.

## 8. Phase E dispatch spec

Dispatch `gpu-bench` with:
- Read 08-umap.md + this doc.
- Implement `ours_adam_sgd` first as the baseline.
- Implement the 4a/4b/4c/4d prototypes in the bench driver.
- Implement the `ours_fused_everything` combo.
- Run the 8-config × 4-scale benchmark with all 4 correctness metrics.
- Write to benchmark-registry.md, pareto-frontier.md (if promoted), novel-attempts.md.

## 9. Open questions

- **How does cuml UMAP handle the graph input?** It internally does kNN+fuzzy-simplicial-set construction from the raw embedding. Our API should take the pre-computed kNN graph from feature 6 directly — this is an architectural difference to document.
- **Trustworthiness calculation wall time**: computing trustworthiness for a 20k-cell benchmark is O(n × k × log n) and may dominate the bench wall. Cache ground-truth embedding + kNN once per scale.
- **Persistent kernel occupancy ceiling**: at what n_cells does the graph stop fitting in smem? Measure during implementation — probably ~10k-20k cells for a 15-NN graph (150k edges × 8 bytes = 1.2 MB, within smem). For the 100k-cell medium scale, persistence is block-local only.
- **Is natural gradient actually O(1/t) in the non-convex UMAP loss?** The UMAP cross-entropy IS non-convex (multiple local minima for cluster placement). The theoretical bound is O(1/√t) for non-convex + smooth, same as SGD. Natural gradient may only win via better conditioning not rate. Measure, don't assume.

## 10. Links

- Original: `state/designs/08-umap.md`
- cuml UMAP: https://github.com/rapidsai/cuml/tree/main/cpp/src/umap
- umap-learn: https://github.com/lmcinnes/umap
- McInnes, Healy & Melville 2018 UMAP paper
- Hinton & Roweis 2003 SNE (the theoretical grandparent of UMAP's loss)
- Böhm 2022 "Attraction-repulsion spectrum in neighbor embeddings" — theoretical framework for the 4a natural gradient claim
