---
feature: 7
module: graph/leiden.h (+ graph/louvain.h)
cycle: 62 or later
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy pass
extends: 07-leiden.md
status: draft
depends_on: feature 6 (kNN/SNN adjacency CSR on device)
---

# Feature 7 — Leiden + Louvain Phase E frontier push

Feature 7 owns graph-based clustering. Per CLAUDE.md roadmap the goal is "beat cuGraph Leiden by ≥2× at ≥10M edges". This doc specs that push and a much higher-leverage novel-pursuit: **one-shot multi-resolution sweep** that eliminates the need for users to run Leiden 5–10× at different resolutions.

## 1. Success metrics

On GSM4037629 PCA-50 → kNN(k=15) SNN(Jaccard prune) → Leiden at resolution r ∈ {0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0}:

- **Wall p50 (single resolution, r=0.8)**: ≤ cuGraph `leiden` wall × 0.5 at ≥10M edges.
- **Wall p50 (multi-resolution sweep over 8 resolutions)**: ≤ cuGraph single-resolution wall × 1.5 — i.e. the sweep costs almost nothing extra once one resolution is computed.
- **Modularity match**: within 1e-3 of cuGraph at every resolution (same partition up to label permutation).
- **ARI vs igraph Leiden (CPU gold standard)**: ≥ 0.98 at every resolution.

Frontier promotion gate: **dominance on the multi-resolution sweep wall** AND correctness match AND Rule 31 autonomy path within 10% of manual.

## 2. SOTA baselines

| Baseline | Path | Notes |
|---|---|---|
| **cuGraph `leiden`** | RAPIDS cuGraph CUDA | Primary GPU baseline; rapids-singlecell default |
| **cuGraph `louvain`** | RAPIDS cuGraph | Secondary GPU baseline |
| **leidenalg (igraph, CPU)** | Python / C | Gold-standard Leiden reference |
| **scanpy `tl.leiden`** | calls leidenalg | scanpy default — CPU bottleneck |
| **Seurat `FindClusters`** | R / C++ Louvain | R gold standard |
| **libleidenalg pre-compiled Louvain** | C++ | Pure-CPU highest-performance Louvain |

## 3. Bench configurations

Scales:
- **small**: GSM4037629 PCA-50 kNN(15) → ~313k edges, single resolution
- **medium**: 5-sample concat ~100k cells kNN(15) → ~1.5M edges, 8-resolution sweep
- **large**: 1M synthetic cells kNN(15) → ~15M edges (this is the "10M edge" threshold cycle log mentions), 8-resolution sweep

Configurations:

1. `ours_leiden_single` — classic single-resolution Leiden
2. `ours_louvain_single`
3. **`ours_leiden_sweep`** — one-shot multi-resolution sweep (Rule 30 novel, §4)
4. `ours_leiden_auto` — Rule 31 autonomy: auto-picks resolution via modularity-stability criterion
5. `cugraph_leiden` (per-resolution separately, summed wall for the sweep)
6. `cugraph_louvain`
7. `leidenalg_igraph`
8. `scanpy_leiden`
9. `seurat_findclusters` — R subprocess, small scale only

Metrics: wall p50, peak dev mem, modularity Q per resolution, cluster count per resolution, ARI vs gold standard, throughput (edges/sec).

## 4. Novel pursuit (Rule 30) — one-shot multi-resolution sweep

### The observation

Users almost never run Leiden at a single resolution. The standard workflow is:

1. Run Leiden at resolution 0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0 (or similar grid).
2. Inspect cluster counts + silhouette + biological markers.
3. Pick the resolution where cluster count is "stable" (small resolution change → small cluster change).

This is **modularity-stability clustering** (Lambiotte et al. 2008): the best resolution is the one at which the partition is most robust to small resolution perturbations. Published implementations do this by running Leiden N times, which is obviously wasteful — there is enormous shared work across resolutions.

### The fused sweep algorithm

```
Input: CSR adjacency A, resolution grid R = [r_1, ..., r_n]
Output: One partition per resolution, modularity per resolution, stability score per resolution

1. Run Leiden at the middle resolution r_mid to build an initial partition P_mid.
2. For each r in R sorted by distance from r_mid (nearest first):
     Warm-start from the neighboring already-computed partition P_neighbor.
     Run Leiden's local-move phase only (skip the expensive initial aggregation)
       with the Constant Potts Model objective parameterized by r.
     Partition P_r converges in 1–3 local-move passes instead of the typical 10+.
3. After all resolutions are done, compute the stability score:
     stab(r) = ARI(P_r, P_{r+dr}) averaged over a small dr
   The resolution with maximum stability is the auto-picked one.
```

**Shared-work ratio**: Leiden's wall is dominated by the local-move phase. Warm-starting from a nearby-resolution partition typically needs 1–3 passes vs 10+ from scratch. At 8 resolutions, the sweep cost is ~`1 initial full run + 7 × (2–3 warm passes)` = roughly `1 + 7*0.25` = `2.75× single-resolution wall`. Compared to running 8 separate Leidens (`8× single-resolution wall`), this is a **~3× wall reduction for the same 8-resolution output**.

**Memory**: all 8 partitions stored simultaneously as uint32 labels — 8 × n_cells × 4 bytes = 640 KB for 20k cells, 32 MB for 1M cells. Negligible vs the adjacency matrix.

### Success gates

- Multi-resolution sweep at 8 resolutions on the medium-scale benchmark: wall ≤ `3 × cuGraph single-resolution wall`. (cuGraph running all 8 sequentially would be `8× single` so we win if we're anywhere under that and hit our own gate.)
- Modularity per resolution: within 1e-3 of cuGraph's single-resolution Leiden run at the same resolution.
- Auto-picked resolution: matches the one a human would pick via cluster-count stability, verified on 3 reference samples.
- On the small scale (313k edges), the sweep wall must beat `cuGraph single × 2` — if we can't amortize on small graphs, the sweep is only a medium/large-scale optimization.

If gates pass, the sweep API becomes `leiden::sweep(graph, resolution_grid)` returning a `LeidenSweepResult` with all partitions + modularity + stability scores + the auto-picked best index. The single-resolution API stays as `leiden::run(graph, resolution)`. Failed gates → log in novel-attempts.md with warm-start convergence trace.

### Additional novel variants (stretch goals for subsequent cycles)

#### 4b. Deterministic Leiden via segmented-scan local-move

cuGraph's Leiden is non-deterministic via thread-order dependent local moves. Provide an opt-in deterministic path using segmented-scan for the `Δmodularity` computation and stable ordering of cells within each iteration. Expected cost: 1.5× wall. Useful for reproducibility-critical workflows.

#### 4c. Small-world-aware initial partition

Most real scRNA SNN graphs have a small-world topology (high local clustering, low average path length). Exploit this by using connected-components as the initial partition instead of singleton cells — saves ~30% of the initial aggregation wall.

#### 4d. Resolution-free clustering via Map Equation (Infomap on GPU)

The Map Equation (Rosvall 2008) avoids the resolution hyperparameter entirely. Infomap is normally CPU-only because of the random-walk simulation. A GPU variant using batched random walks + warp-level flow accumulation could match Infomap's output at competitive wall. Defer as a future cycle — too large for Phase E.

## 5. Autonomy pass (Rule 31)

| Config field | Auto-tune strategy |
|---|---|
| `resolution` | Auto: run `leiden::sweep` over `[0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0]`, return the max-stability resolution. |
| `algorithm` | Auto: Leiden for `n_cells ≤ 1M`, Louvain for larger (Leiden's guaranteed-connectivity invariant is less important at scale; Louvain is faster). |
| `n_iterations` | Auto: run until no cell changes partition, bounded at 20. |
| `random_seed` | Auto: `42` (documented) unless user pins. |
| `max_comm_size` | Auto: unbounded unless the user supplies. |
| `quality_function` | Auto: CPM for Leiden, modularity for Louvain. |

No-args `leiden::cluster(graph)` returns a partition with the auto-picked resolution + the full sweep result as a side product (users get both for free).

## 6. OOC streaming contract

Leiden is inherently graph-local — the entire adjacency must be resident for the local-move phase to work correctly. OOC options:

- **Block-wise graph partitioning**: cut the graph into k components using METIS + run Leiden per-block + merge. Approximate but scales.
- **Edge streaming**: stream edges from a `.1pz` adjacency file; Leiden's initial pass accumulates the k-core approximation in O(edges) time and O(cells) memory. Subsequent phases still need full residency.
- **Pair-ish resolution sharding**: run different resolutions on different GPUs. Not streaming, but scale-out.

Document; implement with feature 16.

## 7. Determinism contract

- Default: non-deterministic via thread-order local moves.
- Opt-in deterministic: segmented-scan Δmodularity + stable cell ordering (§4b). 1.5× wall cost.
- Reproducibility seed always exposed as `random_seed` config field, even in non-det mode — it controls initial partition assignment and tie-break order.

## 8. Phase E dispatch spec

Dispatch `gpu-bench` with:
- Read 07-leiden.md + this doc.
- Implement the `leiden::sweep` prototype with warm-starting.
- Run the 9-config × 3-scale × 8-resolution bench table.
- Verify modularity match against cuGraph per-resolution.
- Verify ARI match against igraph Leiden on small scale.
- Write to benchmark-registry.md, pareto-frontier.md (if promoted), novel-attempts.md.

## 9. Open questions

- **cuGraph Leiden quality**: cuGraph's Leiden historically had quality issues vs igraph (see 2021 benchmarks). Quality has improved, but verify on fresh benchmarks — if cuGraph partitions are ARI < 0.9 vs igraph, our correctness gate must be anchored on igraph not cuGraph.
- **Warm-start convergence bound**: the "1–3 passes" estimate is empirical. The theoretical bound is that warm-started Leiden reaches a local optimum in O(n × Δr × avg_degree) work where `Δr` is the resolution gap. For small `Δr` this is tight; large `Δr` means we're back to from-scratch cost.
- **Does Louvain benefit from the sweep approach as much as Leiden?** Louvain is simpler but more prone to bad-quality partitions. Sweep on Louvain may amplify quality issues. Start with Leiden-only for the novel pursuit.

## 10. Links

- Original: `state/designs/07-leiden.md`
- cuGraph Leiden: https://github.com/rapidsai/cugraph
- Traag, Waltman & van Eck 2019 Leiden paper
- Lambiotte et al. 2008 modularity-stability framework
- Rosvall 2008 Map Equation (for the §4d defer)
