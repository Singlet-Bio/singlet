---
feature: 6
module: graph/knn.h (+ graph/ann.h, graph/snn.h planned)
cycle: 61 or later
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy pass
extends: 06-knn.md
status: draft
mandate_v2_ref: §C (ANN + SNN — the most-optimized module)
---

# Feature 6 — kNN (ANN + SNN) Phase E frontier push

Feature 6 owns graph construction. Per Mandate v2 §C this is **the most-optimized module in the library** because every downstream graph-family kernel (Leiden, Louvain, UMAP, label transfer, scVI minibatch sampler, BBKNN integration) inherits its wall-time and memory cost. kNN wins here compound into wins everywhere above.

The feature splits into two kernel families that share the same CSR adjacency output:

1. **ANN** (`graph/ann.h`): brute-force fp16 → IVF-Flat → IVF-PQ → HNSW-GPU with single `ann::build(emb, k, params)` entry and auto-selection.
2. **SNN** (`graph/snn.h`): Seurat `FindNeighbors`-equivalent — kNN → Jaccard overlap pruning → weighted CSR adjacency. Zero host transfer between ANN result and SNN construction.

This doc specs both in a single Phase E because they must share configuration and a single auto-tune path.

## 1. Success metrics

On GSM4037629 PCA-50 embedding (20,866 cells × 50 dims) at k ∈ {15, 30, 100}:

- **Wall p50**: ≤ best of {cuml `NearestNeighbors`, RAFT brute_force_knn, FAISS-GPU IVF-Flat, pynndescent CPU}.
- **Peak dev mem**: ≤ cuml `NearestNeighbors`.
- **Correctness (recall@k)**: ≥ 0.99 vs exact brute-force ground truth at k=15 and k=30. k=100 tolerated at ≥ 0.98.

At 1M-cell scale (synthetic concat): wall ≤ FAISS-GPU IVF-PQ, recall@15 ≥ 0.95.

Frontier gate: dominance on wall AND (recall ≥ 0.99 at small k OR recall ≥ 0.95 at medium k).

## 2. SOTA baselines

| Baseline | Path | Notes |
|---|---|---|
| **cuml `NearestNeighbors`** | cuVS / RAFT | rapids-singlecell default |
| **RAFT brute_force_knn** | RAFT CUDA | pure brute force baseline |
| **FAISS-GPU IVF-Flat** | FAISS | industry IVF baseline |
| **FAISS-GPU IVF-PQ** | FAISS | medium-scale PQ compression baseline |
| **HNSWlib (CPU)** | C++ | Reference for recall — HNSW is the published SOTA for recall-vs-wall tradeoff on CPU |
| **pynndescent** | Python | scanpy's default CPU approximate NN |
| **Annoy** | C++ | older CPU reference — often still used in Seurat workflows |

## 3. Bench configurations (Cycle 61 dispatch)

Scales:
- **tiny**: 500 cells × 50 dims, k=10 (smoke)
- **small**: 20,866 cells × 50 dims (GSM4037629 PCA-50), k ∈ {15, 30, 100}
- **medium**: 100,000-cell concat × 50, k ∈ {15, 30, 100}
- **large**: 1M synthetic cells × 50, k=15 (scaling limit smoke, only the winning configs)

Configurations × every scale × every k:

1. `ours_brute_force_fp16`
2. `ours_ivf_flat`
3. `ours_ivf_pq`
4. `ours_hnsw_gpu`
5. `ours_auto` (Rule 31 — picks by scale, dim, k)
6. `ours_snn_jaccard` (layered on each ANN above; time SNN alone)
7. `cuml_nearest_neighbors`
8. `raft_brute_force`
9. `faiss_gpu_ivf_flat`
10. `faiss_gpu_ivf_pq`
11. `hnswlib_cpu`
12. `pynndescent_cpu`

Metrics: wall p50, peak dev/host mem, recall@k vs exact brute-force ground truth (computed once per scale and cached), throughput (cells/sec), SNN-construction-only wall (for the SNN-layered rows).

## 4. Novel pursuit (Rule 30) — the scRNA-tuned IVF-PQ + implicit refinement

### The observation

scRNA kNN has a specific shape the literature does not target well:

- **d (embedding dim) is tiny**: almost always ≤ 200, usually 50. FAISS and HNSW target d ≥ 512 and have not been micro-tuned for small d. Shared-memory blocking strategies that assume large d are suboptimal here.
- **k is small**: usually 15–100. PQ codebook size 256 is wasteful at k=15.
- **Embeddings are low-rank**: they came from PCA, so there's residual structure a general ANN method does not exploit.
- **Recall-at-k is what matters downstream (Leiden/UMAP), not ordered distances**: we only need the SET of k neighbors, not their precise ranking.

### Novel variants

#### 4a. Small-d IVF-PQ with fp16 codebook fit in registers

- d ≤ 64: PQ subvectors fit entirely in registers. No shared-memory table lookup — directly store the codebook in register-resident fp16 and do warp-shuffle broadcast.
- Expected win: ~2× over FAISS-GPU IVF-PQ at d=50 because the shared-memory LUT stall is the FAISS bottleneck.
- Risk: register pressure caps occupancy. Verify with Nsight before full implementation.

#### 4b. PCA-conditioned Voronoi cells (IVF with PCA-aware centroid placement)

- Standard IVF uses k-means on the raw embedding to pick `nlist` cells. On a PCA embedding, the first few components have a Gaussian profile with known variance-scale ratios — pick centroids from a deterministic lattice on the top-3 PC axes instead of k-means.
- Benefit: **zero training time** (deterministic centroids), cell assignments are closed-form (`argmin_c ||x_projected - c||²` on the top-3 PC axes only).
- Expected win: eliminates the IVF training phase entirely (typically 20–40% of total wall time at build time), AND gives deterministic output across runs.
- Caveat: only works because the input embedding is a PCA. Document as "PCA-aware-only" mode and fall back to k-means IVF when the user passes a non-PCA embedding.

#### 4c. Implicit refinement via Cycle-57 feature 0 matrix

- After ANN returns approximate neighbors, we have the option to refine distances using the raw count matrix (not the PCA). The raw counts give a more cell-biology-faithful distance than the PCA embedding.
- Novel: run approximate ANN on the 50-dim PCA for candidate selection, then refine distances on a tiny candidate set (candidates × raw CSC column dot products, streaming through `singlet_gpu::io::load()`).
- Expected win: **recall as good as brute force on raw counts, wall time as good as approximate ANN on PCA**. Best of both worlds.
- Risk: the candidate set must be small (e.g. 5k per cell) or the refinement cost exceeds the ANN savings. Cache the raw-count-column dot products in a device-resident LRU of hot cells.

#### 4d. Single-pass SNN construction (fused into ANN)

- Standard pipeline: ANN returns (n × k) neighbor indices, then SNN kernel computes Jaccard overlap by sorting + comparing k×k pairs per cell.
- Novel: **fuse Jaccard computation into the ANN hit-list assembly.** When cell A's kNN is being computed, we already know cell B is a neighbor; simultaneously accumulate the edge weight `|N(A) ∩ N(B)| / |N(A) ∪ N(B)|` in a single pass using bitsets (k ≤ 128 fits in 2 × uint64).
- Expected win: eliminates the SNN construction pass entirely — SNN becomes free on top of ANN.
- Caveat: requires knowing target ANN hits before hand, which rules out pure ANN. Works for brute-force and IVF variants where the candidate set is enumerated.

### Success gates for the novel cluster

- On small-real at k=15: our auto variant beats cuml `NearestNeighbors` wall by ≥ 30% AND recall@15 ≥ 0.99.
- On medium at k=30: our auto variant beats FAISS-GPU IVF-PQ wall by ≥ 50% AND recall@30 ≥ 0.98.
- Fused ANN+SNN wall ≤ standalone ANN wall × 1.1 (SNN is essentially free).
- PCA-conditioned Voronoi eliminates ≥ 90% of IVF training wall time in the PCA embedding case.

Gate-failure logging: per Rule 30, failed variants append to `state/novel-attempts.md` with the residual trajectory.

## 5. Autonomy pass (Rule 31)

Everything auto-tunes. The no-args `ann::build(emb, k)` returns a CSR adjacency with every dial picked on-device.

| Config field | Auto-tune strategy |
|---|---|
| `backend` | Auto: `(cells ≤ 50k, d ≤ 128) → brute_force_fp16`; `(50k–500k) → ivf_flat`; `(500k–5M) → ivf_pq_smalld`; `(5M+) → hnsw_gpu`. Also check the recall@k gate and re-pick if needed. |
| `k` | No default — user-supplied (this is the one dial we refuse to auto-tune; it has no universal optimum). |
| `nlist` (IVF) | Auto: `max(16, sqrt(n_cells))`. |
| `nprobe` (IVF) | Auto: `max(4, nlist / 64)`, bounded by target recall. |
| `pq_m` (IVF-PQ subvector count) | Auto: `d / 8` capped at 32. |
| `pq_bits` (bits per subvector code) | Auto: 8 (256-entry codebook), downgraded to 4 when `n_cells > 10M`. |
| `hnsw_M` | Auto: 16 (HNSW default). |
| `hnsw_ef_construction` | Auto: 200. |
| `hnsw_ef_search` | Auto: `max(k × 2, 50)`. |
| `metric` | Auto: `euclidean` for PCA embeddings, `cosine` for raw count vectors, detected from input variance signature. |
| `prune.SNN` threshold | Auto: 1/15 (Seurat default) when k ≥ 15; else 0. |
| `fuse_snn` | Auto: true for brute_force + ivf_flat; false for ivf_pq (PQ distances are approximate and the fused SNN Jaccard would be noisy). |

## 6. Out-of-core streaming (Rule 14)

Billion-cell kNN is dominated by the need to keep the full embedding resident. Options:

- **HNSW with memory-mapped graph**: the HNSW layered graph can live on-disk-backed host pinned memory with device queries. ~2× wall hit but unlimited scale.
- **IVF-PQ with on-device codebook + host-side inverted lists**: lists stream on demand. Works up to the point where host RAM limits.
- **Block-wise brute-force**: chunk the embedding cells into groups of 10k, iterate, maintain a per-query top-k heap with min-heap replacement. O(n²) wall but fully streamable with O(k) memory per cell.

Document the contract; implement when feature 16 lands.

## 7. Determinism contract

- Brute-force is deterministic by construction (no ties at fp32 unless two cells have bit-identical coordinates — document the tie-break as cell-id-ascending).
- IVF-Flat and IVF-PQ are non-deterministic via atomicAdd accumulators in the PQ distance. Opt-in determinism via segmented scan.
- HNSW-GPU is non-deterministic via the thread-ordering of layer-zero insertion. Seed-controlled pseudo-deterministic path available via serial single-thread insertion (slow, opt-in only).
- SNN construction is deterministic once the ANN output is pinned.

## 8. Phase E dispatch spec (Cycle 61)

Dispatch `gpu-bench` with:
- Read 06-knn.md + this doc.
- Implement the small-d IVF-PQ prototype in the bench driver.
- Implement the PCA-conditioned Voronoi prototype.
- Implement the fused ANN+SNN Jaccard bitset path for brute-force + ivf_flat.
- Run the 12-config × 3-scale × 3-k bench table.
- Write outputs to benchmark-registry.md, pareto-frontier.md (if promoted), novel-attempts.md.

## 9. Open questions

- **Does RAFT already ship small-d optimized IVF-PQ?** cuVS v25+ might. Check via code-reader before re-implementing.
- **Recall benchmark methodology**: cuml and FAISS use different ground-truth definitions. Pick one and use it consistently (suggest: exact brute force on fp32 euclidean).
- **HNSW-GPU availability**: the main HNSW-GPU implementation is in RAFT / cuVS. If not available, our HNSW-GPU is a novel contribution (rare — HNSW is harder on GPU than it looks).
- **The 4c refinement idea depends on the raw count matrix being loaded per-cell on demand.** Requires a fast column-access mode on `DeviceCSC` that Cycle 56's loader does not optimize for. File as a follow-up for the streaming driver.

## 10. Links

- Original: `state/designs/06-knn.md`
- Mandate v2 §C: `CLAUDE.md` search for "ANN + SNN"
- cuml NearestNeighbors: `https://github.com/rapidsai/cuml`
- FAISS-GPU IVF-PQ reference: Jeff Johnson et al. 2017
- HNSW reference: Malkov & Yashunin 2016
