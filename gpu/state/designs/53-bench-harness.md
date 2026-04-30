---
feature: bench_harness_cycle_53
roadmap_id: 53
module: bench/*.cpp + bench/CMakeLists.txt + bench/refs/*.py + bench/run_all.sh
status: design
tolerance: N/A — benchmark harness builds the framework, not the numbers
target_perf: at GPU dispatch, each feature runs its bench at tiny/10k/100k scales with timings
ooc_plan: N/A — bench harness itself doesn't stream, it exercises the kernels
---

## Why this exists

CLAUDE.md §9 requires: *"Every feature ships with a benchmark driver that runs at three scales using real sample concatenations from quant/: **Small**: 10k cells, **Medium**: 100k cells, **Large**: 1M+ cells."*

**Current state**: after 40 features + 50 cycles, the `bench/` directory is effectively empty. We have exhaustive correctness harnesses in `tests/` (`*_correctness.cpp`) but no `bench/` drivers logging `{wall_ms, peak_mem_mb, cells_per_s, sota_wall, ratio}` per `state/benchmark-registry.md`. Every cycle has marked "bench: SKIPPED (no nvcc)" but nobody has actually built the driver infrastructure.

Cycle 53 fills this gap: builds the bench harness framework, writes drivers for the most important features, and seeds `benchmark-registry.md` with placeholder rows that will be populated on first GPU dispatch.

## What gets built

### Framework (bench harness core)
- `bench/include/singlet_gpu/bench/harness.h` — header-only bench utility:
  - `BenchTimer` class wrapping cuEvents for accurate GPU wall timing.
  - `PeakMemoryTracker` using `cudaMemGetInfo` before/after each kernel run.
  - `throughput(n_cells, wall_ms)` helper.
  - `BenchRow` struct matching `state/benchmark-registry.md` schema.
  - `log_row(BenchRow)` writes to both stdout + `state/benchmark-registry.md` (appends).

- `bench/refs/common.py` — shared Python reference launcher (subprocess invocation of SOTA tools with common timing interface).

### Per-feature bench drivers

One `bench_{feature}_perf.cpp` per major feature. Total ~25 drivers. Each driver:
1. Loads the canonical sample (GSM4037629 for 10k scale; 5-sample concat for 100k; streaming for 1M).
2. Runs the GPU kernel with a fixed seed, warms up with 3 discarded iters, times 5 real iters, reports min/median/max.
3. Runs the SOTA reference via subprocess (scanpy/scran/cellrank/etc.) and times it.
4. Computes ratio, logs a `BenchRow` to `state/benchmark-registry.md`.
5. Skips with clear message when no GPU or no sample.

### Features covered (25 drivers)

**Core** (cycles 0-9):
- `bench_io_pz_loader_perf.cpp` — read_10x vs our loader.
- `bench_preprocess_lognorm_perf.cpp` — scanpy.pp.normalize_total.
- `bench_preprocess_hvg_perf.cpp` — scanpy.pp.highly_variable_genes (3 methods).
- `bench_reduce_svd_perf.cpp` — cuml PCA vs our factornet 5-backend.
- `bench_reduce_nmf_perf.cpp` — sklearn NMF vs factornet.
- `bench_graph_knn_perf.cpp` — cuml NearestNeighbors vs our compute_knn.
- `bench_graph_leiden_perf.cpp` — cugraph Leiden vs our wrapped Leiden.
- `bench_embed_umap_perf.cpp` — cuml UMAP vs our wrapped UMAP.
- `bench_de_wilcoxon_perf.cpp` — scanpy rank_genes_groups vs our Wilcoxon.

**Unique and first-GPU** (cycles 11-50):
- `bench_gsea_fgsea_perf.cpp` — fgsea R vs our fgsea + AUCell.
- `bench_anno_mt_lineage_perf.cpp` — mgatk R vs our mt_lineage.
- `bench_de_donor_pseudobulk_perf.cpp` — DESeq2 R vs our donor_pseudobulk.
- `bench_embed_cell2fate_perf.cpp` — Cell2fate Python vs our GPU.
- `bench_spatial_stagate_perf.cpp` — STAGATE Python.
- `bench_generative_discrete_diffusion_perf.cpp`.
- `bench_perturbation_cpa_perf.cpp`.
- `bench_spatial_flash_deconv_perf.cpp`.
- `bench_atac_chromvar_perf.cpp`.
- `bench_cna_numbat_perf.cpp`.
- `bench_grn_granie_perf.cpp`.
- `bench_comm_cellchat_perf.cpp`.
- `bench_eqtl_nebula_perf.cpp`.
- `bench_qc_omnidoublet_perf.cpp`.
- `bench_ase_daesc_perf.cpp`.
- `bench_fate_cospar_perf.cpp`.
- `bench_variants_monopogen_perf.cpp`.
- `bench_fate_cellrank2_perf.cpp`.
- `bench_enrich_ssgsea_perf.cpp` (covers progeny too).
- `bench_fate_palantir_perf.cpp`.
- `bench_network_hdwgcna_perf.cpp`.
- `bench_abundance_milo_perf.cpp`.
- `bench_disease_scdrs_perf.cpp`.

### Orchestration
- `bench/CMakeLists.txt` — builds every driver against the header-only library + factornet + GoogleBenchmark (if available, else plain cuEvents).
- `bench/run_all.sh` — shell driver that runs every benchmark sequentially on a compute node, logs results.
- `bench/refs/requirements.txt` — pip list for SOTA baselines (scanpy, rapids-singlecell, cuml, scvelo, cellrank, scvi-tools, scdrs).

## Scope split

This is too big for one cycle. Split:
- **Cycle 53a** (this turn): framework + 10 core bench drivers (cycles 0-9 features) + `run_all.sh` + `benchmark-registry.md` seed.
- **Cycle 53b**: 10 first-GPU bench drivers (cycles 11-35 features).
- **Cycle 53c**: final 10 bench drivers (cycles 36-50 features) + SOTA requirements.

## Cycle 53a parallel dispatch plan
Three parallel kernel-devs:
1. **Framework + IO/preprocess bench** (6 drivers): harness.h + loader + lognorm + hvg + svd + nmf + common.py.
2. **Graph/embed/de bench** (4 drivers): knn + leiden + umap + wilcoxon.
3. **benchmark-registry.md seed + bench/CMakeLists.txt + run_all.sh**: infrastructure/orchestration split out.

## HARD CONSTRAINTS
1. **No new kernels** — bench drivers ONLY exercise existing kernels.
2. **cuEvents for timing** — not host-side `chrono`.
3. **Warm-up iters discarded** — 3 warmup + 5 timed iters per feature.
4. **Reference subprocess pattern** — same idiom as correctness tests but logs wall time instead of diff.
5. **Peak memory via `cudaMemGetInfo`** — report delta, not absolute.
6. **Register row schema**: `{feature, scale, wall_ms, mem_mb, cells_per_s, sota_wall, sota_mem, ratio_wall, ratio_mem, commit}`.
7. **Skip cleanly** when no GPU / no sample / SOTA absent.

## Success criteria
- `ls bench/` shows framework + 10 driver `.cpp` files + CMakeLists + run_all.sh after cycle 53a.
- `state/benchmark-registry.md` has a header row + 1 seed placeholder row per driver (populated at first GPU dispatch).
- `bench/run_all.sh` is executable and self-documenting (exits 0 when no GPU with "no GPU — skipping bench" message).

## Risks
1. **GoogleBenchmark dependency** — prefer a minimal cuEvents wrapper over a new dep. Header-only.
2. **SOTA install weight** — requirements.txt can be heavy. Document how to install.
3. **Bench harness drift from cycle 0 in future** — add a `bench/README.md` explaining the pattern so future features get bench drivers.
