# singlet-gpu — Follow-ups (long tail)

Post-cycle-N items grouped by module. Pulled from `state/dag.md` when an entry has been active >7 days without movement, or filed directly when a kernel ships with a known optimization deferred.

## Conventions

- 🟡 = pending, low/medium priority
- 🔴 = should be addressed in next dedicated cycle
- ⛔ = blocked on something in `state/blockers.md`
- ✅ = done; this entry moves to `state/cycle-log.md` and is removed
- Each entry: `**ID** (origin-cycle, area)` — one-line summary + estimated LOC + when to address.

---

## io / loader

- 🟡 **CYCLE-5-FACTORNET-DEVICE-OVERLOAD** (cycle 5, factornet) — submit PR adding `*_svd_gpu_device(int* d_col_ptr, ...)` overloads that skip H2D copy. ~50 LOC factornet patch + adapter rewrite.
- 🟡 **CYCLE-5-FACTORNET-STREAM-OVERLOAD** (cycle 5, factornet) — PR for stream-aware GPU APIs.
- 🟡 **CYCLE-5-SHAREDPTR-OWNERSHIP** (cycle 5, io) — invert `PzDeviceMatrix` shared_ptr ownership so `PinnedBuffer` holds a weak_ptr; the deleter calls `cudaFreeHost`. ~30 LOC.
- 🟡 **CYCLE-6-FACTORNET-DEVICE-LOADER** (cycle 6, factornet) — PR adding `nmf_chunked_gpu` overload that takes a custom loader interface returning device pointers, skipping Eigen.
- 🟡 **CYCLE-7-MULTI-INPUT-NMF** (cycle 7, streaming) — streaming pipeline driver only passes first file to `chunked_fit` for NMF path. True multi-file NMF needs `FactorGraph::SharedNode` wiring.
- 🟡 **CYCLE-54-WRITE-PZ-HELPER** (cycle 54, io) — implement `singlet_gpu::io::write_pz(path, matrix)` so the streaming_pipeline tests (currently `#if 0`'d) can be re-enabled. ~100 LOC.
- 🟡 **CYCLE-86-H5AD-REFLOADER** (cycle 86, bench) — sklearn/scanpy medium-scale refs broke because star_Solo MTX was cleaned. Add `.1pz` → h5ad (or CSC-direct) loader in `bench/refs/`. ~80 LOC.

## preprocess

- 🟡 **CYCLE-3-DEFER-EXCEPTION** (cycle 3, lognorm) — align `lognorm.h` deferred-mode signaling with validator's Test 5 expectation (`std::exception` vs status code).
- 🟡 **CYCLE-4-PEARSON-CLOSEDFORM** (cycle 4, hvg) — `hvg.h` PearsonResiduals path is O(nnz_g + n_cells) per gene; design called for O(nnz + m) via separable `sum(cell_sums²)` precomputation. Optimization, not correctness.
- 🟡 **CYCLE-20-DEDICATED-LOG1P** (cycle 20, wrappers) — `log1p` Python binding currently aliases `lognorm` with `target_count=1`. Add a dedicated `preprocess::log1p_inplace` kernel.
- 🟡 **CYCLE-55-LOGNORM-HVG-RUNTIME** (cycle 55, lognorm/hvg) — 15 LognormTest + HvgTest runtime failures on g001 from cycles 2–3 kernels. Pre-existing bugs newly visible after Cycle 54 compile gate.
- 🟡 **CYCLE-55-SVD-NMF-PZLOADER-RUNTIME** (cycle 55, reduce) — 24 SVD/NMF/PzDataLoader runtime failures on g001 from cycles 4–5 kernels.

## reduce / nmf

- 🟡 **CYCLE-6-FALLBACK-DISPATCHER** (cycle 6, nmf) — wrap factornet's `nmf::fit` (CPU+GPU dispatcher) for users who want CPU fallback when GPU is unavailable.
- 🟡 **CYCLE-20-GRAPH-TOPOLOGY** (cycle 20, wrappers) — `nmf_graph_factorize` only supports `topology="shared_h"`; add `"concat"` and `"hierarchical"`.

## graph

- 🟡 **CYCLE-GATE-2F** (cycle gate, knn/leiden) — replace static_assert guards in `graph/knn.h:430`, `graph/leiden.h:388,442` with `#ifdef SINGLET_GPU_HAS_CUGRAPH` / `HAS_CUVS` preprocessor guards. Compile-time assertion fails even when enclosing test is GTEST_SKIP-gated.
- 🟡 **CYCLE-35-KNN-WRAPPER-FIELD-STYLE** — already resolved cycle 49a; remove from active.

## de

- 🟡 **CYCLE-21-LOGREG-DE** (cycle 21, wrappers) — `rank_genes_groups(method='logreg')` currently dispatches to Wilcoxon with a warning. Add GPU logreg DE.
- 🟡 **CYCLE-85-BENCH-HARNESS-OOM** (cycle 85, de) — large-scale DE bench (100k × 30k) skipped due to host `vector::reserve` OOM before GPU OOM check. Move OOM-gate before vector reserves in bench drivers. ~30 LOC.

## factornet (upstream PRs)

- 🔴 **CYCLE-104-FOLLOWUP-FACTORNET-LOSS-ALGORITHM** (cycle 104, factornet, **CRITICAL — blocks pip install**) — submit upstream PR to factornet covering at minimum: (a) `factornet/gpu/loss.cuh` lines 328, 336, 391, 428, 456, 498, 557, 569 — unqualified `min(int, int)` calls; (b) `factornet/gpu/batch_nnls.cuh` lines 265, 266, 274, 276 — unqualified `max(...)` and `__shfl_sync(...)` calls. Add `#include <algorithm>` and qualify with `std::min` / `std::max` for host-side; ensure `<cuda_runtime.h>` is included before any template body that uses CUDA intrinsics. Likely additional sites in other factornet headers — **do a comprehensive audit when submitting** (CYCLE-104-FOLLOWUP-FACTORNET-LATENT-AUDIT). Until merged: full pybind11 `_core.so` build is blocked. C++ users unaffected. ~30–50 LOC factornet patch.
- 🟡 **CYCLE-104-FOLLOWUP-FACTORNET-LATENT-AUDIT** (cycle 104, factornet) — one-time grep across all factornet `gpu/*.cuh` and `nmf/*_gpu.cuh` headers for `\b(min|max|abs|__shfl|__ballot|__any|__all)\b\s*\(` outside of `std::` / qualified contexts. Build the comprehensive list before submitting the upstream PR. Otherwise we'll discover more sites the same way we discovered `batch_nnls.cuh`.
- 🟡 **CYCLE-104-FOLLOWUP-DIRECT-CMAKE-VERIFY** (cycle 104, infra) — write a verify script that runs `cmake -S python -B build -DSINGLET_GPU_BUILD_PYTHON=ON ... && cmake --build build --target _core` directly, bypassing pip / scikit-build-core / ensurepip / pyproject metadata. Skips 5 of the 9 paper-cut layers in CYCLE-104. Useful for CI pre-merge checks.

## pybind / wrappers

- 🟡 **CYCLE-104-FOLLOWUP-PYBIND-DEVICEMEMORY-AUDIT** (cycle 104, pybind) — one-time audit to catch any other `DeviceMemory::data()` strays in the bindings. Run `grep -rn "DeviceMemory.*\.data()" python/src/ r/src/ | grep -v "vector\|host_indptr\|host_indices\|host_values"` and fix every site to `.get()` (Cycle 54 sweep missed `_bind_loader.hpp`).
- 🟡 **CYCLE-104-FOLLOWUP-DEFERRED-BINDINGS-GATE** (cycle 104, pybind) — apply the `SINGLET_GPU_BUILD_DEFERRED` gate consistently to all bindings whose underlying features are in the deferred-indefinitely scope: `_bind_spatial_phaseb`, `_bind_generative`, `_bind_perturbation`, `_bind_enrich`, `_bind_atac`, `_bind_cna`, `_bind_eqtl`, `_bind_grn`, `_bind_disease`, `_bind_abundance`, `_bind_comm`, `_bind_ase`, `_bind_variants`, `_bind_network`. Keeps foundational build small + fast. ~30 LOC of `#ifdef` gates.

## website / publishing

- 🟡 **CYCLE-93-FOLLOWUP-FEATURE-ID-CROSSCHECK** (cycle 93, infra) — add a tiny script (Python or Node) that asserts every short ID in `scripts/frontier_sync.py` `_LONG_TO_SHORT` is present in `singletai-website/src/pages/Benchmarks.tsx` `FEATURES[].id`, and warns when a frontier feature has no website entry. ~40 LOC. Wire into a pre-publish check or the cycle 90 cron.
- 🟡 **CYCLE-92-FOLLOWUP-NVCC-ONLY-DOCS** (cycle 92, docs) — update `docs/install.md` to make the nvcc-only requirement explicit: header-only, but the umbrella header includes CUDA kernel launch syntax inline, so consumers compile their own TUs with nvcc (or nvcc -ccbin g++). ~10 LOC.
- 🟡 **CYCLE-92-FOLLOWUP-CMAKE-COMMIT-SHA** (cycle 92, build) — once git is initialized at workspace root, add `git describe`/`git rev-parse HEAD` to the top-level CMakeLists.txt and pass `-DSINGLET_GPU_COMMIT_SHA="..."` to compilation so `singlet_gpu::commit_sha()` returns a real value instead of the literal "pre-1.0".

## tests / infra

- 🟡 **CYCLE-54-EIGEN-AUTOFIND** (cycle 54, cmake) — auto-discover Eigen from `FACTORNET_ROOT/subprojects/eigen-3.4.0`. 5 LOC.
- 🟡 **CYCLE-54-PER-CYCLE-COMPILE-GATE** — already enforced by orchestrator Rule 12.
- 🟡 **CYCLE-54-LIBSTDCPP-NODE-VARIANCE** (cycle 54, cmake) — document per-node libstdc++ variance in `state/style-rules.md`. Add CMake auto-detection or explicit link rule. 30 LOC.
- 🟡 **CYCLE-54-NO-MONITOR-LEAK** (cycle 54, agents) — update `agents/gpu-kernel-dev.md` + `agents/analysis-validator.md` to forbid spawning background Monitor sub-workers (notifications leak to orchestrator context window).
- 🟡 **CYCLE-54-GPU-NODE-GUARD** (cycle 54, tests) — 49 ctest failures on c006 (CPU-only) were "no CUDA-capable device" exceptions instead of `GTEST_SKIP`. Add a common `gpu_available()` helper to `tests/common/`. 80 LOC.
- 🟡 **CYCLE-55-MILO-WIRING** (cycle 55, deferred-features) — milo tests are GTEST_SKIP stubs. Need real fixture wiring + correctness vs R `MASS::glm.nb`. ~400 LOC. Deferred per scope narrowing — only revisit if user reverses scope decision.
- 🟡 **CYCLE-55-SCDRS-WIRING** (cycle 55, deferred-features) — finish wiring + fix p-value calibration (`Scdrs_PvalueCalibration_Uniform` KS=0.214 > 0.10). Deferred per scope.
- 🟡 **CYCLE-55-E2E-REVALIDATE** (cycle 55, integration) — `tests/integration_e2e_correctness.cpp` deferred via cmake option. Realign against the 9-stage API surface after CYCLE-55-LOGNORM-HVG and CYCLE-55-SVD-NMF runtime cycles stabilize. ~1500 LOC.
- 🟡 **CYCLE-55-BENCH-RELINK** (cycle 55, bench) — 9 bench drivers `EXCLUDE_FROM_ALL`-ed in Phase 3 based on misdiagnosed linker issue. Re-include with explicit `gcc-toolset-13 libstdc++.a` link.

## wrappers (Python / R)

- 🟡 **CYCLE-18-NANOBIND-MIGRATION** (cycle 18, python) — rapids-singlecell migrated to nanobind. Cycle 18 shipped pybind11. Migrate post-1.0.
- 🟡 **CYCLE-19-SCANPY-SIG-PARITY** (cycle 19, python) — wrappers used `inplace`/`backend`/`seed`; scanpy uses `copy`/`svd_solver`/`rng`. ~50 LOC of parameter renames.
- 🟡 **CYCLE-23-CYCLE-22-BINDING-NAMES** (cycle 23, python) — wrappers reference `_core.fgsea`, `_core.aucell` etc.; cycle 22 may have used different names. One-line sed in `_bind_kernels.hpp`.
- 🟡 **CYCLE-24-RCPPML-PATTERN** (cycle 24, R) — apply RcppML's better patterns: drop reticulate, three-state `resource` parameter, `gpu_stubs.cpp` dual-compilation, `sparse_from_csc()` zero-copy, `.gpu_env` lazy detection.

## advanced / deferred-indefinitely

These features have headers in-tree but receive no development per scope narrowing in `state/roadmap.md`. Listed here so future cycles can find them if scope reverses.

- Cell2fate, CSI-GEP, STAGATE, discrete_diffusion, PerturbGraph, FlashDeconv, chromVAR, Numbat, GRaNIE, CellChat, NEBULA, OmniDoublet, DAESC, Cospar, Monopogen, CellRank2, ssGSEA, PROGENy, Palantir, hdWGCNA, Milo, scDRS — all 22 carry follow-ups in their original cycle entries (cycles 27–50). Pull when scope reopens.
