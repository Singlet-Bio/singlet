# Changelog

All notable changes to singlet-gpu. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning per `state/release-policy.md`.

## [unreleased]

### Added

- **Native GPU linear-algebra kernels (CYCLE-105)**: ~2,500 LOC of internal CUDA replacing factornet — `core/{types,handles,memory}.h` (DeviceCSC, DeviceDense, DeviceMemory, GPUContext), `reduce/svd/{deflation,randomized,auto_select}.h` (truncated SVD via successive rank-1 deflation + Halko–Martinsson), `reduce/nmf/{fit,cv,chunked}.h` (Lee–Seung MU + Hsieh–Dhillon CD + speckled-mask cross-validation + multi-shard chunked). Algorithm credit to factornet (Zach DeBruine, 2021–2026, GPL-2.0) preserved in SPDX + per-file derivation comments.
- **Native streaming types (CYCLE-106)**: `io/chunk.h` `singlet_gpu::io::Chunk` host CSC slab. `streaming/pz_data_loader.h` no longer inherits factornet's loader interface.
- **5 new Python wrappers (CYCLE-103/107)** — `pip install singlet-gpu` now exposes `pp.calculate_qc_metrics`, `pp.filter_cells`, `pp.filter_genes`, `pp.scale`, `pp.regress_out` (scanpy-style). 1,331 LOC of bindings + Python wrapper code. Wheel: `singlet_gpu-0.1.0-cp311-cp311-linux_x86_64.whl` (8.0 MB).
- **`pip install -e python/` works end-to-end** (CYCLE-107, job 368478) — first successful build since the verify chain started.
- **Three-state tier ladder** for features: `frontier → documented → released`. See `state/release-policy.md`.
- **Phase H — Document** in the cycle protocol. Every frontier promotion now also writes `docs/api/{feature}.md` in the same cycle.
- **Public API surface** tracked in `state/public-api.md`. Anything outside this file is internal and may break in any PATCH.
- Docs scaffold under `docs/`: `install.md`, `quickstart.md`, `api/`, `notebooks/`, mdBook config.
- New state files: `infrastructure.md`, `release-policy.md`, `website-contract.md`, `public-api.md`, `blockers.md`, `followups.md`.

### Removed

- **factornet as runtime backend** (CYCLE-105/106/107). `FACTORNET_INCLUDE_DIR` is now optional (migration safety-valve), no longer required. Multi-modal `FactorGraph` (`reduce/nmf/graph.h` `nmf_graph_factorize`), CSI-GEP, and OmniDoublet remain in deferred-indefinitely scope (gated behind `SINGLET_GPU_BUILD_DEFERRED`); they still reference factornet internally but don't affect foundational builds.
- 14 deferred-scope binding modules (DAESC, Numbat, Monopogen, GRaNIE, NEBULA, chromVAR, Cospar, CellRank2, Palantir, CellChat, hdWGCNA, Milo, scDRS, FlashDeconv, STAGATE, Cell2Fate, DiscreteDiffusion, PerturbGraph, ssGSEA, PROGENy) gated behind `SINGLET_GPU_BUILD_DEFERRED` (default OFF). Foundational `_core.so` is now factornet-free.

### Changed

- `CLAUDE.md` is now a thin pointer; the full agent definition lives in `agents/singlet-gpu-orchestrator.md`.
- Rule 14 rewritten: streaming is mandatory, not optional. In-memory-only PRs are rejected unless `O(n_cells × constant)` device memory.
- Rule 34 rewritten: signal-driven wakeup cadence replaces the previous "60s indefinite" loop.
- Rule 35 rewritten: persistence via `state/cycle-log.md`, not git. Auto-commit/push removed from Phase F.
- `state/feature-roadmap.md` renamed to `state/roadmap.md` (single source of truth).
- DAG split into `dag.md` / `followups.md` / `blockers.md`.

### Frontier (pre-tier-ladder; pending backfill to `documented` and `released`)

- io/pz_device_loader (#0) — 6.4× anndata-gpu, 9.4× memory savings.
- preprocess/lognorm total-count + log1p (#2) — 370× scanpy CPU.
- preprocess/lognorm scran deconvolution (#2 sub-variant) — first GPU-native scran-style deconvolution.
- preprocess/hvg seurat-v3 (#3) — 107× scanpy.
- preprocess/hvg pearson-residuals (#3) — 12,609× scanpy.
- reduce/svd deflation (#4 winner) — 27× scanpy.
- reduce/nmf (#5) — 1.82–8.66× sklearn across k=10–100.
- qc/metrics + filter + doublet (#6) — 429× scanpy.
- preprocess/scale + regress_out (#7).
- graph/knn exact (#8 partial) — 2.1× sklearn.
- de/wilcoxon (#11) — 6.5–388.8× scanpy.
- de/ttest (#11) — 8.4–10.4× scanpy.

## Pre-makeover history

See `state/cycle-log.md` for the full per-cycle history (Cycle 0–88). The cycle log is the authoritative record until git is initialized at the workspace root.
