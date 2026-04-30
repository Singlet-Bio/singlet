---
feature: wrapper_catchup_cycle_52
roadmap_id: 52
module: python/singlet_gpu/{comm,fate,grn,eqtl,ase,cna,variants,atac,spatial,generative,perturbation,network,abundance,disease,enrich,qc}/ + r/R/*
status: design
tolerance: every new feature from cycles 27-50 importable + smoke-test passes in clean venv/R env
target_perf: N/A — wrapper consolidation cycle
ooc_plan: N/A
---

## Why this exists

**Second consecutive NO STRONG CANDIDATE from cycle 51 lit-scout.** The first-GPU feature pipeline has exhausted obvious targets in the 2023-2026 literature. Scout explicitly recommends: "Pivot to wrapper consolidation — Python/R API hardening, benchmarking against Scanpy/Seurat/scran across all 50 cycles."

**Current state**:
- Python wrapper sprint: cycles 18-23 (built out core + preprocess + reduce + streaming + de + enrichment + integrate + velocity + lineage + pseudobulk bindings).
- R wrapper sprint: cycles 24-26 (built out RcppEigen integration with hybrid dispatch).
- **Features shipped cycles 27-50**: 22 features with NO wrapper coverage:
  - 27 Cell2fate, 28 CSI-GEP, 29 STAGATE, 30 discrete diffusion, 31 doublet_score, 32 CPA, 33 FlashDeconv, 34 chromVAR, 35 Numbat CNA, 36 GRaNIE, 37 CellChat, 38 NEBULA, 39 OmniDoublet, 40 DAESC, 41 Cospar, 42 Monopogen, 43 CellRank 2, 44 ssGSEA+PROGENy, 45 Palantir, 46 hdWGCNA, 47 Milo, 50 scDRS.

## What gets built

### Python side (pybind11)
- New binding modules under `python/src/_bind_*.hpp` grouped by domain:
  - `_bind_fate.hpp` — cospar, cellrank2, palantir (3 functions).
  - `_bind_grn.hpp` — granie (1 function).
  - `_bind_eqtl.hpp` — nebula (1 function).
  - `_bind_ase.hpp` — daesc (1 function).
  - `_bind_cna.hpp` — numbat (1 function).
  - `_bind_variants.hpp` — monopogen (1 function).
  - `_bind_atac.hpp` — chromvar (1 function).
  - `_bind_spatial_phaseb.hpp` — flash_deconv, stagate, cell2fate (3 functions).
  - `_bind_generative.hpp` — discrete_diffusion (1 function).
  - `_bind_perturbation.hpp` — perturb_graph (1 function).
  - `_bind_comm.hpp` — cellchat (1 function).
  - `_bind_network.hpp` — hdwgcna (1 function).
  - `_bind_abundance.hpp` — milo (1 function).
  - `_bind_disease.hpp` — scdrs (1 function).
  - `_bind_enrich.hpp` — ssgsea, progeny (2 functions).
  - `_bind_qc_new.hpp` — omnidoublet, doublet_score (2 functions).
  - `_bind_nmf_new.hpp` — csi_gep (1 function).

- New Python module files:
  - `python/singlet_gpu/fate/{cospar,cellrank2,palantir}.py`
  - `python/singlet_gpu/grn/granie.py`
  - `python/singlet_gpu/eqtl/nebula.py`
  - `python/singlet_gpu/ase/daesc.py`
  - `python/singlet_gpu/cna/numbat.py`
  - `python/singlet_gpu/variants/monopogen.py`
  - `python/singlet_gpu/atac/chromvar.py`
  - `python/singlet_gpu/spatial/{flash_deconv,stagate,cell2fate}.py`
  - `python/singlet_gpu/generative/discrete_diffusion.py`
  - `python/singlet_gpu/perturbation/perturb_graph.py`
  - `python/singlet_gpu/comm/cellchat.py`
  - `python/singlet_gpu/network/hdwgcna.py`
  - `python/singlet_gpu/abundance/milo.py`
  - `python/singlet_gpu/disease/scdrs.py`
  - `python/singlet_gpu/enrich/{ssgsea,progeny}.py`
  - `python/singlet_gpu/qc/{omnidoublet,doublet_score}.py`

### R side (Rcpp)
- Parallel R wrapper module for each Python module; use the dual-compilation gpu_stubs.cpp pattern from cycle 24.
- New R files: `r/R/{fate,grn,eqtl,ase,cna,variants,atac,generative,perturbation,comm,network,abundance,disease,enrich,qc}.R`.

### Smoke tests
- `python/tests/test_new_features_smoke.py` — import and one-tiny-input run per new feature.
- `r/tests/testthat/test-new-features-smoke.R` — parallel.

## Scope split across sub-cycles

### Cycle 52a (this turn) — Python bindings tier 1
Dispatch 3 parallel kernel-devs (max per message):
1. `gpu-kernel-dev` (Sonnet) — write `_bind_fate.hpp` + `_bind_disease.hpp` + `_bind_comm.hpp` + `_bind_network.hpp` + `_bind_abundance.hpp` + corresponding Python module files (cospar, cellrank2, palantir, scdrs, cellchat, hdwgcna, milo → 7 features).
2. `gpu-kernel-dev` (Sonnet) — write `_bind_grn.hpp` + `_bind_eqtl.hpp` + `_bind_ase.hpp` + `_bind_cna.hpp` + `_bind_variants.hpp` + `_bind_atac.hpp` + corresponding Python module files (granie, nebula, daesc, numbat, monopogen, chromvar → 6 features).
3. `gpu-kernel-dev` (Sonnet) — write `_bind_spatial_phaseb.hpp` + `_bind_generative.hpp` + `_bind_perturbation.hpp` + `_bind_enrich.hpp` + `_bind_qc_new.hpp` + `_bind_nmf_new.hpp` + corresponding Python modules (flash_deconv, stagate, cell2fate, discrete_diffusion, perturb_graph, ssgsea, progeny, omnidoublet, doublet_score, csi_gep → 10 features).

23 features total split across 3 parallel dispatches.

### Cycle 52b — R bindings (next turn after 52a)
Dispatch 3 parallel kernel-devs mirroring the split for R wrappers via Rcpp.

### Cycle 52c — smoke tests (after 52b)
Dispatch 1 validator to write the combined smoke test sweep.

## Budget
- ~5000 LOC total across ~100 files (many small wrappers are 50-150 LOC each; binding headers 200-300 LOC each).
- Split across 6-7 parallel dispatches.

## HARD CONSTRAINTS
1. **No new kernels**. Wrappers only. Every function marshals arguments to a header-only kernel and returns cupy/scipy/anndata-compatible outputs.
2. **Reuse cycle 18-26 patterns**. The existing `_core.cpp` module registration mechanism is proven.
3. **pybind11 cuda_array_interface** for zero-copy cupy/cudf pass-through on device arrays.
4. **R dual-compilation** via `gpu_stubs.cpp` from cycle 24.
5. **Every new wrapper function takes explicit `stream=None` and `seed=0` parameters** (consistent API).

## Risks
1. **Binding scope is large** — 23 features × (1 binding + 1 Python module + 1 R file) = ~70 new files. Must split across sub-cycles.
2. **cuda_array_interface for custom result structs** (e.g., `NumbatResult`, `CellChatResult`) requires careful marshalling — not just raw cupy arrays.
3. **Python import path**: ensure all new modules follow `from singlet_gpu.{domain}.{method} import {function}` convention.
4. **R type mapping** for complex result structs: use named list with matrix/vector fields.

## Success criteria
- `python -c "import singlet_gpu; singlet_gpu.fate.palantir.run_from_csc(...)"` works end-to-end on a mini fixture.
- `R -e "library(singletgpu); singletgpu::run_cellchat(...)"` works.
- All 22 features in cycles 27-50 importable from both Python and R after cycle 52 completes.
