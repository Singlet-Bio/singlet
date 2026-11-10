# Move to the private repository (owner list; delete this file before the v1.0.0 tag)

Source: consolidation-audit.md section 5 and 6. Nothing here has been removed yet.

## Order of work

1. Move `bench/gpu/m5/stitch.cuh` into `src/engine_gpu` (the engine includes it), update `bench/count_parity/counting_sources.txt`
   and `.github/workflows/count-parity.yml`, and confirm the build and a checksum-identical run. Also `m6.cu`, `pileup.cu`,
   `pileup_rec.h` if the engine includes them from `bench/gpu`.
2. Change the `packs_build.py` defaults that point at `bench/`.
3. Move `bench/` and `scripts/`.

## Paths

| Group | Paths | Files | Lines |
|---|---|---:|---:|
| Benchmarks and results | `bench/**` except `bench/count_parity/**` (CI gate, stays) and `bench/gpu/m5/stitch.cuh` | about 2,700 | about 570,000 |
| Campaign runner and corpus tooling | `scripts/atlas_rerun/**`, `scripts/alpha_qc/**`, `scripts/celltype_nn/**`, `scripts/gate2`, `scripts/cpu_perf`, `scripts/engine_gpu`, `scripts/genotype`, `scripts/editing`, `scripts/s1_*.py`, `scripts/s1_*.sbatch`, `scripts/annotate_*`, `scripts/doublet_*`, `scripts/features_*`, `scripts/restrict_published_bundles.*`, `scripts/build_*` data builders | 188 | 26,052 |
| Sprint state and agent logs | `state/**` | 172 | 25,257 |
| Manuscripts and strategy | `papers/**` (highest priority: `papers/manuscripts/{business-analysis,investor-pitch,risk-assessment,product-strategy,competitive-analysis,development-roadmap,genomics-intelligence}`) | 196 | 37,840 |
| Campaign docs | `docs/research-beta/**`, `docs/research-beta.md`, `docs/rename-to-1-0/**`, `docs/rename-to-1-0.md`, `docs/atlas-rerun-*.md`, `docs/s1-migration.md`, `docs/ROADMAP_ANALYSIS.md`, `docs/gpu-engine/STATUS-2026-11.md`, `docs/gpu-engine/fable/` (review first) | 43+ | 8,123+ |
| Agent definitions and notes | `.github/agents/**`, `GPU_BRANCHES.md`, `IDEAS.md`, `PROFILE.md`, `CLAUDE.md` | 13 | 2,133 |
| Python campaign CLIs (verify first) | `python/singlet/s1/{migrate,verify,packs_build,policy,cellcall,labels_step,modality_members}.py`; `cellcall.py` is the reference for `tests/data/called_ed_golden`, so check that first | 7 | 4,375 |

## Merged into this branch in this pass and bound for the private repo

`bench/expression_sota/**` (102 files added from claude/expression-sota-31cf6d), `bench/readlevel/**` (oracles), `bench/cellcall_gap/**`,
`bench/stream/**` (loader compare), `bench/ambient_prior/**`. Keep them in a public `bench/` folder only if the owner wants the evidence public.

## Cleanup after the move

- Personal and cluster paths remain in files under `bench/`, `state/`, `papers/`, `.github/agents/`, `tests/python/conftest.py`,
  `tests/python/test_s1_campaign_catalog.py` and 34 files in `tests/cpp/gpu/` (one line each), `notebooks/01_load_and_explore.ipynb`,
  `notebooks/splicing.ipynb`.
- Operational identifiers (restricted bucket sidecar design) in `CHANGELOG.md`, `docs/format-s1-cell-index.md`,
  `include/singlet/pileup/export.h`, `include/singlet/pileup/summary_json.h`, `python/singlet/s1/recipes_pzw.py`: check the text
  names no private hostnames or bucket names.
- `CYCLE-NNN` markers (440 hits in 116 files) and TODO/FIXME comments date the code to the old agent process.
