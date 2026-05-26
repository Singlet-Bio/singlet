# Singlify → Singlet Monorepo Port — Reconciliation Manifest

**Date**: 2026-05-26
**Singlify branch**: `port-checkpoint-2026-05-26` (commit `fee1f03`, local-only safety branch on operator's workstation)
**Singlify tip before checkpoint**: `0ea4f80` "rename: singlify → singlet (binary, CLI, env vars, logs, JSON fields, docs)"

## Decision matrix

For each file modified in singlify's WIP relative to the rename commit, the reconciliation verdict:

### Headers — SINGLET_WINS (singlet is canonical, no changes required)

| Singlify path | Reason |
|---|---|
| include/singlet-pileup/cascade_router.h | Singlet has independently evolved superset |
| include/singlet-pileup/cell_calling.h | Singlet pileup module set is larger (100 vs 96) |
| include/singlet-pileup/export.h | Singlet's export.h is ~4× larger with multiome/doublet/ambient/validation features |
| include/singlet-pileup/summary_json.h | Singlet's PipelineSummary has 60+ fields vs singlify's 45 (singlet adds mismatch_rate_per_base, cascade.*, low_cell_diagnostic.*, species_mismatch.*, avg_input_read_length) |
| include/lib1fq/protocol.h | Functionally identical; only namespace/include-path differences |

### Headers — PARTIAL MERGE applied

| File | Action |
|---|---|
| include/singlet-pileup/species_detect.h → singlet/include/singlet/pileup/species_detect.h | Ported singlify-only bio data: `GRCh38-GRCm39-barnyard` multi-species detection (human+mouse mixed samples), assembly version upgrade routing (ARS-UCD1.2 → 1.3), ~50 additional species (primates, plants, parasites, fungi, bacteria). Kept singlet's SPDX/kmer_util.h/pragma diagnostics. |

### Tests — SINGLET_WINS (45+ files)

Singlet has 111 cpp tests vs singlify's 105. All test files modified in singlify relative to its rename commit are stale relative to singlet's independent evolution; singlet's versions are the canonical state. Net-new singlify tests (13) were ported in singlet@f4a71d6.

### Sources / build — SINGLET_WINS

| Singlify path | Singlet path | Reason |
|---|---|---|
| src/singlet.cpp (281 mod, 8708 lines) | src/pipeline/singlet.cpp | Pipeline drivers diverged independently; singlet's version is the canonical orchestrator |
| src/singlify.cpp (8467 lines, untracked) | — | Obsolete legacy pre-rename source; superseded by the renamed singlet.cpp |
| src/star/readLoad.cpp (1 line) | src/star/readLoad.cpp | Trivial, no port needed |
| CMakeLists.txt (53 lines mod) | CMakeLists.txt | Singlet's monorepo build (header-only library + optional components) supersedes singlify's flat layout |

### Docs

| Singlify path | Singlet path | Action |
|---|---|---|
| docs/DEVLOG.md (429 lines added) | — | NOT PORTED — historical singlify-internal devlog; kept on `port-checkpoint-2026-05-26` branch for reference |
| docs/TEST_SUITE.md | docs/TEST_SUITE.md | Ported in singlet@f4a71d6 |

## Binary / CLI / namespace rename pass

Singlify's commit `0ea4f80` renamed binary/CLI/env-vars/JSON-fields/docs from `singlify` → `singlet`. The monorepo's pipeline binary was already named `singlet` in `src/pipeline/`, so the rename pass here focused on lingering string references:

- `python/singlet/pipeline/_binary.py`: fixed candidate lookup paths
- `src/star/STAR.cpp`, `src/star/CMakeLists.txt`: comment cleanup
- `include/singlet/pileup/export.h` (3 hits): comment cleanup
- `include/singlet/star/star_api.h` (2 hits): comment cleanup

Two intentional `singlify` references remain (both labeled "legacy" or "ported from singlify"):
- `python/singlet/pipeline/_binary.py:43` — legacy lookup path for users who still have a built singlify checkout
- `tests/cpp/CMakeLists.txt:170` — historical comment from the 2026-05-26 port

## Outstanding operator items

See cowork prompt issued 2026-05-26-A.
