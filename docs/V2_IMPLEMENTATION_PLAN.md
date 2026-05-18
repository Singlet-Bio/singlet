# Singlet v2 Pipeline Migration — Implementation Plan

> Companion to: [`CANONICAL_OUTPUT_FORMAT.md`](CANONICAL_OUTPUT_FORMAT.md)
> Phase 1 (cancel + wipe + plan) completed: 2026-05-15
> Status: Implementation phases 2–8 pending

This plan stages the migration from today's heterogeneous 31-file-per-sample
layout to the canonical v2 format. Each phase is a self-contained PR;
phases are ordered so each is independently testable.

## Phase 1 — Cancel, wipe, plan (this session) ✅

- ✅ Cancelled all `singlet-v3` array jobs and the `singlet-orch` driver
  (~80 running tasks, ~60 pending).
- ✅ Moved 270 GB of v1 outputs to
  `/mnt/projects/debruinz_project/singlet_pipeline/quant.trash.20260515_1243`
  (reversible — `mv` not `rm`).
- ✅ Archived `results/` and `orchestrator_v3_state.json` similarly.
- ✅ Wrote [`CANONICAL_OUTPUT_FORMAT.md`](CANONICAL_OUTPUT_FORMAT.md) (the spec).
- ✅ Scaffolded reader API stubs in `python/singlet/io/sample.py` and view
  functions in `python/singlet/views/`.
- ✅ This document.

## Phase 2 — Reference bundle builders ✅

Completed: 2026-05-15.

- ✅ `python/singlet/refbundle/` — shared format module (used by both
  builders and Phase-6 readers): `_features.py` (FeaturesBundle reader +
  writer, §3.1 layout), `_snp.py` (SnpPanel reader + writer, §3.2 layout).
- ✅ `scripts/build_features_fbin.py` — single-pass GTF parser; emits
  merged exons, derived introns, and one junction per intron.
- ✅ `scripts/build_snp_sites_fbin.py` — single-pass VCF parser; filters
  to biallelic A/C/G/T SNPs; honors `--min-af`.
- ✅ `tests/python/test_refbundle.py` — 14 tests covering round-trip
  (features + SNP), bad-magic rejection, chrom-table overflow, gzip
  GTF input, multi-allelic / indel rejection, AF filtering.

Pending for production reference build: run both scripts against
`reference/GRCh38-2024-A/genes/genes.gtf` and
`reference/GRCh38-2024-A/snps/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.vcf.gz`
to materialize the production fbins, then update each `manifest.json`
with `features_fbin_sha256` and `snp_sites_fbin_sha256`. The format is
locked; this is a one-shot reference-build job, not a code task.

## Phase 3 — Pileup engine refactor: enforce UMI partition

**Goal**: every deduplicated transcript molecule lands in exactly one of
`exon_body`, `intron_body`, `junctions`. Fix the latent double-counting in
[`include/singlet-pileup/pileup_engine.h`](../../singlify/include/singlet-pileup/pileup_engine.h).

Tasks:
- Replace the three independent dedup tables (`umi_dedup_`,
  `intron_umi_dedup_`, `sj_umi_dedup_`) with a single unified table keyed
  by (cell_barcode, umi).
- Add a single-molecule classifier (§4.3 of spec):
  1. Any `N` cigar op → `junctions`
  2. Else footprint in one exon → `exon_body`
  3. Else footprint in one intron → `intron_body`
  4. Else EI/IE straddle → `junctions`
- Emit three blocks (exon_body, intron_body, junctions) with a shared
  cell-barcode axis.
- Regression test: on 100-sample panel, sum across three blocks per cell
  must equal raw `umi_unique`. Reject the PR if the invariant fails.

**Acceptance**: `tests/cpp/test_pileup_partition_invariant.cpp` passes on
100 GSM panel; per-block nnz totals logged in `pileup_stats.json`.

## Phase 4 — `.1pz` format extension (multi-block + two-layer)

**Goal**: extend [`include/singlet/pz/writer.h`](../include/singlet/pz/writer.h)
to support the layouts the spec needs.

Tasks:
- **Multi-row-block layout**: header carries a block table with name,
  row_offset_in_global_axis, row_dim, nnz, payload offsets, CRC32. One
  zstd stream per block. Shared CBOR-encoded header with
  `cell_barcodes`. Bump magic to `1PZ02`.
- **Two-data-layer CSC**: a single block can carry two `data` arrays
  sharing one `indptr`/`indices`. Used for `snp.1pz` (alt_ad + dp) and
  `mt.1pz` (alt_ad + dp).
- Backward-compat: reader detects `1PZ01` vs `1PZ02` magic and dispatches.
- Add `pz::Reader::block_names()`, `block(name)` view API.

**Acceptance**: round-trip tests in `tests/cpp/test_pz_v2.cpp` for both
new layouts; existing v1 files still readable.

## Phase 5 — Singlet binary outputs

**Goal**: `singlet process` writes the canonical per-sample layout from §5
of the spec.

Tasks:
- Rewrite [`src/pipeline/singlet.cpp`](../src/pipeline/singlet.cpp) output
  block:
  - `counts.1pz` (Phase 3 partition × Phase 4 multi-block writer)
  - `snp.1pz` (Phase 4 two-layer; rows = `snp_sites.fbin`)
  - `mt.1pz` (Phase 4 two-layer; rows = chrM positions)
  - `nonhost.json` + `nonhost_species.1pz` (Kraken2 + Bracken raw outputs;
    per-cell × NCBI taxid via existing `NonHostCellMatrix`)
  - `cell_meta.parquet` (replaces 7 TSVs; written via arrow-cpp or
    parquet-cpp)
  - `summary.json` (one file replacing 8 JSONs + `metrics_summary.csv`)
- **Delete** all v1 writers: `gene_counts.1pz`, `gene_counts_em.1pz`,
  `spliced.1pz`, `unspliced.1pz`, `ambiguous.1pz`, `splice_psi.1pz`,
  `gene_expression.tsv`, `mt_variants.tsv`, `donor0_coverage.tsv`,
  `metrics_summary.csv`, `read_stats.tsv`, `auto_barcodes.tsv`,
  `ambient_contamination.tsv`, `nonhost_per_cell.tsv`,
  `nonhost_em_abundance.tsv`, `pileup_stats.json`, `rrna_report.json`,
  `sex_call.json`, `ancestry_call.json`, `ambient_contamination.json`,
  `nonhost_summary.json`.
- Conditional outputs: `donor_consensus.fa`, `donor_variants.vcf.gz`,
  `ambient_profile.npy`, `splice_events.tsv`, `guides.1pz`,
  `antibodies.1pz`, `vdj_gene_usage.1pz` (gated by CLI flags).

**Acceptance**: end-to-end run on one sample produces exactly the file list
in §5 of the spec; `singlet process --validate` runs schema validation
against the spec.

## Phase 6 — Python reader API

**Goal**: implement the stubs in `python/singlet/io/sample.py` and
`python/singlet/views/`.

Tasks:
- `SingletSample(path)` — loads `summary.json`, knows what files exist.
- `SingletCounts` — `.exon_body()`, `.intron_body()`, `.junctions()` CSC views.
- `SingletSnp` — `.ad()`, `.dp()`, `.vaf()` views.
- `SingletMt` — analogous.
- `SingletNonhost` — `.species_table()`, `.per_cell()`.
- `views.gene_counts(sample)` — sum exon_body + intron_body + junctions
  mapped to gene IDs via `features.fbin`.
- `views.usa(sample)` — three matrices (spliced, unspliced, ambiguous).
- `views.psi(sample)` — per-junction PSI.
- `views.cell_meta(sample)` — column projection on parquet.
- All views lazy-evaluate; benchmark < 200 ms on 12K-cell sample.

**Acceptance**: `tests/python/test_sample_io.py` covers every helper;
round-trip against C++ writer outputs.

## Phase 7 — Orchestrator + SLURM scripts

**Goal**: rebuild the orchestrator around v2 outputs and the
trimmed scripts directory.

Tasks:
- Move all legacy job scripts in
  `/mnt/projects/debruinz_project/singlet_pipeline/scripts/` to
  `scripts.trash.20260515/`. Keep only:
  - `batch_template_v4.sh` (NEW, calls v2 binary, writes canonical layout)
  - `singlet_orchestrator_v2.sh`
  - `create_and_submit_batch_v2.py`
- New scripts emit per-sample manifest validating that all required v2
  files exist before marking the sample SUCCESS.
- Catalog DB schema migration: drop legacy columns referencing
  `gene_counts.1pz` etc., add `counts_path`, `snp_path`, `mt_path`,
  `nonhost_path`, `cell_meta_path`, `summary_path`.

**Acceptance**: pilot run of 100 samples through new orchestrator;
success rate >90% with all outputs validated.

## Phase 8 — Legacy code removal

**Goal**: delete dead code paths so the codebase reflects v2 only.

Files/directories to remove:
- `singlify/` repo entirely (superseded by `singlet/`; verify no remaining
  imports first).
- `geo-reprocess/` if catalog-build superseded by `singlet/python/singlet/catalog/`.
- `singlet/python/singlet/io/` legacy SPZ paths (`read_spz`, `write_spz`,
  `spz_info`) — `_io.py` SPZ functions and the convert.py SPZ shims.
- `singlet/include/singlet/pz/` v1-only code paths once v2 readers stable.
- Old USA/PSI computation code that previously wrote derived `.1pz` files.

Add deprecation shims for one minor version (`singlet.io.read_spz` →
`DeprecationWarning` then `read_1pz`), then remove.

**Acceptance**: `git grep -i "spz\|gene_counts_em\|nonhost_per_cell"` in
the main package returns no matches; tests pass.

## Phase 9 — Full reprocessing

**Goal**: reprocess the GEO catalog with v2 pipeline.

Tasks:
- Smoke-test 100 representative samples (10x v3, v2, drop-seq, indrop,
  smartseq2, ATAC).
- Re-run cohort. With cleaned UMI partition, the canonical output is
  smaller (~125 MB vs ~170 MB per sample). Across 90K samples, ~5 TB
  saved.
- After 30-day soak, `rm -rf` the trash directories.

---

## Module locations

| Component | Path |
|---|---|
| C++ writer extensions | `singlet/include/singlet/pz/writer.h` |
| C++ pileup classifier | `singlet/include/singlet/pileup/` |
| C++ pipeline binary | `singlet/src/pipeline/singlet.cpp` |
| Reference builders | `singlet/scripts/build_features_fbin.py`, `build_snp_sites_fbin.py` |
| Python readers | `singlet/python/singlet/io/sample.py` |
| Python derived views | `singlet/python/singlet/views/` |
| SLURM scripts (new) | `singlet_pipeline/scripts/batch_template_v4.sh`, `singlet_orchestrator_v2.sh`, `create_and_submit_batch_v2.py` |
| Tests | `singlet/tests/cpp/test_pz_v2.cpp`, `tests/cpp/test_pileup_partition_invariant.cpp`, `tests/python/test_sample_io.py` |

## Risk log

- **Bracken integration**: not yet in pipeline. Phase 5 must wire
  Bracken into the non-host path, store `bracken_reads` in
  `nonhost.json`. If Bracken proves flaky, fallback is raw Kraken2 reads
  with empty Bracken section.
- **Parquet C++ dependency**: requires linking against arrow-cpp. If
  build complexity grows, alternative is a minimal CBOR-encoded sidecar
  with column-major arrays — still column-projectable, smaller link
  footprint.
- **Cross-block UMI dedup edge cases**: the §4.3 classifier needs unit
  tests for ambiguous reads (e.g. soft-clipped reads partially overlapping
  exon-intron boundary). Risk is mitigated by the partition invariant
  test on real data.
