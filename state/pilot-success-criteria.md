# Pilot Success Criteria — 1,000-Sample Human 10x Droplet Pilot

**Freeze target**: `v0.3.0-pilot-freeze`
**Pilot universe**: 1,000 stratified samples from 37,055 Homo sapiens × 10xv1/v2/v3/v3_5prime × high|medium protocol-confidence in `processing_catalog.parquet` (2026-05-13 snapshot)
**Reference**: GRCh38-2024-A (GRCh38.p14 + GENCODE v44), pinned by `state/reference-manifest-v1.yaml`
**Author**: singlet master orchestrator
**Date**: 2026-05-27

## Purpose

The pilot exists to answer a single question: **Is the v0.3.0-pilot-freeze codebase fit-for-purpose to run the full 56K-sample campaign without code changes?**

A pilot pass means we tag `v0.3.0-production` from the same commit and start the full campaign. A pilot fail means we triage, hotfix, re-tag, and re-pilot a smaller subset before proceeding.

## Go / no-go thresholds

All criteria must be met simultaneously for a pass.

### Stability (binary correctness)

| # | Metric | Threshold |
|---|---|---|
| S1 | Unhandled crashes (SIGSEGV / SIGABRT / un-caught exceptions) | 0 |
| S2 | Samples lost to OOM that should have fit the requested memory tier | 0 (any OOM must be classifiable as data-driven, not pipeline-driven) |
| S3 | `.1pz` files that fail decode-roundtrip validation | 0 |
| S4 | `summary.json` files that fail schema-1.1 validation | 0 |
| S5 | `summary.json` files missing `git_sha` or `reference_manifest_sha256` | 0 |
| S6 | Reference-manifest sha256 mismatch across the 1,000 samples | 0 (all samples must record the same hash) |

### Throughput (production scaling)

| # | Metric | Threshold |
|---|---|---|
| T1 | Overall success rate (`status == "success"`) | ≥ 85% (≥ 850 / 1,000) |
| T2 | Samples in classified failure modes | ≥ 99% of non-success (≤ 1% unclassified) |
| T3 | Median wall-clock per sample @ 16 cores | ≤ 6 h |
| T4 | p95 wall-clock per sample @ 16 cores | ≤ 18 h |
| T5 | Median peak RSS | ≤ 32 GB |
| T6 | p95 peak RSS | ≤ 96 GB (must fit Clipper bigmem queue) |
| T7 | Median disk footprint per sample (final output bundle) | ≤ 15 MB |
| T8 | Total throughput at MAX_QT=200 concurrent jobs | ≥ 150 samples / day extrapolated |

### Biology (signal correctness)

Random spot-check 20 successful samples; manual review required.

| # | Metric | Threshold |
|---|---|---|
| B1 | Median mapping rate for 10xv3 samples | ≥ 70% |
| B2 | Median mapping rate for 10xv2 samples | ≥ 60% |
| B3 | Median mapping rate for 10xv3_5prime samples | ≥ 60% |
| B4 | Median cells called per sample (where input read count > 100M) | ≥ 1,000 |
| B5 | Median genes per cell | ≥ 800 for 10xv3, ≥ 500 for 10xv2 |
| B6 | Fraction of samples with `pct_mt` distribution looking biologically reasonable (p50 < 25%) | ≥ 90% |
| B7 | Spliced/unspliced ratio plausible for tissue type (spot check) | qualitative pass on 20/20 |

### Catalog representativeness

The 1,000-sample pilot must include:

| # | Stratum | Min count |
|---|---|---|
| C1 | 10xv3 | 500 |
| C2 | 10xv2 | 200 |
| C3 | 10xv3_5prime | 100 |
| C4 | 10xv1 | 0 (only include if present in catalog; unlikely) |
| C5 | Distinct GSEs | ≥ 250 |
| C6 | Distinct tissue annotations (where present) | ≥ 15 |
| C7 | Read-count tiers: <50M / 50–200M / 200–500M / >500M | each ≥ 100 |
| C8 | Previously-completed samples (for regression check) | ≥ 100 |
| C9 | Previously-failed samples (to test failure classification) | ≥ 50 |

C8 + C9 together provide a regression test against the prior `simpleaf`-based pipeline.

## Failure-mode taxonomy (allowed `status` values)

These are the five statuses defined in `summary_json.h::classify_outcome()`. Any sample falling outside these is an **unclassified failure** and counts against T2.

| Status | Meaning | Pilot acceptance |
|---|---|---|
| `success` | all gates passed | ✅ |
| `align_low_map` | mapping rate < 50% (or 30% for ATAC) | ✅ classified |
| `align_zero_cells` | 0 cells called | ✅ classified |
| `align_low_cells` | < 10 cells called | ✅ classified |
| `align_low_genes` | median genes/cell < 200 | ✅ classified |
| *(any other)* | unclassified — pipeline bug | 🔴 fails T2 |

## Regression check vs prior runs

For the ≥ 100 previously-completed samples (C8), the pilot's `summary.json` outputs must agree with the prior `simpleaf`-based outputs within these tolerances:

| Metric | Tolerance |
|---|---|
| Mapping rate | ±5 percentage points |
| Cells called | ±20% relative |
| Median genes per cell | ±15% relative |
| Median UMIs per cell | ±15% relative |

Larger deltas are not automatic fails — they may reflect legitimate algorithmic improvements — but each must be investigated and signed off before tagging `v0.3.0-production`.

## Operator sign-off

Pilot pass requires the operator to sign the following declarations after reviewing `pilot_qc_report.html`:

1. [ ] All Stability criteria (S1–S6) met
2. [ ] All Throughput criteria (T1–T8) met
3. [ ] All Biology criteria (B1–B7) met on spot-check
4. [ ] All Catalog criteria (C1–C9) met
5. [ ] Regression deltas reviewed; no concerning drift
6. [ ] Reference-manifest sha256 stable across all 1,000 samples
7. [ ] No P1 issues raised during pilot run

Signed: __________________________  Date: __________

## Failure response

If the pilot **fails** any criterion:

1. Stop immediately — do not let downstream waves launch from a failing freeze.
2. Classify each failure into: data-driven (catalog bug), pipeline-driven (code bug), or infra-driven (cluster bug).
3. Pipeline-driven failures → fix on `hotfix/pilot` branch, cherry-pick into `main`, re-tag `v0.3.0-pilot-freeze.1`, re-pilot a 100-sample subset reproducing the failure mode.
4. Data-driven failures (≤ 5%) → patch the catalog filter, document in `state/catalog-exclusions.md`, no re-tag needed.
5. Infra-driven failures → operator handles (queue, NFS, disk); no code change.

A hotfix that fundamentally alters the output schema (e.g., bumping `schema_version` to 1.2) requires a full re-pilot (1,000 samples), not the 100-sample subset.
