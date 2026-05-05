# Publication Pipeline DAG

> Updated by publish-orchestrator. Do not edit manually unless correcting errors.

---

## Active Manuscripts

### 1pz (SinglePress Format)

| Metric | Value |
|--------|-------|
| **phase** | 7 (complete) |
| **refs_count** | 31 |
| **refs_doi_pct** | 87% (+ 13% justified NO_DOI = 100% coverage) |
| **benchmark_trials** | 25 datasets × 4 formats + GPU/threading/BPCells/R |
| **pdf_errors** | — |
| **reviewer_confidence** | 0.85 / 0.90 / 0.88 (Methods/Writing/Novelty) |
| **ecosystem_tests** | Colab notebook + R Markdown created |
| **docs_sections** | 9 (overview, install, quickstart, CLI, API, benchmarks, FAQ, citation, R) |

**Phase Status**:
- [x] Phase 0: Draft exists (`manuscripts/singlepress-format/main.tex`)
- [x] Phase 1: FEATURE_BRIEF.json
- [x] Phase 2: refs.bib 31 entries (27 DOI + 4 NO_DOI justified), SOTA_TABLE.md (11 competitors)
- [x] Phase 3: benchmark_results_v3.json (25 datasets, 8 species, 4 formats) + 19 CSVs + GPU/threading data
- [x] Phase 4: main.tex polished + main.pdf compiles clean (29 pages, 0 fatal errors)
- [x] Phase 5: Reviewer confidence 0.85/0.90/0.88 — all ≥0.85
- [x] Phase 6: Colab notebook + R Markdown companion
- [x] Phase 7: Docs page at singletai-website/docs/singlepress.md (9 sections)

**Blockers**: None

---

## Manuscript Queue

| Slug | Title | Status |
|------|-------|--------|
| `1pz` | SinglePress: A Purpose-Built File Format for Single-Cell Omics Matrices | **All 7 phases complete** — ready for bioRxiv |
