# Publication Pipeline Context Index

> Hierarchical summary for publish-orchestrator. Updated by workers after each phase.

---

## System Level (≤80 lines)

### Active Slug: 1pz

**What**: SinglePress (.1pz) — a purpose-built sparse matrix file format for single-cell omics.
**Package**: `singlepress` v1.0.0 (Python + R). Exports: `write_1pz`, `read_1pz`, `read_1pz_torch`, `cbind_1pz`, `rbind_1pz`, `subset_1pz`, `sample_1pz`, `open_1pz`, `colsums_1pz`. Interop: AnnData, scipy, HDF5, 10x.
**Draft**: `manuscripts/singlepress-format/main.tex` — 3 Results subsections, 6 figures, targets Genome Biology Software.
**Key claims**: 9.5× median compression, 868 MB/s decode, 2–4× smaller than H5AD, 3–4× faster decode.
**Refs**: `refs.bib` has 20 entries (19 DOIs).
**Style**: `templates/singletai-preprint.sty` — 2-column, Times, teal hyperlinks, `\correspondingauthor{}`.

### Infrastructure

**State dir**: `papers/state/` — dag.md, context-index.md, episodes.md
**Agent dir**: `papers/.github/agents/` — 8 agent files (orchestrator + 7 specialists)
**LaTeX style**: `papers/templates/singletai-preprint.sty`
**Brand color**: `#0C8B84` (teal)
**Author**: Zach DeBruine, GVSU IST + Singlet Bio Inc.

---

## Module Level

### manuscripts/singlepress-format/
- `main.tex`: Full draft, ~11 pages two-column. Sections: Introduction, Results (VOCSC compression, I/O performance, ecosystem interop), Discussion, Methods.
- `refs.bib`: 20 entries, 19 with DOIs. Missing: BPCells DOI needs verification.
- `figures/`: 6 main figures (fig1–fig6) + 2 supplementary (figS1–figS2). All PDF.
- `benchmark_results.json`: Exists from prior benchmarking runs.
- `generate_figures.py`: Not yet created (figures generated ad-hoc via R and Python scripts).

### templates/
- `singletai-preprint.sty`: 2-column preprint style. Has ORCiD icon, `\correspondingauthor{}`, `\keywords{}`. Missing: brand color definition, TikZ logo, `\AIdisclosure` environment.
