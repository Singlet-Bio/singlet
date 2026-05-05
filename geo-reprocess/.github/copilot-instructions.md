# Copilot Instructions

## Repository Overview

`geo-reprocess` is an HPC-native toolkit for building single-cell RNA-seq catalogs from GEO and reprocessing FASTQs at scale using simpleaf/alevin-fry. Part of the [Singlet AI](https://github.com/Singlet-AI) organization.

## Large File Creation

When creating files larger than ~200 lines, **always break the work into phases**:
1. Create a skeleton file first (imports, exports, basic structure) — under 200 lines
2. Add content in successive edits, each under 200 lines of new content
3. Never attempt to create a file >400 lines in a single `create_file` call

## Project Structure

- **Package:** `scgeo/` — Main Python package
  - `catalog/` — GEO discovery, metadata enrichment, protocol inference
  - `pipeline/` — Download, detect, quantify, QC, kraken2
  - `indices/` — Reference index building (genome FASTA + GTF → piscem)
  - `slurm/` — SLURM batch submission and monitoring
  - `cli/` — Command-line interface (`sc-geo`)
  - `config/` — Configuration system (dataclass-based, nested configs)
  - `io/` — I/O utilities (ENA client)
  - `utils/` — Shared utilities (NCBI E-utilities client)
- **Docs:** `docs/` — Python Sphinx documentation (furo theme, myst-parser)
- **Tests:** `tests/` — pytest suite
- **Examples:** `examples/` — 7 tutorial scripts (01-07)
- **Scripts:** `scripts/` — 6 operational analytics scripts
- **Controlled Access:** `controlled_access_catalog/` — 8-step discovery pipeline

## HPC Environment

- Login node: file edits, git, grep only — **never run Python/node on login node**
- Compute nodes: all Python, pip, pytest via SSH
- GPU jobs: SLURM only (sbatch)
- Pattern: `ssh <node> "source venv/bin/activate && cd /path && python script.py"`

## HPC Path Conventions

All data, indices, and pipeline outputs live on project storage (terabytes available). The home directory (`/mnt/home/debruinz/`) has a **100 GB quota** — never write data, temp files, or SLURM outputs there.

| Purpose | Path |
|---------|------|
| **Project base** | `/mnt/projects/debruinz_project/cellarium/` |
| **Workspace (code)** | `/mnt/projects/debruinz_project/cellarium/workspace/geo-reprocess` |
| **Pipeline outputs** | `/mnt/projects/debruinz_project/cellarium/pipeline/` |
| **Dataset (migrated)** | `/mnt/projects/debruinz_project/cellarium/dataset/` |
| **Catalog** | `/mnt/projects/debruinz_project/cellarium/catalog/` |
| **Reference indices** | `/mnt/projects/debruinz_project/cellarium/index/` |
| **Logs** | `/mnt/projects/debruinz_project/cellarium/pipeline/logs/` |
| **simpleaf home** | `/mnt/projects/debruinz_project/cellarium/af_home` |

Key environment variables (set automatically by SLURM scripts via `scgeo.slurm.submit`):
- `SCGEO_BASE` — project base directory
- `SCGEO_WORKSPACE` — workspace code directory (where `scgeo` package is importable)
- `ALEVIN_FRY_HOME` — simpleaf home for index caching

SLURM scripts should always:
- Write `--output` / `--error` logs to `pipeline/logs/`
- `cd` to the workspace on project storage, not the home directory
- Use `module load miniconda3/25.5.1` + `conda activate cellarium` for the Python environment

## Related Repositories

- [singlet](https://github.com/Singlet-AI/singlet) — Python client for SingletDB
- [singlepress](https://github.com/Singlet-AI/singlepress) — .1pz compression format
- [singlet-intelligence](https://github.com/Singlet-AI/singlet-intelligence) — ML models
- [singlet-strategy](https://github.com/Singlet-AI/singlet-strategy) — Business strategy
- [singletai-website](https://github.com/Singlet-AI/singletai-website) — Website (React + TypeScript)
- [papers](https://github.com/Singlet-AI/papers) — LaTeX manuscripts & reports

## Documentation

- Built with Sphinx + furo theme + myst-parser
- API docs auto-generated from docstrings via sphinx-autodoc
- Intersphinx links to singlet, singlepress docs
- Deployed to GitHub Pages via `.github/workflows/docs.yml`
