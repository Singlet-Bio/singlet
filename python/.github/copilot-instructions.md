# Copilot Instructions

## Repository Overview

`singlet` is a Python client for SingletDB — the world's largest uniformly processed single-cell database (19,790+ datasets, 24 species). Part of the [Singlet AI](https://github.com/Singlet-AI) organization.

## Large File Creation

When creating files larger than ~200 lines, **always break the work into phases**:
1. Create a skeleton file first — under 200 lines
2. Add content in successive edits, each under 200 lines
3. Never attempt to create a file >400 lines in a single `create_file` call

## Project Structure

- **Package:** `singlet/` — Main Python package
  - `_catalog.py` — Dataset browsing and search
  - `_loader.py` — Download/load from Zenodo or AWS
  - `_io.py` — Read/write .spz files
  - `_auth.py` — API key management
  - `_query.py` — Cross-atlas structured search
  - `torch.py` — Zero-copy PyTorch sparse tensors (CSR/COO, GPU)
  - `convert.py` — Format conversion (HDF5, Zarr, TileDB-SOMA)
  - `nmf.py` — Server-side NMF projection and annotation
  - `preprocessing/` — FASTQ→.spz pipeline
- **C++ Extension:** `src/_singlepress.cpp` — pybind11 bindings
- **C++ Headers:** `include/singlepress/singlepress.h` — Header-only compression
- **Docs:** `docs/` — Sphinx documentation (furo theme)
- **Tests:** `tests/` — pytest suite (io, torch, singlepress, integration)

## HPC Environment

- Login node: file edits, git, grep only — **never run Python on login node**
- Compute nodes: all Python, pip, pytest via SSH

## Related Repositories

- [singlepress](https://github.com/Singlet-AI/singlepress) — Compression engine
- [geo-reprocess](https://github.com/Singlet-AI/geo-reprocess) — HPC pipeline
- [singlet-intelligence](https://github.com/Singlet-AI/singlet-intelligence) — ML models
- [singlet-strategy](https://github.com/Singlet-AI/singlet-strategy) — Business strategy
- [singletai-website](https://github.com/Singlet-AI/singletai-website) — Website
- [papers](https://github.com/Singlet-AI/papers) — LaTeX manuscripts & reports
