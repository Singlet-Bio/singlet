# Installing singlify

`singlify` ships as two sister packages sharing the same header-only C++
reader: a Python package at `singlify/python/` and an R package at
`singlify/r/`. The C++ core lives at `singlify/include/singlet-pileup/pz_reader.h`
and is staged into both packages at build time, so both bindings are
**byte-identical** for decoding `.1pz` pipeline outputs.

Everything below assumes you want to read the singlify single-cell
reprocessing pipeline's outputs (the `.1pz` files and their sidecar
TSVs) into your analysis environment of choice. If you want to call
the live BAM pileup engine instead, see
[`singlify/python/docs/quickstart.rst`](python/docs/quickstart.rst).

## Python (PyPI name: `singlify`)

### From GitHub (always available)

```bash
pip install "git+https://github.com/zdebruine/Singlet-AI.git#subdirectory=singlify/python"
```

### From PyPI (once released)

```bash
pip install singlify
```

### For Scanpy / AnnData workflows

```bash
pip install "singlify[scanpy]"
```

### Requirements

| Dependency    | Reason                              |
|---------------|-------------------------------------|
| Python ≥ 3.9  | Base runtime                        |
| numpy, scipy  | Sparse matrices                     |
| zstandard     | Metadata / introspection in pure Python |
| anndata       | Optional — for `singlify.interop.anndata` |
| scanpy        | Optional — for `singlify.interop.scanpy`  |

Building the `_pz_io` pybind11 extension from source additionally needs
a C++17 compiler and `libzstd` (≥ 1.4). Wheels (once published) bundle
the extension for Linux, macOS, and Windows.

### Verify the install

```bash
python -c "import singlify; print(singlify.__version__)"
singlify info path/to/pipeline_output/
singlify verify path/to/pipeline_output/gene_counts.1pz
```

## R (CRAN name: `singlify`)

### From GitHub (always available)

```r
install.packages("remotes")
remotes::install_github("zdebruine/Singlet-AI", subdir = "singlify/r")
```

### From CRAN (once released)

```r
install.packages("singlify")
```

### Requirements

| Dependency | Reason                              |
|------------|-------------------------------------|
| R ≥ 4.2    | Base runtime                        |
| Rcpp       | C++ binding                         |
| Matrix     | dgCMatrix                           |
| SingleCellExperiment, SummarizedExperiment | Optional — for `as_sce()` |
| Seurat     | Optional — for `as_seurat()`        |
| libzstd ≥ 1.4 | System package (see `singlify/r/inst/INSTALL_NOTES.md` for per-platform install commands) |

### Verify the install

```r
library(singlify)
mat <- read_1pz("path/to/pipeline_output/gene_counts.1pz")
dim(mat)
attr(mat, "user_kv")[["gsm_id"]]
```

## Development install (both)

```bash
git clone https://github.com/zdebruine/Singlet-AI.git
cd Singlet-AI

# Python — editable
cd singlify/python
pip install -e ".[dev]"
pytest tests/

# R — install from local source
cd ../r
R CMD INSTALL .
R -e 'testthat::test_check("singlify")'
```

## Header-only reader sync

The C++ reader lives at `singlify/include/singlet-pileup/pz_reader.h`. It
is copied into `singlify/r/inst/include/singlet-pileup/` as part of the
R package build. The Python package imports it directly via `-I../include`
in its `CMakeLists.txt` — no copy needed.

When you edit `pz_reader.h`:

```bash
# Sync the R copy (Python picks it up automatically)
cp singlify/include/singlet-pileup/pz_reader.h    singlify/r/inst/include/singlet-pileup/
cp singlify/include/singlet-pileup/pz_writer.h    singlify/r/inst/include/singlet-pileup/
cp singlify/include/singlet-pileup/sparse_accumulator.h singlify/r/inst/include/singlet-pileup/
```

The R package's GitHub Actions workflow does this automatically before
each `R CMD check` run.

## CI

Both packages are exercised by GitHub Actions on every push that
touches their sources:

| Workflow                     | What it does                                              |
|------------------------------|-----------------------------------------------------------|
| `python-ci.yml`              | `pytest tests/` on Ubuntu+macOS × Python 3.9/3.10/3.11/3.12 |
| `python-docs.yml`            | Sphinx HTML build with `-W` (warnings as errors)          |
| `python-build-check.yml`     | `python -m build --sdist` + `twine check`                 |
| `R-CMD-check.yml`            | `R CMD check --as-cran` on Ubuntu+macOS+Windows × R release/devel |
