# Installation

## From PyPI

```bash
pip install singlepress
```

## Requirements

- Python 3.9+
- C++17 compiler (GCC ≥ 7, Clang ≥ 5, MSVC ≥ 19.14)
- libzstd (system library)
- numpy ≥ 1.21
- scipy ≥ 1.7

### Optional dependencies

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch dataloaders (`singlepress.torch`) |
| `anndata` | AnnData interop (`singlepress.interop`) |
| `pandas` | DataFrame metadata in .1pz files |
| `h5py` | HDF5 / 10x .h5 conversion |

Install all optional dependencies:

```bash
pip install singlepress[torch,bio,hdf5]
```

## From source

```bash
git clone https://github.com/Singlet-AI/singlepress.git
cd singlepress
pip install -e ".[dev]"
```

## HPC cluster

```bash
module load python/3.11 gcc/11
pip install --user singlepress
```

## R package

The R bindings require Rcpp, Matrix, and a system zstd installation:

```r
# Install from source
Rcpp::sourceCpp("R/singlepress_rcpp.cpp")
source("R/singlepress.R")

# Or install as a package
devtools::install_github("Singlet-AI/singlepress", subdir = "R")
```

## Verify installation

```python
import singlepress
print(singlepress.__doc__[:80])
# singlepress — Python interface for the .1pz sparse matrix format.
```
