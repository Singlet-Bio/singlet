# Installation

## Requirements

- Python ≥ 3.9
- C++ compiler with C++17 support (for building the SinglePress extension)

## Basic Install

```bash
pip install singlet
```

This installs the core package with AnnData and scipy support.

## Optional Dependencies

```bash
# PyTorch integration (zero-copy sparse tensors, GPU support)
pip install singlet[torch]

# Zarr format support
pip install singlet[zarr]

# TileDB-SOMA support
pip install singlet[tiledb]

# Everything
pip install singlet[all]
```

## From Source

```bash
git clone https://github.com/CellariumAI/singlet.git
cd singlet
pip install -e ".[dev]"
```

Building from source requires:
- A C++17-capable compiler (GCC ≥ 7, Clang ≥ 5, MSVC 2017+)
- pybind11 (installed automatically)

## Verify Installation

```python
import singlet
print(singlet.__version__)

# Verify C++ extension
from singlet._singlepress import sp_info
print("SinglePress extension loaded successfully")
```
