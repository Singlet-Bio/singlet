# Contributing to Singlet

Thank you for your interest in contributing to Singlet! This document provides
guidelines and instructions for contributing.

## Development Setup

```bash
# Clone the monorepo
git clone https://github.com/Singlet-Bio/singlet.git
cd singlet

# Install Python package in development mode
pip install -e ".[dev]"

# Build C++ tests (requires GCC 13+, htslib, zstd, zlib)
make build

# Run all tests
make test
```

## Project Structure

```
singlet/
├── include/singlet/       C++ header-only libraries
│   ├── pz/                .1pz sparse matrix codec
│   ├── fq/                .1fq encoded FASTQ codec
│   ├── pileup/            streaming BAM pileup engine (70+ headers)
│   ├── gpu/               CUDA analysis kernels
│   └── star/              STAR aligner API
├── python/singlet/        Python package
│   ├── gpu/               GPU-accelerated analysis (cupy/CUDA)
│   ├── torch/             PyTorch DataLoaders and sparse tensors
│   ├── mcp/               MCP server for AI assistants
│   └── preprocessing/     Pipeline preprocessing utilities
├── r/                     R package (singlet)
├── src/                   C++ source (pipeline binary, GPU kernels)
├── tests/
│   ├── cpp/               102 C++ unit tests
│   └── python/            584 Python tests
├── docs/                  Documentation
├── pipeline/              singlify pipeline scripts
└── notebooks/             Jupyter notebooks and examples
```

## Running Tests

```bash
# All tests
make test

# C++ only (102 tests, ~72s)
make test-cpp

# Python only (584 tests, ~16s)
make test-python

# Lint check
make lint
```

## Code Style

### Python
- **Formatter**: `ruff format` (line-length 100)
- **Linter**: `ruff check` (E, F, I rules)
- **Type annotations**: Add return types to all public functions
- Run `make lint` before committing

### C++
- C++17, namespace `singlet`
- Header-only where possible
- No external dependencies beyond htslib, zstd, zlib, HDF5 (optional)

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, atomic commits
3. Ensure `make test` and `make lint` pass
4. Update documentation if relevant
5. Submit a PR with a clear description

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(module): short description
fix(module): what was broken and how it's fixed
docs: what was documented
test(module): what tests were added
style: formatting/lint changes only
```

## Architecture Notes

- **C++ libraries are header-only**: No compiled `.so`/`.a` needed for most use cases
- **CMake targets**: `Singlet::pz`, `Singlet::fq`, `Singlet::pileup`
- **Python package uses bundled parquet**: No network needed for catalog/sample queries
- **GPU code requires CuPy**: All GPU functions are in `singlet.gpu.*`
- **MCP server**: Exposes atlas data via Model Context Protocol (stdio transport)

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (see [LICENSE](LICENSE)).
