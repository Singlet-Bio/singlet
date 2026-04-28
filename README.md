# singlet

A unified single-cell genomics platform: raw SRA reads → processed atlas in one command.

## Repository Layout

| Directory | Contents |
|-----------|----------|
| `pipeline/` | C++ `singlify` binary — SRA download, STAR alignment, UMI dedup, cell calling, .1pz output |
| `python/` | Python `singlet` package — load, annotate, analyze .1pz files |
| `singlepress/` | .1pz format codec (Python + R bindings) |
| `gpu/` | CUDA/cuSPARSE GPU analysis kernels (lognorm, HVG, PCA, NMF, kNN, leiden, UMAP, DE) |
| `star/` | STAR aligner fork (singlet-lite branch — PGO + header-only integration) |
| `papers/` | Scientific manuscripts |
| `docs/` | User-facing documentation (rendered on singlet.bio) |

## Quick Start

```bash
# Install Python client
pip install "singlet @ git+https://github.com/Singlet-Bio/singlet#subdirectory=python"

# Load a processed sample from the Singlet Atlas
import singlet
adata = singlet.load("GSM1234567")
```

## Building from Source

### Pipeline (C++)
```bash
cd pipeline && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) singlify
```

### GPU Library (CUDA 12+)
```bash
cd gpu && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## License

MIT — see [LICENSE](LICENSE)
