# singlet-gpu — Documentation

A bare-metal cuBLAS / cuSPARSE / cuSOLVER GPU library for single-cell analysis. Reads singlet `.1pz` outputs zero-copy, runs the entire foundational EDA workflow on device, scales out-of-core to billion cells.

## Site map

- **[install.md](install.md)** — build from source, pip, R `install_github`, supported CUDA + GPU arch matrix.
- **[quickstart.md](quickstart.md)** — load a `.1pz`, run the standard pipeline (QC → norm → HVG → PCA → kNN → Leiden → UMAP → DE), write results.
- **[api/](api/)** — per-feature reference pages. One page per public function.
- **[notebooks/](notebooks/)** — reproducibility notebooks. Each notebook runs singlet-gpu and a reference tool (Scanpy / rapids-sc / Seurat) on real data, computes correlation, plots speedup at 3 scales.

## Render locally

```bash
# Install mdBook (one-time)
cargo install mdbook

# Build the static site
cd singlet-gpu/docs && mdbook build

# Serve locally with live reload
mdbook serve
```

The output is a static site that gets deployed to `singlet.bio/docs`.

## Audience

- **Bioinformaticians** — `quickstart.md` + the notebooks under `notebooks/`.
- **Library developers** — `install.md` from source + the API reference under `api/`.
- **Reviewers / methods readers** — every API page links to the original method paper, the singlet-gpu design doc, and the correctness equivalence notebook.

## Versioning

See [`../state/release-policy.md`](../state/release-policy.md). The umbrella header `include/singlet-gpu/singlet_gpu.hpp` is API-frozen across MINOR. Internal headers under module subdirs may break in any PATCH.
