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

# Or load directly from a singlify output directory
adata = singlet.load_dir("/path/to/sample")
# → AnnData with gene counts, QC, doublets, cell cycle, ancestry, sex, summary
```

## Notebooks (18 ready)

| Notebook | Topic |
|----------|-------|
| [quickstart](notebooks/quickstart.ipynb) | Catalog API, browse 2,407 samples |
| [gene_counting](notebooks/gene_counting.ipynb) | STARsolo equivalence (r=0.9995) |
| [sex_calling](notebooks/sex_calling.ipynb) | Sex/karyotype calling validation |
| [ambient_rna](notebooks/ambient_rna.ipynb) | Ambient RNA contamination profiling |
| [doublet_detection](notebooks/doublet_detection.ipynb) | UMI-based doublet detection |
| [corpus_analytics](notebooks/corpus_analytics.ipynb) | Atlas-wide QC distributions |
| [01_load_and_explore](notebooks/01_load_and_explore.ipynb) | Full scanpy pipeline (PCA→UMAP→Leiden) |
| [cell_cycle](notebooks/cell_cycle.ipynb) | Cell cycle phase scoring |
| [sample_qc_report](notebooks/sample_qc_report.ipynb) | Complete one-call QC report |
| [saturation_curve](notebooks/saturation_curve.ipynb) | Sequencing depth analysis |
| [ancestry_calling](notebooks/ancestry_calling.ipynb) | Genetic ancestry inference |
| [mt_variants](notebooks/mt_variants.ipynb) | Mitochondrial heteroplasmy |
| [splicing](notebooks/splicing.ipynb) | Alternative splicing events |
| [rna_velocity](notebooks/rna_velocity.ipynb) | Spliced/unspliced for scVelo |
| [pipeline_outputs](notebooks/pipeline_outputs.ipynb) | Complete outputs reference |
| [cell_calling](notebooks/cell_calling.ipynb) | EmptyDrops deviance testing |
| [protocol_detection](notebooks/protocol_detection.ipynb) | 15+ protocol auto-detection |
| [1pz_format](notebooks/1pz_format.ipynb) | .1pz format: 8.7× smaller than h5ad |

## Atlas Stats

- **2,407 samples** processed (1,001 SUCCESS)
- **1,181 GEO series** covered
- **7 species**: human, mouse, macaque, fruit fly, chicken, zebrafish, chimpanzee
- **2.94M cells** in the atlas
- **18 notebooks** with embedded matplotlib plots ([view on singlet.bio](https://singlet.bio/notebooks))
- **Text search**: `singlet.samples(search="lung")` across GEO titles

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
