# singlet Notebooks

Tutorial and reproducibility notebooks for the singlet ecosystem.

## Getting Started

| Notebook | Description |
|----------|-------------|
| [quickstart.ipynb](quickstart.ipynb) | Explore the singlet atlas — catalog, species, filters, statistics |
| [01_load_and_explore.ipynb](01_load_and_explore.ipynb) | Load a .1pz file, QC, cluster, and visualize |
| [02_gpu_analysis.ipynb](02_gpu_analysis.ipynb) | Full GPU-accelerated scRNA-seq pipeline with singlet-gpu |

## Reproducibility & Analysis

| Notebook | Feature | E2E Panel |
|----------|---------|-----------|
| [gene_counting.ipynb](gene_counting.ipynb) | Gene quantification equivalence vs STARsolo (r=0.9995) | A |
| [sex_calling.ipynb](sex_calling.ipynb) | Sex chromosome calling (100% concordance) | F |
| [ambient_rna.ipynb](ambient_rna.ipynb) | Ambient RNA contamination profiling | G |
| [doublet_detection.ipynb](doublet_detection.ipynb) | Computational doublet detection | H |
| [corpus_analytics.ipynb](corpus_analytics.ipynb) | Atlas-wide QC distributions and filtering | — |

## Requirements

```bash
pip install singlet-bio
# For GPU notebooks: pip install singlet-bio[gpu]
```

## Atlas Stats (auto-updated)

- 2,250 samples processed
- 924 successful (41%)
- 1,131 GEO series
- 7 species
- 2.7M total cells
- Median 1,167 cells/sample
- Median 80.4% mapping rate
