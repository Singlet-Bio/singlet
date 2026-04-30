# singlet Notebooks

17 executed Jupyter notebooks demonstrating the singlet ecosystem.

## Getting Started

| Notebook | Description |
|----------|-------------|
| [quickstart.ipynb](quickstart.ipynb) | Browse the atlas catalog — 2,358 samples, filters, statistics |
| [01_load_and_explore.ipynb](01_load_and_explore.ipynb) | Load 75K cells with `load_dir()`, cluster with scanpy (PCA→UMAP→Leiden) |
| [sample_qc_report.ipynb](sample_qc_report.ipynb) | Complete one-call QC report — UMIs, genes, doublets, cell cycle, ancestry |
| [pipeline_outputs.ipynb](pipeline_outputs.ipynb) | Reference guide: all 40+ files singlify produces per sample |

## QC & Quality Control

| Notebook | Description |
|----------|-------------|
| [cell_calling.ipynb](cell_calling.ipynb) | EmptyDrops deviance testing — 74K cells called |
| [doublet_detection.ipynb](doublet_detection.ipynb) | UMI-based doublet detection (13.8% rate, 20× score separation) |
| [ambient_rna.ipynb](ambient_rna.ipynb) | Ambient RNA contamination profiling |
| [cell_cycle.ipynb](cell_cycle.ipynb) | Cell cycle phase scoring (G1/S/G2M) |
| [saturation_curve.ipynb](saturation_curve.ipynb) | Sequencing depth vs gene/UMI discovery |

## Genomic Features

| Notebook | Description |
|----------|-------------|
| [rna_velocity.ipynb](rna_velocity.ipynb) | Spliced + unspliced matrices ready for scVelo |
| [splicing.ipynb](splicing.ipynb) | 37,909 alternative splicing events per sample |
| [mt_variants.ipynb](mt_variants.ipynb) | Mitochondrial heteroplasmy — lineage tracing markers |
| [ancestry_calling.ipynb](ancestry_calling.ipynb) | Genetic ancestry inference (5 super-populations) |
| [sex_calling.ipynb](sex_calling.ipynb) | Sex/karyotype calling (100% concordance) |

## Validation & Corpus

| Notebook | Description |
|----------|-------------|
| [gene_counting.ipynb](gene_counting.ipynb) | Formal equivalence vs STARsolo (r=0.9995) |
| [corpus_analytics.ipynb](corpus_analytics.ipynb) | Atlas-wide QC distributions across 975 samples |
| [protocol_detection.ipynb](protocol_detection.ipynb) | 15+ protocols auto-detected, QC by protocol |

## Requirements

```bash
pip install "singlet-bio @ git+https://github.com/Singlet-Bio/singlet#subdirectory=python"
# Or: pip install singlet-bio  (when published to PyPI)
```

## Atlas Stats

- **2,358** samples processed
- **979** successful (42%)
- **1,164** GEO series
- **7** species (human, mouse, rat, zebrafish, pig, chicken, macaque)
- **2.89M** total cells
- **17** notebooks with executed outputs
