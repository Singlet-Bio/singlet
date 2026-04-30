# singlet Notebooks

Tutorial and reproducibility notebooks for the singlet ecosystem.

## Getting Started

| Notebook | Description |
|----------|-------------|
| [quickstart.ipynb](quickstart.ipynb) | Explore the singlet atlas — catalog, species, filters, statistics |
| [01_load_and_explore.ipynb](01_load_and_explore.ipynb) | Load a .1pz file, QC, cluster, and visualize |
| [02_gpu_analysis.ipynb](02_gpu_analysis.ipynb) | Full GPU-accelerated scRNA-seq pipeline with singlet-gpu |

## Reproducibility

| Notebook | Feature | E2E Panel |
|----------|---------|-----------|
| gene_counting.ipynb | Gene quantification equivalence vs STARsolo | A |
| donor_demux.ipynb | Donor demultiplexing | B |
| atac_fragments.ipynb | ATAC fragment generation | C |
| cite_seq_adt.ipynb | CITE-seq ADT counting | D |

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
