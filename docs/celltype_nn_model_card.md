# Model card: celltype-nn=0.1.0;census=2025-11-08;cl=2026-06-08

- Task: per-cell probability over Cell Ontology types (softmax; rows sum to 1), calibrated by one temperature.
- Training data: CZ CELLxGENE Census 2025-11-08 (LTS), primary data, droplet assays; 2,809,764 cells from 561 datasets in 268 collections (human 2,335,155, mouse 474,609), 70 tissues. Licence CC BY 4.0 (Census); this model and its outputs: CC0.
- Held out: the 16 CELLxGENE collections of the benchmark studies; 0 copies of benchmark cells found elsewhere were dropped.
- Weak labels: refinements of the author label agreed by at least two of: Singlet guide, CellTypist 1.7.1 (16 models, CL crosswalk), SingleR-style correlation (rebuilt, cross-fitted by dataset).
- Output: 822 CL terms (CL 2026-06-08); inputs 6000 genes (human Ensembl gene id; mouse via Ensembl one-to-one orthologs).
- Temperature 1.798 (fitted on the reliability of the call, validation collections; closure-likelihood fit 2.111).
- Singlet counts, 16 held-out studies, mean over studies: depth 1/2/3/4 = 0.825/0.654/0.333/0.360, wrong 0.251.

Per study (network): d1/d2/d3/d4, wrong

| study | d1 | d2 | d3 | d4 | wrong | cells |
|---|---|---|---|---|---|---|
| GSE125970 | 1.00 | 0.93 | 0.19 | nan | 0.69 | 8809 |
| GSE130973 | 0.96 | 0.94 | 0.33 | 0.86 | 0.20 | 7164 |
| GSE131882 | 0.99 | 1.00 | 0.67 | 0.60 | 0.10 | 13633 |
| GSE144870 | 0.99 | 0.22 | 0.01 | 0.98 | 0.24 | 71466 |
| GSE157480 | 1.00 | 0.99 | 0.76 | 0.06 | 0.19 | 3000 |
| GSE161089 | 0.63 | nan | nan | nan | 0.00 | 4970 |
| GSE164291 | 0.18 | 0.17 | 0.10 | nan | 0.03 | 5467 |
| GSE166766 | 0.90 | 0.27 | 0.16 | 0.03 | 0.04 | 49280 |
| GSE184111 | 0.95 | 0.49 | 0.19 | nan | 0.14 | 23935 |
| GSE185222 | 0.36 | 0.70 | 0.57 | 0.59 | 0.73 | 4686 |
| GSE185224 | 0.99 | 0.88 | 0.56 | nan | 0.16 | 10716 |
| GSE185477 | 0.95 | 0.94 | 0.42 | 0.14 | 0.56 | 1545 |
| GSE201773 | 0.71 | 0.14 | 0.04 | 0.00 | 0.62 | 4133 |
| GSE260685 | 0.71 | 0.36 | 0.07 | nan | 0.02 | 19380 |
| GSE270225 | 0.91 | 0.81 | 0.30 | 0.11 | 0.07 | 28365 |
| GSE280852 | 0.98 | 0.96 | 0.63 | 0.23 | 0.21 | 21145 |
