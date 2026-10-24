# Cell type identity probabilities (`singlet.celltype_nn`)

Package `python/singlet/celltype_nn/`, scripts `scripts/celltype_nn/`. A network that gives every cell a calibrated probability
distribution over Cell Ontology (CL) types. It is trained on CELLxGENE Census author labels together with the calls of an
ensemble of annotators, and evaluated on Singlet's own reprocessed counts. Status, numbers and commands: sections 5-8.

## 1. What it outputs

`<GSM>.celltype.parquet`, one row per called cell, joined to the cell index and the other sidecars on `(gsm, col)`
(`docs/CELL-ARCHITECTURE.md`; schema version 1, `singlet.celltype_nn.output`):

| column | type | content |
|---|---|---|
| `gsm`, `col` | string, int64 | the cell (as in the label file) |
| `cell_type_cl`, `cell_type_label` | string | the call: the most specific CL term whose rolled-up probability is at least 0.5; null when only the root qualifies |
| `cell_type_prob` | float32 | rolled-up probability of the call |
| `argmax_cl`, `argmax_prob` | string, float32 | the output term with the most probability mass |
| `top1_cl` .. `top5_cl`, `top1_prob` .. `top5_prob` | string, float32 | the five terms with the most mass |
| `model_version` | string | `celltype-nn=<version>;census=<release>;cl=<release>` |

- **Probabilities.** The softmax runs over a fixed list of CL terms, internal terms included. Mass on "T cell" means a T cell
  whose subtype is unresolved. Rows sum to 1.
- **Roll-up.** The probability of any term, whether an output term or an ancestor of one, is the mass on that term and its
  descendants (`TermSpace.rollup`). The top five cover most of the mass, so summing them gives a lower bound for any ancestor;
  the model gives the exact value.
- **No forced call.** A cell whose mass is spread over sibling subtypes gets their common parent as its call. A cell spread over
  unrelated branches gets a general term with a low probability.
- **Fit with the label file.** `cell_type_cl` and `cell_type_label` mean the same as in the label file, so the cell index can
  use this call in place of the guide's.
- **Guard.** `assert_public_safe` runs before every write. It accepts exactly these columns, CL ids and names, probabilities
  with top-k descending and summing to at most 1, unique `col`, and one GSM. Genes, counts, barcodes and genotypes have no
  column to travel in. `tests/python/test_celltype_nn.py` checks it.

## 2. Annotator ensemble: survey and choice

Requirements for an ensemble member: an independent source of evidence (markers, other references, other models), CL-mappable
output, a licence that allows the use, active maintenance, and a cost that fits a corpus of about a million cells.

| tool | licence (GitHub, 2026-10-24) | last push | used | how / why not |
|---|---|---|---|---|
| Singlet guide (`singlet.celllabels`) | MIT (ours) | 2026-10 | yes | marker hierarchy keyed to CL; independent of every reference |
| CellTypist 1.7.1 | MIT | 2026-05 | yes | run as published, per cell (no majority voting): `Immune_All_Low` on every human cell plus the model matched to the tissue (16 models); labels mapped to CL by a crosswalk (section 3) |
| SingleR | GPL-3.0 | 2026-09 | rebuilt | GPL: rebuilt from the paper (Aran et al. 2019), no code used (`ensemble.SingleRRef`): pseudobulk references, pairwise markers, Spearman, 0.8 quantile, fine-tuning |
| Azimuth (Seurat reference mapping) | GPL-3.0 | 2026-05 | no | R only, and the pinned runtime has no R; its references (PBMC, lung/HLCA, kidney, heart, ...) are themselves CELLxGENE datasets, so they add little beyond the corpus at the cost of an R stack |
| scANVI (scvi-tools) | BSD-3-Clause | 2026-10 | no (follow-up) | usable directly, but a scANVI model trained on the corpus learns from the same author labels as the network, so it is not an independent vote; the network is itself the supervised reference model |
| popV | MIT | 2026-10 | no | wraps CellTypist, scANVI, OnClass and others over Tabula Sapiens, which is in Census; the two members it would add are covered above or are not independent |
| scType | GPL-3.0 | 2024-11 | no | GPL, unmaintained since 2024, and the same kind of evidence as the guide (marker scoring) |

The CellTypist models (`models.json`) carry no licence statement. Their calls are used only as weak training labels. They are
never redistributed, and neither the network nor its outputs contain them. If that is not acceptable, the network can be
retrained without that member (`train.py --no-ensemble` drops all refinements, or `targets` can be rebuilt without CellTypist).

**Training data licence.** CZ CELLxGENE Census data are CC BY 4.0. The trained weights and the per-cell probabilities are model
outputs, not the data, and are released CC0 with the atlas. The model card credits the Census release and lists its datasets
(`census_info/datasets` ids in `plan.json`).
