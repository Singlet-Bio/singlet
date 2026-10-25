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

## 3. Training corpus (phase 1)

`scripts/celltype_nn/corpus.py` (plan, fetch, orthologs, leakcheck, dedup), `prepare.py`; on Clipper in `$WORK/data/celltype_nn`.

- **Source.** CZ CELLxGENE Census 2025-11-08 (the current LTS), primary data only (`is_primary_data`), from the droplet assays
  Singlet processes: 10x 3'/5' (all versions), multiome RNA, Flex, Drop-seq, Seq-Well, BD Rhapsody WTA. Cells under 200 genes
  and labels that name no type (`unknown`, `cell`, `native cell`, ...) are dropped.
- **Stratified sample.** Per (cell type, tissue) at most 1,500 cells, taken round-robin across datasets so that no study
  dominates a type; at most 20,000 cells per dataset.

  | | primary cells (assay ok) | sampled | datasets | collections | cell types | tissues |
  |---|---|---|---|---|---|---|
  | human | 90.3M | 2,335,155 | 518 | 239 | 828 | 68 |
  | mouse | 6.6M | 474,609 | 43 | 38 | 386 | 33 |

- **Held out.** The 16 CELLxGENE collections linked to the benchmark studies are excluded (274k human and 136k mouse primary
  cells). Twelve datasets in those collections are not Census primary data, so their cells could have a primary copy elsewhere.
  A copy keeps its raw counts, so `corpus.py dedup` matched every corpus cell's (organism, raw_sum, nnz) against the 531k
  cells of the benchmark collections. No corpus dataset matched above chance: the highest was 10%, against a 2% median, all in
  small brain datasets with few UMIs. The non-primary datasets are therefore subsets of primary datasets inside the same
  excluded collections, and no copy of a benchmark cell is in the training data.
- **Validation split.** 20 whole collections (190,019 cells) are held out for early stopping and calibration. The benchmark is
  never used for either.
- **Genes.** The vocabulary has 6,000 human Ensembl ids:
  - per-tissue highly variable genes (human, plus mouse through orthologs), with every guide marker that passes the filters
    (2,001 of the 6,000);
  - only genes that Singlet's annotation has;
  - without mitochondrial, ribosomal-protein, XIST and chrY sex-model genes.

  Mouse genes map through 17,146 Ensembl one-to-one orthologs (BioMart), which cover 5,133 of the vocabulary genes.
- **Intron/exon ratios.** For the augmentation, each gene's intron/exon ratio is measured on Singlet's spliced/unspliced
  layers of two non-benchmark studies (GSE149689 PBMC, GSE202109): median 0.64, 90th percentile 4.6.

## 4. Model (phase 2)

**Inputs.** Counts are projected onto the vocabulary by Ensembl id. Each detected gene then gets its rank among the cell's
detected genes, scaled to (0, 1] (`transform="rank"`). The rank depends only on the order of the cell's own counts, so depth
and per-gene scale factors cancel. Optionally the input also carries the sample's tissue as a multi-hot over the guide's 30
tissue groups (`tissue_context`). In training that context comes from Census `tissue_general`; at inference it comes from the
same GEO text the guide reads, through the same matcher. It is withheld for 30% of training cells, so the model also works
without it.

**Network.** An MLP, 6,000 (+30) → 2,048 → 1,024 → 822 CL terms. LayerNorm, GELU, dropout 0.4, input dropout 0.3; AdamW
(lr 1e-3, weight decay 0.1), one-cycle schedule, 8 epochs. The epoch with the best validation closure likelihood is kept
(clean plus augmented validation cells). Training takes about 7 minutes on one H100 for 2.6M cells.

**Output space.** 822 CL terms: every term with at least 50 training cells, internal terms included. Rarer labels fold into
their most specific qualifying ancestor (`ontology.build_space`).

**Loss.** For a cell with target term t and output distribution p, the loss is `-log Σ_v p_v · credit(v, t)`:
- credit 1 for t or any descendant (a finer answer than the author's is never penalised);
- credit β·IC(v)/IC(t) for a strict ancestor v (a coarser answer earns partial credit), with IC the structural information
  content over the output terms;
- credit 0 for any other branch.

β matters. With β = 1 (full IC-ratio credit), the first run learned to hedge. Its argmax earned full credit on only 18% of
validation cells, and its closure likelihood was 10 nats, because "T cell" earns 0.45 against any T-cell subtype. β = 0 is the
pure subtree likelihood: on the validation collections the roll-up call was consistent with the author for 0.72 of cells, and
the model overfit after one epoch. β = 0.25 gave 0.80 and is the default.

**Ensemble targets.** Where at least two members call strict descendants of the author term, and the deepest term their calls
share is still below it, that term becomes a second target with weight 0.5. This applies to 305,372 corpus cells.

Members against author labels on corpus training cells, CL level:

| member | exact | finer | coarser | other branch | no call |
|---|---|---|---|---|---|
| guide | 0.135 | 0.313 | 0.060 | 0.478 | 0.015 |
| CellTypist (crosswalk) | 0.120 | 0.037 | 0.548 | 0.184 | 0.112 |
| SingleR-style (cross-fitted) | 0.180 | 0.312 | 0.062 | 0.447 | 0 |

**Augmentation** (on the GPU, per cell):
- binomial thinning to 5-100% of depth (p = 0.8);
- intronic reads added in proportion to each gene's measured intron/exon ratio, up to 1.5x (p = 0.5);
- ambient RNA from the batch's mean profile, up to 15% (p = 0.5);
- per-gene protocol bias from a bank of 64 log-normal profiles, sd 0.3 (p = 0.7);
- random gene dropout up to 10% (p = 0.5).

**Calibration.** One temperature, fitted after training on the validation collections, clean and augmented, half each.
- The closure likelihood is not used for this. Validation cells whose author label conflicts with every other study (another
  branch, probability near 0) dominate it and push the temperature up. On Singlet counts the result was an underconfident
  call: 0.54 stated against 0.74 observed in the largest bin.
- The temperature is chosen instead from a grid by the expected calibration error of the call itself (`calibrate.py`). That
  error measures what the output promises: among calls made with probability p, the truth lies inside the call's subtree a
  fraction p of the time.
- The model keeps both temperatures (`model_nll.pt` holds the likelihood fit).

**Decisions made after looking at the benchmark.** These are disclosed because they followed a benchmark result.
- The calibration criterion above was changed after the author-only model's first benchmark showed the underconfidence.
- The tissue context was added after the same run. Its coarse calls clustered in tissues where the guide relies on tissue
  context, which the guide's own ablation (`docs/cell_annotation.md` 3.1) had already measured at 0.15 lineage accuracy.

Hyperparameters were chosen on the validation collections only. The sweep covered:
- dropout and weight decay;
- width, 1,024-512 against 2,048-1,024;
- transform: log, z-scored log, rank;
- learning rate.

Rank with the wider network gave the best roll-up consistency at the same call depth (0.814 at depth 8.0, against 0.801 for
the log baseline).
