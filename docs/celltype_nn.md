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

**Network.** An MLP, 6,000 (+30) → 2,048 → 822 CL terms (one hidden layer). LayerNorm, GELU, dropout 0.4, input dropout 0.3; AdamW
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
- width: two layers of 1,024 and 512 against one layer of 2,048. The sweep runs meant to have 2,048 and 1,024 got a single
  2,048 layer, because `sbatch --export` splits values at commas; that single layer is the configuration measured and kept;
- transform: log, z-scored log, rank;
- learning rate.

Rank with the wider network gave the best roll-up consistency at the same call depth (0.814 at depth 8.0, against 0.801 for
the log baseline).

## 5. Evaluation on Singlet's counts (phase 3)

**Setup.** This is the benchmark of `docs/cell_annotation.md` section 3: the same 16 studies and 66 samples, 291,331 cells with
author labels transferred by barcode, none of them in the training corpus (section 3).
- Every method is scored on Singlet's own counts (exon + intron, as served) with `singlet.celllabels.evaluate` on the full
  guide: depth-d accuracy, and the exact / coarse / finer / wrong fractions.
- Means are over studies; each study pools its samples' cells.
- The guide was rerun in the same harness, which gives 0.838/0.734/0.420/0.320 and wrong 0.336. The published figures
  (0.842/0.747/0.428/0.320, 0.350) come from a different per-sample aggregation; the comparison below is like for like.
- Commands: `bench.py` (one job per study for the members), then `bench_summary.py`.

**Network.** `m1_notissue`: Census author terms plus ensemble refinements, rank inputs, augmentation, no tissue context.
`network, τ` is the most specific term whose rolled-up probability is at least τ; the published call (`cell_type_cl`) uses
τ = 0.5.

| method | d1 | d2 | d3 | d4 | consistent | wrong | CL consistent | ECE |
|---|---|---|---|---|---|---|---|---|
| guide (bar to beat) | 0.838 | 0.734 | 0.420 | 0.320 | 0.582 | 0.336 | 0.501 | |
| SingleR-style member | 0.877 | 0.741 | 0.380 | 0.428 | 0.634 | 0.344 | 0.475 | |
| CellTypist member (crosswalk) | 0.277 | 0.202 | 0.082 | 0.085 | 0.180 | 0.220 | 0.601 | |
| vote of the three members | 0.728 | 0.615 | 0.300 | 0.271 | 0.580 | 0.189 | 0.765 | |
| **network, τ = 0.3** | **0.895** | **0.740** | 0.405 | **0.465** | 0.632 | 0.345 | 0.619 | |
| network, τ = 0.4 | 0.850 | 0.698 | 0.372 | 0.409 | 0.619 | 0.298 | 0.682 | |
| **network, τ = 0.5 (published call)** | 0.825 | 0.654 | 0.333 | 0.360 | **0.638** | **0.251** | 0.732 | 0.079 |
| network, τ = 0.7 | 0.765 | 0.551 | 0.252 | 0.267 | 0.651 | 0.167 | 0.796 | |
| network, argmax | 0.903 | 0.744 | 0.418 | 0.498 | 0.619 | 0.374 | 0.587 | |
| network on exon-only counts | 0.798 | 0.607 | 0.297 | 0.340 | 0.633 | 0.227 | 0.750 | 0.102 |

- **Against the bar.**
  - At τ = 0.3 the network has the guide's error rate (0.345 against 0.336). At that rate it is ahead at depth 1 (+0.057),
    depth 2 (+0.006) and depth 4 (+0.145), and 0.015 behind at depth 3.
  - At τ = 0.5 it makes a quarter fewer wrong calls (0.251 against 0.336) and is more often consistent with the author
    (0.638 against 0.582). It is 0.08-0.09 behind at depths 2-3, because it stops at the parent where it is unsure.
  - The probability lets a user choose a point on this curve; the guide offers only one point.
- **CellTypist** is weak alone: its crosswalk maps many labels to coarse terms, and many of those terms fall outside the
  guide's nodes. Its value is as a voter.
- **The SingleR-style member** is the strongest single member on Singlet's counts.
- **Exon-only counts** cost 0.02-0.05 at every depth, about what the guide loses. Intronic reads still help.

**Calibration** (the τ = 0.5 call: its rolled-up probability against "the truth lies inside the call's subtree"; 242,045
checkable cells; ECE 0.079):

| probability | cells | mean probability | observed |
|---|---|---|---|
| 0.5-0.6 | 84,378 | 0.542 | 0.697 |
| 0.6-0.7 | 43,063 | 0.647 | 0.740 |
| 0.7-0.8 | 32,338 | 0.748 | 0.739 |
| 0.8-0.9 | 35,466 | 0.855 | 0.846 |
| 0.9-1.0 | 47,800 | 0.941 | 0.970 |

Above 0.7 the probability is right to within 0.03. Below 0.7 it is conservative: the true rate is higher than stated. The
author-only model (`sweep_s6`) reached ECE 0.038 on the same benchmark, so the ensemble targets cost some calibration under
the shift to Singlet's counts. On the validation collections the ECE is 0.021.

**Per study** (d1/d2/d3, wrong in brackets):

| study | tissue | cells | guide | SingleR-style | network τ = 0.5 | network τ = 0.3 |
|---|---|---|---|---|---|---|
| GSE125970 | colon / rectum / ileum epithelium | 8,809 | 1.00/0.79/0.15 (0.83) | 1.00/0.90/0.22 (0.78) | 1.00/0.93/0.19 (0.69) | 1.00/0.96/0.23 (0.73) |
| GSE130973 | skin | 7,164 | 0.95/0.94/0.34 (0.26) | 0.92/0.87/0.31 (0.23) | 0.96/0.94/0.33 (0.20) | 0.97/0.96/0.41 (0.25) |
| GSE131882 | kidney cortex, snRNA | 13,633 | 0.85/0.85/0.76 (0.16) | 1.00/0.99/0.90 (0.10) | 0.99/1.00/0.67 (0.10) | 1.00/1.00/0.71 (0.11) |
| GSE144870 | thymus, Lin-CD34+ | 71,466 | 1.00/0.88/0.16 (0.57) | 1.00/0.97/0.30 (0.11) | 0.99/0.22/0.01 (0.24) | 0.99/0.53/0.01 (0.45) |
| GSE157480 | mouse marrow / brain | 3,000 | 0.96/0.90/0.84 (0.15) | 0.96/0.95/0.81 (0.23) | 1.00/0.99/0.76 (0.19) | 1.00/0.99/0.81 (0.26) |
| GSE161089 | airway basal cells | 4,970 | 1.00/-/- (0.00) | 0.99/-/- (0.00) | 0.63/-/- (0.00) | 0.98/-/- (0.00) |
| GSE164291 | fallopian tube (species mix-up) | 5,467 | 0.18/0.18/0.10 (0.03) | 0.22/0.17/0.09 (0.83) | 0.18/0.17/0.10 (0.03) | 0.72/0.18/0.10 (0.27) |
| GSE166766 | bronchial epithelium, culture | 49,280 | 1.00/0.92/0.53 (0.27) | 0.90/0.34/0.20 (0.18) | 0.90/0.27/0.16 (0.04) | 0.99/0.56/0.32 (0.21) |
| GSE184111 | ureter | 23,935 | 0.98/0.86/0.55 (0.33) | 0.90/0.85/0.34 (0.30) | 0.95/0.49/0.19 (0.14) | 0.90/0.70/0.33 (0.30) |
| GSE185222 | dental pulp | 4,686 | 0.35/0.64/0.52 (0.74) | 0.63/0.68/0.57 (0.43) | 0.36/0.70/0.57 (0.73) | 0.37/0.70/0.60 (0.74) |
| GSE185224 | small intestine | 10,716 | 1.00/0.91/0.69 (0.26) | 1.00/0.91/0.59 (0.39) | 0.99/0.88/0.56 (0.16) | 1.00/0.93/0.67 (0.24) |
| GSE185477 | liver | 1,545 | 0.89/0.88/0.45 (0.66) | 0.95/0.94/0.45 (0.71) | 0.95/0.94/0.42 (0.56) | 0.96/0.95/0.45 (0.70) |
| GSE201773 | mouse ependyma | 4,133 | 0.67/0.26/0.25 (0.52) | 0.78/0.50/0.14 (0.70) | 0.71/0.14/0.04 (0.62) | 0.76/0.17/0.03 (0.83) |
| GSE260685 | ovary | 19,380 | 0.80/0.43/0.15 (0.06) | 0.93/0.47/0.09 (0.07) | 0.71/0.36/0.07 (0.02) | 0.75/0.61/0.21 (0.07) |
| GSE270225 | bladder | 28,365 | 0.90/0.77/0.18 (0.21) | 0.89/0.74/0.13 (0.18) | 0.91/0.81/0.30 (0.07) | 0.95/0.88/0.45 (0.11) |
| GSE280852 | liver | 21,145 | 0.88/0.80/0.62 (0.32) | 0.98/0.83/0.56 (0.26) | 0.98/0.96/0.63 (0.21) | 0.98/0.97/0.75 (0.27) |

The network beats the guide in bladder (where the guide fails on ambient smooth muscle), liver, skin, colon and kidney. It is
far behind in two places:
- **Thymic progenitors (GSE144870).** On the authors' own matrix of these cells, the network's argmax gets depth 2/3 =
  0.98/0.93 (section 6). The loss is therefore reprocessing, not knowledge. The rank-correlation member is robust to that
  shift; the network is not yet.
- **Cultured bronchial epithelium (GSE166766).** On the authors' matrix the network is no better (argmax depth 2 0.73).
  Census has few cultured airway cells, so this is a corpus gap.

## 6. Same cells, two processings

The network was run on the CELLxGENE author matrices of the benchmark cells (`bench_cxg.py`: raw counts from `raw.X`, the
same cells by barcode core and label). Model `m1`:

| counts | d1 | d2 | d3 | d4 | wrong |
|---|---|---|---|---|---|
| authors' matrices, τ = 0.5 | 0.883 | 0.654 | 0.369 | 0.292 | 0.229 |
| authors' matrices, argmax | 0.897 | 0.736 | 0.493 | 0.440 | 0.347 |
| Singlet's counts, τ = 0.5 | 0.799 | 0.606 | 0.323 | 0.357 | 0.257 |

Reprocessing costs 0.05-0.08 at depths 1-3 on average, concentrated in the thymus. Kidney snRNA goes the other way: depth 1 is
0.55 on the authors' exon-only nuclei counts against 0.99 on Singlet's exon + intron counts.

## 7. Ablations (Singlet's counts, τ = 0.5 call)

| model | d1 | d2 | d3 | d4 | wrong | ECE |
|---|---|---|---|---|---|---|
| author labels only (`sweep_s6`) | 0.822 | 0.574 | 0.302 | 0.382 | 0.227 | 0.038 |
| + ensemble refinements (`m1_notissue`, the model above) | 0.825 | 0.654 | 0.333 | 0.360 | 0.251 | 0.079 |
| + tissue context (`m1`) | 0.799 | 0.606 | 0.323 | 0.357 | 0.257 | 0.092 |
| author labels + tissue context (`m0t`) | 0.801 | 0.575 | 0.307 | 0.349 | 0.231 | 0.092 |
| ensemble + tissue context, no augmentation (`m1_noaug`) | 0.824 | 0.615 | 0.332 | 0.373 | 0.259 | 0.096 |

- **Ensemble refinements** add 0.08 at depth 2 and 0.03 at depth 3.
- **Tissue context** does not help: the network reads tissue from expression as well as from the GEO text. It is off by
  default.
- **Augmentation** leaves accuracy level but brings calibration error on Singlet's counts from 0.096 down to 0.079-0.092.
- **Singlet pseudo-labels** (`m1p`). Training adds 1,241,338 cells from 491 non-benchmark Singlet studies, labelled by the
  consensus of at least two members, at weight 0.5 (`singlet_pseudo.py`). At τ = 0.5 this gives 0.843/0.676/0.354/0.407,
  wrong 0.329, ECE 0.039: deeper and better calibrated on Singlet's counts.
  - But it makes more wrong calls at every threshold (τ = 0.3: 0.867/0.742/0.409/0.458, wrong 0.386). At a matched error
    rate, `m1_notissue` is ahead: at τ = 0.3 it is 0.895/0.740/0.405/0.465 with wrong 0.345.
  - The consensus carries the members' shared mistakes into training. It does not fix the thymus either (depth 2: 0.40).
  - `m1_notissue` stays the default. `m1p` is the choice where calibration on Singlet's counts matters more than the error
    rate.

## 8. Atlas inference plan (phase 4, not run)

`scripts/celltype_nn/infer_campaign.{py,sbatch}`. The plan and study shards come from `/api/manifest`, the same as the
annotation campaign. Each array task runs the network on every called cell of its studies:
- v2 studies are read by range requests; legacy bundles are downloaded to node-local scratch and deleted afterwards;
- each task writes `<OUT>/<GSE>/<GSM>.celltype.parquet` and `<GSM>.celltype.json` (QC: cells, call counts, mean
  probability, `barcode_digest`);
- every table passes `assert_public_safe` before it is written;
- a sample whose sidecar exists at the current `model_version` is skipped;
- nothing is written to R2 or D1.

**Smoke test** (Clipper job 784357, 4 CPUs, outputs in `$WORK/out/celltype_nn/smoke`): GSE145370 (v2, 14 samples, 40,614
cells) and GSE149689 (legacy, 20 samples, 53,646 cells). Every file passed the guard.

| | measured | atlas, 193.1M cells |
|---|---|---|
| prediction (projection + network, CPU) | 14.4 s for 94,260 cells on 4 threads: 0.17 CPU-h per M cells | 33 CPU-h |
| reading, v2 (range requests) | 1.4 s for 40,614 cells | small |
| reading, legacy (decode, after download) | 52 s for 53,646 cells on 4 threads: 1.1 CPU-h per M cells | about 210 CPU-h while studies are legacy |
| sidecar size | 4.54 MB for 94,260 cells: 48 B per cell | 9.3 GB |

This comes to about 250 CPU-hours, or about 5 hours of wall time at 50 concurrent 4-CPU tasks on `short-cpu`, plus the 1.5 TB
of legacy downloads the annotation campaign also made. Clipper is enough; Anvil is not needed. A GPU is not needed either:
prediction is 15% of the cost and reading is the rest. Running the network inside the next annotation campaign would read
each study once for both.

Commands, for Zach to approve; nothing has been submitted:

```
# plan (a job, not the login node): 1,982 human + mouse studies
python scripts/celltype_nn/infer_campaign.py plan --shards 200 --out $OUT/plan.tsv
# run, 50 at a time
sbatch --array=0-199%50 --export=ALL,OUT=$OUT,MODEL=$WORK/data/celltype_nn/models/<final>/model.pt,\
MODEL_VERSION="celltype-nn=0.1.0;census=2025-11-08;cl=2026-06-08" scripts/celltype_nn/infer_campaign.sbatch
python scripts/celltype_nn/infer_campaign.py status --out $OUT
```

Ingest after the run is not part of this change. The cell index would read `cell_type_cl` / `cell_type_prob` like the label
file's `cell_type_cl` / `cell_type_score`, plus the top-k columns as feature columns. It would also write `cell_feature` rows
for the network's outputs and `sample_cell_summary` rows from the per-sample JSON.

## 9. Comparison with CellTypist, scANVI, Seurat and SingleR (public benchmark, capability `cell-type`)

The result is `bench/public/results/cell-type.json`, in the Singlet 1.0.0 contract (`bench/public/CONTRACT.md`). The module
`bench/public/capabilities/cell_type.py` builds it from artifacts with `compute(artifact_dir)`, and the shared scoring is
`singlet.celltype_nn.benchmark`.

**Setup.**
- **Dataset:** cxg16, the same 16 studies and 291,331 cells as section 5, scored on Singlet's counts. Every tool reads the
  same counts on the same 6,000-gene vocabulary.
- **Reference:** every tool learns from the same Census training corpus (benchmark collections excluded).
  - CellTypist 1.7.1 (`celltypist.train`, SGD with mini-batches) and Singlet's network train on all of it.
  - Seurat 5.0.0 reference mapping (the Azimuth method: FindTransferAnchors / TransferData), SingleR 2.0.0 (`aggr.ref`) and
    scANVI (scvi-tools 1.4.2; scVI then scANVI, scArches query mapping) use a stratified subsample of at most 200 cells per
    term: 145k human and 62k mouse cells.
  - Labels are the curated CL terms, so no tool needs a crosswalk.
- **How each tool was run:** the R tools run as published, in their own process (`scripts/celltype_nn/compete/run_r.R`,
  Seurat 5 container image); the Python tools through `run_py.py`.
- **Runtime:** Singlet and CellTypist are timed back to back in one job with the same 8 threads (`timing.py`). The slower
  tools are timed as they ran.

**Results** (mean over 16 studies; strict match-or-beat against the best competitor, no tolerance; 95% CIs, from a
bootstrap over studies, are in the JSON):

| metric | Singlet | CellTypist | scANVI | Seurat | SingleR | pass |
|---|---|---|---|---|---|---|
| accuracy, depth 1 (compartment) | **0.903** | 0.830 | 0.835 | 0.797 | 0.856 | yes |
| accuracy, depth 2 (lineage) | **0.744** | 0.671 | 0.729 | 0.628 | 0.708 | yes |
| accuracy, depth 3 (cell type) | **0.418** | 0.376 | 0.352 | 0.332 | 0.392 | yes |
| accuracy, depth 4 (subtype) | **0.498** | 0.326 | 0.289 | 0.241 | 0.384 | yes |
| consistent with the author's label | **0.619** | 0.582 | 0.565 | 0.519 | 0.586 | yes |
| wrong (another branch), forced label | 0.374 | 0.397 | 0.380 | **0.369** | 0.380 | **no** |
| risk-coverage AURC (lower is better) | **0.293** | 0.422 | 0.650 | 0.561 | 0.519 | yes |
| ECE of the forced label's probability | 0.216 | 0.364 | 0.514 | **0.135** | - | **no** |
| wrong, published label (Singlet: calibrated call) | **0.251** | 0.397 | 0.380 | 0.369 | 0.380 | yes |
| ECE of the published probability (Singlet: call) | **0.079** | 0.364 | 0.514 | 0.135 | - | yes |
| depth-2 accuracy, tissues thin in the reference (2 studies) | 0.507 | 0.457 | **0.615** | 0.449 | 0.532 | **no** |
| CPU-seconds per million cells | **1,096** | 1,948 | 144,182 (GPU) | 18,937 | 533,230 | yes |

**Reading.**
- **Labels.** Singlet's forced label (`argmax_cl`) is the like-for-like comparison: every competitor names one term per cell.
  It is the most accurate at every depth, by 0.015 (depth 2) to 0.11 (depth 4) over the best competitor.
- **The calibrated call** (`cell_type_cl`, which backs off to the parent term when unsure) makes a third fewer wrong calls
  than any competitor's label. Its probability is the best calibrated: ECE 0.079 against Seurat's 0.135.
- **Fails, reported as fails:**
  - **Forced-label wrong rate.** At 0.374 the forced label makes 0.005 more wrong calls than Seurat, which is inside every
    CI. The rule is strict, so it fails.
  - **Forced-label calibration.** The argmax term's probability is worse calibrated than Seurat's prediction score. Its own
    temperature, fitted on the validation collections after this benchmark had been seen (section 4, disclosed), did not
    help on Singlet's counts: 0.181 → 0.216. It is kept rather than tuned on the benchmark.
  - **Tissues thin in the reference** (dental pulp GSE185222, mouse ependyma GSE201773). scANVI's integration
    generalises better there. Only two studies fall in this group.
- **Runtime.** Singlet labels a million cells in about 18 CPU-minutes, 1.8x faster than CellTypist, 17x faster than Seurat
  mapping and about 500x faster than SingleR. scANVI needs a GPU and a per-sample query training.

**Not covered yet.**
- Azimuth's own pretrained references, which cover 3 of the 16 tissues; the Seurat mapping with the shared reference stands
  in for the method.
- The cohort's other cell-type datasets (Ding 2020 in 7 protocols, the 10x and mouse sets, Segerstolpe, Tabula Muris).
  They need the harness's Singlet counts, which do not exist yet.

**Reproduce** (Clipper):
`scripts/celltype_nn/compete/{env,prep,scanvi_train,method,ours,timing,score,publish}.sbatch`. `export.py` writes the shared
reference and queries; `artifacts.py` collects the per-cell calls (`$WORK/bpub/artifacts/cell-type/`, 39 MB);
`publish.sbatch` runs `compute()` and the contract validator.
