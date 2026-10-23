# Per-cell labels: cell type, chromosomal sex, genetic ancestry

Package `python/singlet/annotate/`, scripts `scripts/annotate_*.py`. One label file per sample, ingested by the atlas cell index.
Validation run 2026-10-23 on Clipper; every number below comes from the artefacts named in its section.

## 1. What is written

`<OUT>/<GSE>/<GSM>.labels.parquet`, one row per called cell (schema version 1, `singlet.annotate.output`):

| column | type | content |
|---|---|---|
| `gsm` | string | sample accession |
| `col` | int64 | column of the cell in the sample's v2 `cells` table / matrix column order (legacy bundles: its column in `exon_counts.1pz`, the order `v2/migrate.py` gives the cells table) |
| `cell_type_cl`, `cell_type_label` | string | Cell Ontology id and name (CL 2026-06-08); null when the method abstains |
| `cell_type_score` | float32 | confidence of the call, in [0, 1] (guide: product of the gating confidences along the path; a ranking, see 3) |
| `sex_chrom` | string | `female` (PATO:0000383), `male` (PATO:0000384), `ambiguous` |
| `sex_score` | float32 | posterior of that label |
| `ancestry_hancestro` | string | HANCESTRO genetic-ancestry category of the cell's donor; null when not computed |
| `ancestry_prob` | float32 | estimated proportion of that ancestry in the donor |
| `model_version` | string | versions of every model that produced the row |

Plus `<GSM>.summary.json` for internal QC (counts per CL id, per-donor sex and ancestry, timings, and `barcode_digest`, the
SHA-256 of the called barcodes in `col` order, so ingest can check that the rows line up with the v2 cells table).

**Privacy.** Genotypes, allele depths, variant positions and anything per-site never reach a label file. Ancestry reads the
per-donor depth tables from the restricted sidecar on the compute node, in memory, and returns only a group, its HANCESTRO id and
a proportion. `assert_public_safe` runs before every write. It rejects any column outside the schema and any string that looks
like a variant id, a chromosome position, a genotype or an allele, and it checks every vocabulary.
`tests/python/test_cell_labels_privacy.py` fails if the schema grows, if a genotype-like value gets through, or if the end-to-end
pipeline fed restricted depths writes anything genotype-like (it also scans the written bytes and the summary).

## 2. Methods

**Cell type** (`celltype.py`). This is a port of the cell-phenotype engine (Singlet-Bio/singlet-agents `feat/cell-phenotype`,
`engine/docs/phenotype/REPORT.md`). Its components:
- a 513-node marker hierarchy keyed to the Cell Ontology, for human and mouse (MGI orthologs);
- a cell-intrinsic rank signal (UCell-like);
- kNN smoothing over the sample's own PCA graph;
- top-down gating with a minimum score and margin, plus look-ahead for heterogeneous parents.

A cell stops at the deepest node its evidence supports; that is the abstention. The default adds per-sample self-training: a
logistic regression trained on the guide's most confident calls relabels every cell and is voted within over-clustered
neighbourhoods. A probability below 0.5 backs off to the guide call, and disagreement with the guide's path backs off to the
common ancestor. The guide is restricted to plausible nodes for the sample's tissue, read from GEO source and characteristics
text. When the text names no organ, every node stays possible. Cells with fewer than 100 detected genes get no call.

**Sex** (`sex.py`). Only the split of sex-informative UMIs is used: XIST against non-PAR chrY genes (human: RPS4Y1, DDX3Y, UTY,
KDM5D, EIF1AY, ZFY, USP9Y, TXLNGY, NLGN4Y, PRKY; mouse: Ddx3y, Eif2s3y, Uty, Kdm5d). In a female cell about 97% of these UMIs are
XIST; in a male cell about 3%. Absolute rates and the absence of XIST are not used, because they vary several-fold with protocol
and depth (the first version used them and miscalled both female pilot samples). Each cell's evidence is a capped
log-likelihood ratio. Cells are pooled per donor where genotype demultiplexing found several donors, otherwise per sample: the
group's male fraction is fitted by EM and each cell's posterior combines its own evidence with that fraction. Cells with no
informative UMI inherit the donor's sex; cells with strong contrary evidence keep their own. A cell with strong XIST and chrY
together is `ambiguous`.

**Ancestry** (`ancestry.py`). This is supervised admixture with fixed allele frequencies, as in ADMIXTURE's projection mode. The
inputs are the donor's reference/alternate read depths at the pipeline's SNP panel (1000 Genomes phase 3, AF >= 0.05, 7.4M
sites; the restricted `donor/donor<k>_coverage.tsv`) and per-super-population allele frequencies of the same release on GRCh38
(`scripts/annotate_build_af.py`: 6.75M panel sites with frequencies). The model:
- genotypes are drawn under Hardy-Weinberg at `f = q·F`, and reads are binomial;
- depth is capped at 8 reads per site;
- the heterozygous read fraction is estimated per donor, because reference-mapping bias under-counts alternate reads by 2-5
  points in RNA.

The call is the super-population with `q >= 0.8`, otherwise HANCESTRO:0306 "admixed ancestry". There is no call when:
- fewer than 300 sites are covered;
- fewer than half of the donor's sites match the table (another build or species);
- the donor's alternate-read fraction is implausible (a defective depth table);
- the donor looks pooled (more than 15% of deep sites with intermediate allele fractions).

Labels map to the genetic-ancestry children of HANCESTRO:0004 (release 2025-10-14): AFR 0010, AMR 0014, EAS 0009, EUR 0005,
SAS 0006, admixed 0306.

## 3. Cell type: benchmark on Singlet's own counts

**Truth.** No public per-cell labels exist for Singlet's own cells, so author labels were transferred by barcode from CZ
CELLxGENE, whose curators map every author label to a Cell Ontology id (`scripts/annotate_truth_cxg.py`):
- barcodes on both sides are reduced to their 16-nt core;
- CELLxGENE cells are grouped by a sample-like column;
- a group is assigned to the GSM holding most of its barcodes (at least 50%, runner-up under 10%);
- only cells that both sides kept are scored.

Studies were drawn from the 78 GEO series linked from a CELLxGENE collection. The six used to develop the guide were excluded
(GSE149689, GSE166895, GSE202109, GSE192742, GSE115469, GSE132042). The result is 16 studies and 66 samples (13 human, 3 mouse),
332,755 called cells, of which 291,331 have a truth label. Truth was built independently on the laptop and again on Clipper from
public downloads (`scripts/annotate_bench_prepare.py`), with identical results. Every sample is annotated from Singlet's own counts
(exon + intron at gene level, as served), and scored with the singlet-agents ontology rules (`singlet.annotate.evaluate`).
Accuracy at guide depth d compares depth-d ancestors over cells whose truth reaches depth d (1 compartment, 2 lineage,
3 cell type, 4 subtype), so stopping early counts as wrong at deeper levels. Means are over studies.

Methods (`scripts/annotate_bench.py`, Clipper `short-cpu`, 8 CPUs, `out/annotate_validation/bench/bench.json`):

| method | d1 | d2 | d3 | d4 | consistent | wrong | cells called | CPU-h per M cells |
|---|---|---|---|---|---|---|---|---|
| **guide** (marker gating + kNN smoothing) | 0.842 | 0.747 | 0.428 | 0.320 | 0.571 | 0.350 | 0.915 | **0.5** |
| guide + self-training (back-off) | 0.845 | 0.748 | 0.427 | 0.337 | 0.576 | 0.349 | 0.919 | 4.2 |
| guide + self-training, no back-off | 0.845 | 0.727 | 0.386 | 0.298 | 0.600 | 0.317 | 0.909 | 3.9 |
| guide on exon-only counts | 0.808 | 0.709 | 0.401 | 0.302 | 0.530 | 0.374 | 0.897 | 0.4 |
| guide without the 100-gene filter | 0.845 | 0.750 | 0.430 | 0.320 | 0.574 | 0.350 | 0.924 | 0.5 |
| `singlet.annotate_cell_types` (existing tool: marker overlap per Leiden cluster, guide markers to depth 3) | 0.859 | 0.517 | 0.288 | 0.000 | 0.752 | 0.176 | 0.923 | 1.6 |
| `singlet.transfer_labels` (existing tool: kNN in a reference PCA; reference = the other studies' labelled cells, same species) | 0.674 | 0.261 | 0.100 | 0.012 | 0.237 | 0.685 | 0.930 | 0.9 |

(n = 16 studies at d1, 15 at d2/d3, 10 at d4. Pooled over cells, the guide's d1/d2/d3/d4 are 0.923/0.838/0.423/0.229.)

**What this says:**
- **The guide is the method to run.** It is as accurate as self-training at depth 1-3, at an eighth of the cost; the 0.02 gain at
  depth 4 does not justify 4 CPU-hours per million cells. The atlas default is therefore `method="guide"`.
- **The existing cluster-marker tool** is good at compartment level (d1 0.86) but stops there by design, so it never resolves a
  subtype.
- **Cross-study kNN transfer fails.** A reference built from other tissues cannot label cells it has never seen; that approach
  needs a reference matched to every tissue.
- **Exon-only counts lose 2-4 points**, so keeping intronic reads helps.
- **The 100-gene filter changes nothing.** Cells below 100 genes are 0.2% of the scored cells.

**Compared with the singlet-agents benchmark** (guide d1/d2/d3/d4 0.979/0.890/0.807/0.563 on 15 CELLxGENE datasets; 0.868/0.741/
0.521 at d2-d4 on 5 held-out ones), accuracy on Singlet's counts is much lower at depth 3-4. Per study (guide):

| study | species | tissue (GEO text) | samples | cells with truth | d1/d2/d3 | consistent | wrong | most frequent confusion (truth > call) |
|---|---|---|---|---|---|---|---|---|
| GSE125970 | human | ileum epithelium | 6 | 13,317 | 1.00/0.76/0.18 | 0.21 | 0.79 | colonocyte > SI enterocyte |
| GSE130973 | human | skin | 3 | 7,164 | 0.95/0.94/0.32 | 0.71 | 0.26 | macrophage > cDC2 |
| GSE131882 | human | kidney (snRNA) | 6 | 13,633 | 0.85/0.85/0.76 | 0.77 | 0.16 | principal cell > connecting tubule |
| GSE144870 | human | thymus, Lin-CD34+ | 7 | 71,466 | 1.00/0.88/0.16 | 0.43 | 0.57 | thymocyte > CLP |
| GSE157480 | mouse | (none) | 1 | 3,000 | 0.96/0.90/0.84 | 0.85 | 0.15 | B cell > TA cell |
| GSE161089 | human | airway basal stem cells | 4 | 13,537 | 1.00/-/- | 1.00 | 0.00 | (truth is compartment-level only) |
| GSE164291 | human | (mouse oviduct text) | 2 | 5,852 | 0.23/0.23/0.18 | 0.20 | 0.04 | species mix-up, no calls |
| GSE166766 | human | bronchial epithelium culture | 4 | 49,280 | 1.00/0.92/0.53 | 0.49 | 0.51 | basal > suprabasal |
| GSE184111 | human | ureter | 12 | 23,935 | 0.98/0.86/0.55 | 0.67 | 0.33 | urothelial basal > intermediate |
| GSE185222 | human | dental pulp | 4 | 4,686 | 0.35/0.62/0.47 | 0.23 | 0.74 | "epithelial" > fibroblast |
| GSE185224 | human | small intestine | 3 | 10,837 | 1.00/0.91/0.69 | 0.74 | 0.26 | submucosal gland > goblet |
| GSE185477 | human | liver | 1 | 1,545 | 0.89/0.88/0.45 | 0.29 | 0.66 | macrophage > classical monocyte |
| GSE201773 | mouse | (none) | 1 | 4,189 | 0.67/0.26/0.25 | 0.35 | 0.53 | radial glia > Müller glia |
| GSE260685 | human | ovary | 3 | 19,380 | 0.80/0.63/0.21 | 0.80 | 0.06 | stromal > lymphatic (truth coarse) |
| GSE270225 | human | bladder | 4 | 28,365 | 0.90/0.77/0.20 | 0.78 | 0.21 | fibroblast > visceral smooth muscle |
| GSE280852 | human | (none) | 5 | 21,145 | 0.88/0.80/0.62 | 0.63 | 0.32 | B cell > IgG plasma cell |

The dominant errors are of four kinds, and each points at a fix:
1. **Tissue context.** Where the GEO sample text names no organ (GSE280852, GSE201773, GSE157480; the series title did not help
   either), every one of the 513 nodes stays possible. Cells then drift to out-of-tissue types: cholangiocyte → corneal
   epithelium, Kupffer cell → decidual macrophage, radial glia → Müller glia. The earlier benchmark always had CELLxGENE's curated
   tissue. Section 3.1 measures this.
2. **Truth coarser than, or different from, the guide's split.**
   - Thymic CD34+ progenitors labelled "thymocyte" (GSE144870: 35k cells called common lymphoid progenitor);
   - ileal enterocytes labelled "colonocyte" (GSE125970);
   - airway basal vs suprabasal, and club vs goblet, in cultured epithelium (GSE166766);
   - urothelial basal vs intermediate (GSE184111).
   Most of these are gradient states that the guide resolves more finely than the authors did, or differently.
3. **Real marker failures.**
   - In bladder (GSE270225), fibroblasts, endothelium and urothelium go to visceral smooth muscle. This looks like a strong
     smooth-muscle ambient signal; detrusor muscle dominates the ambient RNA.
   - In kidney snRNA, intercalated and principal cells go to HSC/MPP (GSE131882).
4. **Species mix-ups in mixed studies.** GSE164291 has 0% calls on its human-labelled sample, whose metadata describes mouse
   oviduct.

**Calibration.** For guide calls, `cell_type_score` is the product of the gating confidences along the cell's path in the guide.
That product is better calibrated than the last-step confidence (`out/annotate_validation/bench/bench3.json`):

| score | < 0.5 | 0.5-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-0.95 | >= 0.95 |
|---|---|---|---|---|---|---|
| cells (path product, published) | 76,302 | 72,253 | 28,273 | 25,769 | 18,076 | 46,419 |
| consistent with truth | 0.47 | 0.51 | 0.58 | 0.68 | 0.80 | 0.80 |
| consistent, last-step confidence (not used) | 0.50 | 0.42 | 0.42 | 0.50 | 0.65 | 0.78 |

The published score rises monotonically, but it is optimistic above 0.9. It should be used to rank calls and to set filter
thresholds, not read as a probability. A per-tissue recalibration fitted on held-out studies is a follow-up.

### 3.1 Tissue context and smoothing

Same 16 studies, guide only (`bench3.json`):

| variant | d1 | d2 | d3 | d4 |
|---|---|---|---|---|
| guide, tissue from GEO text (default) | 0.842 | 0.747 | 0.428 | 0.320 |
| guide, CELLxGENE's curated tissue instead (oracle) | 0.848 | 0.731 | 0.413 | 0.320 |
| guide, no tissue context | 0.808 | 0.599 | 0.336 | 0.270 |
| guide, no kNN smoothing | 0.831 | 0.717 | 0.406 | 0.291 |

- **Tissue context matters:** without it, lineage accuracy falls from 0.75 to 0.60.
- **The GEO text already gives as much as the curated tissue.** Curated tissue helps where GEO names nothing (GSE280852, d3
  0.62 → 0.67). It hurts where the curated term is narrower than the cells (GSE185224, 0.69 → 0.44). So the lower depth-3
  accuracy on Singlet's counts is not a tissue-context problem. It comes from author labels coarser than or different from the
  guide's split, and from the specific marker failures listed above.
- **Smoothing over the sample's own kNN graph** is worth 2-3 points at every depth.
- **Next steps for depth 3-4:**
  - curate the guide on the confusions above (bladder smooth muscle, kidney snRNA, thymic progenitors, airway basal/suprabasal);
  - add a per-sample ambient-RNA correction before scoring.

## 4. Chromosomal sex: 157 samples with GEO sex metadata

Validation run `out/annotate_validation/run1`: 21 studies (the v2 pilot GSE145370, 12 human and 8 mouse legacy studies whose GEO
characteristics record sex, both sexes in each), 167 samples, 443,530 cells, through the campaign script itself
(`scripts/annotate_campaign.py`, `scripts/annotate_validation_summary.py`).

| | samples | studies | sample call matches GEO | ambiguous | wrong |
|---|---|---|---|---|---|
| human | 106 | 13 | 105 | 1 | 0 |
| mouse | 51 | 8 | 48 | 0 | 3 |
| all | 157 | 21 | 153 of 156 called (98.1%) | 1 | 3 |

- The pilot GSE145370 is 14 of 14, including both female donors. Most of its cells have under 100 UMIs; only 22 and 38 cells of
  the two female samples carry any XIST or chrY UMI, and every cell inherits the donor call.
- Median fraction of a sample's cells whose label agrees with the sample's sex: 1.00. Mean fraction with the opposite label:
  1.9%. These are doublets with other donors, ambient RNA and mislabelled barcodes.
- **The three disagreements are GEO metadata errors, not model errors.** The expression is unambiguous in each case:
  - GSE254999 GSM8062050: labelled male, 13,881 XIST UMIs against 10 chrY UMIs;
  - GSE254999 GSM8062059: labelled female, 7 XIST against 3,295 chrY (with the previous sample, the pair looks swapped);
  - GSE193416 GSM5799927: labelled female, 5 XIST against 1,839 chrY.
- The ambiguous sample, GSE228056 GSM7113331, has 10 informative cells out of 1,000. It is left `ambiguous` rather than guessed.

The earlier cell-phenotype benchmark (CELLxGENE donor metadata, 15 datasets) found per-cell agreement >= 0.97 on 13 of 15
datasets with every donor-level call right.

## 5. Genetic ancestry

### 5.1 Out of sample on 1000 Genomes (`scripts/annotate_ancestry_validate.py`)

The real depths are restricted, so donors were simulated from public 1KGP phase 3 genotypes (chr18-22, 755k biallelic SNVs with
AF >= 0.05):
- individuals were split 80/20 stratified by the 26 populations, and frequencies were computed from the 80% only (501 held-out
  individuals);
- depth per site is 1 + Poisson(LogNormal(ln 0.5, 1.5)): median 1, 25% of sites at >= 3 reads, like the real tables;
- allele reads are beta-binomial with overdispersion rho 0.02 or 0.1, standing in for allele-specific expression.

| covered sites | correct | admixed | accuracy of named calls |
|---|---|---|---|
| 300 | 0.727 | 0.263 | 0.986 |
| 1,000 | 0.830 | 0.168 | 0.998 |
| 3,000 | 0.886 | 0.110 | 0.996 |
| 10,000 | 0.894 | 0.104 | 0.998 |
| 30,000 | 0.892 | 0.106 | 0.998 |
| 100,000 (100 donors) | 0.900 | 0.100 | 1.000 |

(rho 0.1.) Confusion at 10,000 sites (rows true, columns called):

| | AFR | AMR | EAS | EUR | SAS | admixed |
|---|---|---|---|---|---|---|
| AFR | 126 | 0 | 0 | 0 | 0 | 6 |
| AMR | 0 | 35 | 0 | 1 | 0 | 34 |
| EAS | 0 | 0 | 102 | 0 | 0 | 0 |
| EUR | 0 | 0 | 0 | 100 | 0 | 0 |
| SAS | 0 | 0 | 0 | 0 | 85 | 12 |

- **A named call is almost always right.** The calls are calibrated: 0.99-1.00 correct in every q bin from 0.8 up, at >= 1,000
  sites. The misses are `admixed` calls, and they come from admixed populations:
  - PUR: q is about 0.33 AMR and 0.51 EUR, so these are all called admixed;
  - CLM: 0.32 correct;
  - ASW/ACB: 0.81 correct;
  - PJL: 0.53 correct.
  This reflects their real genetic mixture, and it does not improve with depth.
- **Threshold.** `q_min` 0.8 keeps named-call accuracy at 0.998 at 10,000 sites, with 10% admixed. A threshold of 0.7 gives 7%
  admixed at 0.996.
- **Leakage.** Frequencies computed from all individuals, including the test ones, overstate accuracy by 0.6-1.4 points. That
  bias is what the split avoids.

### 5.2 Real donors (`scripts/annotate_ancestry_real.py`, restricted sidecars read on Clipper, `out/annotate_validation/ancestry_final`)

These are 147 donors in 50 human studies whose sidecars hold coverage tables (final code, 20-replicate bootstrap). 140 donors are
on the human panel; 94-99% of their sites are found in the frequency table (median 95%). Covered sites per donor range from 5 to
1.15M (median 56k).

**Self-reported race or ethnicity in GEO characteristics: 22 donors, 2 studies.**

| self-report | called |
|---|---|
| caucasian / white (13) | European 11, admixed 1 (EUR 0.61, EAS 0.26, SAS 0.12), Latin American 1 (AMR 0.83) |
| hispanic / latino (6) | Latin American 4 (AMR 0.83-0.98), admixed 2 (EUR 0.68-0.73, AMR 0.27-0.32) |
| black or African American (2) | African 2 (AFR 0.83 and 0.98, the rest EUR) |

- **No call contradicts the reported group.** The two "admixed" Hispanic donors and the white donor called Latin American have
  the European / Indigenous American mixtures typical of Hispanic self-report.
- **Admixed African American donors** show 2-17% European ancestry, consistent with US cohorts.

**The pipeline's sample-level call** (`ancestry_call.json`, 50 markers) agrees with this call for 11 of 59 donors. It mostly
says Latin American (14) or East Asian (11) where this method says European with q > 0.98 over 10^5 sites, so the 50-marker call
is not usable.

**Guards and what they removed:**
- **Defective tables.** 46 donors had alternate-read fraction 0.000, which is impossible for a person (single donors read
  0.28-0.33 alternate). These are the 45 donors of GSE207310 and one other. Their coverage tables record ad = 0 at every site.
  Before the guard, all 45 GSE207310 donors were called the same "35% AFR / 60% EUR" mix.
- **Few sites.** 6 donors had too few covered sites; 7 more were not on the human panel.
- **Pooling.** The intermediate-allele-fraction statistic of single donors is 0.036-0.060; the no-call threshold is 0.15. It has
  not been tested on a known pooled sample.
- **Reference bias.** The estimated heterozygous read fraction is 0.40-0.50 (median 0.48), so reference bias is real but small.

**Runtime:** median 0.6 s per donor, at most 5 s (1.1M sites with the bootstrap), after subsampling the multi-start screening and
the bootstrap to 20k sites.

Bootstrap `stability` is reported per donor in the summary. It flags uncertainty at <= 1,000 sites (accuracy 0.61 below 0.8
against 0.94 at 1.0) but saturates above about 10k sites, so it cannot flag admixture-driven `admixed` calls.

## 6. Cost and the atlas run

Validation run (8 threads per task, Clipper `short-bigmem`): 7,400 s wall per million cells per task, about 16 CPU-hours per
million cells.

| stage | s per million cells |
|---|---|
| read counts | 1,296 |
| cell type, guide + self-training (validation-run setting) | 5,639 |
| cell type, guide only (benchmark; the atlas default) | 279 |
| sex | 13 |
| ancestry (incl. sidecar parsing) | 450 |

With the guide as the cell-type method, one task spends about 2,040 s of wall clock per million cells, which is about 4.5
CPU-hours per million cells at 8 threads. Reading the counts is now the largest stage. v2 studies, read by range requests, should
read faster; the legacy studies are downloaded whole.

`scripts/annotate_campaign.py plan` takes its study list from `https://singlet.bio/api/manifest`. Human and mouse studies only:
1,982 studies, 1.46 TB of bundles, 266.5M cells, which is about 1,200 CPU-hours. They go into 400 shards balanced by bytes
(3.3-3.4 GB each; one study of 29 GB is a shard of its own). At 60 concurrent 8-CPU tasks (480 cores) that is about 3 hours of
wall time on Clipper, plus download time (about 1.5 TB in total). It fits `short-cpu`/`short-bigmem` 4-hour slots, except the
29 GB study, which should get a `cpu`/`bigmem` slot. Anvil is not needed.

The default is now the guide. With self-training (`--method selftrain`) the campaign would cost about 4,400 CPU-hours for the
small depth-4 gain measured in section 3.

Submission (not launched; it needs Zach's go-ahead):

```
python scripts/annotate_campaign.py plan --shards 400 --out $OUT/plan.tsv            # inside a job
sbatch --array=0-399%60 --export=ALL,OUT=$OUT,SRC=$SRC scripts/annotate_campaign.sbatch
python scripts/annotate_campaign.py status --out $OUT
```

Each task downloads its studies' bundles (and, with R2 read credentials, the restricted sidecars) to node-local scratch and
deletes them after each study. It writes only label and summary files, and it never writes to R2 or D1. A study whose files
exist at the current `model_version` is skipped, so failed tasks can simply be resubmitted.

## 7. Findings for other owners

- **GSE207310 (pipeline):** `donor/donor<k>_coverage.tsv` has ad = 0 at every site for every donor, while dp is populated, so
  the alternate allele counts were lost. The donor VCFs of that study are presumably wrong too.
- **GEO metadata:** the sex of GSE254999 GSM8062050/GSM8062059 looks swapped, and GSE193416 GSM5799927 looks mislabelled
  (section 4).
- **`ancestry_call.json`:** the pipeline's 50-marker call disagrees with a 10^5-site call on 36 of 39 donors. The website should
  stop showing it once these labels are served.
- **data.singlet.bio:** it answers 403 to Python's default urllib User-Agent; clients must send one (the v2 transport does).
- **CELLxGENE truth (section 3):** Singlet's called cells and the authors' kept cells often differ. Matched fractions are listed in
  the benchmark manifest.
