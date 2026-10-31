# Per-cell feature scores: gene-set activity, cell-cycle phase and doublet score

Package `python/singlet/features/`, scripts `scripts/features_run.py` (+ `.sbatch`), `scripts/features_build_data.py`, pilot
`bench/features/pilot.py`. Writes the feature sidecars of `docs/CELL-ARCHITECTURE.md` section 3; the cell index turns them into
shard columns and `sample_cell_summary` rows. Malignancy and perturbation calls are a design note only (section 7).

## 1. What is written

Per sample and feature set, `<OUT>/<GSE>/<GSM>.features.<set>.parquet`, one row per called cell, sorted by `col`:

| column | type | content |
|---|---|---|
| `gsm` | string | sample accession |
| `col` | int64 | column of the cell in the sample's store `cells` table (legacy bundles: its column in `exon_counts.1pz`), unique |
| one per feature | float32 / uint16 | score in [0, 1], NaN = not computed; category code, 65535 = not computed |

Parquet metadata `singlet_features_schema = "1"`, `feature_set`, `method_version`. `output.assert_public_safe` runs before every
write: exact registered columns and types, one GSM, unique non-negative `col`, value ranges and category codes, metadata keys, no
column named like a gene symbol, no genotype-like string.

Also per sample: `<GSM>.features.rows.json` (its `sample_cell_summary` rows) and `<GSM>.features.json` (internal: gene-set
coverage, dropped sets, timings).

| Feature set | Features | Kind |
|---|---|---|
| `hallmark` | 50 MSigDB Hallmark sets, e.g. `hallmark/hypoxia`, `hallmark/interferon_alpha_response` | score |
| `cell_cycle` | `cell_cycle/phase` (G1, S, G2M), `cell_cycle/s_score`, `cell_cycle/g2m_score` | category, score, score |
| `doublet` (on request, section 8) | `doublet/score_v2` | score |

53 features in all: the cell index takes at most 64 per index version, so a further collection of this size needs either a new
index version with a different selection or a larger cap there.

## 2. Score

Per cell, genes are ranked by count (ties share their average rank; undetected genes rank last). A gene at rank `r` carries
`s = max(0, 1 - (r - 1)/1500)`. For a set of `n` genes present in the reference, the rank AUC is
`auc = mean(s) / (1 - (n - 1)/3000)`: the Mann-Whitney U of the set's ranks among the cell's top 1500 genes, scaled to [0, 1]
(0 = no set gene in the top 1500, 1 = the set's genes are the cell's `n` most abundant). It orders cells exactly as UCell does
(UCell = `mean(s) + (n - 1)/3000`; `tests/python/test_cell_features.py` checks both identities).

The rank AUC alone tracks depth and the abundance of the set's genes (pilot: median |Spearman| with the detected-gene count
0.35-0.57). Each set therefore has 16 null sets: every member gene is replaced by a random gene from the same abundance bin
(50 equal-count bins of the per-organism reference in `data/abundance_<organism>.parquet`), with a seed fixed by set and
organism. The null sets are constants of the method, not of the sample. In each cell,
`z = (auc - mean of the null aucs) / max(their sd, 0.002, 0.1 * their mean)`, and the written score is

    score = clip(z / 10, 0, 1)

so 0 = no higher than abundance-matched random genes in this cell, 0.3 = z 3, 0.5 = z 5. Everything it uses is the cell's own
counts and fixed tables, so `comparable = 'atlas'`. Cells with fewer than 100 detected genes get NaN (as in the labels), as do
samples of other organisms and sets with fewer than half of their genes (or fewer than 5) in the sample. Clipping merges all
cells at or below the null into 0; filters ("score >= t") are unaffected, rankings below the null are not kept.

**Cell-cycle phase.** `z = (auc - null_mean)/null_sd` for the S and G2/M gene lists (Tirosh et al. 2016, Seurat 2019 names).
A cell is S or G2M when that phase has `z >= 3` (the larger z wins), otherwise G1 (not cycling: G0 or G1). Seurat's
`CellCycleScoring` instead compares module scores against controls drawn from the sample's own genes and calls every cell with a
positive score cycling; that depends on the sample and is not used.

**Thresholds** (`cell_feature.thresholds`, counted as `n_hi1` / `n_hi2`): 0.3 and 0.5 (z 3 and z 5) for every score. In
the pilot, 0.2-17% of a sample's cells reach 0.3 for a given set, and 0-6% reach 0.5 (`bench/features/results/pilot.json`,
`share_at_thresholds`); with 16 null sets, a z of 3 is reached by a few percent of cells by chance, so 0.5 is the stricter
filter.

## 3. Gene sets and licences

| Collection | Release | Licence | Mapping |
|---|---|---|---|
| MSigDB Hallmark: human H, mouse MH | 2026.2 (`h.all.v2026.2.Hs.json`, `mh.all.v2026.2.Mm.json`, sha256 in `data/hallmark.json`) | CC BY 4.0 (MSigDB licence terms for v2022.1 and later, checked 2026-10-23) | symbols -> GRCh38-2024-A / GRCm39-2024-A Ensembl ids: human 7,305 of 7,322, mouse 7,189 of 7,191 (unmapped symbols listed in the file) |
| Cell cycle S, G2/M | Tirosh et al. 2016 with Seurat `cc.genes.updated.2019` names | gene names from a publication; no collection licence | human 97 of 97; mouse (same symbols, mouse case) 97 of 97 |

Attribution for Hallmark is in `cell_feature.source_ref` / `licence` and in `data/hallmark.json`. Collections to add next and
their licences, checked on the MSigDB licence page: C2 Reactome (`REACTOME_*`, CC BY 4.0 within MSigDB; Reactome itself is CC0),
C5 GO (CC BY 4.0). C2 `KEGG_*` (KEGG licence), `KEGG_MEDICUS_*` (CC BY-SA 4.0) and `BIOCARTA_*` (BioCarta terms) are not open
enough for a CC0 atlas and must stay out.

Abundance references (`scripts/features_build_data.py`): mean log1p CP10k over called cells with at least 200 detected genes, at
most 1,500 per sample. Human: 44,079 cells of GSE138867 (blood), GSE294109 (ccRCC) and GSE293707 (T cells). Mouse: 10,063 cells
of GSE214960 and GSE178941. The sample list is in the files' metadata. A broader reference (an atlas-wide aggregate) is a version
bump: it changes the null sets and therefore `method_version`.

## 4. Pilot

Twelve samples, 90,433 called cells, laptop (one core, bundles on local disk); `bench/features/pilot.py` and
`bench/features/contrasts.py`, results in `bench/features/results/`.

| Study | Samples | Organism | Cells | Median genes | Scoring s / 10k cells |
|---|---|---|---|---|---|
| GSE145370 (store, ESCC) | GSM4317426 | human | 2,928 | 184 | 8.2 (includes drawing the null sets once) |
| GSE138867 (blood) | GSM4120733, GSM4120734 | human | 14,739 | 794-865 | 4.8-4.9 |
| GSE294109 (ccRCC) | GSM8898802, GSM8898804 | human | 4,812 | 315-322 | 2.7-3.0 |
| GSE293707 (blood, COVID-19) | GSM8888963 (healthy), -64, -65, -66 | human | 44,788 | 125-391 | 1.0-2.3 |
| GSE214960 | GSM6619183 | mouse | 6,413 | 788 | 4.7 |
| GSE178941 | GSM5401596, GSM5401599 | mouse | 16,753 | 634-928 | 2.6-4.1 |

**Cost.** Scoring (ranks, 53 sets and 848 null sets, cell-cycle) took 24.3 s for 90,433 cells: 2.7 s per 10k cells, about
2 to 10 s per 10k cells depending on depth. Reading counts took 6.4 s per 10k cells from local bundles. On Clipper (job 781594: GSE214960, GSE293707; 51,201 cells) scoring took
0.88 s and reading 1.9 s per 10k cells. A billion cells is about 25-75 core-hours of scoring; the atlas run is bound by reading counts. Sidecars: 121 bytes per cell for `hallmark` (50 float32 scores, zstd) and 6 for `cell_cycle` (about 127 GB at a billion cells); the
cell index stores them as u8.

**Which statistic.** Median |Spearman| with log detected genes over the 50 sets (per sample), and AUROC of expected
contrasts (cell groups from canonical markers, for this check only):

| Statistic | Depth |rho| | heme: erythroid vs other (blood) | E2F: S+G2M vs G1 | IFN-gamma: myeloid vs T | inflammatory: myeloid vs T | IFN-alpha: NK, COVID-19 vs healthy | IFN-alpha: myeloid, COVID-19 vs healthy |
|---|---|---|---|---|---|---|---|
| raw rank AUC (UCell order) | 0.35-0.57 | 0.43 | 0.93 | 0.79 | 0.84 | 0.89 / 0.47 | 0.50 |
| z against the null sets | 0.07-0.23 | 0.85 | 0.93 | 0.89 | 0.89 | 0.68 / 0.58 | 0.67 |
| written score (z / 10, clipped) | 0.07-0.18 | 0.60 | 0.91 | 0.86 | 0.89 | 0.68 / 0.57 | 0.69 |

The raw AUC follows depth: of the two COVID-19 samples, the deeper one (391 genes) separates from the healthy donor (178 genes)
at 0.89 and the shallower one (125 genes) at 0.47; after calibration the two give 0.68 and 0.58. Clipping lowers the AUROC where
most cells sit near the null (heme), not the share of cells above a threshold.

**Interferon.** Share of cells with `interferon_alpha_response >= 0.3`: healthy donor 2.5%; COVID-19 donors 8.1%, 7.3%, 1.8%.

**Cell cycle.** Cells called S or G2M: 2-28% per sample; most in the mouse leukemia samples (GSE178941, 21-28%). MKI67 or
TOP2A is detected in 87-96% of G2M cells of GSE214960 and GSE178941 (630-930 genes per cell), in 41-59% in GSE145370 and
GSE294109 and 4-32% in GSE293707 (125-390 genes), and in 0.3-33% of G1 cells. In blood (GSE138867, 1% of cells called G2M)
only 3-6% of G2M calls have either gene, so in samples with few cycling cells most calls at z 3 are background.

**Not shown by the pilot.** Hypoxia: the ccRCC samples have CA9 at zero counts, so there was no hypoxic tumour group to test;
a positive control (a hypoxia-treated cell line or ccRCC tumour cells) is still needed before hypoxia is offered as a filter.
GSE145370 (store, 184 genes per cell) loses 3% of its cells to the 100-gene minimum, GSE293707's shallowest sample 29%.

## 5. Running

Local, one study (store from data.singlet.bio, or a legacy bundle downloaded to `--scratch`):

    PYTHONPATH=python python scripts/features_run.py run --gse GSE138867 --out out/features
    PYTHONPATH=python python scripts/features_run.py registry --out out/features
    PYTHONPATH=python python scripts/features_run.py collect --out out/features

Clipper, a study list sharded over an array (one line per study: `GSE`, `GSE GSM ...` or `GSE path.singlet`):

    sbatch --array=0-199%60 --export=ALL,OUT=$OUT,LIST=$OUT/studies.txt,SHARDS=200,SRC=$SRC scripts/features_run.sbatch

`SRC` must have the compiled readers (`python setup.py build_ext --inplace`); `LABELS_SRC` can point at a worktree that
has them instead.

## 6. Publishing (not run by the job)

1. Registry, once per method version (53 rows):
   `curl -fsS -X POST https://singlet.bio/api/ingest/cell_feature -H "X-Ingest-Token: $INGEST_TOKEN" -H "Content-Type: application/json" --data-binary @$OUT/ingest/cell_feature.json`
2. Summaries (<= 2000 rows per request):
   `for f in $OUT/ingest/sample_cell_summary.*.json; do curl -fsS -X POST https://singlet.bio/api/ingest/sample_cell_summary -H "X-Ingest-Token: $INGEST_TOKEN" -H "Content-Type: application/json" --data-binary @$f; done`
   The cell index's `--d1-export` writes the same rows from the shards; load one of the two, not both.
3. Sidecars: `singlet.s1.atlas.normalize_feature_sidecar` (cell-index branch) validates each file against the registry and the
   sample's cells, returns the canonical bytes and key `s1/cf/<GSM>/<set>.<d16>.parquet`; upload those to `singlet-data`
   (`wrangler r2 object put singlet-data/<key> --file <file> --remote`) and attach the entries to the study manifest with
   `attach_features`. The index builder reads the sidecars as shard columns.

## 7. Design note: malignancy and perturbation calls (not built)

**Malignant vs normal (`malignancy_cnv_1/score`, flag `malignancy_cnv_1/malignant`).** Copy-number inference per sample, against
the sample's own non-malignant cells:
- Reference cells: cells labelled immune, endothelial or stromal by the annotation pipeline (`cell_type_cl` under CL:0000738
  leukocyte, CL:0000115 endothelial cell, CL:0002320 connective tissue cell / fibroblasts), at least 100 of them; otherwise no
  call for the sample.
- Expression: log1p CP10k of genes expressed in at least 10% of cells, ordered by genomic position (the reference GTF), minus
  the reference cells' mean, clipped at +-3, smoothed along each chromosome arm with a 101-gene window (inferCNV / copyKAT
  approach; code written independently, not ported).
- Per cell: CNV burden = mean squared smoothed signal; correlation of the cell's profile with the mean profile of the top-burden
  cells (malignant clones share their CNVs). Score in [0, 1] from both, calibrated against the reference cells' own distribution
  (leave-one-out), so 0.5 means "outside 99% of the sample's normal cells".
- `comparable = 'sample'` for the score (it is relative to the sample's reference); the flag combines it with the cell type
  (epithelial / unassigned tissue cells only) and the sample's disease term (MONDO cancer descendants), and only the flag is an
  atlas filter. Validation: studies with author-labelled malignant cells (e.g. ccRCC GSE294109, ESCC GSE145370) and normal-tissue
  samples, where the call rate must stay near 0.
- Needs: gene positions per reference build (one table per build), cell-type labels first (run after the annotation job).

**Perturbation calls (`perturbation_guide_1/*`, `hashtag_1/*`).** Only for samples whose assay carries guide or hashtag
libraries (the modalities vocabulary marks them; most atlas samples have none):
- CRISPR screens: guide UMI counts per cell from the feature-barcode library, a per-guide two-component mixture (Poisson
  background vs signal) as in Replogle et al. 2022, and the call = the single guide above the posterior cut-off (multiple guides
  = `multiple`, none = `unassigned`). Stored as a category with the target gene symbol as the category name; this is the one
  place a gene symbol is a value, so the guard needs an allow-list for this set.
- Hashtags: demultiplexing (HTO counts, CLR-normalised, per-tag mixture) gives singlet / doublet / negative and the tag; tags map
  to sample conditions through GEO metadata, so the per-cell value is the condition's `sample_property` (treated / control),
  not the tag name.
- Both are `comparable = 'atlas'` categories; `sample_cell_summary` counts per category value.
- Blocker: the .s1 objects do not yet carry feature-barcode counts for most screens (the screen-atlas session is checking which
  studies have them).

## 8. `doublet/score_v2`: a frozen cross-study doublet stacker

Code `python/singlet/features/doublet.py` (+ `doublet_atlas.py`, `doublet_sdfl.py`, `doublet_util.py`), model
`data/doublet_v2_stacker.{npz,json}`, jobs `scripts/doublet_v2.py` + `scripts/doublet_v2.sbatch`, tests `tests/python/test_doublet_v2.py`.
Study and recipe: `docs/latent-doublet-ambient.md` (branch study/latent-qc) and `bench/expression_sota/doublets/` (branch
claude/expression-sota-31cf6d). No R anywhere.

**What is written.** Feature set `doublet`, one feature `doublet/score_v2` (float32 in [0, 1], higher = more doublet-like; NaN = not
computed), one row per called cell sorted by `col`, through the same writer, guard and registry as the sets above (`comparable =
'sample'`: it ranks cells within a sample and is not a calibrated doublet rate; thresholds 0.5 and 0.8). The version that produced a
sidecar is in its Parquet metadata (`method_version`), in the registry row (`cell_feature.method_version`) and, through
`normalize_feature_sidecar` / `attach_features`, in the study's `.s1` manifest entry. `method_version` =
`singlet-features=<pkg>;doublet=score_v2;stacker=hgb300/<sha256[:8] of the trees>(sklearn <training version>);inputs=...;
atlas_encoder=celltype_nn/m1_notissue(<sha256[:8] of model.pt>);centroids=<sha256[:8]>;sdfl=a,c;min_cells=200`. Any change to the
trees, the encoder or the centroids changes it. The set is not in `FEATURE_SETS` (the default run); ask for it with
`features_run.py run --sets doublet --prod-dir <engine run dirs> [--device cuda]` and register it with
`features_run.py registry --sets doublet`. It is the 54th feature, inside the cell index's cap of 64.

**What is computed.** Nine inputs per sample, each replaced by `rank / n` over the sample's called cells:

| input | definition |
|---|---|
| `umi`, `genes` | log1p total counts, log1p detected genes |
| `prod` | the engine's doublet score (`doublet_scores.tsv`, column `doublet_score`, of the sample's run directory; `--prod-dir DIR/<GSM>/doublet_scores.tsv`) |
| `atlas_knn`, `atlas_clf`, `ct_entropy`, `atlas_nnls` | the cell type network `singlet.celltype_nn` (m1_notissue, trained on 2.8M CELLxGENE cells) as a fixed encoder of the real cells and 2n simulated doublets (summed counts of random pairs): fraction of simulated cells among the k = 0.5 sqrt(n) nearest neighbours in PCA-50 of its hidden layer; 5-fold cross-fitted MLP simulated vs real on the same PCA; entropy of the network's calibrated cell type distribution; relative residual drop of the best two-centroid non-negative fit over the best single one (856 atlas cell-type centroids). Port of `latent_gpu.py`; CPU or GPU (`device`) |
| `sdfl`, `sdflc` | R-free artificial-doublet scorer in the style of scDblFinder (HVG, PCA-20, k-means, artificial doublets from cluster pairs, kNN features, two-round gradient boosting, out-of-fold scores); `sdflc` adds one third same-cluster pairs. Port of `sdfl.py`; the two variants share the PCA and clustering. The port returns exactly the values of the reference script (maximum absolute difference 0 on pbmc-1A-dm) |

The stacker is `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
random_state=0)`, trained once on the 17 real-label datasets (labels from genotype, hashtag, MULTI-seq and species mixing;
182k cells), each dataset weighted equally, on the stored precomputed scores. The trained trees are kept as plain arrays with the
feature order, the scikit-learn version that trained them (1.6.1) and a SHA-256 over the arrays (`Stacker.load` refuses a file
whose hash differs) and evaluated in numpy, so the scores do not depend on the installed scikit-learn; the exported trees equal
sklearn's `predict_proba` to 0.0 on kang21 and to 1e-12 in the unit test. Retraining needs scikit-learn only
(`doublet_v2.sbatch MODE=train`). Samples with fewer than 200 called cells, other organisms (only human and mouse), or no engine
score get NaN (`features.json` lists the set under `dropped`); cells with a non-finite input get NaN.

**Evidence (hold-out only).** Leave-one-study-out over the 17 datasets (the 17 datasets form 6 studies; a dataset is scored by a
model trained on the datasets of the other studies), mean AUPRC:

| inputs | mean AUPRC |
|---|---|
| this stacker, held-out inputs recomputed by the package (what a deployment computes) | **0.5707** |
| the same, held-out inputs from the stored scores (the stack.py recipe; the package reproduces its per-cell scores to 1e-9) | 0.5694 |
| trained and scored on package-recomputed inputs | 0.5724 |
| same stacker without `sdfl`, `sdflc` (stack.py `combo_prod_atlas`, from the study) | 0.5573 |
| scDblFinder (R, from the study) | 0.5587 |

Against scDblFinder this is a statistical tie, not a win: the mean difference is +0.012 AUPRC, 95% CI about [-0.019, +0.040] over
the 17 datasets (study-clustered CI wider). Against the previous combiner (`combo_prod_atlas`) the paired gain is +0.011 to +0.013,
better on 16/17 datasets. The reason to ship it is the same accuracy as scDblFinder without R.

The AUPRC of the frozen model on the 17 datasets is not evidence: it was trained on them. The 17 labels see only heterotypic
doublets across donors, hashtags or species. The stored `sdfl`, `sdflc` and atlas scores of the training set agree with the
package's recomputation only up to Spearman 0.89-0.98 (kang21 and pbmc-1A-dm: `sdfl` 0.92 / 0.95, `atlas_clf` 0.93 / 0.89; the
rest of the difference is not explained: the port equals the reference script exactly, and the stored `sdfl` files were rewritten
on 2026-10-31), which is why the package-input row is reported; it does not lower the score (0.5707 vs 0.5694).

**Scope.** For single-donor samples. Where a pool has two or more donors, genotype doublets (`donor_assignments`
`prob_doublet`, Kang 2.1 AUROC 0.969 / AUPRC 0.897 against 0.935 / 0.698 for the expression stacker, and recall 0.89 on doublets
that look like one cell type against about 0.5) stay authoritative; this feature does not replace them, and neither catches
same-donor homotypic doublets. It is a ranking within a sample.

**Cost per sample** (Clipper `short-bigmem`, 8 threads, CPU only, `OMP_WAIT_POLICY=PASSIVE`; `doublet_v2.sbatch MODE=feats`
over the 17 datasets and `MODE=score`; CPU seconds = `time.process_time()`, user + system of all threads; peak = resident set of
the process, cumulative over datasets in the `feats` run, so only the `score` runs give a clean peak):

| sample | called cells | wall s | CPU s | of which atlas / sdfl wall s | peak memory |
|---|---|---|---|---|---|
| pbmc-1A-dm | 3,298 | 21 | 59 | 13 / 7 | 2.1 GB |
| kang21 | 13,737 | 54 | 201 | 38 / 15 | 2.8 GB |
| hm-12k | 12,820 | 72 | 215 | 37 / 35 | |
| mkidney-ch | 21,179 | 101 | 324 | 69 / 32 | 5.3 GB |
| HMEC-orig-MULTI | 26,426 | 142 | 413 | 82 / 60 | 9.3 GB |

About 30 CPU-seconds plus 14.5 CPU-seconds per 1,000 called cells (500 cells: 21 s; 26k: 413 s), memory 2 GB at 3k cells to 9 GB at
26k; 1 million cells is about 4 CPU-hours. The stacker itself takes under 1 s. A first measurement on a busy node with
spinning OpenMP threads gave 3,305 CPU-seconds for pbmc-1A-dm (the same work, 608 s wall); with passive waiting it is 59, so
`OMP_WAIT_POLICY=PASSIVE` belongs in every job that runs this. A GPU (`short-gpu` node, `--device cuda`, kang21) did not help at
these sizes: 62 s wall, 106 CPU-s, because the encoder and the kNN are small next to the classifier loop and sdfl, which stay
on the CPU; ask for CPUs. For comparison the engine's own doublet score costs about 2 s per 10,000 cells.

**Where it runs: a post-pass right after the drain, not inside it.** (1) Cost: 1-7 minutes of 8 cores per typical sample, 30-100
times the engine's whole doublet step, would lengthen every drain task and its memory request (2-9 GB, plus the 2.8M-cell
network and torch) for a score that is not needed to finish a sample. (2) It needs only things the drain leaves behind (the called
cells' counts and `doublet_scores.tsv`), so it can run as a Slurm array over finished samples with sized requests (4-8 CPUs,
16 GB up to 30k cells, more above; not tested above 26k cells), restart per sample (`done` skips written sidecars) and be re-run
for a `score_v3` or a new encoder without touching the drain's outputs. (3) The called-cell set must be final: the score ranks
cells within the called set, and a sidecar built on a set that is later changed (for example by the EmptyDrops calls of
docs/called-ed.md) has to be recomputed. (4) It keeps torch and the encoder files out of the production drain image. Run it per
study, right after its samples finish: `sbatch --array=... scripts/features_run.sbatch` with `--sets doublet --prod-dir <run
dirs>`, `OMP_WAIT_POLICY=PASSIVE` (model files: below).

**Install and model files.** `pip install "singlet[doublet]"` (the extra adds `torch>=2.0`; scikit-learn, scipy, pyarrow are core
dependencies; the stacker's `.npz` ships in the wheel). The two large files are not in the repository: the cell type network (56 MB) and
the atlas centroids (20 MB) are assets of the GitHub release `doublet-v2-model-b7c4a08f5397` of Singlet-Bio/singlet (public, no
credentials; tag = first 12 hex of the sha256 of the two `sha256sum` lines). `doublet_assets.fetch` looks each file up in this order, and every candidate is checked against the pinned sha256 (after a download
and on every use; a mismatching download or cache entry is deleted and `AssetHashError` raised, a mismatching store entry is skipped
with a warning): (1) the content-addressed model store `$SINGLET_MODEL_STORE` (default
`/mnt/projects/debruinz_project/debruinz/singlet/models/sha256`, files `<sha256>.npz` / `<sha256>.pt`, mode 444): the copy campaign workers on
Clipper use; (2) the user cache `$SINGLET_DOUBLET_CACHE` (default `$XDG_CACHE_HOME/singlet/doublet_v2`, `~/.cache/singlet/doublet_v2`, or
`%LOCALAPPDATA%` on Windows; one directory per sha256); (3) a download of the pinned release URL into that cache. The pins are also
compared with the hashes recorded in the stacker file. The GitHub release (not yet created) is for Anvil and public reuse.
Overrides for offline workers: `SINGLET_CELLTYPE_NN_MODEL`, `SINGLET_DOUBLET_CENTROIDS` (local copies, hash-checked too) and
`SINGLET_DOUBLET_BASE_URL` (a mirror or `file://` directory holding the same file names). Pre-fetch on a node with network access
with `python -c "from singlet.features import doublet_assets as A; [A.fetch(k) for k in A.ASSETS]"`.

**Not done / limits.** Reading the engine score from the `.s1` cells table (`dbl_doublet_score`) instead of the run directory is not
wired (the run directory is). Neither the sidecars nor the registry rows were uploaded or registered.
