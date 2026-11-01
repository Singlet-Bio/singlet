# Expression doublet score (`qc/doublet_score`)

Package `bench/doublets/legacy_doublets/`, job `bench/doublets/doublets_run.py`, evidence `bench/doublets/` and
`bench/public/capabilities/doublets.py`. Implements the Beta decision of `docs/beta-doublet-decision.md` (evidence: study/latent-qc,
`docs/latent-doublet-ambient.md`). One registered, versioned `cell_feature` computed in batch on stored counts, per
`docs/CELL-ARCHITECTURE.md` section 3: the sidecar is a per-sample Parquet joined on `(gsm, col)`, D1 gets the feature's registry row
and per-sample summaries, never a row per cell, and nothing runs in a Worker or in the engine.

## 1. What is written

`<OUT>/<GSE>/<GSM>.features.qc.parquet` in the layout of `docs/cell_features.md` section 1 (`gsm`, `col`, `doublet_score` float32
in [0, 1], NaN = not computed), written through `singlet.features.output` so the public-safety guard runs first. Also
`<GSM>.features.rows.json` (the `sample_cell_summary` rows), `<GSM>.doublets.json` (internal: status, timings, peak memory,
`method_version`) and `ingest/cell_feature.json` (`bench/doublets/doublets_run.py registry`). Feature id `qc/doublet_score`,
thresholds 0.5 and 0.8.

`method_version` = `singlet-doublets=<package>;combiner=<model id>;celltype-nn=<network>;dbl-run=<commit>;min_cells=100`: a new
combiner, atlas network or engine scorer is a new version.

**Not scored (NaN):** samples with two or more genetic donors (`donor_assignments.tsv`; their doublet call is `prob_doublet`:
Kang 2.1 AUROC 0.969 / AUPRC 0.897, recall 0.89 on doublets that look like one cell type, where expression methods reach
0.47-0.59), samples with fewer than 100 called cells, and organisms other than human and mouse. The score orders the cells of one
sample; it is not a calibrated probability (the share above 0.5 depends on the sample's doublet rate).

## 2. Method

Per sample, from the called cells' counts:

1. `umi`, `genes`: log1p of total counts and of detected genes.
2. `prod`: the engine's `detect_doublets` score (Scrublet-style simulation plus classifier, `doublet_detect.h` v7). The public
   store withholds the doublet columns, so the job recomputes it with the `dbl_run` harness built by
   `bench/doublets/build_dbl_run.sh` from a pinned commit of `prod/modality-fixes`. **`prod/campaign` and `main` still carry the older
   v6 scorer**: the engine's own `doublet_scores.tsv` in a campaign run is not this score. The combiner was trained on v7, so
   merging the v7 header into the campaign engine is a separate change (files owned by the engine sessions).
3. Four atlas-network features (`singlet.doublets.atlas`, a port of `bench/latent_qc/doublets/latent_gpu.py`): `atlas_knn`,
   `atlas_clf`, `ct_entropy`, `atlas_nnls` (definitions in the module docstring). They need `singlet.celltype_nn` (merged, PR #32), a
   `model.pt` and the atlas cell-type centroids (`atlas_centroids.npz`, built by `bench/latent_qc/doublets/atlas_centroids.py`).
4. Each feature is replaced by its within-sample rank, and a gradient-boosted ensemble (300 trees, 15 leaves, learning rate 0.05)
   returns the score. The model is trained once (`bench/doublets/train_eval.py`) on the real labels of 17 datasets (genotype,
   hashtag, MULTI-seq, species mixing; each dataset weighs the same) and packaged as `data/combiner_prod_atlas.npz`; evaluating it
   needs numpy only. Training needs scikit-learn and is not part of the job.

Non-atlas variant: the atlas features are the part of the job that needs a GPU-class network; without them (`prod` + `umi` +
`genes`) the cheap combiners reach 0.52 mean AUPRC, no better than adding Scrublet to the production score (section 4). Because
`celltype_nn` is merged, the atlas variant is the shipped one and no non-atlas variant is provided.

## 3. Evidence

Leave-one-study-out on the 17 real-label datasets (never a simulated doublet in the evaluation; the combiner scoring a dataset was
trained without every dataset of its study). Source: the cached per-cell scores of study/latent-qc on Clipper; table and artifacts
from `bench/doublets/train_eval.py`, public file `bench/public/results/doublets.json`.

| method | mean AUPRC, 17 datasets | Kang lanes (6) | Kang 2.1 on Singlet counts | s / 10k cells |
|---|---|---|---|---|
| production `detect_doublets` v7 | 0.501 | 0.482 | 0.632 | 2 |
| scDblFinder (R) | 0.559 | 0.544 | 0.642 | 44 |
| `combo_prod_atlas` (shipped) | 0.559 | 0.585 | 0.701 | about 10 |

Per-study means (six studies, each once): see the capability file's `auprc_mean_by_study`. The pooled difference to scDblFinder is a
statistical tie, as the decision said; the margin is smaller than the bootstrap CI over datasets.

## 4. Cheap non-R features (the 0.003 question)

`bench/doublets/cheap_features.py` on the cached scores, same leave-one-study-out recipe:

| combiner features | mean AUPRC (17) | Kang lanes (6) |
|---|---|---|
| production + Scrublet + library size (no atlas) | 0.520 | 0.497 |
| production + library size + atlas k-NN | 0.542 | 0.559 |
| production + library size + atlas k-NN + Scrublet | 0.551 | 0.559 |
| unsupervised rank average, production + atlas k-NN + Scrublet | 0.553 | 0.551 |
| **shipped: production + four atlas features** | **0.559** | **0.585** |
| shipped + Scrublet | 0.563 | 0.575 |
| scDblFinder | 0.559 | 0.544 |

A rank-averaged Scrublet score does not close the gap by itself (Scrublet's python score is also close to the engine's own
Scrublet-style score). Adding Scrublet to the shipped feature set reaches 0.563 (above 0.56) but loses 0.010 on the Kang lanes, the
gating real-truth set, and costs another 8 s per 10k cells and a dependency; the +0.004 is also inside the noise (it was found by
comparing eleven variants on the same 17 datasets, so it carries a selection effect). It is not adopted. The shipped pooled mean
(0.559 on a re-run of the study's recipe on Clipper; the study reports 0.557) equals scDblFinder's; exceeding 0.56
outright still needs `combo_all` (adds scDblFinder: 0.572), i.e. R.

## 5. Run

```
bash bench/doublets/build_dbl_run.sh $WORK/bin/dbl-run              # inside a Slurm job
python bench/doublets/doublets_run.py run --gse GSE138867 --out $OUT --model .../m1_notissue/model.pt \
    --centroids .../atlas_centroids.npz --dbl-run $WORK/bin/dbl-run/dbl_run --dbl-commit <sha> --celltype-version m1_notissue
python bench/doublets/doublets_run.py registry ...   # ingest/cell_feature.json
python bench/doublets/doublets_run.py collect --out $OUT   # ingest/sample_cell_summary.*.json (nothing is uploaded)
```

`bench/doublets/e2e.sbatch` runs the job on two store samples and the parity check (`parity_kang.py`) on Clipper. Measured on a V100S
(GSE138867, 8,121 and 6,618 cells): 19.7 s and 11.4 s per sample (engine score 3.0 s and 2.5 s, atlas features 16.3 s and 8.7 s,
including a cold CUDA start), peak memory 2.0 GB. Parity with the study's cached features on Kang 2.1 (13,737 cells): Spearman 1.000
(`prod`, `ct_entropy`, `atlas_nnls`), 0.984 (`atlas_knn`), 0.927 (`atlas_clf`, a cross-fitted classifier with its own seeds).

Publishing is separate and needs a go-ahead: the registry row and the summaries go to `/api/ingest/cell_feature` and
`/api/ingest/sample_cell_summary`; the sidecars go through `singlet.s1.atlas.normalize_feature_sidecar` like the other feature sets.
