# Cell type labels: research beta status (2026-10-26)

Singlet 1.0.0 research beta. Package `python/singlet/celltype_nn/`, full report `docs/celltype_nn.md`, public benchmark
`bench/public/results/cell-type.json` (contract 1.1.0, draft). PRs: Singlet-Bio/singlet#32 (package, harness, report;
into `feat/singlet-v2`) and Singlet-Bio/singlet#44 (capability module and result; into the bench-public branch). Neither is
merged.

## What it is

A network gives every cell a calibrated probability over 822 Cell Ontology terms. Two labels come out of it:
- `argmax_cl`: the most probable term;
- `cell_type_cl`: the most specific term whose probability, summed over its subtypes, is at least 0.5. It backs off to the
  parent term when the subtypes share the probability.

The network is trained on 2.8M CELLxGENE Census cells (author labels) plus refinements that an annotator ensemble agrees
on: the Singlet guide, CellTypist and a rebuilt SingleR. Per-cell output is a `(gsm, col)` sidecar of 48 B per cell with a
public-output guard.

## Results so far

Real data only. cxg16: 16 GEO studies, 291,331 cells with CELLxGENE curators' labels, scored on Singlet's counts. None of the
16 studies was used for training, and every tool learned from the same Census reference.

| | Singlet | CellTypist | scANVI | Seurat mapping (Azimuth method) | SingleR |
|---|---|---|---|---|---|
| accuracy, depth 1 / 2 / 3 / 4 | **0.903 / 0.744 / 0.418 / 0.498** | 0.830 / 0.671 / 0.376 / 0.326 | 0.835 / 0.729 / 0.352 / 0.289 | 0.797 / 0.628 / 0.332 / 0.241 | 0.856 / 0.708 / 0.392 / 0.384 |
| wrong calls: forced label / calibrated call | 0.374 / **0.251** | 0.397 | 0.380 | 0.369 | 0.380 |
| calibration error: forced label / calibrated call | 0.216 / **0.079** | 0.364 | 0.514 | 0.135 | - |
| abstention (risk-coverage area) | **0.293** | 0.422 | 0.650 | 0.561 | 0.519 |
| CPU-seconds per million cells | **1,096** | 1,948 | 144,182 (GPU) | 18,937 | 533,230 |

Against the previous Singlet guide in the same harness (0.838/0.734/0.420/0.320, wrong 0.336), the network's forced label
is ahead at every depth.

## Known issues

- **Three public metrics fail, under strict rules:**
  - the forced label's wrong rate is 0.005 behind Seurat;
  - the forced label's probability is worse calibrated than Seurat's (the calibrated call is fine);
  - in tissues thin in the reference (2 studies, dental pulp and mouse ependyma), lineage accuracy is 0.51 against
    scANVI's 0.61.
- **Thymic CD34+ progenitors (GSE144870)** lose most of their accuracy on Singlet's counts (depth 2 0.22 for the call)
  but not on the authors' matrix (0.98). A reprocessing difference worth investigating in the pipeline.
- **Cultured airway epithelium** is thin in Census; the network is weak there with either counts.
- **Choices made after looking at cxg16, disclosed in the report and the JSON caveats:**
  - the call's calibration criterion;
  - trying tissue context, which was not kept;
  - reporting the forced label as Singlet's comparison value;
  - a second temperature for `argmax_prob`, which made it worse on cxg16 and was kept.
- **CellTypist pretrained models** carry no licence statement; their calls are used only as weak training labels.

## What remains

- **The rest of the cohort's cell-type datasets** (Ding 2020 in 7 protocols, the 10x and mouse sets, Segerstolpe, Tabula
  Muris) need the bench-public harness runs and their author labels, then `scripts/celltype_nn/compete/*` on them.
- **Merges:** the bench-public branch, then #44 on it; #32 into `feat/singlet-v2`. The capability's `compute()` needs
  `singlet.celltype_nn.benchmark` from #32.
- **Uploads, needing approval:** the per-cell artifacts (`$WORK/bpub/artifacts/cell-type/`, 39 MB) to R2.
- **Atlas-wide inference, needing approval:** about 250 CPU-h on Clipper, 9.3 GB of sidecars, not run
  (`docs/celltype_nn.md` section 8).
- **Possible improvements:**
  - fix the thymus reprocessing gap;
  - more cultured and thin-tissue reference data;
  - Azimuth's own references for the 3 tissues they cover;
  - a model trained with the Singlet pseudo-labels at a lower weight (`m1p` was better calibrated but made more wrong
    calls).
