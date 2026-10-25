# `called_v2`: a cell call computed from stored data

Design note and prototype (2026-10-23). It follows the empty-droplet study (`docs/empty-droplets.md`), which showed that the
stored `called` flag misses many real cells. Code: `bench/called_v2/`. The prototype is `call_v2.py`; the validation is
`validate.py` and `aggregate.py`, with per-sample results in `called-v2/samples.csv`, `called-v2/tables.md` and
`called-v2/outliers.jsonl`. Nothing in `python/singlet/v2/` or the pipeline was changed.

## Proposal

`called_v2` = the stored `called` set, plus every uncalled barcode that EmptyDrops (Lun et al. 2019) finds different from
ambient RNA at BH FDR <= 0.001. The test runs on the counts a v2 sample already holds:

| parameter | value | why |
|---|---|---|
| counts | exon + intron per gene (what `n_umi_exon + n_umi_intron` counts) | best agreement with the authors' calls; exon-only loses nuclei (below) |
| ambient pool | uncalled barcodes with 0 < total <= `lower` | DropletUtils' definition |
| `lower` | "plateau": 3 x the mode of log10(total) below half the 5th percentile of called totals, clipped to [5, 100] (median 42 over usable samples, range 5-100) | the fixed `lower = 100` puts low-RNA cells into the ambient pool; that is the blind spot the hashtags exposed |
| tested | uncalled barcodes with total > max(`lower`, 50) | below 50 UMIs the test has little power and adds BH burden |
| retain | barcodes at or above the knee are called without a test | DropletUtils default |
| FDR | 0.001 for the flag; the per-barcode FDR is stored so 0.01 can be used | 0.01 scores 0.003 higher Jaccard against the authors but 0.009 lower precision |
| Monte Carlo | 10,000 draws, seed 20261023 | deterministic: a re-run reproduces the flag exactly |
| union with `called` | always | `called_v2` never removes a stored cell (0 dropped in 65 samples) |

Per protocol: one parameter set for droplet and microwell protocols (10x 3'/5' v2/v3, Multiome GEX, nuclei, Seq-Well, BD
Rhapsody). For combinatorial indexing (Parse, SPLiT-seq), `called_v2` = `called`, because there is no ambient pool among the
stored barcodes. inDrop and Drop-seq are computed but marked unvalidated (see "Where it fails").

## Validation

Data: the 65 samples (18 studies) the empty-droplet study fetched, 42 of them usable gene-expression samples. Usable means
>= 200 called cells with median >= 100 UMIs, excluding the pilot's TCR libraries and the broken Parse calls. Ground truth:

* the authors' own Cell Ranger calls in GEO, available for 9 studies and 19 samples. This is four studies more than the
  empty-droplet report used: GSE179165, GSE230451, GSE274381 and GSE292909 also deposit filtered matrices. GSE213957's deposited
  table lists only cells that passed the authors' QC, so it was left out;
* hashtags in the hashed PBMC library (GSE108313/GSM2895282);
* species purity in the barnyard references, and the empty-droplet study's profile test, as weaker checks.

The prototype's defaults were fixed before the run. One caveat: the plateau rule was designed knowing the hashtag blind spot
from the empty-droplet study. In the grid of 480 variants per sample, the best variant (identical except FDR 0.01) scores
within 0.003 Jaccard of the defaults.

### Against the authors' Cell Ranger calls

Studies weighted equally (per-sample values averaged within a study first). J = Jaccard, R = recall of the authors' cells,
P = share of our cells the authors also called:

| study | stored `called` J / R / P | Cell Ranger 3-like on stored data J | **`called_v2`** J / R / P | added (median per sample) |
|---|---|---|---|---|
| GSE157827 brain nuclei | 0.26 / 0.26 / 0.99 | 0.61 | **0.75 / 0.94 / 0.81** | 11,420 |
| GSE179165 mouse gut | 0.41 / 0.41 / 0.99 | 0.55 | **0.82 / 0.99 / 0.83** | 6,642 |
| GSE180113 macrophages | 0.60 / 0.60 / 1.00 | 0.67 | **0.92 / 0.92 / 1.00** | 5,859 |
| GSE230451 mouse nuclei | 0.64 / 0.64 / 1.00 | 0.72 | **0.94 / 0.96 / 0.98** | 2,824 |
| GSE230559 AML | 0.45 / 0.45 / 1.00 | 0.56 | **0.79 / 0.86 / 0.90** | 9,813 |
| GSE246624 PBMC | 0.92 / 0.92 / 1.00 | 0.95 | **0.97 / 0.99 / 0.98** | 726 |
| GSE256089 PBMC 5' | 0.63 / 0.63 / 0.99 | 0.77 | **0.91 / 0.92 / 0.99** | 5,150 |
| GSE274381 Seq-Well PBMC | 0.51 / 0.51 / 0.99 | 0.86 | **0.94 / 0.98 / 0.96** | 4,704 |
| GSE292909 cell-line village | 0.55 / 0.55 / 1.00 | 0.59 | **0.90 / 0.90 / 1.00** | 8,953 |
| **all (study-weighted)** | **0.55 / 0.55 / 1.00** | **0.70** | **0.88 / 0.94 / 0.94** | |

Leave one study out: choosing the best of the 480 variants on the other 8 studies picks the defaults' family in 7 of 9 cases
(`lower` = plateau, test > 50, retain = knee, union; FDR 0.01). The other 2 pick `lower` = 20 with test > 100. On the held-out
study the fixed defaults score as well as or better than the leave-one-out choice in 6 of 9 studies (differences from -0.015 to
+0.071 Jaccard). The defaults are not overfitted to these studies.

Where precision against the authors is below 0.9, the extra barcodes sit at 50-300 UMIs. Cell Ranger 3 does not test those:
it only considers candidates with >= 500 UMIs in its own full-depth counts. They are not ambient:

* GSE157827/GSM4775563 (P = 0.44 against the authors) adds 12,209 barcodes. At 100-300 UMIs their intronic fraction is
  0.60-0.64, like the called nuclei (0.64) and unlike ambient (0.20), and 59-80% are profile-test cell-like. Their
  mitochondrial fraction is about twice that of the called nuclei (0.05-0.06 vs 0.027), so some are damaged nuclei. The
  authors' own call for this sample is 6,166 against 15,357 for a sister sample, which suggests a different cutoff, not
  missing nuclei.
* GSE179165/GSM5410927 (P = 0.81): the additions at 50-200 UMIs are 79-96% profile cell-like, with intronic fraction 0.26-0.31
  (called 0.28, ambient 0.20).

### Against hashtags (GSE108313 PBMC, super-loaded)

The empty-droplet study found 14,920 uncalled barcodes carrying a single dominant hashtag, five times the 2,950 called cells.

| call | cells | hashtag singlets called | singlet recall (>= 20 UMIs) | added: singlet / multi / negative |
|---|---|---|---|---|
| stored `called` | 2,950 | 548 | 3.8% | |
| Cell Ranger 3-like on stored data | 3,615 | 848 | 5.8% | 300 / 143 / 169 |
| EmptyDrops, `lower` = 100 (DropletUtils default) | 6,835 | 3,148 | 22% | 2,600 / 580 / 624 |
| **`called_v2`** (`lower` = 17 here) | **21,818** | **13,600** | **93%** | **13,052 / 1,148 / 4,481** |

69% of the added barcodes are confirmed stained single cells, and 6% are hashtag multiplets (super-loading makes doublets).
The 24% that are "negative" have a median hashtag count of 707, 3.5x the empty-droplet level (203) but under the
positive threshold (1,015). Many are probably weakly stained cells; none is confirmed either way. The called set itself is
61% hashtag-negative: large unstained cells with a median 5,038 UMIs. So "negative" is not the same as "empty" in this
library. The ambient bound decides this sample: with `lower` = 100 the ambient pool is mostly PBMC RNA and the PBMCs are
untestable.

### Species purity (GSE163793, human and mouse lines 1:1:1:1)

In the 3' v2 and 5' samples, 91-93% of additions above 30 UMIs are single-species. In the two 3' v3 samples only 40-48% are,
and part of the low-UMI end is a real false-positive class. In GSM4987146, 719 of 2,402 additions (30%) have a human fraction
of 0.3-0.7, mostly at 50-200 UMIs, with intronic fraction 0.06 (called cells 0.17, ambient 0.10). They are neither ambient
(human fraction 0.14) nor single cells. GSM4987148 has 336 of 2,659 (13%). Above 300 UMIs the additions are single-species.
Users who want only confident cells can filter on `called_v2_fdr` and total UMIs; the defaults stay because raising the test
floor to 100 costs more recall elsewhere than it saves here (study-weighted Jaccard 0.877 vs 0.886 at FDR 0.01).

### Where it fails

* **Drop-seq** (GSE163793/GSM4987478; stored call 168): adds 11,605 barcodes, 8,476 of them at 50-100 UMIs, with no
  mitochondrial reads and 2% profile cell-like. These are ambient-like.
* **inDrop** GSE266546/GSM8249613: adds 3,976 barcodes, 0.3-3% profile cell-like, with an intronic fraction like ambient. The
  empty-droplet study had already found this sample's stored call mostly ambient. The other two inDrop samples look cell-like
  (83-86% profile cell-like).
* Feature-barcode libraries processed as RNA (HTO/ADT/TCR) get a meaningless call either way. They should be flagged by
  library type, not by the caller.

So inDrop and Drop-seq get `called_v2` computed, with a sample-level `validated: false`. Nothing in this study gives ground
truth for them.

## Exon/intron double counting

The pileup deduplicates exon and intron UMIs in separate sets (`include/singlet/pileup/pileup_engine.h`: `umi_dedup_` vs
`intron_umi_dedup_`, and `dir_exon_store_` vs `dir_intron_store_`). Exon and intron hits are also decided independently for
each read. One molecule is therefore counted in both matrices when a read straddles an exon-intron boundary, or when
different reads of the same UMI land in an exon and an intron. Introns are gene body minus merged exons
(`gene_model.h`), so isoform overlap is not a cause. Measured from stored data:

* In ambient barcodes, a median 7.5% of intron UMIs duplicate an exon UMI of the same gene (range 0.4-21% over usable samples;
  the pilot's TCR libraries ~20%). This is the excess of same-gene exon/intron pairs over what independent molecules from the
  ambient profile give. In the barcodes holding exactly 1 exon + 1 intron UMI, a median 9% are one molecule (pilot 46-67%).
* In called cells, the upper bound sum_g min(exon_g, intron_g) / total is a median 3.6% of UMIs (max 10%).
* For the call it does not matter: exon + intron is the best input (study-weighted Jaccard 0.886), per-gene
  max(exon, intron) is close (0.868), and exon only is clearly worse (0.813; mouse nuclei 0.72 vs 0.95, the cell-line village
  0.71 vs 0.91). In the hashed library, exon-only EmptyDrops recovers 4,238 hashtag singlets against 13,600.

`called_v2` therefore uses exon + intron as stored. The double counting should be fixed in the pipeline: one label per
(barcode, gene, UMI), as in velocyto and STARsolo GeneFull. The pipeline-benchmark session owns that fix. It changes counts,
not this design: the plateau rule adapts to the new totals.

## What to store

Per barcode, as optional columns of the v2 cells table (`docs/format-v2.md` 5.3 allows them):

| column | dtype | content |
|---|---|---|
| `called_v2` | 7 (bitmap) | the flag |
| `called_v2_fdr` | 5 (f32) | BH FDR of tested barcodes. NaN = not tested: in `called`, in the ambient pool, or below the test floor. A barcode at or above the knee has p = 0 |

Per sample, in the manifest's sample record (or a `cell_call_v2` JSON in the sample object):
`{"method": "singlet-emptydrops-v2.0", "counts": "exon+intron", "lower_rule": "plateau", "lower": 42, "test_min": 50,
"retain": 584.0, "fdr": 0.001, "niters": 10000, "seed": 20261023, "alpha": 490.6, "n_ambient": 79379, "ambient_umi": ...,
"n_tested": ..., "n_called_v2": 14725, "validated": true}`. Alpha and the ambient pool make the test reproducible even if
low-UMI barcodes are later dropped. The p-value is not stored: FDR is what users threshold, and the call is deterministic.

Cost: one EmptyDrops run is a few CPU-seconds per sample (the full 11-run grid took a median 39 s on 8 cores). Storage is
one bitmap plus one f32 column that is NaN for most barcodes; under zstd this should be tens of KB per sample, to be measured
in the writer.

**Recommendation: compute it in the migration and write it into the cells table.** There are three reasons.
(1) v2 sample objects are immutable, so adding it later means a sidecar. Per-cell feature sidecars hold called cells only
(`docs/CELL-ARCHITECTURE.md`), which is the wrong shape for a flag about uncalled barcodes. (2) The ambient pool sits mostly
below 20 UMIs (median `lower` = 42, ambient barcodes mostly < 10 UMIs). If v2 ever applies the "drop < 20 UMIs" floor from
the empty-droplet study, the ambient pool is gone; then only the stored alpha and summed profile keep a re-call possible.
(3) The cost is negligible next to the rest of the migration.

Keep `called` unchanged as provenance (STARsolo `CellRanger2.2 3000 0.99 10`) and as the default selection in the first
release. Making `called_v2` the default for reads (`gather`, cell index, `n_called` in D1) is a separate decision. It changes
cell counts by a median +65% per sample (range +386 to +18,868 cells), and users should see that announced. The cell index
would need a `called_v2` bit to filter on.

## Decisions (Zach, 2026-10-23)

1. Approved: `called_v2` as specified, computed in the migration and stored in the cells table (`called_v2`, `called_v2_fdr`).
   Implementation: `python/singlet/v2/cellcall.py` and `migrate.py` on branch `claude/called-v2-migration` (PR into
   `feat/singlet-v2`).
2. `called` stays the default selection for now.
3. inDrop and Drop-seq: stored with `validated: false` (as proposed; not separately decided).

## In the pipeline (C++ port, prod/atlas-rerun)

`include/singlet/pileup/called_v2.h` is a port of `cellcall.call()` with the same parameters. It is the pipeline's
default cell caller (`--cell-caller called_v2`; `star_solo` and `emptydrops` keep the old callers):

* The knee set `called` is STARsolo's `CellRanger2.2 3000 0.99 10` rule (`SoloFeature_cellFiltering.cpp`), computed
  on the pileup's exon counts after strand detection. STARsolo's own filtered barcodes are no longer used: STARsolo
  counts with `--soloStrand Forward`, so for 5' libraries its knee came from wrong-strand counts. Strand is only known
  after the pileup probe (#25), so passing `--soloStrand Reverse` would not cover undeclared 5' libraries.
* `called_v2` runs on exon + intron counts per gene, keyed and sorted as in `cellcall.gene_counts`.
* `cell_calls.tsv` has one row per barcode: `is_cell` (= `called`, what the migration reads), `called`, `called_v2`,
  `umi_exon`, `umi_total`, `fdr`, `pvalue`, `log_prob`. The parameters (lower, retain, alpha, pool, counts) are in
  `pileup_stats.json` under `provenance.cell_call`. Downstream per-cell outputs use the `called_v2` set.

The port follows the reference down to floating-point order wherever a discrete decision depends on it:

* the Monte Carlo is the same numbers: MT19937 seeded per path (`seed + i`), drawing as numba's `random()` and
  `randint()`;
* cephes `lgam` matches `scipy.special.gammaln` bit for bit;
* the FITPACK `curfit` port reproduces `UnivariateSpline`'s knots and coefficients, including the `nest`
  reset. It matched in 300 of 300 random barcode-rank curves;
* numpy's pairwise sums, `quantile` and `histogram` edges, and scipy's bounded Brent minimiser are reproduced.

One step is not reproduced bit for bit: `np.polyfit` in the Good-Turing smoothing (LAPACK `gelsd`). The ambient
profile can therefore differ in the last bit, and alpha by up to the minimiser's `xatol` (1e-4 in log alpha) when
the likelihood is flat. That can move a Monte Carlo count by one. `tests/cpp/test_called_v2.cpp` checks the port
against `cellcall.py` output (`bench/called_v2/make_cpp_fixture.py`). It requires identical `called_v2` sets and
FDRs equal except for Monte Carlo ties. `bench/called_v2/equiv_cpp.py` runs the same comparison on real pipeline
outputs.

## Limitations

* The authors' calls are Cell Ranger, itself OrdMag + EmptyDrops on full-depth counts (Singlet caps input at 30M reads), so
  agreement partly reflects a shared method. The hashtags, the intronic and mitochondrial fractions and species purity are the
  independent checks, and they agree except at the low-UMI end of the barnyard 3' v3 samples.
* Ground truth covers 10x and Seq-Well only. BD Rhapsody additions are 82-92% profile cell-like but unverified.
* Our EmptyDrops is the empty-droplet study's implementation of DropletUtils' algorithm, not checked against the R package on
  these data.
* `called_v2` inherits every stored cell, including the ambient-like ones the empty-droplet study found (median 0.3% per
  sample, more in inDrop). A separate per-cell "ambient-like" flag would serve those.

## Reproduce

On Clipper from this branch: `sbatch --array=0-17%9 --export=ALL,MODE=validate bench/called_v2/run.sbatch` (reads
`$WORK/data/emptydrop`, fetched by `bench/empty_droplets/run.sbatch MODE=fetch`), then `MODE=aggregate` and `MODE=inspect`.
Results are under `$WORK/state/called_v2/`. On a single fetched sample: `python -P bench/called_v2/call_v2.py SAMPLE.npz`.
