# `called_v2` against ground truth: over-calling, its cause and the fix

Study of 2026-10-25. Question: does `called_v2` (docs/called-v2.md; Python `python/singlet/v2/cellcall.py`, C++
`include/singlet/pileup/called_v2.h`) over-call cells, and what change makes it match ground truth without losing the low-RNA
cells it was designed to recover? Code: `bench/called_v2_truth/`. Results on Clipper: `$WORK/state/called_v2_truth/`. No
pipeline code was changed.

## Answer

Yes, it over-calls, and the cause is one bug: **the `retain` knee lands just above `lower`**. In that case every tested barcode
gets p = 0 and is called without a test.

`cellcall.knee` (ported unchanged to C++) fits the barcode-rank spline to all totals above `lower`. DropletUtils' `barcodeRanks`
first restricts the fit to the region between the left edge and the steepest drop (`.find_curve_bounds`); our port leaves that
step out. With `lower` at the plateau (5-100 instead of DropletUtils' 100), the ambient shoulder of the curve is inside the fit,
and the point of maximum curvature can fall there. The production logs show it:

| dataset | `lower` | knee used as `retain` | tested | added | knee with DropletUtils bounds |
|---|---|---|---|---|---|
| Kang 2018 lane | 5 | **8** | 2,597 | 2,597 (all) | 1,194 |
| hgmm1k_v3 | 100 | **115** | 1,616 | 1,038 | 19,857 |
| neuron1k_v3 | 53 | **76** | 2,828 | 1,652 | 12,809 |
| pbmc1k_v3 | 34 | 8,766 | 430 | 56 | 7,590 |

When the bug is fixed, a second limit shows up. At FDR 0.001 with 10,000 Monte Carlo draws, the smallest possible p-value is
1/10,001. BH can therefore pass nothing when fewer than n_tested/10 barcodes sit at that floor, which is what happens in the
small 10x samples (hgmm, neuron1k, nuclei2k). Extrapolating the null tail for barcodes below every simulated path fixes this.

**Recommendation (for the C++/Python implementer):**

1. **Knee with DropletUtils' curve bounds.** On the (log10 rank, log10 total) points with total > `lower`, compute
   `d1n = diff(y)/diff(x)`. Let `skip` = number of points with rank <= 50, `right = skip + argmin(d1n[skip:])` and
   `left = skip + argmax(d1n[skip:right+1])`. Fit the existing smoothing spline on points `left..right` only, and take the knee
   at minimum curvature as now. Reference: `bench/called_v2_truth/common.py:knee_bounded`.
2. **Tail p-values at the Monte Carlo floor.** For a tested barcode whose DM log-probability is below all 10,000 simulated
   paths at its total (`n_le == 0`), set `p = min(Phi((lp - mean_t) / sd_t), 1/(niters+1))`, where mean and sd are taken over
   the simulated log-probabilities at that total. Leave all other p-values unchanged. The call stays deterministic. Reference:
   `common.py:raw_pvalues` (`pv_tail`).
3. Keep everything else: plateau `lower`, test floor 50, FDR 0.001, union with `called`, seed, 10,000 draws.

The method string should change (e.g. `singlet-emptydrops-v2.1`), so stored calls record which knee they used.

## Per dataset (production vs. recommended)

Truth: Cell Ranger filtered barcodes (10x datasets), the demuxlet table (Kang, ctrl lane, all 14,619 rows) and CellBender
(hgmm, 1,048 cells). Extras = called but not in the truth set. They are classified (`datasets.py`) as ambient-like (the
multinomial likelihood favours the ambient profile over every truth-cell cluster profile; in the barnyard, a human fraction
consistent with ambient), doublet (Singlet demux doublet, or mixed species unlike ambient), damaged (mito >= 20%), or likely
cell. Calibration at the extras' depth: truth cells downsampled to that depth score cell-like in 98.5-100% of cases, and draws
from the ambient profile in 0-0.8%.

| dataset | truth | prod `called_v2` | recall / precision | extras: cell / ambient / doublet / damaged | **recommended** | recall / precision | extras: cell / ambient / doublet / damaged |
|---|---|---|---|---|---|---|---|
| Kang (demuxlet) | 14,619 | 17,004 | 1.000 / 0.860 | 1,451 / 924 / 4 / 6 | **16,573** | 1.000 / 0.882 | 1,368 / 580 / 4 / 4 |
| hgmm1k_v3 (Cell Ranger) | 1,046 | 1,919 | 1.000 / 0.545 | 75 / 386 / 154 / 258 | **1,026** | 0.961 / 0.980 | 11 / 4 / 6 / 0 |
| hgmm1k_v3 (CellBender) | 1,048 | | 1.000 / 0.546 | | | 0.951 / 0.972 | |
| neuron1k_v3 | 1,301 | 2,758 | 1.000 / 0.472 | 428 / 787 / 0 / 242 | **1,355** | 0.952 / 0.914 | 86 / 30 / 0 / 0 |
| nuclei2k_v2 (CR2 OrdMag) | 2,352 | 2,175 | 0.909 / 0.983 | 37 / 0 / 0 / 0 | **2,376** | 0.910 / 0.901 | 208 / 26 / 0 / 2 |
| pbmc1k_v3 | 1,222 | 1,190 | 0.959 / 0.985 | 18 / 0 / 0 / 0 | 1,190 | same | same |
| pbmc5p_v1 | 8,258 | 8,399 | 0.984 / 0.967 | 210 / 6 / 0 / 59 | 8,399 | same | same |

(The production run of pbmc1k_v3 used here, `singlet-modfix-t8`, calls 1,190; `prod-t4/t8` call 1,196.)

* **hgmm1k_v3:** the 873 production extras have a median of 147 UMIs, 28% mitochondrial reads and an intronic fraction of 0.03
  (cells: 23,000 UMIs, 6.5%, 0.21). 381 of them are species-pure, 222 are mixed but unlike ambient, and 270 sit at the ambient
  human fraction. They are empty droplets and debris, not cells. After the fix, 21 extras remain.
* **neuron1k_v3:** the production extras have a median of 119 UMIs and an intronic fraction of 0.03 (cells 0.34), so they are
  ambient. The recommended call keeps 116 extras (median 401 UMIs, intronic fraction 0.14, 74% cell-like).
* **nuclei2k_v2:** the stored call alone misses 214 truth nuclei. The tail p-values recover 201 barcodes at around 500 UMIs.
  Their intronic fraction is 0.52 (nuclei 0.62), so these are low-RNA nuclei that Cell Ranger 2's OrdMag does not test, not
  ambient.
* **pbmc1k_v3, pbmc5p_v1:** the knee was sane, so nothing changes. The 59 damaged extras in pbmc5p are high-mito barcodes at
  around 220 UMIs.

### Kang: genotype check (`kang_genotype.py`)

The method: cellsnp-lite on the 19,743 informative SNPs of the reference run, then vireo with the 8 known donor genotypes.
Singlet's own demux is not used. To set the reference rates at the same depth (about 350 UMIs), we added two comparison groups:
truth cells thinned to that depth, and synthetic empty droplets (sums of real 20-50 UMI empty barcodes).

| group | n | median UMIs | median SNPs covered | confident single donor (prob >= 0.9) |
|---|---|---|---|---|
| extras kept by the fix | 1,956 | 350 | 41 | **56.5%** |
| extras removed by the fix | 429 | 68 | 6 | 7.7% |
| truth cells thinned to that depth | 1,495 | 348 | 45 | 74.4% |
| synthetic ambient at that depth | 2,000 | 344 | 26 | 8.6% |

Mixture estimate: about 73% of the remaining extras are real single cells (about 1,430). With >= 50 SNPs covered, 86-88% of
the extras are confidently one donor, against 87-90% for the thinned truth cells and 3.6% for synthetic ambient. Their donor
agrees with Singlet's demux in 89% of cases. Of the extras, 1,368 are expression-cell-like, with median mito 2.0% and 219 genes.
These are low-RNA cells below the authors' Cell Ranger 1 cutoff (truth cells start at about 1,000 UMIs). The remaining
roughly 27% (mostly fewer than 20 SNPs) are ambient-like. Kang's plateau `lower` hits the clip floor (5). With `lower` = 100,
the estimated real-cell share of the extras rises to 0.93 (1,514 extras). But `lower` = 100 is clearly worse on the
validation set (below), so this residual is documented rather than fixed. Users who want only confident cells can filter on
`called_v2_fdr` or total UMIs.

## The 42-sample validation, re-scored (`revalidate.py`)

These are the same samples and truth as docs/called-v2.md, with p-values recomputed by the production kernel (identical to the
stored ones). Studies are weighted equally. "Kept" counts the production additions that the authors also called and that the
variant keeps.

| variant | J / R / P vs authors | kept author-confirmed recoveries | hashtag sample: cells, singlets added, singlet recall | auto-retained (samples) |
|---|---|---|---|---|
| production (unbounded knee) | 0.883 / 0.938 / 0.940 | | 21,818, 13,052, 0.93 | 47,833 (30) |
| **bounded knee + tail p, FDR 0.001** | **0.882 / 0.938 / 0.940** | **109,280 / 109,681 (99.6%)** | 19,212, 11,302, 0.81 | 26,960 (29) |
| bounded knee, FDR 0.001 (no tail) | 0.882 / 0.938 / 0.940 | 99.6% | 19,212, 11,302, 0.81 | |
| retain none (+/- tail), FDR 0.001 | 0.879 / 0.933 / 0.941 | 97.3% | 19,212, 11,302, 0.81 | 0 |
| bounded knee (+/- tail), FDR 0.01 | 0.886 / 0.950 / 0.931 | 99.8% | 20,462, 12,124, 0.87 | |
| bounded knee, test > 100 | 0.873 / 0.917 / 0.951 | 93.5% | 17,113, 10,234, 0.74 | |
| `lower` = 100, bounded knee + tail | 0.867 / 0.906 / 0.956 | 89.3% | 6,835, 2,600, 0.22 | |

* Only one validation sample had a degenerate knee: the hashed PBMC library GSE108313/GSM2895282, where the knee was 112 against
  3,223 with bounds and 15,832 barcodes were auto-retained. There, the 2,606 barcodes the fix drops were 67% hashtag singlets
  and 31% hashtag-negative: low-RNA cells EmptyDrops cannot separate from ambient at FDR 0.001 in a super-loaded library. The
  earlier report's hashtag recall of 93% depended on the bug. The honest number at FDR 0.001 is 81%, or 87% at FDR 0.01.
* Elsewhere the fix changes almost nothing: per-study Jaccard is identical in 7 of 9 studies (GSE230559 0.791 -> 0.788,
  GSE256089 0.913 -> 0.911). The species purity of additions in the barnyard references is unchanged (0.40-0.93).
* The tail p-values change nothing in these 42 samples, because the floor is not binding there. They matter in small 10x
  samples (hgmm 881 -> 1,026, neuron1k 1,106 -> 1,355).
* FDR 0.01 is the alternative if hashtag-like recall matters more than precision. It gives +0.004 Jaccard and 87% hashtag
  recall, at the cost of more extras: Kang 16,674, neuron1k precision 0.899, nuclei2k 0.881.

## Limitations

* The demuxlet table and the Cell Ranger calls are themselves cutoff-based. Extras below their cutoffs are not false by
  definition, which is why the extras are classified rather than counted as errors.
* The expression classes use truth-cell clusters, so a cell type absent from the truth set would score ambient-like.
* The vireo genotypes come from the reference run on the truth barcodes. That run separates 7 of the 8 donors well; donor 1039
  is not recovered (recall 0.002). This lowers the confident-singlet rate of truth cells and extras alike, and so does not
  bias the comparison.

## Reproduce

Scripts in `bench/called_v2_truth/` (put into `$WORK/jobs/called_v2_truth` together with `cellcall.py` from branch
`claude/called-v2-migration`):

* `run.sbatch` with `MODE=datasets`, array 0-5: `datasets.py` per dataset, giving `datasets/<name>.json` and per-barcode
  `.barcodes.tsv.gz`;
* `MODE=revalidate`, array 0-17, then `MODE=aggregate`: `revalidate.py`, giving `report/revalidate.json`;
* `kang_genotype.sbatch`, giving `kang_genotype/kang_genotype.json`;
* `MODE=kang_lower`, giving `kang_lower.json`.
