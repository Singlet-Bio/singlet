# Are Singlet's uncalled barcodes empty droplets?

Study for the v2 decision "may v2 drop the matrix entries of uncalled barcodes?" (2026-10-23). Code: `bench/empty_droplets/`.
Data: 65 samples of 17 studies read from the legacy `.singlet` bundles, 49 of them usable as gene-expression libraries (the rest
are hashtag/antibody/guide or TCR libraries processed as RNA, or Parse sub-libraries with a broken call; see "Samples").

## Verdict

**No. The uncalled barcodes are not demonstrably empty, and v2 should not drop them.** In 47 of 49 usable samples, between
hundreds and ~18,000 uncalled barcodes per sample (median 3,317, i.e. 57% as many as the called cells) carry RNA from single cells:

* their expression rejects "ambient RNA only" at FDR 1% against a depth-matched, overdispersion-aware null, and still does with
  a null four times more overdispersed (252k of 3.52M uncalled barcodes; 183k under the 4x null);
* split halves of each barcode's counts independently pick the same cell type, far above what any ambient draw does;
* in species-mixing samples they are single-species (83-97% of uncalled barcodes above 300 UMIs; 350-1,112 hidden single cells
  per sample) while the ambient plateau never is;
* their intronic and mitochondrial fractions are those of the called cells (median 0.27 and 0.017), not those of the ambient pool
  (0.10 and 0.000), although the test never looked at introns;
* they are enriched up to 10-60x in specific low-RNA cell types (microglia-like and oligodendrocyte clusters in brain nuclei,
  plasma cells, neutrophils, cytotoxic T cells);
* the authors' own Cell Ranger calls include them: in human brain nuclei (GSE157827) the authors called 3.2-4.6x more nuclei than
  Singlet, a superset of ours;
* every standard caller run on the stored matrix calls more cells than Singlet: EmptyDrops a median 1,911 more per sample
  (range 389-10,972), Cell Ranger 3-style a median 808 more, and 77-92% of those extra calls are cell-like by the profile test.

The barcodes that are demonstrably empty are the low-UMI ones, and they hold few entries: everything below 20 UMIs is a median
**5% of the matrix bytes** (p10-p90 2-13%), not the 30-55% that "drop uncalled" would save. The safe rule and its saving are
in "Recommendation".

Two facts change the framing of the question:

1. **`called` is not EmptyDrops.** No pipeline code passes `--soloCellFilter`, so STARsolo applies its default
   `CellRanger2.2 3000 0.99 10`: a single UMI threshold at one tenth of the 99th percentile of the top 3,000 barcodes, on STAR's
   own counts (`export.h` takes `Solo.out/Gene/filtered/barcodes.tsv` whenever it exists; the pipeline's own EmptyDrops runs only
   for protocols without Solo.out). That rule is known to miss cells with little RNA, and it is what the data show: a single
   threshold on the stored totals reproduces the call to within 0-5% of barcodes in every sample (median 0.6%).
2. **The v2 pilot GSE145370 is the wrong library type.** Its 14 samples (GSM4317423-36) are TCR-enriched V(D)J libraries
   ("S133_TCR Sample 1" ... in GEO); the mRNA libraries are GSM4317409-22 and are not in the bundle. The pilot's shallow counts,
   TRBV/TRAV cluster markers and large vdj member follow from that. Measurements on the pilot (29.9 MB -> 13 MB; 39-56% of
   entries in uncalled barcodes) describe amplicon libraries. Over the 49 usable gene-expression samples uncalled barcodes hold a
   median 32% of exon+intron entries (p10-p90 11-63%) and 33% of the estimated matrix bytes.

## What the stored barcodes are

* Matrix columns are the barcodes with >= 100 raw reads in the `.1fq` barcode dictionary (`src/pipeline/singlet.cpp`, barcode
  auto-discovery), not the whitelist and not the cells. In pileup counts many of them hold 0-2 UMIs (median uncalled barcode:
  8 UMIs in the PBMC sample, 1 in the pilot).
* `called` comes from STAR's call on STAR's counts; the stored counts are the pileup's. The two agree on rank: the best single
  threshold on stored totals mismatches 0-5% of barcodes (median 0.6%).
* The pileup sometimes counts one molecule once in the exon and once in the intron matrix: among uncalled barcodes with exactly
  2 UMIs, the two sit on one gene in a median 8% of cases (up to 70% in BD Rhapsody, 30-54% in the pilot), and in about half
  of those as 1 exon + 1 intron count. This inflates apparent overdispersion at low depth, so every expression test here uses exon counts only.
* The pipeline caps input at 30M reads per sample (`input_reads_capped`), so "deep" samples are subsampled; depth differences
  between studies are mostly protocol and library complexity.

## Samples

| study | protocol | reference | usable/fetched | called (median) | called UMI q05 | uncalled share of entries | cell-like uncalled (median) | EmptyDrops extra | CR3-like extra |
|---|---|---|---|---|---|---|---|---|---|
| GSE108313 cell hashing | 10x v2 | GRCh38 | 1/4 | 2950 | 2070 | 0.31 | 691 | 3874 | 665 |
| GSE137539 neutrophils | 10x v2 | barnyard (mouse only) | 3/3 | 5022 | 1142 | 0.17 | 2185 | 1453 | 1266 |
| GSE145370 pilot (TCR libs) | 10x 5' | GRCh38 | 7/14 | 2473 | 60 | 0.36 | 394 | 1002 | 517 |
| GSE157827 brain nuclei | 10x v3 | GRCh38 | 3/3 | 1936 | 1242 | 0.46 | 11821 | 10024 | 3026 |
| GSE163793 human+mouse lines | 10x v2/v3/5' + Drop-seq | barnyard | 4/5 | 3596 | 1839 | 0.15 | 4050 | 1208 | 400 |
| GSE179165 mouse gut immune | 10x v2 | GRCm39 | 2/2 | 2986 | 499 | 0.44 | 7174 | 5986 | 1158 |
| GSE180113 macrophages | 10x v3 | GRCh38 | 1/1 | 11074 | 720 | 0.21 | 6677 | 6018 | 1197 |
| GSE213957 HEK + 3T3 (CaRPool) | 10x v3 | barnyard | 4/4 | 9938 | 560 | 0.09 | 1102 | 749 | 136 |
| GSE216584 liver | BD Rhapsody | GRCh38 | 3/3 | 3017 | 614 | 0.36 | 16032 | 1911 | 583 |
| GSE216784 schwannoma cells + nuclei | Multiome GEX | GRCh38 | 4/4 | 4242 | 878 | 0.24 | 3546 | 3476 | 1028 |
| GSE230451 mouse hippocampus nuclei | 10x v3 | GRCm39 | 2/2 | 5590 | 653 | 0.23 | 884 | 2057 | 582 |
| GSE230559 AML marrow/blood | 10x v3 | GRCh38 | 3/3 | 15376 | 460 | 0.32 | 11539 | 8518 | 2584 |
| GSE233574 organoids | Parse | GRCh38 | 0/3 | (1-101 called: broken call) | | | | | |
| GSE246624 PBMC multiplexing | 10x v3 | GRCh38 | 2/3 | 7910 | 916 | 0.09 | 398 | 694 | 330 |
| GSE256089 PBMC, hashed | 10x 5' | GRCh38 | 2/3 | 10977 | 570 | 0.31 | 7851 | 5196 | 2462 |
| GSE266546 gut biopsies | inDrop | GRCh38 | 3/3 | 374 | 639 | 0.84 | 7182 | 2547 | 1436 |
| GSE274381 PBMC | Seq-Well | GRCh38 | 3/3 | 5593 | 220 | 0.32 | 4271 | 3952 | 3675 |
| GSE292909 cell-line village | 10x v3 | GRCh38 | 2/2 | 14096 | 314 | 0.33 | 5758 | 6708 | 1101 |

"Usable" = at least 200 called barcodes with median >= 100 UMIs. Excluded: GSE108313 GSM2895283/84 and GSM3501447 (HTO/ADT
libraries), GSE256089 GSM8085473 (HTO library, 6 barcodes), GSE246624 GSM7873659 (144 called), GSE163793 GSM4987478 (Drop-seq,
168 called), GSE233574 (Parse: 1-101 called per sub-library), 7 pilot samples (median called < 100 UMIs). "Uncalled share" is the
mean of the exon and intron shares of feature-level non-zeros; "extra" = called by that caller and not by Singlet (medians over
the usable samples of the study).

## Evidence

### 1. Profile test against ambient (Figure 2)

For each sample: ambient profile = Good-Turing proportions of the summed exon counts of uncalled barcodes with at most
`clip(0.5 x q05(called), 2, 100)` exon UMIs (any cells hiding there make the test conservative); Dirichlet-multinomial
concentration by maximum likelihood over those barcodes (as EmptyDrops); cell profiles = Leiden clusters of the called cells
(scanpy defaults). Each barcode is fitted as `w q_k + (1 - w) a` over clusters k and a grid of w; the statistic is the log
likelihood ratio against pure ambient. Its null is simulated, per barcode depth, from Dirichlet-multinomial ambient draws pushed
through the same fit, so the p-values account for depth, overdispersion and the fitting itself; beyond the simulated range an
exponential tail fitted above the null's 90th percentile extrapolates (calibration on fresh nulls: 1.0% below 0.01, 0.08% below
0.001). BH within the uncalled barcodes.

Power: called cells thinned to fixed depths and scored against clusters built from the other half of the cells reject ambient
in 35% / 67% / 84% / 93% / 98% of cases at 10 / 20 / 30 / 50 / 100 exon UMIs (medians over samples). So a 30-100 UMI barcode
that the test leaves alone is ambient-like with good confidence; at 10-20 UMIs the test cannot tell.

Pooled over the 49 usable samples, fraction of uncalled barcodes rejecting ambient at FDR 1% (fitted null / 4x overdispersed null):

| total UMIs | barcodes | cell-like | cell-like, 4x null |
|---|---|---|---|
| < 10 | 1,520,876 | 0.7% | 0.5% |
| 10-20 | 912,944 | 2.2% | 1.2% |
| 20-30 | 377,967 | 4.2% | 2.5% |
| 30-50 | 300,676 | 6.0% | 3.6% |
| 50-75 | 104,625 | 13.9% | 8.9% |
| 75-100 | 47,591 | 23.3% | 15.9% |
| 100-150 | 58,868 | 33.4% | 23.8% |
| 150-200 | 42,742 | 48.9% | 36.5% |
| 200-300 | 59,138 | 68.1% | 52.0% |
| 300-500 | 56,263 | 81.0% | 63.4% |
| 500-1000 | 31,809 | 88.9% | 78.3% |
| 1000-2000 | 6,344 | 92.9% | 87.5% |

The fitted cell fraction w of significant barcodes is 0.8-0.9 above ~200 UMIs: these are cells with some ambient, not ambient
with a trace of cell. Where the transition sits depends on the sample: in the PBMC library GSE108313 the uncalled plateau at
30-500 UMIs is ambient (median w = 0.01, 1-5% significant) and only barcodes above 500 UMIs are cell-like; in brain nuclei
GSE157827 everything above ~150 UMIs is (88-100%); in BD Rhapsody liver a minority of 10-30 UMI barcodes is already cell-like.

### 2. Split-half consistency (Figure 3)

Each barcode's exon counts are split 50/50 at random and the best cluster is fitted to each half. Ambient, however overdispersed
around the ambient mean, agrees only at a chance rate that rises slowly with depth (measured on null draws with alpha/4). In brain
nuclei the uncalled barcodes agree 73-87% at 100-300 exon UMIs against 10-15% for the null; in the human/mouse lines 55-63% at
200-500 against 14-19%. Thinned held-out called cells agree less often at the same depth (brain: 31-51% at 100-200 exon
UMIs), because the hidden barcodes come mostly from a few very distinct clusters (section 5).

### 3. Species mixing (Figure 4)

GSE163793 (Jurkat + TALL104 human, EL4 + IVA12 mouse, 1:1:1:1) and GSE213957 GSM6596913 (HEK293FT + NIH/3T3) are mapped to the
combined reference. A barcode is single-species if >= 90% of its UMIs come from one genome and a binomial test against the
ambient mix rejects (p < 0.001); counted only at >= 30 UMIs.

| sample | ambient human fraction | uncalled 30-75 UMIs single-species | uncalled >= 300 UMIs single-species | hidden single cells (>= 30 UMIs) | called |
|---|---|---|---|---|---|
| GSE163793/GSM4987142 (3' v2) | 0.07 | 0.0% | 350 / 359 | 383 | 2300 |
| GSE163793/GSM4987144 (5') | 0.10 | 2.3-5.0% | 587 / 600 | 913 | 3421 |
| GSE163793/GSM4987146 (3' v3) | 0.14 | 0.3-2.2% | 1112 / 1212 | 1838 | 3771 |
| GSE163793/GSM4987148 (3' v3) | 0.12 | 0.3-0.6% | 587 / 671 | 1284 | 7881 |
| GSE213957/GSM6596913 | 0.97 | 0.3-0.5% | 518 / 581 | 794 | 13527 |

Of the barcodes the profile test calls cell-like, 42-65% are single-species; of those it leaves alone, under 0.5%. Two
independent tests agree barcode by barcode, and the species test needs no model of ambient expression.

### 4. Intronic and mitochondrial fractions

Medians over the 49 samples of the per-sample medians:

| group | intronic fraction | mitochondrial fraction |
|---|---|---|
| called | 0.267 | 0.019 |
| uncalled, cell-like | 0.267 | 0.017 |
| uncalled, not cell-like | 0.103 | 0.000 |
| ambient pool | 0.102 | 0.000 |

In brain nuclei (GSE157827/GSM4775562): called 0.63, cell-like uncalled 0.61, ambient 0.20 intronic. The profile test only
saw exon counts, so the intronic fraction is an independent check: cell-like uncalled barcodes are nuclei/cells, the rest is
ambient-like.

### 5. Which cells are missed

The cell-like uncalled barcodes are not a random sample of the called cells. The most enriched cluster per sample (share of
cell-like uncalled vs share of called): brain nuclei RGS1+ cluster 48% vs 4% and an oligodendrocyte (PLP1, CLDN11) cluster
48% vs 5%; AML marrow cytotoxic T/NK (LCK, PRF1, CD160) 34% vs 4% and B cells (MS4A1, CXCR5) 29% vs 5%; neutrophil samples
Retnlg/Mmp8 (4.4x) and Il1b/Cxcl2 (3.6x) clusters; PBMC plasma cells (IGHA1, TNFRSF17) 19% vs 0.3%; human/mouse lines a mouse
B-lineage cluster 44% vs 5%; schwannoma T cells 21% vs 3%. Low-RNA cell types are what a UMI threshold drops, so dropping their
barcodes would bias every composition estimate made from v2.

Some cell-like barcodes are damaged cells or fragments (one brain sample's enriched cluster is mitochondrial transcripts). They are
still not empty droplets, and deciding whether they are cells belongs to whoever re-calls.

### 6. Independent callers on the stored matrix

Medians over the 49 usable samples (EmptyDrops: our implementation of DropletUtils' algorithm, 10,000 Monte Carlo draws,
FDR 0.001, retain = knee; "adaptive" lower = clip(0.5 x q05(called), 5, 100)):

| caller | extra cells vs Singlet | Singlet cells it drops | Jaccard | extra calls that are profile-cell-like |
|---|---|---|---|---|
| EmptyDrops, lower = 100 | 1911 | 12 | 0.62 | 81% |
| EmptyDrops, adaptive lower | 1911 (389-10,972) | 8 | 0.64 | 79% |
| EmptyDrops, exon only | 1374 | 121 | 0.69 | 92% |
| Cell Ranger 3-like (OrdMag + EmptyDrops) | 808 (89-4,808) | 12 | 0.84 | 83% |
| OrdMag on stored counts (CellRanger2.2) | 470 | 67 | 0.85 | 77% |
| knee (barcodeRanks) | 30 | 794 | 0.70 | 89% |
| inflection | 719 | 1 | 0.64 | 82% |

No caller finds Singlet's call too generous except the knee, which sits above it. CellBender: see "CellBender" below.

### 7. Called barcodes that look empty

The same test applied to called cells (held-out clusters) finds 1.2% of called barcodes indistinguishable from ambient (p > 0.05),
mostly in the pilot's TCR libraries (6-27% of their called barcodes); median 0.3% per sample. One real exception: inDrop
GSE266546/GSM8249613, where 163 of 374 called barcodes (~300 UMIs) are ambient-like, i.e. that sample's call is mostly
ambient; GSE292909 has 1.4-1.7% (cells at ~300 UMIs). A minimum-UMI floor on called cells is not needed for storage (called cells are kept anyway); a per-cell
"ambient-like" flag would serve users better than a floor.

## Recommendation

**Keep the matrix entries of every stored barcode in v2.** The cell call can be redone from stored data only if the uncalled
barcodes above the ambient plateau keep their profiles, and the plateau's position varies by sample (30-500 UMIs in the
samples here), so no fixed rule drops much without dropping cells.

If storage must shrink, the only rule this study supports is:

> keep the called barcodes and every uncalled barcode with >= 20 total (exon + intron) UMIs; for the dropped barcodes keep
> their per-barcode totals (already in the `cells` table: `n_umi_exon`, `n_umi_intron`, `n_feat_*`) and one summed profile per
> sample (exon and intron counts per feature over the dropped barcodes), plus the EmptyDrops concentration alpha estimated on
> the full data at the default lower bound.

What it saves and loses (49 usable samples; bytes estimated as member bytes x share of that member's non-zeros kept):

| rule | matrix bytes saved, median (p10-p90) | cell-like uncalled lost, pooled (median per sample) | extra EmptyDrops cells lost |
|---|---|---|---|
| drop all uncalled | 33% (12-64%) | 100% | 100% |
| keep >= 100 UMIs | 15% (5-39%) | 36% (18%) | 3.3% |
| keep >= 50 UMIs | 11% (4-28%) | 26% (9%) | 1.1% |
| **keep >= 20 UMIs** | **5% (2-13%)** | **12% (2.8%)** | **0%** |
| keep >= 10 UMIs | 2% (0.2-8%) | 4% (1.9%) | 0% |

The pooled loss at >= 20 comes mostly from BD Rhapsody and inDrop, where a minority of 10-30 UMI barcodes is cell-like; at
10-20 UMIs the test has 35-67% power, so these barcodes are neither demonstrably empty nor demonstrably cells.

What survives under that rule:

* EmptyDrops re-calling: its inputs are the summed ambient profile (stored), alpha (stored scalar), the profiles of barcodes
  above `lower` (kept, since any sensible lower is >= 20) and every barcode's total (cells table). With alpha re-estimated from
  the remaining ambient barcodes instead of stored, re-calls agree with the full data to within 1% in 26 and within 3% in 36
  of 49 samples; the larger differences (PBMC GSE108313 67%, the pilot's TCR libraries 16-42%, Seq-Well 4-9%, mouse nuclei
  3-9%) come only from alpha, since the tested barcodes and the ambient profile are identical. With the stored alpha the
  re-call is exact by construction. Alpha depends on the chosen `lower`; storing it for lower = 100 (and the adaptive lower)
  covers the defaults.
* SoupX: its soup profile is the summed counts of droplets in `soupRange` (default 0-100 UMIs), recoverable exactly from the
  stored sum plus the kept barcodes.
* Knee/OrdMag/Cell Ranger 2 calls and barcode-rank plots: need only totals (cells table).
* Lost: per-droplet profiles below 20 UMIs. CellBender models individual empty droplets and would have to be fed droplets
  simulated from the stored profile at the stored totals; DecontX, scDblFinder and clustering do not use them.

A cheaper way to the same goal than dropping: keep uncalled barcodes, but in their own chunk run (or member) after the called
cells, so a default read never fetches them; the cell-index session already proposes variable-width chunks. Storage then costs
33% more than "called only", reads cost nothing extra, and nothing is irreversible.

Follow-ups this study points to (not done here): fix the pilot (migrate the mRNA libraries GSM4317409-22, not the TCR ones);
record the STARsolo filter actually used and consider an EmptyDrops-based `called_v2` flag computed from the stored data; screen
the atlas for feature-barcode and V(D)J libraries processed as RNA.

## CellBender

PENDING

## Ground truth from the authors' deposits

PENDING

## Limitations

* The profile test needs clusters of called cells; a cell type with no called representative cannot be detected (it would look
  "not ambient" only if far from ambient, which is then counted against the best cluster anyway).
* Barcodes below ~20 exon UMIs are not testable one by one; what is said about them is collective.
* Byte savings are estimated from non-zero shares; low-UMI barcodes are mostly 1-counts, which the codec stores cheaply, so the
  true saving of a floor is probably a little smaller than estimated.
* One reference implementation of EmptyDrops (ours) was used; it follows DropletUtils but was not cross-checked against the R
  package on these data.

## Reproduce

All runs on Clipper from this branch (`bench/empty_droplets/run.sbatch`): `MODE=all` over `studies.tsv` (fetch by HTTP range,
analyse), `MODE=groundtruth`, `MODE=rules`, `MODE=aggregate`; CellBender with `cellbender.sbatch`. Results under
`$WORK/state/emptydrop/`; the figures and `summary.json` here come from `MODE=aggregate`.
