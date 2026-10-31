# Pre-registration: hashtag-oligo demultiplexing on an independent set

Written 2026-10-31, committed before any hashtag caller was run on the independent set (git history of this file is the
timestamp). Code: `hto_callers.py`, `lodo_bff.py`, `eval_independent.py` in this directory.

## 1. Question

Does Singlet's HTO caller (`include/singlet/pileup/hto_demux.h`, fold-change rule plus Gaussian-mixture doublet veto) reach the best
open competitor on a hashed dataset that played no part in designing or selecting it? The release gate measured 0.919 / 0.814 /
0.748 / 0.726 (Howitt 2023 BAL1/2/3/cell lines) against 0.935 / 0.846 / 0.785 / 0.874 for BFF-cluster, but the caller's setting
was chosen by leave-one-dataset-out on those four datasets, so they are not independent of the caller design.

## 2. Independent sets

The caller family, grid and defaults were fixed on Howitt et al. 2023 (Zenodo 8304003) only. `bench/adt/hto_lodo.py` states that
Stoeckius 2018 "awaits genotype truth, so it is not used for selection". No other data entered the design.

* **I1 (primary): Stoeckius et al. 2018, GEO GSE108313, PBMC from 8 donors, one hashtag per donor, super-loaded, 10x 3' v2**
  (GSM2895282 RNA, GSM2895283 HTO; ENA SRR8281306 / SRR8281307; public GEO data, cite the paper).
  * Cells: the barcodes called by Singlet's own pipeline (`singlet-fixed-det --pipeline`, `cell_calls.tsv`) on the first 200 M
    read pairs of the RNA library. Hashtag counts: the authors' deposited table (`Hashtag-HTO-count.csv.gz`), restricted to those
    barcodes; a called cell absent from the table has all-zero counts. (If the deposited table cannot be fetched, counts are
    made from SRR8281307 with the Cell Ranger feature-barcoding rule in `python/singlet/adt/count.py`; the choice is recorded.)
  * Truth: genotype. cellSNP-lite (1000 Genomes AF >= 5% panel, `--minMAF 0.1 --minCOUNT 20`) on the Singlet tagged BAM, then
    vireo `-N 8 --randSeed 42`: donor0..donor7 / doublet / unassigned. Genotype demultiplexing sees RNA reads only, never the
    hashtags. Cells vireo calls `unassigned` (low SNP coverage, not evidence of an empty droplet) are excluded from scoring and
    their number is reported. Vireo is the truth tool of the Howitt benchmark as well.
  * Hashtag to donor map: the one-to-one assignment (Hungarian on the contingency table of top tag against vireo donor over
    vireo singlets), the same for every caller, as in Howitt et al.
  * Known bias: HTODemux was developed on this dataset. It is expected to be favoured; that is part of the comparison, not
    removed.
* **I2 (secondary): Stoeckius 2018 mixed cell lines, GSM3501446 (RNA) / GSM3501447 (HTO).** HEK cells carry three hashtags (HEK_A,
  HEK_B, HEK_C), THP1 cells one (THP1_A); K562 and KG1 cells were added unlabelled. Truth from RNA: the cell line of each cell
  (marker-gene scores, fixed in `eval_independent.py` before the run), so unlabelled K562 / KG1 cells are truth negatives, THP1 cells
  truth THP1_A singlets, HEK cells singlets of any of the three HEK tags (a HEK singlet call with a non-HEK tag is wrong, one with
  a wrong HEK tag is not detectable), and cross-line RNA doublets (a cell with a clear mixture of two lines) truth doublets;
  within-HEK doublets are invisible. I2 therefore tests negatives and cross-line doublets, not all errors. Reported, not gating.

## 3. Metrics

Cells are classified singlet (S) / doublet (D) / negative (N). **Primary: macro F1 = mean over the classes present in the
truth of F = tp / (tp + (fp + fn) / 2)**; a singlet call counts as correct only if its tag maps to the cell's truth donor / line
(a singlet call on the wrong donor is a false positive and a false negative of S). On I1 there are no truth negatives (vireo
`unassigned` are excluded), so the primary metric is the mean of F_S and F_D; on I2 all three. **Secondary: genetic doublets
called singlet** = share of truth doublets called singlet (lower is better). The Howitt-style mean per-hashtag F is reported for
comparison with the gate, not for the pass rule.

Intervals: 95% percentile bootstrap over cells (1000 resamples); differences between callers use paired resamples.

**Pass rule (I1):** Singlet's primary metric >= (best competitor's primary metric) - 0.01, where the best competitor is the
highest-scoring of HTODemux, BFF-cluster, BFF-raw, hashedDrops (defaults) and the GMM baseline *as implemented here*. The
secondary rate is reported with its CI and judged only descriptively. I2 is judged by the same rule but does not gate.

## 4. Callers, frozen before the independent set is touched

| caller | what | settings |
|---|---|---|
| `current` | `HtoDemux` at repo default (numpy reference `bench/adt/hto_lodo.py::call_hashtags`, which the header reproduces call for call) | confident_min 0.5, doublet_nmads 3, doublet_min 1, pseudo 5, mixture_positive 0.5, min_cells 20 |
| `bffc` / `bff` | candidate from section 6, only if it passes its rule | fixed in `frozen.json` |
| `hasheddrops` | DropletUtils fold-change rule, no veto | confident.min 2, doublet.min 2, nmads 3, pseudo 5 |
| `gmm` | per-tag 2-component Gaussian mixture on log1p, posterior >= 0.5 positive; 0 / 1 / >= 2 positives = N / S / D | as stated |
| `htodemux` | Stoeckius 2018: CLR over cells, k-means k = tags + 1, background cluster = lowest mean, NB fit after dropping the top 0.5%, 0.99 quantile threshold; 0 / 1 / >= 2 positives | as stated |
| `bff_raw` | Boggy 2022: log10(count + 1 + U(0,1)), SJ-bandwidth KDE widened by 10% steps until the first derivative has < 4 local maxima, threshold = density minimum between the two tallest modes; 0 / 1 / >= 2 positives | seed 0 |
| `bff_cluster` | Boggy 2022: bimodal quantile normalisation, hi and 2nd per droplet, Tn / T2 from densities | alpha = beta = 0.05, Td = 0 |

These are **reimplementations from the published descriptions** (Singlet is MIT; no GPL code was read or copied). They are not
the cellhashR, Seurat or DropletUtils packages themselves, because no R install is available in a job here. Where the BFF paper's
text is ambiguous (the Tn rule when negatives form their own mode; the meaning of "peaks of the first derivative") the choice is
documented in the docstrings of `hto_callers.py`. The BFF reimplementation is validated on Howitt against the authors' own
published BFF_raw / BFF_cluster calls; the agreement is reported next to every result, and a poor agreement lowers the claim
made about "BFF-cluster" accordingly.

## 5. Freeze

After section 6 is complete, `frozen.json` (caller name, every setting, the commit of `hto_callers.py`) is committed. Nothing in a
caller may change after that commit. If a competitor implementation is later found to be buggy, the corrected numbers are added and
the pre-fix numbers stay in the report; Singlet's own caller is not touched.

## 6. Howitt leave-one-dataset-out test of a BFF-style caller

On BAL1, BAL2, BAL3 and the cell lines, captures called separately, scored over the whole dataset:

* fixed-default callers (no tuning): `current`, `hasheddrops`, `gmm`, `htodemux`, `bff_raw`, `bff_cluster(0.05, 0.05)`;
* family `bffc`: `bff_cluster` over alpha, beta in {0.02, 0.05, 0.1, 0.2}^2, chosen by nested LODO (for each held-out dataset the pair
  with the best mean macro F1 over the other three); a hyper-parameter is never chosen on the dataset being scored.

**Decision rule.** The BFF-style caller replaces `current` as Singlet's default (a C++ implementation behind a parameter, `current`
kept) if, over the four held-out scores, its mean macro F1 is higher than `current`'s *and* no dataset loses more than 0.01 *and*
the cell-line or BAL3 score improves by >= 0.02. Otherwise `current` remains the frozen Singlet caller and the BFF numbers are
reported as the frontier. A hybrid designed after seeing these numbers is allowed only if its hyper-parameters are chosen by nested
LODO and it is listed in `frozen.json` before the independent set is scored; it is then labelled exploratory.

## 7. Single run

The frozen choice and every competitor are run once on I1 and once on I2 by `eval_independent.py`; the output document is committed
unedited (`results/independent.json`). Nothing is re-run after looking at it; a failure to run is fixed and the fix reported.

## 8. Amendment 1 (2026-10-31, before any caller was run on I1 or I2 and before the LODO scores were read)

Found while fetching the data and checking the BFF reimplementation; none of it depends on a hashtag-caller result on I1 / I2.

1. **I2 has 12 tags, not 4**: the deposited table (GSM3501447) has HEK_A/B/C, THP1_A/B/C, K562_A/B/C, KG1_A/B/C (three tags per
   line) plus the columns no_match, ambiguous, total_reads, bad_struct (not tags). There are no unstained lines, so I2 has no truth
   negatives and its primary metric is the mean of F_S and F_D. A singlet call is correct if the tag's line equals the cell's RNA
   line. Cells are the authors' deposited RNA cells (>= 500 UMIs), not Singlet's cell caller (no reads are used for I2).
   RNA truth (`eval_independent.py::rna_truth`, fixed): log1p CP10k, top 2,000 genes by variance / mean, 20 SVD components,
   k-means (k = 4, 20 starts, seed 0); clusters named by marker genes (listed in the script) with a one-to-one assignment;
   cross-line doublets = cells whose 30 nearest neighbours (real cells plus as many simulated cross-cluster sums) are mostly
   simulated (> 50%). Within-line doublets are invisible to this truth. The concordance of the RNA lines with the tag families is
   reported as a check and is not an input.
2. **The authors' HTO table is available for I1** (GSM2895283, fetched by Range requests), so counts come from it, as planned.
3. **BFF reimplementation fidelity** (`fidelity.py`, `results/fidelity.json`): BFF_raw agrees with the published calls on 86-95% of
   cells; no reading of the BFF_cluster text (the Tn and T2 definitions are ambiguous) reproduces the published doublet counts, so
   `bff_cluster` here is a BFF-style approximation. Default reading = best mean class agreement with the published calls (no truth
   labels used): Tn `edge`, T2 `top`. The family in section 6 gains the T2 reading as a third axis: `t2_rule` in {edge_M, top}.
   `bff_cluster` and `bff_raw` on I1 / I2 are therefore **proxies**: the claim "within 0.01 of the best competitor" is relative to
   the callers as implemented here, and the cellhashR output itself was not run.

## 9. Amendment 2 (2026-10-31, after the I2 truth-only dry run, before any caller was run on I1 or I2)

The truth-only run (`eval_i2_truth.sbatch`, no caller involved) showed that the marker-gene naming of the four RNA clusters swapped
THP1 and KG1 (the clusters named THP1 / KG1 carry the KG1 / THP1 tags in 99% of their cells; HEK and K562 were right). The
marker lists cannot separate those two myeloid lines. Fix, fixed before any caller runs: the RNA clusters (and the RNA doublet
set) stay exactly as computed, and each cluster is **named** by the one-to-one assignment (Hungarian) to the tag family that
holds most of its non-doublet cells, the same caller-free label matching used for tag to donor on I1. The marker naming is kept in
the output as a QC field. The partition, the doublets and every disagreement between RNA cluster and tag remain counted as
errors; only a swap of names is repaired. I2 is secondary and non-gating.
