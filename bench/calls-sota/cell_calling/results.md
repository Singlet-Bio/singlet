# Cell calling: gate results (2026-11-11)

Gate: `bench/calls-sota/gates/cell_calling.md`, committed before any held-out scoring (commit 028f9f9f), with Amendment A (nuclei truth)
committed before the nuclei library was scored. Engine frozen at v1.3.7. The Singlet call is `python/singlet/s1/cellcall.py` (called_v2,
version 2.1, file sha256 51d9dce4...) on stored or deposited counts. Raw outputs are in `results/`. Scripts are in this directory; every analysis
ran as a Slurm job on short-cpu, no GPU.

## Development libraries (class G genotype, class S species)

Truth tables are the stored `state/cellcall_truth` tables. Most barcodes there have unknown truth, so the scored quantity is the expected
precision, recall and F1 under the per-barcode posterior of the registered EM (`jobs/cellcall_truth/score.py`, recomputed by
`score_posterior.py`). Comparators are the stored STARsolo EmptyDrops_CR emulation and the stored CellBender 0.3.2 output. CellBender
outputs exist only for the b2 control lane and the hgmm libraries; the three Kang b1 lane outputs are not on disk (parameter and log files only).
Scoring only decided barcodes (`score_tables.py`, `results/dev_scores_decided_only.json`) is degenerate on the Kang lanes, because every tool calls
the same decided set, so it is not used for the verdict.

| Library | Class | Singlet F1 | EmptyDrops_CR F1 | CellBender F1 | Singlet minus best | Matched recall, Singlet minus best | Within 0.005 |
|---|---|---|---|---|---|---|---|
| Kang b1 A | G | 0.9892 | 0.9264 | none | +0.063 | +0.003 | yes |
| Kang b1 B | G | 0.9873 | 0.9482 | none | +0.039 | +0.000 | yes |
| Kang b1 C | G | 0.9832 | 0.9437 | none | +0.040 | -0.000 | yes |
| Kang b2 control | G | 0.9910 | 0.9542 | 0.9847 | +0.006 | +0.005 | yes |
| hgmm1k v2 | S | 0.7747 | 0.7675 | 0.7795 | -0.0048 | +0.000 | yes |
| hgmm1k v3 | S | 0.7198 | 0.7326 | 0.7374 | -0.0176 | +0.000 | no |
| hgmm5k v3 | S | 0.5240 | 0.5449 | 0.5345 | -0.0209 | +0.000 | no |

Matched count: N = expected number of cells, each tool contributes its called barcodes first and then barcodes by UMI. Singlet minus each tool is
between -0.0003 and +0.0048 on every library (intervals in `results/dev_scores_posterior.json`).

Dev gate against the fixed rule: 5 of 7 libraries are within 0.005 of the best tool on F1 (the bar needs 60%), and matched recall passes on all 7.
Clause (b), no library worse than the best tool by more than 0.02, fails on F1 for hgmm5k v3 by 0.0009 (EmptyDrops_CR 0.5449 against Singlet
0.5240; paired 95% interval of the difference -0.0236 to -0.0184). The species libraries carry weak truth (the EM puts the expected cell
count of hgmm5k v3 at about 14,300 barcodes while every tool calls about 5,000, so all recalls sit near 0.36). The decision rule was fixed
before scoring, so the formal dev result is a fail on that one cell.

Tool-agreement block (class T, not ground truth, restated from `bench/cellcall_gap/matched_count_v1.3.4.json`, not rescored): matched-count recall
of Singlet minus the best other tool is between -0.0010 and +0.040 on the eight gate libraries; the two negative values include zero.

## Held-out libraries (scored once, at defaults)

Comparator on held-out libraries: `DropletUtils::emptyDrops` 1.26.0 (R 4.4.3, `lower = 100`, 10,000 iterations, knee retain, FDR 0.001), called as an
external program. CellBender was not run (no GPU), so these rows are judged against EmptyDrops only. The input is the deposited all-barcode matrix
with every barcode of at least one UMI, which differs from the campaign's stored universe (barcodes with at least 5 reads, so no singletons); see the
diagnostic below.

| Library | Truth | Singlet calls | EmptyDrops calls | Singlet F1 | EmptyDrops F1 | Singlet minus EmptyDrops (95% interval) | Matched recall |
|---|---|---|---|---|---|---|---|
| 10x CellPlex PBMC 10k (hashed cells) | H, 15,982 tag-confirmed cells among 16,057 decided barcodes | 14,273 | 13,164 | 0.9391 (P 0.9977, R 0.8869) | 0.9023 (P 0.9995, R 0.8224) | +0.0368 [+0.0346, +0.0390] | 0.9955 both (N exceeds the calls, so UMI order decides) |
| McGinnis nuclei MULTI-seq, GSE129578 POC nuclei (hashed nuclei) | H, 5,779 tag-confirmed nuclei, recall only (Amendment A) | 118,666 | 14,639 | not defined | not defined | recall at own call +0.0235 [+0.0202, +0.0276] | 0.4925 both |
| scHi-HOST pooled LCLs (genotyped pool) | G | not scored | not scored | | | | blocked, see below |

The CellPlex library passes. The nuclei library passes the recall-only rule it was gated on, but the result is not credible as a cell call: Singlet
calls 118,666 barcodes against 14,639 for EmptyDrops and 10,415 retained by the authors, and 91% of its calls lie outside the authors' set (class T
information). The recall of 1.000 is bought by calling everything. The cause is visible in the fitted parameters: with singleton barcodes in the
universe the plateau rule fits an ambient bound of 5 UMIs and a knee at 54 UMIs, and every barcode above the knee is retained untested.

## Post-hoc diagnostic: barcode universe (not part of the verdict)

Run after the nuclei failure was seen, on the unchanged call (`universe_sensitivity.py`, `results/universe_sensitivity.json`). The campaign keeps only
barcodes with at least 5 reads, which removes most singletons, so the universe was cut at total UMIs of 1, 2, 3 and 5.

| Library | Universe cut | Calls | Ambient bound | Knee | Result |
|---|---|---|---|---|---|
| CellPlex | 1 / 2 / 3 / 5 UMIs | 14,273 / 14,304 / 13,307 / 13,307 | 5 / 7 / 85 / 85 | 4,695 | F1 0.9391 / 0.9402 / 0.9075 / 0.9075 (EmptyDrops 0.9023) |
| Nuclei | 1 / 2 / 3 / 5 UMIs | 118,666 / 118,663 / 14,770 / 14,750 | 5 / 7 / 100 / 100 | 54 / 54 / 1,084 / 1,084 | recall of tag-confirmed 1.000 / 1.000 / 0.992 / 0.993; calls outside the authors' set 91% / 91% / 30% / 30% |

At a universe of 3 or more UMIs the nuclei call count falls to the EmptyDrops level, and CellPlex stays above EmptyDrops (0.9075 against 0.9023, a
margin of 0.005). The call therefore depends on how many singleton barcodes the input holds. This does not mean the stored campaign matrices behave
like the 3-UMI cut; it means the result on the actual stored universe has to be checked on a campaign nuclei sample.

## Runtime

Single-threaded, CellPlex raw matrix (1.05 million barcodes, 14,273 calls): 4.6 s for the call (8.4 s with numba compilation in the first call), which is
0.053 core-minutes per 10,000 called cells, 3% of the 2 core-minute ceiling. Nuclei (587,606 barcodes, 118,666 calls): 0.9 s. The R comparator took
about 92 to 104 s on the same matrices and is not part of the ceiling.

## Not done

* Genotyped held-out pool: scHi-HOST (GSE205796) has reads only on Clipper and deposited filtered matrices (about 10,000 to 15,000 barcodes per
  library, no empties). Stored all-barcode counts and genotype truth need the registered engine run on the staged reads plus the genotype pipeline
  against the staged 1000 Genomes genotypes (about 10 CPU-hours or more per library).
* CellBender on held-out libraries (GPU) and on the Kang b1 lanes (outputs absent).
* Cell Ranger filtered lists for CellPlex (the 10x filtered files return 403).
* Only the main branch of Singlet-Bio/singlet was searched for earlier use of the held-out libraries; the all-branch search timed out.

## Follow-up (2026-11-11 evening), post-hoc diagnostics on hgmm5k v3

Labelled post hoc; the registered bars are unchanged. `diag_hgmm5k*.py`.

* The 311 barcodes EmptyDrops_CR calls and Singlet does not are mostly real cells (282 species-pure) of 500 to 9,000 UMIs. Singlet tests all of them (p-values present) and most get p near 1.
* Neither the knee retain nor the FDR (0.01 instead of 0.001) explains the gap (F1 0.5240 to 0.5252).
* The ambient bound is the lever. The plateau rule fits 100 UMIs here (its clip maximum). Fixed bounds of 5, 10, 20, 50, 200 give F1 0.5382, 0.5310, 0.5287, 0.5268, 0.5275 against EmptyDrops_CR 0.5449 and CellBender 0.5345. A bound of 5 closes the gap to -0.0067, but it is also the bound that fails on the singleton-rich nuclei input, so a rank-based ambient pool (EmptyDrops_CR takes ranks 45,000 to 90,000, independent of singletons) is the candidate fix. Not yet implemented or scored on held-out data.

### Follow-up 2 (post hoc, labelled; registered bars unchanged)

* **p near 1 is not a bug.** On hgmm5k v3 the simulated null and the DM log-probability agree (`diag_lp.py`). The missed cells have a log-probability above the null (z of +3 to +22): their counts are more clumped than any ambient draw, and the one-sided test only looks for lower probability. EmptyDrops has the same one-sided form.
* **Two-sided candidate** (`cellcall_v23_candidates.py`, `results/dev_twosided.json`): inconsistent. F1 hgmm1k v3 0.8665 (default 0.7194), hgmm1k v2 0.7908, but hgmm5k v3 0.4822 (default 0.5240) with 7,717 calls. Not adopted. The posterior used for scoring depends on caller-agreement groups, so large changes in call sets are not reliably scored.
* **Rank-based ambient pool** (`cellcall_v22_rank.py`, `results/dev_rank.json`): F1 against the best other tool, default then rank pool: Kang four lanes ahead in both; hgmm1k v2 -0.0048 then -0.0084; hgmm1k v3 -0.0180 then -0.0208; hgmm5k v3 -0.0209 then -0.0085. Four of seven within 0.005, and hgmm1k v3 breaches 0.02, so it fails the registered dev gate. Not adopted.
* **Rank pool on held-out** (`results/heldout_rank.json`): invariant to the barcode universe. CellPlex F1 0.9068 (EmptyDrops 0.9023); nuclei 14,889 calls (EmptyDrops 14,639), recall of tag-confirmed nuclei 0.993, 31% of calls outside the authors' set against 91% for the default on the singleton-rich input.
* **Fair nuclei test** (`nuc_fair.py`, `results/nuc_fair_scores.json`; Gaublomme mouse nuclei with hashtags, stored 5-read universe, real hashtag truth, a development library): production called_v2 and python v2.1 F1 0.9594 (precision 0.9595, recall 0.9593); rank pool 0.9599; DropletUtils emptyDrops 0.9587. So on stored-format input the nuclei call is at parity with EmptyDrops. The 118,666-call failure on the deposited McGinnis matrix is an input artifact (singleton barcodes drive the plateau bound to 5 UMIs and the knee to 54), and the remedy is to feed the call only the stored universe (or at least 3 UMIs) in the call layer.
