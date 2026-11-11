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
