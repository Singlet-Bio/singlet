# V(D)J held-out benchmark: results (pre-registration: PREREG.md)

Truth = published Cell Ranger `filtered_contig_annotations.csv` (10x's for ctrl10k, the authors' 3.0.1 tables for GSE145370). Competitor
TRUST4 1.1.5, same FASTQs. No 10x software, no MiXCR. Primary metric = clonotype pairing accuracy (PREREG.md); 95% CIs by 2,000 bootstrap
resamples of reference-paired cells (paired for the difference). Raw files: `results/*.json`, `results/secondary/*.json`.

**Verdict: the pre-registered pass rule FAILS** on the held-out libraries (pooled TCR and BCR below TRUST4 by more than 0.01), both before and
after the fix. The wrong-lineage criterion passes (+0.0008 vs Cell Ranger, limit 0.005).

## Primary metric on the held-out libraries (Singlet vs TRUST4)

Before = frozen code and reference (runs of 2026-10-26); after = IMGT Cys104 anchor reference (Addendum 1), re-run once.

| group | cells | Singlet before | Singlet after | TRUST4 | diff after [95% CI] | rule (>= -0.01) |
|---|---|---|---|---|---|---|
| BCR (ctrl10k_b) | 436 | 0.934 [0.908,0.956] | 0.959 [0.938,0.975] | 0.986 [0.975,0.995] | -0.028 [-0.046,-0.011] | fail |
| TCR pooled (5 libs) | 17,597 | 0.786 [0.780,0.792] | 0.786 [0.779,0.791] | 0.825 [0.819,0.830] | -0.039 [-0.047,-0.032] | fail |
| all pooled | 18,033 | 0.790 [0.784,0.795] | 0.790 [0.784,0.795] | 0.829 [0.823,0.834] | -0.039 [-0.046,-0.032] | fail |

Per library (after; before differs only for ctrl10k_b 0.934 and ctrl10k_t 0.937):

| library | cells | Singlet | TRUST4 | diff |
|---|---|---|---|---|
| ctrl10k_t (TCR) | 3,992 | 0.936 [0.928,0.943] | 0.938 [0.930,0.945] | -0.002 |
| ctrl10k_b (BCR) | 436 | 0.959 [0.938,0.975] | 0.986 [0.975,0.995] | -0.028 |
| gse145370_S135A | 1,035 | 0.871 [0.849,0.891] | 0.876 [0.855,0.897] | -0.006 |
| gse145370_S149A | 4,189 | 0.742 [0.729,0.756] | 0.845 [0.834,0.856] | **-0.103** |
| gse145370_S150A | 7,381 | 0.698 [0.688,0.709] | 0.735 [0.725,0.745] | **-0.037** |
| gse145370_S133B | 1,000 | 0.925 [0.908,0.941] | 0.899 [0.880,0.916] | +0.026 |

The TCR failure is two libraries: S149A and S150A (11,570 of 17,597 TCR cells). Without them the other three TCR libraries pool to 0.923 (Singlet) and 0.921 (TRUST4). Their
secondary metrics show the mechanism only at the level of symptoms: cell-chain recovery 0.88 vs 1.00 for TRUST4 (Singlet misses a chain of
~12% of Cell Ranger's cells), contig recovery 0.86 vs 0.93-0.95, CDR3 accuracy higher than TRUST4 (0.95-0.97 vs 0.84-0.91). The cause was not
investigated on held-out reads (that would be tuning on held-out data); the development libraries do not show it (cell-chain recovery 0.97).
Candidates: deeper libraries (79 M and 56 M read pairs, 765 k and 319 k UMIs with 87% / 61% under the adaptive read threshold) and the UMI
support filters. It needs its own development library with that profile.

Wrong lineage (GEX markers; ctrl10k_t + ctrl10k_b contigs pooled): Singlet 8/8,891 = 0.0009 (before 0.0010), Cell Ranger 1/8,378 = 0.0001,
difference +0.0008 (rule: <= 0.005): pass. TRUST4 is 0.012 (T library) and 0.058 (B library).

## Secondary metrics on the held-out libraries (Singlet before -> after | TRUST4)

| library | contig F1 | CDR3 nt acc. | J agreement | V agreement | clonotype ARI |
|---|---|---|---|---|---|
| ctrl10k_t | 0.970 -> 0.970 \| 0.747 | 0.985 -> 0.985 \| 0.968 | 1.000 \| 1.000 | 0.999 \| 0.999 | 0.834 \| 0.042 |
| ctrl10k_b | 0.967 -> **0.979** \| 0.918 | 0.983 -> **0.997** \| 0.996 | 0.954 -> 0.954 \| **0.979** | 0.835 -> 0.837 \| 0.869 | 0.857 \| 0.667 |
| S135A | 0.899 \| 0.598 | 0.972 \| 0.923 | 1.000 \| 0.999 | 0.997 \| 0.996 | 0.556 \| 0.078 |
| S149A | 0.876 \| 0.648 | 0.968 \| 0.907 | 1.000 \| 0.999 | 0.999 \| 0.999 | 0.593 \| 0.039 |
| S150A | 0.883 \| 0.637 | 0.953 -> 0.957 \| 0.836 | 1.000 \| 0.999 | 0.999 \| 0.999 | 0.511 \| 0.013 |
| S133B | 0.925 \| 0.557 | 0.979 -> 0.981 \| 0.939 | 1.000 \| 0.999 | 0.999 \| 0.998 | 0.808 \| 0.025 |

Singlet beats TRUST4 on contig F1, precision, CDR3 accuracy (except ctrl10k_b: 0.997 vs 0.996, a tie) and clonotype ARI on all six held-out
libraries; TRUST4 wins cell-chain recovery everywhere (1.00) and BCR J agreement (0.979 vs 0.954) and BCR V agreement.
Wall time and peak memory (Singlet 8 threads, original runs | TRUST4 8 threads): ctrl10k_t 17.7 min, 38 GB | 2 h 04 min, 74 GB; ctrl10k_b 1.3 min,
3.8 GB | 10.7 min, 7.6 GB; GSE145370 libraries 8-19 min, 14-33 GB | 1.2-3.1 h, 39-83 GB. (The "after" runs used 16 threads, 6-15 min.)

## The fix and its validation (development libraries only)

Cause found on pbmc10k_b: V genes that end in two adjacent Cys codons (IGLV2-23, IGLV2-11, IGLV2-5 ...) were anchored at the last Cys; Cell
Ranger >= 5 starts the junction at the first (IMGT Cys104), 25 of 402 IGL top contigs differed by exactly one `TGC`. The missing IGHV1-8 and
IGHV4-30-4 are not the main cause: of 28 pbmc10k_b IGHV1-8 cells 26 are already right via the nearest paralog, so no genes were added.
`reference._first_cys` / `reanchor_v` moved 10 anchors (IGKV1D-27, IGKV2-14, IGKV2-4, IGKV2D-14, IGLV2-11, IGLV2-23, IGLV2-5,
IGLVVI-22-1, IGLVVI-25-1, TRAV28); the reference is `python/singlet/vdj/data/vdj_human_imgtcys.tsv.gz` (not made the default).

| development library (truth made by) | metric | before | fix |
|---|---|---|---|
| pbmc10k_b (CR 5.0) | pairing / CDR3 nt / J | 0.947 / 0.985 / 0.972 | **0.970 / 0.998** / 0.972 (TRUST4 0.974 / 0.995 / 0.985) |
| pbmc2_b (CR 3.0, legacy convention) | pairing / CDR3 nt / J | 0.930 / 0.974 / 0.982 | 0.902 / 0.956 / 0.982 (the expected loss: that truth starts at the 2nd Cys) |
| pbmc2_t, pbmc10k_t (TCR; control rerun at 16 threads: 0.931, 0.940) | pairing | 0.931, 0.940 | 0.929, 0.939 (TRAV28; within the 0.002 limit) |
| wrong lineage, dev pooled | rate | 0.0025 | 0.0025 |

Cell Ranger changed its CDR3-start convention between versions, so no single reference setting reproduces both; this is Addendum 1 of the
pre-registration (written before the held-out re-run), and the fix would fail the rule as first written because of pbmc2_b.
`j_slack` (the J read vote vs junction end): 1 gave the same pairing and J agreement 0.9786/0.9715 vs 0.982/0.9715, 0 lost J agreement
(0.9545/0.9549); default 2 kept, no change.

## Not done / caveats

* Held-out libraries were seen twice (frozen-code "before", fixed "after"); the after numbers are not blind. The ctrl10k_b Singlet block was seen
  before pre-registration (PREREG.md disclosure).
* Only one BCR held-out library (436 paired cells): BCR conclusions rest on it. GSE145370 has TCR only and no GEX in the harness, so wrong lineage is
  only measured on ctrl10k.
* The primary metric breaks UMI ties stably; `compare_vdj.correct_pairing` does not, so TRUST4's value differs by up to 0.03 in some libraries
  (S133B 0.899 vs 0.869 in the old metrics).
* The S149A/S150A TCR deficit and the BCR J / V agreement gap (0.954 vs 0.979, 0.837 vs 0.869; missing IGHV genes) are not fixed.
* The old held-out TRUST4 outputs from 2026-10-26 were reused (not re-run); fixed Singlet runs use 16 threads.
