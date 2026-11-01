# V(D)J to state of the art: pre-registration of a fresh held-out set (vdj2)

Branch `bench/vdj-sota` (from `prod/campaign` + `bench/vdj-heldout`), written 2026-10-31 before any change of the
assembler's default behaviour in this study (options added so far are off by default) and before anything was run on the
libraries below.

## Goal and rules

Bring Singlet's V(D)J assembler to: clonotype pairing accuracy within 0.01 of TRUST4, and a wrong-lineage rate within 0.005 of
Cell Ranger's published calls, while keeping its contig F1 / clonotype ARI lead over TRUST4 and its speed.

* The six libraries of the first held-out study (`bench/expression_sota/vdj/`: ctrl10k_b, ctrl10k_t, GSE145370 S133B, S135A,
  S149A, S150A) are **burned**: not used for anything in this study.
* Development only on: pbmc2_t / pbmc2_b (10x `vdj_v1_hs_pbmc2`, Cell Ranger 3.0.0), pbmc10k_t / pbmc10k_b (10x
  `sc5p_v2_hs_PBMC_10k`, 5.0.0), and GSE145370 S133A, S158A (used before) plus S149B, S150B, S135B (new here; same series,
  authors' Cell Ranger 3.0.1 tables). S149B and S150B are the second libraries of the donors of the two burned libraries that
  failed (S149A, S150A), chosen because they have the same depth profile (about 100 M and 71 M read pairs); they are development
  data, so their use is not a test.
* The held-out set below is downloaded (FASTQs, truth tables, GEX matrices) before the code is frozen, and only the number of
  Cell Ranger cells and reference-paired cells is computed from the truth tables (the denominators below). Singlet and TRUST4
  run on it **once**, after the code and settings are frozen (a commit named in RESULTS.md), and the result is reported
  whatever it is.
* Truth = each library's published Cell Ranger `filtered_contig_annotations.csv` (productive, high-confidence contigs of
  `is_cell` barcodes). No 10x software is run, no MiXCR.

## Fresh held-out set

None of these libraries has been run through Singlet or TRUST4 in this project (checked: no output under `$WORK/state/vdjc/runs`
or `$WORK/expression-sota`, no job log). Disclosure: the FASTQs of S134A (SRR11094261) and S159A (SRR11094266) were
downloaded on 2026-10-25 by an earlier session under wrong directory names and deleted without being run.

| id | content | chains | truth (Cell Ranger version) | convention | URL / accession | CR cells | reference-paired cells |
|---|---|---|---|---|---|---|---|
| gse145370_S134A | ESCC donor S134, 10x 5' V(D)J | TCR | authors' 3.0.1 | 2nd Cys | GSM4317424, SRR11094261 | 2,643 | 1,828 |
| gse145370_S134B | ESCC donor S134, 2nd library | TCR | authors' 3.0.1 | 2nd Cys | GSM4317431, SRR11094268 | 2,100 | 1,108 |
| gse145370_S159A | ESCC donor S159 | TCR | authors' 3.0.1 | 2nd Cys | GSM4317429, SRR11094266 | 10,536 | 7,443 |
| gse145370_S159B | ESCC donor S159, 2nd library | TCR | authors' 3.0.1 | 2nd Cys | GSM4317436, SRR11094273 | 3,720 | 2,230 |
| pbmc3_t | 10x `vdj_v1_hs_pbmc3` (5' v1, PBMC) | TCR | 10x 3.1.0 | 2nd Cys | cf.10xgenomics.com/samples/cell-vdj/3.1.0/vdj_v1_hs_pbmc3 | 3,925 | 3,356 |
| ngpbmc3_t | 10x `vdj_nextgem_hs_pbmc3` (5' v1.1 Next GEM) | TCR | 10x 3.1.0 | 2nd Cys | .../3.1.0/vdj_nextgem_hs_pbmc3 | 6,648 | 5,615 |
| pbmc1k_t | 10x `sc5p_v2_hs_PBMC_1k` (5' v2) | TCR | 10x 5.0.0 | 1st Cys | .../5.0.0/sc5p_v2_hs_PBMC_1k | 479 | 416 |
| pbmc3_b | 10x `vdj_v1_hs_pbmc3` | BCR | 10x 3.1.0 | 2nd Cys | .../3.1.0/vdj_v1_hs_pbmc3 | 772 | 721 |
| ngpbmc3_b | 10x `vdj_nextgem_hs_pbmc3` | BCR | 10x 3.1.0 | 2nd Cys | .../3.1.0/vdj_nextgem_hs_pbmc3 | 1,309 | 1,172 |
| pbmc1k_b | 10x `sc5p_v2_hs_PBMC_1k` | BCR | 10x 5.0.0 | 1st Cys | .../5.0.0/sc5p_v2_hs_PBMC_1k | 83 | 81 |
| melanoma_b | 10x `sc5p_v2_hs_melanoma_10k` (5' v2, tumour) | BCR | 10x 4.0.0 | 2nd Cys | .../4.0.0/sc5p_v2_hs_melanoma_10k | 5,644 | 5,293 |

Totals: TCR 21,996 reference-paired cells (7 libraries), BCR 7,267 (4 libraries). Seven TCR libraries (three Cell Ranger versions, three chemistries, tumour and blood) and four BCR libraries (three versions,
including a tumour with plasma cells). The 10x libraries have the GEX matrix of the same cells (`filtered_feature_bc_matrix.h5`),
used for the called-cell filter and the wrong-lineage labels; GSE145370 has none in the harness.

## Cell Ranger version convention (rule 3)

For V genes that end in two adjacent Cys codons (IGLV2-11, IGLV2-23, IGKV2-14, TRAV28 ...), Cell Ranger >= 5 starts the
junction at the first Cys (IMGT Cys104) and Cell Ranger 3.x / 4.x at the second (found on development data, first study,
Addendum 1). This is a reference-version setting matched to the version that produced each truth table, decided by the
version alone (the 10x directory name, or the authors' stated 3.0.1) before any run: `cr5` libraries (>= 5.0) use
`python/singlet/vdj/data/vdj_human_imgtcys.tsv.gz`, `cr3` libraries (3.x, 4.x) use `vdj_human.tsv.gz`. Each library is scored
against its own truth with its own setting. The same rule applies to the development libraries (pbmc10k: cr5; pbmc2 and
GSE145370: cr3). TRUST4 is run with its own IMGT reference unchanged.

## Primary metric: clonotype pairing accuracy

As in the first study (`../vdj/PREREG.md`, `../vdj/primary.py`, used unchanged): reference cells are Cell Ranger `is_cell`
barcodes with a reference pair (TRA+TRB, else TRG+TRD, else IGH+IGK, else IGH+IGL; per chain the productive high-confidence
contig with most UMIs, ties broken by file order); a cell is correct when the method, restricted to called GEX barcodes where the
library has a GEX matrix (the same filter for every method), has a productive top-UMI contig for both chains of the pair with
the reference's `cdr3_nt`. Reported per library, pooled TCR, pooled BCR and all pooled, with 95% CIs from 2,000 bootstrap
resamples of reference-paired cells within each library (seed 0), and the paired Singlet - TRUST4 difference.

## Pass rules (held-out set, single run)

1. Pairing: accuracy(Singlet) - accuracy(TRUST4) >= -0.01 (point estimate) on each of pooled TCR, pooled BCR and all pooled.
   A 95% CI of the difference entirely below -0.01 is called a clear fail.
2. Wrong lineage: pooled over the libraries with GEX (pbmc3_t/b, ngpbmc3_t/b, pbmc1k_t/b, melanoma_b; contigs summed),
   rate(Singlet) - rate(Cell Ranger published contigs) <= 0.005. Wrong lineage = TCR contigs on GEX-B barcodes and BCR contigs
   on GEX-T barcodes (`bench/vdj/gex_lineage.py` markers), as in the first study.
3. Reported as secondary (no pass rule, but the goal "keep the lead"): contig F1 and clonotype ARI of Singlet vs TRUST4 per
   library (`bench/vdj/compare_vdj.py`), V / J agreement, wall time and peak memory at 16 threads for both tools.

Competitors: TRUST4 1.1.5 (`$WORK/envs/vdjcrispr`, `run-trust4 -f hg38_bcrtcr.fa --ref human_IMGT+C.fa`, both mates for
150-nt R1 else R2 only, `--barcodeWhitelist 737K-august-2016.txt -t 16`), Cell Ranger only as the published truth.

## Development protocol and dev pass criterion

Causes are diagnosed on development libraries only (`diag2.py`: for each lost chain, whether no contig was assembled, the
contig was filtered as ambient / low support / below the cell's top / an error variant, or the top was a different junction;
and the V / J disagreements). A change to `python/singlet/vdj/` or its references is kept only if, on the development libraries,
re-run with the same code version for the baseline: pooled pairing accuracy improves, and on no development library does
pairing accuracy, contig F1 or clonotype ARI drop by more than 0.005 or the wrong-lineage rate rise by more than 0.005
(libraries with GEX). The dev results of every tried variant are reported.

At the time of writing, the first development diagnostic (`results/diag/diag2_*_base.json`) points at the cross (ambient)
filter: on S149B 110 of 198 lost chains are the reference junction itself, assembled with a median of 1 UMI and ~2,500 reads
and removed because a sibling cell of an expanded clone holds the same junction with >= 10x the UMIs; 95% of them leave the
chain empty. Candidate changes under test (all off by default until adopted): `ambient_rescue=cell|pair`, and for BCR V
naming `v_open_frac` (an open V gene over a germline-stop paralog: IGHV4-55, IGLV1-41).


## Addendum 1 (2026-10-31, development data only, before any held-out run): the ARI part of the dev criterion

On the six earlier development libraries (the three new GSE145370 ones wait for TRUST4), `ambient_rescue=cell` and `=pair`
raise pooled pairing by +0.009 but fail the dev criterion on pbmc2_t ARI (-0.011); `=pair` also adds BCR wrong lineage
(pbmc2_b +0.0033), so both are dropped. `ambient_rescue=clone` (added after that result: the weak junction is kept only when it
and the droplet's own kept partner chain are a clone's pair seen in another barcode) keeps 141 of the 170 S133A rescues
(pairing +0.028) and 2 of 54 in pbmc2_b (no wrong-lineage change), and still shows pbmc2_t ARI -0.0112, from 4 changed
cells out of 2,982 (`ari_check.py`): two of them become identical to Cell Ranger's call (TRA+TRB where Singlet had TRB only)
and removing either of them alone moves the difference to -0.0135, i.e. ARI falls when cells are corrected (ARI is not
monotone in per-cell agreement; one cell of a large clone moves it by ~0.005). The base ARI's own bootstrap SD over cells is
0.008, larger than the 0.005 tolerance. Deviation declared here: for ARI, a drop larger than 0.005 is accepted when it comes
from cells whose new chain set equals Cell Ranger's (checked with `ari_check.py`, reported). The literal criterion's
verdict (REJECT on pbmc2_t ARI) is reported next to the decision. Pairing, contig F1 and wrong lineage keep the rule as
written.


## Addendum 2 (2026-10-31, development data only, before any held-out run): frozen settings

Scores: `scores/dev_*.md`, from jobs 795043 and 795077, on all 9 development libraries.

**Ambient rescue: `ambient_rescue="clone"` (new default).**
- Pooled pairing goes from 0.9123 to 0.9249 against the base run (8 libraries: S150B's TRUST4 had not finished then).
- With S150B included (9 libraries), pooled pairing is 0.9238 [0.920, 0.928] against TRUST4's 0.9104.
- Literal dev rule: REJECT, because pbmc2_t ARI drops by 0.0112. Accepted under Addendum 1.

**Lone-contig barcodes: `orphan_max_umis` = 0 (rule not adopted).** The rule drops a barcode whose only kept productive
contig has at most 1 or at most 2 UMIs.
- Wrong lineage falls from 47/19,046 to 24/18,982 (at most 1 UMI) or 14/18,885 (at most 2).
- Contig F1 falls by more than 0.005 on GSE145370: S158A by 0.022 / 0.061 and S135B by 0.029 / 0.069, against clone.
  Lone low-UMI chains there are often Cell Ranger contigs.
- Dev rule: REJECT.
- The remaining wrong-lineage rate is 0.0025, against Cell Ranger's 0.0002. That is within the goal's 0.005.

**Germline references: the AIRR-C IG V additions (reference-version change).**
- The genes and their source, versions and licence (CC0 1.0) are recorded in `python/singlet/vdj/data/README.md`.
- No IMGT data is used.

**V-gene vote: `v_open_frac` = 0.3 (new default).** Each run used the AIRR-C references plus clone rescue; the numbers are
Cell Ranger V agreement on pbmc2_b / pbmc10k_b.

| setting | pbmc2_b | pbmc10k_b |
|---|---|---|
| clone, old references | 0.802 | 0.849 |
| AIRR-C only | 0.782 | 0.867 |
| `v_open_frac` 0.3 | 0.802 | 0.868 |
| `v_open_frac` 0.5 | 0.798 | 0.868 |
| `v_open_frac` 0.7 | 0.789 | 0.868 |

- The AIRR-C additions alone lose pbmc2_b V agreement, through IGHV3-30-3 / IGHV3-30 near-identity.
- `v_open_frac` 0.3 restores it while keeping the pbmc10k_b gain.
- Pairing, ARI, J agreement and wrong lineage are unchanged. Contig F1 changes by at most 0.0004.
- TCR libraries are unaffected.
- Literal dev rule: REJECT, because pooled pairing does not improve (+0.0000). Adopted on the V-agreement validation the
  coordinator asked for, with no metric lost beyond the tolerance.

**Frozen settings:**
- `ambient_rescue="clone"`, `v_open_frac=0.3`, `orphan_max_umis=0`; all other parameters unchanged.
- The packaged human references now include the AIRR-C additions.
- The anchor follows the convention of each library's Cell Ranger version (`vdj_human_imgtcys.tsv.gz` for CR >= 5).
- The held-out run uses these defaults, with no `PARAMS`, at the frozen SHA recorded in `RESULTS.md`.
- Held-out pass rule: unchanged, with the Addendum 1 ARI tolerance applied in the same way.
