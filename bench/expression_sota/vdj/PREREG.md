# V(D)J held-out benchmark: pre-registration

Branch `bench/vdj-heldout`, written 2026-10-31 before any held-out scoring or re-run in this session.

## Disclosure (read first)

* The held-out set below was designated by the previous session (`docs/research-beta/vdj-crispr.md`, "Decisions") when the
  assembler was frozen at 094da510, and the libraries were not used for development or tuning of anything.
* Frozen-code Singlet and TRUST4 outputs for these libraries already exist on Clipper
  (`$WORK/state/vdjc/runs/<lib>/{singlet,trust4}`), produced on 2026-10-26. Before writing this file the author of this
  session saw the Singlet block of `ctrl10k_b/metrics.json` (contig F1 0.967, CDR3 nt 0.983, correct pairing 0.933, J 0.954,
  wrong-lineage 0.0036 vs Cell Ranger 0.0012) and part of its TRUST4 block, nothing else held out. The primary metric below is new
  and has not been computed on any library. The frozen outputs are scored as the "before" result; any fix is validated on development
  libraries only and the held-out libraries are then re-run once. Because the held-out libraries are then seen twice (before and
  after), the "after" numbers are a re-run on seen-but-never-tuned-on libraries, not a blind test.

## Libraries

Development (may be used to look at errors, tune, validate fixes): pbmc2_t / pbmc2_b (10x PBMC 5' v1, `pbmc5p_v1`),
pbmc10k_t / pbmc10k_b (10x PBMC 10k 5' v2, `pbmc10k_5p_v2`), gse145370_S133A, gse145370_S158A.

Held-out (never used for development; truth = the published Cell Ranger `filtered_contig_annotations.csv` of each library; no 10x
software is run):

| id | content | truth |
|---|---|---|
| ctrl10k_t | 10x 10k PBMC 5' v2 Chromium Controller, TCR library (Cell Ranger 6.1.0) | 10x published filtered_contig_annotations |
| ctrl10k_b | same sample, BCR library | 10x published filtered_contig_annotations |
| gse145370_S135A, S149A, S150A, S133B | GSE145370 (ESCC, PRJNA606979) 10x 5' TCR libraries | authors' Cell Ranger 3.0.1 filtered_contig_annotations (GEO supplementary) |

Only ctrl10k_b provides BCR. Competitor: TRUST4 1.1.5 on the same FASTQs and threads. No MiXCR, no 10x software.

## Primary metric: clonotype pairing accuracy

Reference cells are Cell Ranger `is_cell` barcodes that have a reference pair: TRA+TRB, else TRG+TRD, else IGH+IGK, else IGH+IGL,
taking for each chain the productive, high-confidence contig with the most UMIs. A cell counts as correct when the method,
restricted to called GEX barcodes (the same filter for every method), has a productive top-UMI contig for both chains of that
pair and both junction nucleotide sequences (`cdr3_nt`) equal the reference's. Accuracy = correct / reference-paired cells. This is
the existing `correct_pairing` of `bench/vdj/compare_vdj.py`, computed per cell so that it can be bootstrapped.

Reported: per library; pooled TCR (the five TCR libraries ctrl10k_t, S135A, S149A, S150A, S133B); BCR (ctrl10k_b); all pooled. 95% CIs
by 2,000 bootstrap resamples of reference-paired cells within library (seed 0); the Singlet minus TRUST4 difference uses the same
resamples (paired).

**Pass rule.** Singlet passes if, on the pooled TCR, the BCR, and the all-pooled estimate, accuracy(Singlet) - accuracy(TRUST4) >=
-0.01 (point estimates; CIs are reported, and a CI entirely below -0.01 is called a clear fail), AND the pooled wrong-lineage rate of
Singlet minus that of Cell Ranger's published contigs is <= 0.005. Wrong lineage is `wrong_lineage_*` of `compare_vdj.py` (TCR contigs
on GEX-B barcodes and BCR contigs on GEX-T barcodes, from GEX marker labels, independent of any caller). It exists only for ctrl10k_t and
ctrl10k_b (no GEX in the harness for GSE145370); pooled by summing contigs.

## Secondary metrics

Those of `bench/public/capabilities/vdj.py` (contig F1, recovery, CDR3 nt/aa accuracy, V/J/C agreement, pairing rate, cells
Jaccard, clonotype ARI, wall time and memory), per held-out library, Singlet vs TRUST4, from `compare_vdj.py`.

## Fix protocol

Only the segment reference (adding IGHV genes missing from GENCODE/RefSeq on GRCh38, from the alt loci already used; no IMGT) or the
contig filters may change. Validate on the development libraries: a change is kept only if BCR correct pairing, CDR3 nt accuracy and J
agreement on pbmc10k_b and pbmc2_b do not get worse, no development TCR metric drops by more than 0.002, and wrong-lineage does not
increase. The held-out libraries are then re-run once with the final settings and both before and after are reported, including when
there is no gain. Nothing is tuned on held-out data.
