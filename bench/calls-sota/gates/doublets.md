# Gate: expression doublets (`doublet/score_v2`)

Fixed 2026-11-11 before any held-out scoring (D-138, D-102, D-109). Scored object: the package scorer `python/singlet/features/doublet.py`
at origin/main b7130d38 (stacker `doublet_v2_stacker.npz`, content hash 54bd13ab...), run from stored counts after the freeze. Best open
competitor: scDblFinder (R, GPL-3, run as an installed package with defaults, never copied). Scrublet and the engine score alone are
reported for context.

## Truth

Real experimental doublets only: genotype (demuxlet), hashtag, MULTI-seq and species. A simulated doublet is never an evaluation label.
Same-type doublets are invisible to every expression method and sit in the truth as undetectable; the effect is shared by all methods.

## Development set

The 17 real-label datasets under `$LQC/dbl/data` (Kang 2018 lanes, Stoeckius 2018, mouse kidney hashing, McGinnis MULTI-seq, 10x hgmm 6k and
12k). They are scored only by leave-one-study-out (LOSO) refits of the stacker: a stacker scoring a study must have been trained without every
dataset of that study. The packaged stacker was trained on all 17, so its scores on them are in-sample and are not evidence. The LOSO result
exists (`$W/doublet_v2/loso_pkg/auprc.json`) and is restated, not rescored.

## Leakage audit (what is checked, and the answer recorded in results.md)

1. Truth enters no feature. `compute_inputs` takes counts, gene names, species and the engine score. Checked by reading it.
2. Training set versus test sets: every held-out library below is absent from the 17 training datasets and from every earlier doublet
   script and result (git grep on origin/main of Singlet-Bio/singlet, run 2026-11-11).
3. The `prod` input is the engine `detect_doublets` score. The v1.3.7 header `include/singlet/pileup/doublet_detect.h` is byte-identical to the
   pinned commit the training scores came from (`git diff` empty). The held-out `prod` is computed with that header (`dbl_run`).
4. The atlas encoder was distilled from the atlas cell-type network. The earlier nearest-neighbor check found no near-duplicate between the
   17 datasets and the atlas (max cosine 0.84, none at or above 0.98). It is not rerun for the held-out libraries (the atlas is built from GEO
   deposits, the held-out libraries are 10x demonstration datasets); recorded as a limit.
5. Selection effect: the feature set and learner were chosen among eleven variants on LOSO over the same 17 datasets. This inflates the
   development number slightly and is the reason the held-out set carries the verdict.
6. scDblFinder receives counts only, default arguments, one fixed seed, no truth.

## Held-out libraries (scored once, never used for a threshold, rule or choice)

| Library | Truth | Doublet rule (fixed here) |
|---|---|---|
| 10x CellPlex PBMC 10k, 3' v3.1 (raw matrix on Clipper from the cell-calling gate; cells = barcodes with `called_v2` = 1 in that gate) | hashtag | CMO UMIs summed over the four tags s; s below 100: dropped. Second-largest tag fraction f2: doublet if f2 >= 0.25; singlet if f2 <= 0.10; between: dropped |
| 10x hgmm 5k v3 (STARsolo filtered matrix on Clipper from the cell-calling gate) | species | Minor species share of gene UMIs >= 0.20: doublet; share <= 0.05: singlet; between: dropped |

Both libraries are new for doublets (hgmm5k v3 and CellPlex were used for cell calling only). The hgmm 1k v2/v3 libraries stay development
(the earlier v2 gate used them).

## Metric and bar

Metric: precision-recall area (AUPRC) per library, paired difference ours minus scDblFinder, with a 2,000-draw bootstrap over cells.

* Parity (needed for every verdict above withheld): LOSO mean difference on the 17 development datasets at least -0.01 (recorded: -0.005), and
  on the held-out libraries the mean difference at least -0.01 with no single library below -0.02.
* Enforcement (default filter at download) needs parity plus non-inferiority with an interval, as the design states: lower 95% bound of the mean
  difference above -0.01, pooled over the 17 LOSO datasets and the held-out libraries, clustered by study.
* Runtime: all expression-doublet work at most 0.5 core-minutes per ten thousand cells (the shared ceiling for all calls is 2).

Verdicts: enforce = parity, interval and runtime met. Metadata only, hidden by default = parity and runtime met, interval not met. Withheld =
parity not met on the development set or on the held-out libraries.

Genotype doublets (donor demultiplexing) stay authoritative wherever a sample holds two or more donors; this scorer is not applied there.
