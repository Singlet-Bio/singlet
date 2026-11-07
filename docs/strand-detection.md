# Strand auto-detection: fix, validation and the campaign re-run list

Follow-up to [pipeline-validation.md §4.6](pipeline-validation.md). Branch `fix/strand-detection`,
built on `fix/molecule-labels` (Singlet-Bio/singlet#22).

## Summary

- **The bug was in both directions.** The atlas engine counted 9,199 of 52,756 GRCh38 campaign
  samples on their minority strand:
  - 5' libraries labelled 3' that were never flipped;
  - **and** sense 3' libraries the old probe flipped to reverse. All 6 Visium samples whose
    BAM was kept are 93% "wrong-strand" in the atlas, yet 97–98% of their reads are sense.
  - `pileup_stats.json`'s `reverse_strand: false` did not show this: the probe flipped only the
    engine's private copy of the config.
- **Fixed:**
  - The probe now samples gene-rich nuclear chromosomes and skips chrM.
  - "Probe ran" and "flip applied" are separate flags.
  - One safety net covers both pileup paths. It is driven by orientation-free nuclear evidence.
  - `encode` now labels 5' 10x libraries correctly.
- **pbmc5p_v1 with auto-detection** equals the forced `--reverse-strand` run. Per-cell gene r vs
  STARsolo goes from **0.04 to 0.99** and marker overlap from **0.02 to 0.96**. The 3' controls
  stay forward.
- **Re-run list:** [8,270 samples](strand-detection/rerun_samples.csv). These are stranded
  libraries counted on their minority strand: 7,739 with wf ≥ 0.7 and 531 with wf 0.6–0.7.
  - The flagged samples are about as stranded as the unflagged ones (median |1 − 2wf| 0.80 vs
    0.89), so these are orientation errors, not antisense-heavy libraries.
  - [929 samples](strand-detection/antisense_heavy_samples.csv) with wf 0.5–0.6 are near-even
    (mostly "other", MARS-seq2, CEL-seq2, 10x v1). They are not a strand error, and a re-run
    gains < 1.5×.
  - [63 chrM-heavy borderline samples](strand-detection/borderline_samples.csv) are listed for
    checking.

## 1. What was wrong

| # | where | bug |
|---|---|---|
| 1 | `run_parallel()` pre-probe | Read the first 20k reads of the busiest chromosome: often chrM, and always one end of one chromosome, so only a few genes. A read was a "hit" if *any* overlapping exon matched, so reads touching genes on both strands (most chrM reads) voted sense. |
| 2 | `strand_flipped_` | Set whenever the probe ran. `strand_was_flipped()` gated the >50% wrong-strand retry in `singlet.cpp`, so the retry never fired. |
| 3 | probe effect | The probe set the engine's `config_.reverse_strand` only. `pileup_stats.json` wrote the caller's `config.reverse_strand`, so a flip was invisible (`reverse_strand: false`, `strand_auto_flipped: true` on every sample). |
| 4 | safety-net metric | `wrong_strand / (wrong_strand + exon_hits)` counts reads on exons of both strands (chrM) as hits. In the synthetic test, 5' data counted forward scores 17% "wrong" by it, and >90% antisense on nuclear reads. |
| 5 | in-stream probe (`run()`) | Also counted chrM reads. It could only flip *to* reverse, so a wrong protocol-derived `reverse_strand` was never cleared. |
| 6 | `encode` | 10x 3' and 5' kits of one generation share R1 length, BC/UMI layout and whitelist. `detect_protocol` picked 3' (the lower id) and enabled 3' poly(A) trimming. |

## 2. The fix

`include/singlet/pileup/pileup_engine.h`, `src/pipeline/singlet.cpp`, `include/singlet/fq/protocol.h`.

- **Orientation-free evidence** (`StrandEvidence`). A primary nuclear read is *sense* if its
  aligned blocks touch exons of genes on one strand only and agree with the forward (10x 3')
  orientation. It is *antisense* if they disagree. Reads touching both strands are not counted.
  Every pass now counts these (`strand_sense`, `strand_antisense` in `PileupStats` and
  `pileup_stats.json`), whatever orientation it counts with.
- **Probe on indexed BAMs** (`run_parallel()`, and `run()` on a BAM file):
  - the 4 nuclear chromosomes with the most mapped reads that carry exons;
  - 16 windows along each, about 20k informative reads in all;
  - MAPQ ≥ 20, primary reads only.

  **Streams** (STAR pipe, unindexed BAM) are probed in-line on the first 10k informative nuclear
  reads.
- **Decision.** The probe flips to reverse above 60% antisense and to forward below 40%. In
  between (unstranded or noisy) it keeps the configured orientation. So it can also clear a
  wrong protocol-derived `reverse_strand`.
  - `strand_was_flipped()` means a flip was applied; `strand_was_probed()` means the probe decided.
- **Safety net** (`rerun_if_wrong_strand()`, both paths):
  - **Trigger:** more than 50% of a finished pass's nuclear informative reads are antisense to
    the strand it counted.
  - **Action:** rebuild the engine with the other orientation, auto-strand off, and run again.
  - **Where it can run:** on the parallel path, and on `--bam-file` (`run()` on a re-readable
    BAM). A STAR pipe cannot be read twice; there the in-line probe decides alone.
  - **When it is off:** when `--reverse-strand` forces the orientation (`auto_strand = false`).
  - **Before the change:** the old engine was destroyed before the retry's references loaded.
    Now it is kept if they fail to load.
- **JSON.**
  - `reverse_strand` is now the orientation actually counted.
  - New fields: `strand_probed`, `strand_probe_informative`, `strand_probe_antisense_frac`,
    `strand_retried`, `strand_sense`, `strand_antisense`, `counted_antisense_frac`.
- **encode.** After a 10x 3' call, the cDNA read decides between 3' and 5'. These signals were
  measured on the benchmark FASTQs (300k reads each, per 1,000 R2 reads):

  | | 3' kit TSO `CAGAGTACATGGG` | poly(A)15 | poly(T)15 |
  |---|---|---|---|
  | PBMC 3' v2 | 41 | 31 | 1.4 |
  | PBMC 3' v3 | 59 | 23 | 2.8 |
  | mouse nuclei 3' v2 | 164 | 32 | 30 |
  | PBMC 5' v1 | **0** | 15 | **32** |

  - **5' requires both:** fewer than 0.2% of reads with the 3'-kit TSO, and more poly(T) than
    poly(A).
  - **Over-sequenced R1:** when R1 is longer than BC+UMI, the 5' TSO tail (`TTTCTTATATGGG`) vs
    poly(dT) after BC+UMI decides alone.
  - **Trimming:** 5' tags do not get 3' poly(A) trimming.
  - I first tested TSO read-through into R2. It is too rare to use: 0.9 per 1,000 reads in the
    5' library.

## 3. Validation

### Unit tests (`tests/cpp/test_strand_detect.cpp`, Clipper)

- **The BAM tests:**
  - **Synthetic BAMs:** chr1/chr2 with 80 single-exon genes on alternating strands. chrM carries
    two overlapping genes on opposite strands and is the busiest chromosome.
  - **Libraries:** 5' (all antisense), 3' (all sense), "noisy" (57% antisense), and an unsorted,
    unindexed 5' stream.
- **What they check:**
  - The probe flips 5' and leaves 3' alone, on both `run_parallel()` and `run()`.
  - It clears a wrong preset `reverse_strand`.
  - A forced orientation is kept.
  - The safety net fires when the probe ran but did not flip; this was the old bug.
  - The in-line probe ignores chrM.
  - `five_prime_evidence` sorts 3'/5' read sets.
- **Results:** 44/44 pass. `test_molecule_labels`, `test_protocol_confidence_override`,
  `test_ws2_protocol_hardening`, `test_indrop_detect`, `test_seqwell_detect`,
  `test_species_short_r2` and `test_1fq_header` still pass.

### Benchmark datasets (same FASTQs and reference as the benchmark, 16 threads)

**Strand calls.**

| dataset | encode label | probe: antisense of informative | counted | nuclear reads antisense to counted strand |
|---|---|---|---|---|
| pbmc5p_v1 (5' v1) | `10x-5p-v2` (was `10x-3p-v2`) | 86% of 17,902 → **reverse** | reverse | 2.8% |
| pbmc1k_v2 (3' v2) | `10x-3p-v2` | 8.9% of 18,702 | forward | 2.7% |
| pbmc1k_v3 (3' v3) | `10x-3p-v3` | 9.6% of 18,791 | forward | 2.9% |

The pbmc5p_v1 run used a binary built before the encode fix. It was still labelled 3' v2, so the
pileup probe alone made the call. The encode label column comes from the fixed `encode` on the
first 2M reads (`strand/encode_check.sbatch`). All six 3' datasets (also neuron1k_v3, nuclei2k_v2,
hgmm1k_v3, pbmc1k_cite_v3) stay 3', with 38–196 3'-kit TSO reads per 1,000.

**pbmc5p_v1 vs STARsolo** (6,605 shared cells; `compare.py`):

| metric | engine of #22 (auto) | `--reverse-strand` forced | **this branch (auto)** |
|---|---|---|---|
| per-cell gene r vs STARsolo GeneFull (exon + intron) | 0.038 | 0.987 | **0.987** |
| per-cell gene r, exonic vs STARsolo Gene | 0.037 | 0.994 | **0.994** |
| per-cell total ratio vs GeneFull | 0.19 | 1.03 | **1.03** |
| pseudobulk Spearman vs GeneFull | 0.57 | 0.96 | **0.96** |
| marker Jaccard vs STARsolo GeneFull labels | 0.016 | — | **0.96** |
| cluster ARI vs STARsolo GeneFull | 0.45 | — | **0.77** |
| cluster ARI vs ADT clusters | 0.46 | — | **0.48** (STARsolo 0.48) |

The auto-detected run matches the forced run on every concordance metric.

### Kept campaign BAMs (`strand/rerun_bam.sbatch`)

Only 214 campaign BAMs were kept, 8 of them GRCh38. Six are flagged, all Visium. The table gives
two orientation-free probes and a re-pile with this branch (`--bam-file`, i.e. `run()` with the
indexed probe and the safety net).

| GSM | atlas wf | probe antisense (engine) | probe antisense (`strand_probe_bam.py`) | atlas exon hits | re-pile exon hits | counted |
|---|---|---|---|---|---|---|
| GSM7090360 | 0.923 | 2.3% | 2.7% | 1.47M | 9.49M | forward |
| GSM7090362 | 0.935 | 2.2% | 2.6% | 1.12M | 3.44M | forward |
| GSM7412962 | 0.936 | 1.5% | 3.0% | 1.07M | 15.6M | forward |
| GSM7596602 | 0.930 | 1.4% | 1.9% | 1.09M | 5.22M | forward |
| GSM7596603 | 0.930 | 1.2% | 1.6% | 0.62M | 2.38M | forward |
| GSM7983360 | 0.914 | 2.2% | 2.7% | 1.31M | 14.4M | forward |

- **These are correctly stranded 3' libraries.** The atlas counted them on the reverse strand
  anyway, with `reverse_strand: false` in their JSON.
- **The fixed engine counts them forward**, without a flip and without a retry.
- **They are not a 5' problem.** So "wf > 0.5" alone says only that a sample was counted on its
  minority strand.

## 4. Campaign classification (52,756 GRCh38 samples)

`analysis/strand_classify.py`.

- **Inputs:** each sample's `pileup_stats.json`, and GEO text from
  `cellarium/catalog/all_gsm_metadata.parquet`.
- **wf:** `wrong_strand / (wrong_strand + exon_hits)`.
- **Strandedness:** |1 − 2wf| does not depend on which way the atlas counted. A stranded library
  scores near 1 whether it was counted right (wf ≈ 0) or wrong (wf ≈ 1); an unstranded one
  scores near 0.

| class | wf | meaning | samples |
|---|---|---|---|
| misoriented | ≥ 0.7 | stranded library counted on its minority strand | **7,739** |
| likely_misoriented | 0.6–0.7 | partly stranded, counted on its minority strand | **531** |
| antisense_heavy | 0.5–0.6 | about even: unstranded, or antisense/noise-dominated; not a strand error | 929 |
| borderline | 0.35–0.5 with chrM > 15% of mapped reads | may be misoriented under the chrM dilution | 63 |

Per protocol (all samples; full table in [per_protocol.csv](strand-detection/per_protocol.csv),
histograms in [wf_hist_per_protocol.csv](strand-detection/wf_hist_per_protocol.csv)):

| protocol | samples | wf median | flagged | misoriented | likely | antisense-heavy | strandedness, flagged / unflagged | flagged studies |
|---|---|---|---|---|---|---|---|---|
| 10x v2 | 20,046 | 0.08 | 3,959 | 3,487 | 148 | 324 | 0.85 / 0.88 | 378 |
| 10x v3 | 23,805 | 0.05 | 2,319 | 2,103 | 79 | 137 | 0.79 / 0.90 | 382 |
| CEL-seq2 | 1,268 | 0.60 | 717 | 556 | 73 | 88 | 0.62 / 0.81 | 30 |
| MARS-seq2 | 1,026 | 0.51 | 530 | 310 | 111 | 109 | 0.46 / 0.50 | 31 |
| Drop-seq | 1,812 | 0.19 | 444 | 342 | 21 | 81 | 0.65 / 0.78 | 47 |
| SPLiT-seq | 713 | 0.20 | 288 | 277 | 2 | 9 | 0.71 / 0.91 | 31 |
| Microwell-seq | 825 | 0.29 | 284 | 211 | 28 | 45 | 0.66 / 0.92 | 25 |
| other | 290 | 0.52 | 148 | 43 | 17 | 88 | 0.15 / 0.77 | 12 |
| 10x ARC GEX | 699 | 0.12 | 115 | 114 | 0 | 1 | 0.73 / 0.80 | 23 |
| 10x Visium | 239 | 0.01 | 78 | 78 | 0 | 0 | 0.85 / 0.98 | 8 |
| inDrop | 150 | 0.27 | 58 | 58 | 0 | 0 | 0.92 / 0.85 | 7 |
| Quartz-seq2 | 457 | 0.16 | 54 | 29 | 9 | 16 | 0.45 / 0.76 | 19 |
| DNBelab C4 | 322 | 0.05 | 48 | 9 | 31 | 8 | 0.32 / 0.90 | 9 |
| BD Rhapsody | 631 | 0.02 | 45 | 41 | 4 | 0 | 0.78 / 0.96 | 15 |
| Seq-Well | 123 | 0.27 | 35 | 22 | 5 | 8 | 0.46 / 0.69 | 6 |
| 10x v1 | 178 | 0.22 | 28 | 13 | 1 | 14 | 0.18 / 0.93 | 6 |
| sci-RNA-seq3 | 28 | 0.86 | 26 | 26 | 0 | 0 | 0.72 / 0.76 | 1 |
| others (5) | 144 | | 23 | 20 | 2 | 1 | | |
| **all** | **52,756** | 0.07 | **9,199** | **7,739** | **531** | **929** | 0.80 / 0.89 | 938 |

Reading per protocol:

- **10x (v2, v3, ARC, Visium), inDrop, BD Rhapsody, SPLiT-seq, sci-RNA-seq3.**
  - **Pattern:** wf is bimodal, with modes near 0.05 and 0.9. Flagged samples are as stranded as
    the rest.
  - **Verdict:** misoriented. A flagged 10x v2 sample may be a 5' library that was not flipped or
    a 3' library that was flipped wrongly.
  - **Weak hint of 5' in 10x v2:** 5'/V(D)J wording appears in the GEO text of 42% of flagged
    vs 25% of unflagged 10x v2 samples. The text match is noisy.
  - **Either case:** the fixed engine decides from the reads.
- **CEL-seq2.**
  - **Pattern:** the bulk of the protocol sits at wf 0.75–0.85. Half the flagged samples are one
    study, GSE163530 (456 CEL-seq2 + 217 MARS-seq2).
  - **Verdict:** a protocol- or study-wide orientation error, not antisense biology.
- **MARS-seq2, Quartz-seq2, Seq-Well, DNBelab C4.**
  - **Pattern:** weakly stranded even when unflagged (strandedness 0.5–0.7 for MARS-seq2 and
    Seq-Well). Several modes.
  - **Verdict:** mixed. Many are still misoriented (MARS-seq2 GSE195452: 80 samples at wf 0.99).
  - **Watch:** the 0.5–0.6 group of these protocols, which is in the antisense-heavy list.
- **"other", 10x v1.**
  - **Pattern:** flagged strandedness 0.15–0.18, piling up at wf 0.5–0.6.
  - **Verdict:** unstranded or mislabelled libraries. Not a strand error; re-counting the other
    strand gains little.
- **Low-depth samples.** The list includes some samples with only a few thousand exonic reads.
  Their wf is reliable as a direction but they matter little.

Per-study view: [flagged_studies.csv](strand-detection/flagged_studies.csv). A study is usually
all one chemistry, so a study-level re-run is natural.

## 5. Re-run list

- **The list:** [`strand-detection/rerun_samples.csv`](strand-detection/rerun_samples.csv),
  8,270 samples (misoriented + likely_misoriented).
- **Columns:** protocol, wf, strandedness, `count_gain` = wf / (1 − wf) (the expected gain in
  counted exonic reads), chrM share and GEO-text flags.
- **Cost:** the samples were run through the double-count fix anyway (#22), so the extra cost is
  only in which samples come first. The fixed engine needs no per-sample flag: it detects the
  orientation from the reads.
- **Optional:** the 929 antisense-heavy samples. Re-counting them gains < 1.5×, and many are
  unstranded protocols.

## 6. Limits and follow-ups

- **Streams can't be retried.** With STAR streaming (`--no-parallel-pileup`, or `/dev/shm` below
  10 GB), the pipe cannot be read twice. The in-line probe (10k nuclear reads) decides alone;
  the reads before its decision keep the starting orientation (<0.1% of a typical run). The
  campaign used the parallel path.
- **Engine built too early.** When the barcodes are known up front, the engine is built before
  the `.1fq` header sets a 5' protocol's `reverse_strand` (`singlet.cpp`, engine at ~3386, header
  at ~3873). The probe now corrects this, but the order is wrong.
- **Unstranded protocols.** There is no unstranded mode on the command line for droplet data. It
  would be the right setting for the antisense-heavy group, if those turn out to be full-length
  protocols.
- **One unexplained crash.** The first pbmc1k_v3 run segfaulted (exit 139) in "Exporting
  feature matrices", after a clean pileup, on node b003. The same binary and input completed on
  c007, and every other run completed. It is not reproduced and probably unrelated (the export
  code is unchanged), but it should be watched in the campaign.
- **Old-probe mechanism not reproduced.** Exactly how the old probe flipped the Visium libraries
  was not reproduced: `--bam-file` takes the `run()` path, not `run_parallel()`. The evidence is
  that the atlas counted them reversed while their reads are 97–98% sense.

## Reproducing (Clipper, `bench/pipeline/strand/`)

| step | script |
|---|---|
| fetch this branch only | `sync.sbatch` |
| binary + unit tests | `build.sbatch [all\|tests]` |
| benchmark runs | `../run_tool.sbatch <dataset> singlet-strandfix` |
| comparison (writes `results/<dataset>__strand`) | `compare.sbatch pbmc5p_v1 --cite` |
| encode labels | `encode_check.sbatch` |
| kept campaign BAMs | `rerun_bam.sbatch` |
| campaign classification | `classify.sbatch` |
