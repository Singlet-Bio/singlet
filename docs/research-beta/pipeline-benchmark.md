# Research beta handoff: pipeline accuracy, atlas re-run, aligner speed

Owner: pipeline-benchmark session. Written 2026-10-26 at the research-beta wind-down. Nothing here was merged into main except
PR #37; every other change is on its own branch or draft PR, listed below.

## 1. What was done

**Counting correctness.**
- **Molecule labels (PR #22):** each (barcode, gene, UMI) gets one label, which removes the exon/intron double count.
- **Determinism fix (775e60d):** 4, 8 and 16 threads give byte-identical output on that binary.
- **Benchmark:** gene counts are equivalent to STARsolo and Cell Ranger (`docs/pipeline-validation.md`).

**Production binary stack.** The branches build on each other in this order; no force-pushes were used.

| PR / branch | contents |
|---|---|
| #31 `prod/atlas-rerun` | #22 + #25 (strand); `called_v2` ported to C++, identical to Python; correct 5′ knee strand; `singlet screen-runs`; provenance |
| #36 `prod/modality-fixes` | 14 modality fixes (list below) |
| `prod/campaign` | the two above plus #34 |
| #34 `perf/star-accel` | exact STAR speedups |

The 14 fixes in #36:
- the false 5′ clip probe;
- doublet threshold calibration, genotype doublets, demux with an automatic donor count;
- `--snps` on by default, with SNP data in `private/`;
- ambient estimator and corrected matrix;
- mtDNA caller (rCRS reference, raw chrM allele counts);
- non-host per-cell table;
- junction filter flags;
- output hygiene and disk BAM sort;
- per-cell aligner QC counters and poly(A) sites.

The campaign binary is Clipper `$PB/bin/singlet-campaign`: sha256 `261d3a50…`, commit 02aa9b9, PGO build.

**`called_v2` 2.1.**
- Knee fitted within the DropletUtils curve bounds, and p-values extrapolated below the Monte Carlo floor.
- Python: PR #37 (merged), with shared golden vectors.
- C++: on #31.
- Tolerance PR #38 (tail p / FDR 1e-5).
- Truth check: `docs/called-v2-truth.md`.

**STAR acceleration (#34).**
- Changes: sequence→alignment cache, lazy winBin, hugepages, faster read parsing, PGO.
- Mapping is 1.54× faster (793 → 515 s); 1.38× re-measured on the campaign binary.
- Output is identical to stock STAR on 60.7M BAM records.
- Real cache hit rate is 46% at 489M reads; the 66% figure was simulated and is not used.
- #34 also removed a dedup bug inherited from #22 (~300 reads changed).

**Transcriptome-first.**
- Not built. Its exact ceiling is about 4% of mapping (PR #41 profile): only reads STAR already maps cheaply can be proven identical.
- PR #41 instead adds two optional speedups:
  - `SINGLET_STAR_SHORTEXIT` (verify mode found 0 mismatches);
  - `SINGLET_STAR_FASTASSIGN` (skips window-assignment calls that have no effect).
- Unmapped "too short" reads take 19% of mapping on 3′ and 55% on 5′.

**Atlas re-run plan (PR #30, this branch).**
- Universe: 70,558 GEX samples.
- **All reads, as Zach decided:** 21.7T read pairs.
- Clipper download ceiling: ~150 MB/s, about 145 days.
- Anvil quote: 365k–535k SU against a 112k balance.
- Orchestrator: PR #42. Whole studies go to one cluster; event ledger; SU guard. Anvil is disabled two ways.
- Smoke test: 7 of 8 wave-a samples done.

**Public benchmark results (contract 1.1.0).** All use real data and 10x's published comparators. Every capability file validates, and every one fails today.

| branch | capability | metrics passing |
|---|---|---|
| `bench/public-counts` | gex-counts | 36/44 |
| | spliced-unspliced | 18/42 |
| | junctions-psi | 40/42 |
| | cell-calling | 5/25 |
| `bench/public-perf` | non-host | 8/10 |
| | alignment-performance | 3/4 |
| | download | 3/6 |
| `bench/public-ambient-doublets` | ambient-rna, doublets | see the agent notes in §5 |
| `bench/public-genotype` | donor-demux, donor-snps, mtdna | see the agent notes in §5 |
| `bench/exact-vs-pseudo` | exact vs pseudoalignment study | see the agent notes in §5 |

**Side studies.**
- PR #43, latent doublet/ambient study: a post-hoc combiner gives +0.06 AUPRC out-of-study and matches scDblFinder. Whether it ships in 1.0.0 is Zach's decision, still open.
- mtDNA lineage session: its own handoff is in `docs/research-beta/mtdna-lineage.md` on its branch.

## 2. Decisions on record (Zach)
- **Depth:** all reads, unconditionally.
- **Clusters:** Clipper first; Anvil only after Zach confirms, following several validated days on Clipper.
- **Data:** real data with ground truth only; simulation only as a corroborated supplement.
- **10x:** use their published outputs; don't run Cell Ranger.
- **Runtime** is shown, not gated, and plotted as a quality/runtime Pareto against alevin-fry (a pseudoaligner).
- **Open, waiting on Zach:**
  - the non-host screen during the re-run (recommended on);
  - a GVSU IT uplink inquiry and a 24 h bandwidth test;
  - private storage of SNP allele counts (implemented in `private/`);
  - Anvil spend / an ACCESS upgrade;
  - whether the doublet combiner ships in 1.0.0.

## 3. Open bugs, highest priority first
1. **NUMT / mt-gene multimapper drop (mouse).** On GRCm39, Singlet drops nearly every molecule of mt-Co2, mt-Atp6, mt-Co3 and mt-Nd3. neuron1k: mt-Atp6 16 vs STARsolo 3,978.
   - Cause: a 2-locus multimapper whose second locus is the chr1 NUMT is discarded, even though only one locus falls in a gene. STARsolo counts it.
   - Fix: count a UMI when all its alignments resolve to one gene. Then measure the effect genome-wide.
   - Branch `fix/intron-assignment` (see §5). It affects any gene with an intergenic pseudogene copy.
2. **Intronic / unspliced assignment.**
   - Singlet's unspliced counts agree less with STARsolo Velocyto and alevin-fry than those two agree with each other.
   - Barnyard intron-layer cross-species: 0.474% vs 0.271% for STARsolo.
   - CITE exon+intron rho: 0.421 vs GeneFull 0.428 (exonic layer equal).
   - Rn18s-rs5 (exon priority) and LINC00486 (a repeat-like 5 kb window with ~2% mismatches, seen in both species) are the largest single genes.
   - Branch `fix/intron-assignment`.
3. **Cell-calling truth (provisionally resolved: called_v2 wins on independent truth; see §5).**
   - The 10x "truth" sets are Cell Ranger's own EmptyDrops-type calls, so they are circular against EmptyDrops_CR.
   - On independent Kang truth, called_v2 has recall 1.00 and precision 0.88, vs 0.97 for EmptyDrops_CR. Genotypes suggest about 73% of the extras are real low-RNA cells.
   - Branch `bench/cellcall-independent-truth` builds genotype- and species-confirmed truth (§5).
4. **Thread non-determinism: cosmetic.** Only the junction-table row order changes with the thread count (§5). Sort junction names before numbering them.
5. **Lossy default download.** The default path drops qualities and trims poly(A) tails, and the mtDNA/SNP callers use base quality (§5). Choose the lossless path for the campaign, or measure the effect.
5b. **Barcode-order dependence.** Ambient, doublet and mt results change with input barcode order (found via multi-run input in #47).
5c. **The smoke test ran without whitelists** (#47). Re-run it before trusting the orchestrator timings.
6. **Ambient calibration.**
   - On Kang genotype truth: estimate 3.43% vs true 2.48%. Per-cell spread is about 10× too wide, and rare clusters are 2–3× too high.
   - The spec is in PR #43; implementation is on branch `fix/ambient-calibration`.
7. **Demux, automatic donor count.** It picks 9 on the 8-donor development lane: a 3% split of donor 1015 (ARI 0.976). It is correct on the held-out batch-1 lanes (4, 4, 8). Suggested fix: merge clusters with near-identical inferred genotypes.
8. **Deep samples.**
   - Disk BAM sort took 18–46 min on shared nodes.
   - Multi-run samples are pooled by decode + re-encode (11 min; OOM at 122M pairs).
   - `download --max-reads` memory is unbounded.
   - `download` and `screen-runs` disagree on chemistry.
   - Branch `perf/deep-samples`.
9. **Cost model.** Deep samples run 1.3–2.5× slower than `cost_model_v2.py` assumes, which makes Clipper compute-bound at about 160 days. Update the model after item 8.

## 4. Lessons
- **Benchmark one dataset per reference.** The cell-calling "regression" (hgmm 654 cells) and part of the profiling were mouse and barnyard data aligned to GRCh38 by a hard-coded harness.
- **Check that every tool can score a metric's inputs.** One NaN from a gene STARsolo doesn't count made a fake CITE gap.
- **Profile before building an aligner shortcut.** The transcriptome-first ceiling was about 4%.
- **Test at depth, on busy nodes.** "Disk sort costs nothing" held on one 66M-read sample on a quiet node and failed at 263M reads on shared nodes.
- **Order event files exactly, not by millisecond timestamp:** they sorted alphabetically within one millisecond (fixed in #42).
- **One bridge, one writer per file.** Five agents stalled together when the Clipper bridge dropped.

## 5. Agent end states and Clipper paths
WORK = `/mnt/projects/debruinz_project/debruinz/singlet`; PB = `$WORK/pipeline-benchmark`.

**Paths**
- **Binaries** in `$PB/bin/`: singlet-campaign (02aa9b9), singlet-campaign-aea204f, singlet-prod, singlet-prod-cv2, singlet-modfix, singlet-fixed-det, singlet-accel-*.
- **Perf matrix:** `$PB/perf/pbmc1k_v3_det/`; harness `$WORK/jobs/atlas-rerun/perf_run.sbatch` (picks the reference per dataset).
- **Campaign runs** on the right references: `$PB/runs/<ds>/singlet-campaign-t8[-reenc]`; regression check results in `$WORK/state/cfix/`.
- **Truth checks:** called_v2 in `$WORK/state/called_v2_truth/`; transcriptome-first eligibility in `$WORK/aligner-bench/txelig/pbmc1k_v3/eligible.json`.
- **Modality benchmarks:** `$WORK/modality-bench/{ambient,genotype_pool,sj_nonhost}/`.
- **Public artifacts:** `$WORK/bench-public/artifacts/` and `$WORK/bpub/artifacts/`.
- **Re-run data:** universe tables and the depth probe in `$WORK/data/atlas-rerun/`; orchestrator smoke test in `$WORK/atlas-rerun/smoke`.

**Agent end states** (written at wind-down from each agent's final report; job ids that were still running are listed there):

**Engine branches** (each is a draft PR into `prod/campaign`; none is merged):

- **#41 `perf/txome-first`.**
  - Transcriptome-first is not built: its exact ceiling is about 4% of mapping time.
  - `SINGLET_STAR_SHORTEXIT` was exact (verify mode found 0 mismatches) but saved no time, so it was dropped.
  - `SINGLET_STAR_FASTASSIGN`, on by default:
    - mapping time −19% (557 → 450 s) and whole run −16% on pbmc1k_v3 with the PGO build;
    - −18 to −19.5% mapping on pbmc10m and pbmc5p;
    - unmapped reasons unchanged.
  - **Not yet proven byte-identical.** Equivalence jobs 787387, 787389, 787391 and 787393 (logs `logs/txf-equiv-*.out`). Timing pairs 787386, 787388, 787390 and 787392, results in `$PB/perf/<ds>_txf/{off,on}_r{1,2}`.
  - Too-short reads take 55% of mapping on 5′ data. Most of what remains is seed search.
- **#46 `fix/intron-assignment`.**
  - Cause of the excess unspliced counts: multimapper rescue (`resolve_mm*` in `pileup_engine.h`). Uniquely mapped U molecules already agree with STARsolo 85–97%.
  - Rules can be switched with `SINGLET_INTRON_RULES`; no default is set yet, so output is unchanged.
  - Rule 1 (multimappers give no intronic evidence): intron-layer cross-species 0.474 → 0.347% (STARsolo 0.271%); hgmm passes 9/9 and pbmc1k_v3 7/9.
  - Rule 128: the CITE check passes (0.427 vs 0.428).
  - NUMT bug, confirmed but not fixed: mt reads hit both chrM and a chr1 copy carrying annotated pseudogenes (Atp6-ps, Gm28437, …), so both hits are exonic for different genes and the read is dropped. The fix needs a pseudogene / NUMT-aware tie-break, e.g. prefer the chrM primary, or the gene in which the read has fewer mismatches.
  - Jobs 787531 (sweep2, not scored), 787550 and 787622.
- **#45 `fix/ambient-calibration`.** Partial. Shrinks cluster rho toward the sample; adds a cell-set guard and tests.

  | lane | quintile slope | rare-cluster ratio | mean ratio |
  |---|---|---|---|
  | Kang 2.1 (development) | 0.21 → 1.06 | 1.62 → 0.81 | 1.05 → 0.68 |
  | stim (held out) | — | — | 2.16 → 1.62 |
  | b1 A/B/C (held out) | still flat | — | 10.7/11.9/9.9 → 8.4/8.6/7.1 |

  - Barnyard Spearman holds at 0.918.
  - Next: replace the per-cluster estimator.
  - Build job 787620 (`bin/singlet-ambfix`), not checked.
- **#47 `perf/deep-samples`.**
  - **Multi-run input:** `singlet A.1fq B.1fq` or `--runs-dir … --run-screen …` joins 4 runs in 0.5 s instead of an 11 min re-encode. All 8 count matrices are identical. Barcode order differs, and ambient, doublet and mt rows change with it, so those steps depend on barcode order: a separate bug.
  - **Download memory:** `download --max-reads` peak memory 19.1 → 1.18 GB (the read-name column), 163 → 22 s, identical reads.
  - **Whitelist dir:** download and compute read `SINGLET_WHITELIST_DIR`. **The orchestrator smoke test ran without whitelists** (the binary had been copied away from them), so its timings are not representative.
  - **Chemistry:** GSM6470208 is now 5′ v2 in both stages.
  - **Barcode refresh:** 31.6 → 12.4 s.
  - **BAM sort:** writing sorted bins straight to the final BAM is not faster yet (315 vs 260 s at 111.6M); profile it.
  - Jobs 787333/787335 (old binary) and 787497/787498 (new) on GSM7656149; 787527 (multi-run vs pooled); 787611. Outputs in `$WORK/deep-perf/`.

**Public result branches** (all draft, none is in a PR yet; target branch `claude/bench-public-cohort-485cf7`):

| branch | capability | status | key numbers |
|---|---|---|---|
| `bench/public-counts` (772eb0cd) | gex-counts | fail, 36/44 | |
| | spliced-unspliced | fail, 18/42 | |
| | junctions-psi | fail, 40/42 | |
| | cell-calling | fail, 5/25 | |
| `bench/public-perf` (d4183b00) | non-host | fail, 8/10 | per-cell metrics lose to Kraken2 by 0.002 |
| | alignment-performance | fail, 3/4 | see thread finding below |
| | download | fail, 3/6 | see lossy-path finding below |
| `bench/public-ambient-doublets` (178cea48) | ambient-rna | draft, fail, 8/10 | Spearman 0.82/0.89 vs CellBender 0.63/0.80; fails on held-out mean-removal bias (0.20 vs 0.03) and own-species removal |
| | doublets | no file | hgmm has fewer than 30 truth doublets; Kang artifacts not built (tool jobs 786074, 786368–786370) |
| `bench/public-genotype` (4eb87a24) | donor-demux | 12/15 | auto-K exact on 6/6 held-out pools (vireo 0/6); mean ARI 0.993 vs vireo 0.986, souporcell 0.994 |
| | donor-snps | 9/30 | concordance 0.961 vs vireo 0.978; false-alt 4.3% vs 0.5–0.8% |
| | mtdna | 12/18 | haplotype precision 0.887 vs mgatk 0.934 |

- **alignment-performance:** the 4/8/16-thread difference is only the row order of the junction tables (`merge_splice_junctions`, `pileup_engine.h` ~3300; sort names before indexing). Every count matrix, call, doublet, demux and mt output is byte-identical. Small unexplained differences remain in `rrna_report.json` and `read_stats.tsv`.
- **download:** Singlet's default path is lossy by design. It drops qualities (all `F`), trims poly(A) of ≥10 A's (`include/fq/writer.h:269, 955`), and stores reads barcode-sorted. **mtDNA and SNP callers use base quality**, so this may explain part of the donor-snps false-alt excess. Use the lossless path for the campaign, or prove the loss is harmless. Diagnostic job 787464 → `$WORK/bench-public/download/lossy_diagnose.txt`.
- **donor-demux:** fails on the two 8-donor lanes and on worst-donor recall. Cause of K = 9: `select_k_heldout` / `count_valid_components` (`donor_demux.h:1879, 1757`) never compare genotypes. A merge at about 0.80 concordance would fix it; the duplicate sits at 0.88 and distinct donors at most 0.68.
- **donor-snps:** the cause is probably the fixed 0.1/0.9 genotype cut-offs at `donor_demux.h:2511`.
- **Genotype data:** needs cohort rows `kang2018_stim`, `maester_k562_bt142_3p` and `maester_bpdcn712_3p`. Artifacts in `$WORK/bench-public/artifacts/{donor-demux,donor-snps,mtdna}`.
- **Rule breach:** the genotype agent ran a short pandas script and `pip download` on the Clipper login node, then cleaned up.

**Exact vs pseudoalignment** (`bench/exact-vs-pseudo`, 8460ed6f; doc marked incomplete). All on real truth:

| | STARsolo | Singlet | alevin-fry |
|---|---|---|---|
| barnyard cross-species | 0.33% | 0.41% | 0.63% (88k human UMIs on mouse mt-Rnr2) |
| chrY in HEK293T, per M human UMIs | 5.2 | 16.4 | 14.5 |
| CITE | 0.428 | 0.422 | 0.415 |
| accuracy score | 1.00 | 0.70 | 0.62 |
| CPU-h per 100M pairs, 4 threads | 1.49 | 1.33 | 0.73 |
| reach (outputs) | 4 | 12 | 2 |

Long-read and spike-in truth and the extra hgmm libraries are missing. Paths: `$WORK/exact-vs-pseudo/`.

**Cell-calling independent truth** (`bench/cellcall-independent-truth`, 90c172e4; README `bench/cellcall_truth/README.md`). F1 on barcodes with at least 50 UMIs:

| dataset | truth | called_v2 | CellBender | EmptyDrops_CR | Cell Ranger |
|---|---|---|---|---|---|
| Kang b2 ctrl | genotype | 0.991 | 0.985 | 0.954 | 0.942 |
| hgmm1k_v3, damaged barcodes counted as non-cells | species | 0.915 | 0.892 | 0.897 | 0.899 |

- On Kang, the confidence intervals of each difference exclude 0.
- **Provisional verdict:** called_v2 wins. Its losses on the 10x-truth metrics come from Cell Ranger calling conservatively, so reframe those metrics as agreement with Cell Ranger instead of changing called_v2.
- Not yet confirmed on hgmm5k_v3, hgmm1k_v2 or the batch-1 lanes: jobs 787567 and 787569, results in `$WORK/state/cellcall_truth/`.


## 6. Exact next steps, in order
1. **Fix bugs 1 and 2 on `fix/intron-assignment`.**
   - Validate on real truth: the neuron1k mt genes vs STARsolo, barnyard cross-species, unspliced agreement and CITE.
   - Merge into `prod/campaign`.
2. **Make outputs order-independent and deterministic:**
   - sort junction names before numbering them;
   - fix the barcode-order dependence;
   - decide the download path (lossless or not).
   Then read the #41 FASTASSIGN equivalence logs, and keep it on only if they are identical.
3. **Land `perf/deep-samples` and `fix/ambient-calibration` into `prod/campaign`.**
   - Rebuild the PGO binary and re-run `bench/called_v2/equiv_cpp.py` plus the perf equivalence checks.
   - Update `cost_model_v2.py` with the measured slope at depth.
4. **Cell calling:** decide from `bench/cellcall-independent-truth`. Either change called_v2 (on Python and C++ together, through the golden vectors) or reframe the 10x-truth metrics as agreement with Cell Ranger.
5. **Re-run all public capability writers** against the new binary, then merge the benchmark branches into a PR into `claude/bench-public-cohort-485cf7`.
6. **50-sample validation** through the orchestrator (PR #42), then several days of Clipper waves a–c. **Anvil only after Zach confirms.**

## 7. After the 1.0.0 rename (main, PR #39)
- `prod/atlas-rerun` (#31) is merged with main (8c3de82f): bench files moved to `bench/called_ed*`; the C++ test and CMake read `tests/data/called_ed_golden` with key `n_called_ed`.
- The stacked branches (#36, `prod/campaign`, #41, #45, #46, #47) still need the same merge, via `prod/atlas-rerun`.
- Decision B is not yet applied to the C++. Still to rename:
  - `include/singlet/pileup/called_v2.h` to the called_ed name;
  - the `cell_calls.tsv` columns `called_v2` and `fdr` to `called_ed` and `called_ed_fdr`;
  - provenance keys.
  The Python store reads `called_v2` as `called_ed` until cleanup.
- #31 state at stop: prod/atlas-rerun 8c3de82f (main merged, rename follow-ups above); CI status unknown.
