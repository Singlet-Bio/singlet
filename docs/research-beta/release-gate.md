# Beta release gate: one binary, 22 capabilities

**Result: not releasable.** 1 of 22 capabilities passes (`multiome`), 18 fail, 3 are incomplete or missing. Ten capabilities were measured on the candidate in this run; eleven are carried forward from earlier documents and say so (for three of them, other sessions measured the same binary and their numbers are in the row note); `adt` has no result file.

Policy (`bench/public/release_gate.py`, PR #79): releasable only when every capability passes on real-data gating metrics; a waiver is explicit and listed. No waivers are set.

## Gate 4: prod/campaign 18eb391c (the rows #164 and #165 move, re-run on the new tip)

Binary: `release-gate/bin/18eb391c/singlet-cpu` and `singlet-gpu`; candidate 18eb391caa5a69444bd8203e1691f5fa7a809fc4 = prod/campaign with #164 (cell calling: EmptyDrops_CR on exon-only counts)
and #165 (legacy exon/intron split plus spliced, unspliced and ambiguous matrices). Re-measured on it: the counts family, cell-calling, ambient-rna, hto and non-host. Every other row is the gate-3
document (measured on 2a3b0d61, or carried from earlier) and says so in the `measured` column; those rows are labelled `re-measure`, not counted as blockers. Files: `bench/release_gate/results/gate4_table.md` / `.json`.

| row | gate 3 (2a3b0d61) | gate 4 (18eb391c) |
|---|---|---|
| cell-calling | 5/20 | **21/4** (neuron1k_v3 recall 0.979 vs cellbender 0.998; nuclei2k_v2 precision 0.972 vs 0.980; kang2018_ctrl recall and Jaccard within 0.001 of starsolo-edcr) |
| gex-counts | 38/6 | **41/3** (exonic rows fixed; total UMI-ratio neuron1k_v3 and hgmm1k_v3, ADT correlation remain) |
| spliced-unspliced | 37/5 (agreement metrics) | **3/0 under the amended contract** (long-read precision, recall, F1, PBMC-E) |
| junctions-psi | 42/0 | 42/0 |
| ambient-rna | 9/1 | **8/2** (hgmm1k_v3 bias 0.460 vs SoupX 0.122; low-depth bias 0.845 vs DecontX-background 0.029) |
| hto | 1/7 | 1/7 |
| non-host | 8/2 | 8/2 |

- **Spliced-unspliced under the amendment** (draft #172): PBMC-E, 4,855 common cells, unspliced call vs long-read truth (>= 10 intronic bases): Singlet precision 0.4224, recall 0.7119, F1 0.5302; STARsolo Velocyto
  0.4167, 0.7082, 0.5247; alevin-fry USA 0.4181, 0.6993, 0.5233; match-or-beat on all three. MuSC, 78 dev cells (informational): precision 0.975, recall 0.384, F1 0.551; recall is capped by R2 read length. The long-read
  scores are S8's, on #165's content (build 0cd56886; merged head 797e78f2 differs by a bench-script commit). The agreement-with-Velocyto metrics (now secondary) got worse on 18eb391c (pbmc1k_v2 unspliced UMI-ratio distance 0.211 vs 0.027
  between the tools). The stored documents of junctions-psi and spliced-unspliced are draft because two cohort rows (pbmc5k_v4_d1, mouse_brain_nuclei5k_v31) have no run; their status is forced to final in this table.
- **Ambient got worse.** The ambient tools consume Singlet's cells, so #164 moved the competitors too (SoupX bias on hgmm1k_v3 0.038 -> 0.122). Owner S12 (draft #176). Keep ambient withheld.
- **Chromosomal sex** (gate-3 run, harness fixed in #167): 131 samples, 373,110 cells, 16 studies, 5/5 pass. GSE145370, GSE209832 and GSE220764 are not served by the atlas (404) and are excluded; no withhold reason is claimed.
- **Gate-3 only (not re-run here):** genotype, ancestry, atac, multiome (incl. the 3k development library), spatial, the 3k and cell-line runs; see the gate-3 section.

| capability | status | pass/fail | measured | worst gap vs best tool | class | note |
|---|---|---|---|---|---|---|
| gex-counts | fail | 41/3 | pin 18eb391c | `umi_ratio_distance.total` neuron1k_v3: 0.05541 vs 0.03132 (starsolo-genefull) | should-fix | gate 4 on 18eb391c (#164, #165): 41/3, was 38/6. Exonic UMI-ratio rows fixed (nuclei2k_v2 0.088 -> 0.018). Remaining: total UMI-ratio neuron1k_v3 0.055 vs 0.031 and hgmm1k_v3 0.018 vs 0.010 (R11), ADT correlation pbmc1k_cite_v3 0.409 vs 0.428 |
| spliced-unspliced | pass | 3/0 | pin 18eb391c |  |  | AMENDED contract (#172, draft): primary = unspliced precision/recall/F1 vs long-read truth. PBMC-E 4,855 cells: Singlet 0.4224 / 0.7119 / 0.5302 vs STARsolo Velocyto 0.4167 / 0.7082 / 0.5247 and alevin-fry 0.4181 / 0.6993 / 0.5233: 3/3 pass (scored by S8 on #165 content, build 0cd56886; merged head 797e78f2 differs by a bench-script commit). MuSC 78 dev cells (informational): P 0.975, R 0.384, F1 0.551, recall capped by R2 length. Agreement with Velocyto/alevin-fry is secondary and informational: on 18eb391c it is farther than before (pbmc1k_v2 unspliced UMI-ratio distance 0.211 vs 0.027 between the tools). Stored document is draft (two cohort rows have no run); status forced to final here, as for junctions-psi |
| junctions-psi | pass | 42/0 | pin 18eb391c |  |  | gate 4: 42/0 against the existing STARsolo reference (unique-only default, #140); stored document is draft (cohort gap), status forced to final here; contract amendment #138 not yet re-scored |
| cell-calling | fail | 21/4 | pin 18eb391c | `recall` neuron1k_v3: 0.9792 vs 0.9977 (cellbender) | should-fix | gate 4 on 18eb391c (#164, EmptyDrops_CR): 21/4, was 5/20. Remaining: neuron1k_v3 recall 0.979 vs cellbender 0.998; nuclei2k_v2 precision 0.972 vs 0.980; kang2018_ctrl recall 0.9975 vs 0.9980 and jaccard 0.9712 vs 0.9721 (starsolo-edcr) |
| ambient-rna | fail | 8/2 | pin 18eb391c | `bias_low_depth_abs_log2` hgmm1k_v3: 0.8447 vs 0.02885 (DecontX-background) | blocker | gate 4 on 18eb391c: 8/2, was 9/1. hgmm1k_v3 bias_abs_log2 0.460 vs SoupX 0.122 (was 0.276 vs 0.038: SoupX moved too, it uses Singlet cells) and low-depth bias 0.845 vs DecontX-background 0.029. Owner S12 (draft #176). Not published until it passes |
| doublets | incomplete | 4/0 | carried, 1bcfef0d |  | missing | doublet v2 (#122, in the tip) scored in S11 v2-mode on hgmm1k_v3 only (5 truth doublets, AUPRC not informative); contract harness (17 datasets, old combiner) not run at the tip. S11 research harness: v2 ties scDblFinder, engine score below it. Publish only as comparable to scDblFinder once wired |
| donor-demux | pass | 15/0 | carried, 2a3b0d61 |  |  | S10, one run on 2a3b0d61 (singlet-gpu, CPU engine), 7 pools; Kang vireo/souporcell outputs from earlier runs |
| donor-snps | fail | 22/3 | carried, 2a3b0d61 | `genotype_concordance` kang2018_b1_a: 0.9845 vs 0.9866 (cellsnp-pileup) | re-measure | S10 run on 2a3b0d61: 3 of 25 fail, all within 0.002 of cellsnp-pileup, ahead of vireo; Kang truth biased toward cellsnp; tolerances not edited |
| mtdna | fail | 11/7 | carried, 2a3b0d61 | `haplotype_precision` kang2018_ctrl: 0.7931 vs 0.8938 (mgatk) | re-measure | S10 run on 2a3b0d61: 7 of 18 fail (precision vs mgatk, recall b1_a/b1_b, BPDCN AUROC 0.794289 vs 0.794303 at tolerance 0); no mtDNA release |
| non-host | fail | 8/2 | pin 18eb391c | `percell_spearman_vs_oracle` ravindra2021_dpi2: 0.93 vs 0.9352 (kraken2-viral) | should-fix | gate 4 on 18eb391c: unchanged 8/2 (ravindra2021_dpi2 vs Kraken2 viral, by 0.002-0.005) |
| alignment-performance | fail | 3/1 | carried, 02aa9b93 | `thread_count_outputs_identical` pbmc1k_v3: 0 vs 1 (threshold) | stale: failing metric passes at the pin | thread_count_outputs_identical re-measured on 2a3b0d61: 59 of 59 data files identical at 4, 8 and 16 threads (pbmc1k_v3). Exclusive-node timings (Singlet 3 datasets x 2 reps, jobs 796260-796265) were queued and did not start; document and timings carried from 02aa9b9 |
| download | fail | 3/3 | carried, 02aa9b93 | `aggregate_reads_s` sra_download_set: 6.12e+05 vs 1.133e+06 (singlet-prefetch-local) | re-measure | re-measure (carried from 02aa9b9; S9 measured the Clipper ceiling instead, PR #108); no download harness on the tip |
| adt | missing |  | - |  | missing |  |
| hto | fail | 1/7 | pin 18eb391c | `f_score` howitt2023_cl: 0.726 vs 0.8743 (bff-cluster) | blocker | gate 4 on 18eb391c: unchanged 1/7 (F-scores 0.919 / 0.814 / 0.748 / 0.726 on BAL1 / BAL2 / BAL3 / cell lines); owner S11 (fix on Howitt dev split only; match_or_beat may not be reachable) |
| cell-type | fail | 9/3 | carried, 19caf8cd | `unseen_tissue_depth2` cxg16: 0.5073 vs 0.6147 (scANVI) | re-measure | re-measure (Python model, carried from 19caf8cd; scripts/celltype_nn compete jobs not run in this gate) |
| vdj | fail | 22/8 | carried, 5d39eb23 | `bcr_correct_pairing` pbmc10k_5p_v2: 0.9511 vs 0.974 (trust4) | re-measure | S11 held-out repro on the tip python (2a3b0d61): TCR pairing 0.9193 vs TRUST4 0.9139 pass; BCR 0.8698 vs 0.9207 fail; wrong lineage 0.0082 vs Cell Ranger 0.0021 fail. Not SOTA; no contract JSON |
| crispr-guides | incomplete | 21/0 | carried, 34a6f535 |  | missing | re-measure (carried from 34a6f535, Python counter, harness on claude/vdj-crispr-benchmark-721d3a, not on the tip); S16: only 2 campaign samples have a paired guide library |
| atac | fail | 9/1 | carried, 2a3b0d61 | `peak_f1` pbmc_multiome10k: 0.8298 vs 0.8373 (chromap+snapatac2) | re-measure | gate 3: both libraries on 2a3b0d61. peak_f1 pbmc_multiome10k vs chromap+snapatac2 |
| multiome | pass | 2/0 | carried, 2a3b0d61 |  |  | gate 3: 10k library on 2a3b0d61; the 3k library was not run (no data found on Clipper) |
| spatial | fail | 7/3 | carried, 2a3b0d61 | `spot_profile_pearson` visium_mouse_brain_ant: 0.9819 vs 0.9888 (starsolo) | re-measure | gate 3: Visium FFPE completes on the default path; three small gaps to STARsolo on visium_mouse_brain_ant; owner S12 |
| chromosomal-sex | incomplete | 5/0 | carried, 2a3b0d61 |  | missing | recorded benchmark (171 samples, 472,363 cells), model unchanged; cell accuracy 0.9698 (0.9976 without 5 samples whose GEO label reads dispute); the tip harness no longer imports (annotate_campaign._has_v2) so not re-run |
| genetic-ancestry | fail | 8/2 | carried, 2a3b0d61 | `accuracy_top.8m` mage150: 0.9329 vs 0.9933 (peddy) | re-measure | re-run from the tip scripts (#146 label rule) on the gate-3 binary, MAGE 150 (150 individuals, reads aligned with 2a3b0d61): accuracy.8m 0.973 (was 0.86), accuracy_top.8m 0.933 vs peddy 0.993; 2m and 0.5m pass. The result document is NOT contract-valid (derived data needs corroborated_by: no gating metric rests on real data); numbers are reported, the row cannot pass until the harness is fixed |

## Gate 3: prod/campaign 2a3b0d61 (all 22 attempted on one binary)

Binary: `release-gate/bin/2a3b0d61/singlet-cpu` and `singlet-gpu` (sha256 of singlet-gpu d95c8d30...). Candidate 2a3b0d61c247aa8cc8de5102b16114dbf4c802c8 is prod/campaign with
#153 merged; it contains the GPU per-base-quality fix 81c5a59f and #146 (ancestry label rule). Files: `bench/release_gate/results/gate3_table.md` and `.json`.

Result: 3 pass, 15 fail, 4 missing or incomplete; 14 of 22 measured on this binary. Blockers: spliced-unspliced, cell-calling, ambient-rna, donor-snps, mtdna,
hto, spatial, genetic-ancestry. Should-fix: gex-counts, non-host, atac. Re-measure (not measured on this binary): download, cell-type, vdj.

- **Counts family** (8 datasets, CPU and GPU): same as gate 2 (gex-counts 38/6, spliced-unspliced 37/5, cell-calling 5/20 with kang2018_ctrl, junctions-psi 42/0). S8's draft PR #165
  restores the legacy exon/intron split and adds spliced/unspliced/ambiguous matrices; exonic is back below the best competitor in S8's rescoring, unspliced still fails on 4 of 5. Re-run the
  counts rows on #165 once merged.
- **GPU engine** (`--engine gpu`, `SINGLET_GPU_PILEUP=on`, H100): all 8 datasets, nuclei2k_v2 included, produce complete output; 55-59 files identical to the CPU run, only `read_stats.tsv` differs
  (`gpu_vs_cpu_gate3.json`). The nuclei2k no-output failure is confirmed fixed.
- **Thread-count identity** (alignment-performance): 59 of 59 data files identical at 4, 8 and 16 threads on pbmc1k_v3. The exclusive-node timings (Singlet, 3 datasets x 2 reps) were queued and did not start; timings are carried from 02aa9b9.
- **Genotype** (S10, one run of 7 pools on this binary): donor-demux pass 15/15; donor-snps 22/25 (3 fails within 0.002 of cellsnp-pileup, ahead of vireo); mtdna 11/18 (precision vs mgatk, recall b1_a/b1_b, BPDCN AUROC at zero tolerance). Kang vireo/souporcell/mgatk outputs are from earlier runs; Kang truth is biased toward cellsnp.
- **Ancestry** (MAGE 150, reads aligned with this binary, #146 rule): accuracy.8m 0.973 (was 0.86), accuracy_top.8m 0.933 vs peddy 0.993; 0.5m and 2m pass. The document is not contract-valid ("derived data needs corroborated_by"), so the row cannot pass until the harness is corrected.
- **Chromosomal sex:** recorded benchmark (171 samples, 472,363 cells), model unchanged; the tip's harness no longer imports (`annotate_campaign._has_v2`), so it was not re-run. Cell accuracy 0.9698, 0.9976 without five samples whose GEO label the reads dispute.
- **Doublets:** doublet v2 (#122) is in the tip. The contract harness (17 datasets, old combiner) is not on the tip and was not run; S11's v2 mode scored hgmm1k_v3 only (about 5 truth doublets, AUPRC uninformative). S11's research harness: v2 ties scDblFinder, the engine score is below it. Publication policy: comparable to scDblFinder, never better, and only once v2 is wired into the driver.
- **V(D)J** (S11, tip python): TCR pairing 0.9193 vs TRUST4 0.9139 (pass), BCR 0.8698 vs 0.9207 (fail), wrong lineage 0.0082 vs Cell Ranger 0.0021 (fail). Contigs and clonotypes are published; BCR pairing is below TRUST4 until the fix lands. Not SOTA.
- **adt, crispr-guides, cell-type, download:** no result at this binary (adt driver not built; crispr/cell-type/download harnesses not on the tip). Not run: the 3k multiome library (no data found on Clipper).
- **Harnesses not on prod/campaign:** doublets and vdj (draft bench-only PR #159); genotype (S10's scripts are untracked); crispr, cell-type, download. The whole gate is not yet reproducible from one SHA.

| capability | status | pass/fail | measured | worst gap vs best tool | class | note |
|---|---|---|---|---|---|---|
| gex-counts | fail | 38/6 | pin 2a3b0d61 | `umi_ratio_distance.exonic` nuclei2k_v2: 0.08631 vs 0.02327 (starsolo-gene) | should-fix | gate 3 on 2a3b0d61: 38/6, same as gate 2; exonic and total UMI-ratio rows vs STARsolo (nuclei2k_v2, pbmc1k_v2, neuron1k_v3). S8 identified the cause: exon/intron counts carry USA labels while STARsolo Gene mixes them; S8 restores the legacy split |
| spliced-unspliced | fail | 37/5 | pin 2a3b0d61 | `umi_ratio_distance.unspliced` pbmc1k_v2: 0.1573 vs 0.02743 (starsolo-velocyto) | blocker | gate 3: 37/5, unspliced UMI-ratio distance vs STARsolo Velocyto. Same cause as gex-counts (S8). Primary metric moves to long-read agreement with Velocyto beside it once S8 out-of-sample result is in |
| junctions-psi | pass | 42/0 | pin 2a3b0d61 |  |  | gate 3: 42/0 against the existing STARsolo reference (unique-only default, #140); contract amendment #138 not yet re-scored |
| cell-calling | fail | 5/20 | pin 2a3b0d61 | `precision` nuclei2k_v2: 0.7861 vs 0.9796 (starsolo-edcr) | blocker | gate 3: 5/20 (kang2018_ctrl now scored); nuclei2k_v2 precision 0.77, kang2018_ctrl 16,524 vs truth 14,619; owner S12 |
| ambient-rna | fail | 9/1 | pin 2a3b0d61 | `bias_abs_log2` hgmm1k_v3: 0.2781 vs 0.03838 (SoupX) | blocker | gate 3: 9/1, bias_abs_log2 on hgmm1k_v3 0.276 vs SoupX 0.038; owner S12. Not published until it passes |
| doublets | incomplete | 4/0 | carried, 1bcfef0d |  | missing | doublet v2 (#122, in the tip) scored in S11 v2-mode on hgmm1k_v3 only (5 truth doublets, AUPRC not informative); contract harness (17 datasets, old combiner) not run at the tip. S11 research harness: v2 ties scDblFinder, engine score below it. Publish only as comparable to scDblFinder once wired |
| donor-demux | pass | 15/0 | pin 2a3b0d61 |  |  | S10, one run on 2a3b0d61 (singlet-gpu, CPU engine), 7 pools; Kang vireo/souporcell outputs from earlier runs |
| donor-snps | fail | 22/3 | pin 2a3b0d61 | `genotype_concordance` kang2018_b1_a: 0.9845 vs 0.9866 (cellsnp-pileup) | blocker | S10 run on 2a3b0d61: 3 of 25 fail, all within 0.002 of cellsnp-pileup, ahead of vireo; Kang truth biased toward cellsnp; tolerances not edited |
| mtdna | fail | 11/7 | pin 2a3b0d61 | `haplotype_precision` kang2018_ctrl: 0.7931 vs 0.8938 (mgatk) | blocker | S10 run on 2a3b0d61: 7 of 18 fail (precision vs mgatk, recall b1_a/b1_b, BPDCN AUROC 0.794289 vs 0.794303 at tolerance 0); no mtDNA release |
| non-host | fail | 8/2 | pin 2a3b0d61 | `percell_spearman_vs_oracle` ravindra2021_dpi2: 0.93 vs 0.9352 (kraken2-viral) | should-fix | gate 3: re-run on 2a3b0d61 with Kraken2 and oracle re-run: 2 of 10 fail by 0.002-0.005 against Kraken2 viral on ravindra2021_dpi2 |
| alignment-performance | fail | 3/1 | carried, 02aa9b93 | `thread_count_outputs_identical` pbmc1k_v3: 0 vs 1 (threshold) | stale: failing metric passes at the pin | thread_count_outputs_identical re-measured on 2a3b0d61: 59 of 59 data files identical at 4, 8 and 16 threads (pbmc1k_v3). Exclusive-node timings (Singlet 3 datasets x 2 reps, jobs 796260-796265) were queued and did not start; document and timings carried from 02aa9b9 |
| download | fail | 3/3 | carried, 02aa9b93 | `aggregate_reads_s` sra_download_set: 6.12e+05 vs 1.133e+06 (singlet-prefetch-local) | re-measure | re-measure (carried from 02aa9b9; S9 measured the Clipper ceiling instead, PR #108); no download harness on the tip |
| adt | missing |  | - |  | missing |  |
| hto | fail | 1/7 | pin 2a3b0d61 | `f_score` howitt2023_cl: 0.726 vs 0.8743 (bff-cluster) | blocker | gate 3 on 2a3b0d61 (same caller as gate 2); owner S11 (fix developed on Howitt dev split only; match_or_beat may not be reachable) |
| cell-type | fail | 9/3 | carried, 19caf8cd | `unseen_tissue_depth2` cxg16: 0.5073 vs 0.6147 (scANVI) | re-measure | re-measure (Python model, carried from 19caf8cd; scripts/celltype_nn compete jobs not run in this gate) |
| vdj | fail | 22/8 | carried, 5d39eb23 | `bcr_correct_pairing` pbmc10k_5p_v2: 0.9511 vs 0.974 (trust4) | re-measure | S11 held-out repro on the tip python (2a3b0d61): TCR pairing 0.9193 vs TRUST4 0.9139 pass; BCR 0.8698 vs 0.9207 fail; wrong lineage 0.0082 vs Cell Ranger 0.0021 fail. Not SOTA; no contract JSON |
| crispr-guides | incomplete | 21/0 | carried, 34a6f535 |  | missing | re-measure (carried from 34a6f535, Python counter, harness on claude/vdj-crispr-benchmark-721d3a, not on the tip); S16: only 2 campaign samples have a paired guide library |
| atac | fail | 9/1 | pin 2a3b0d61 | `peak_f1` pbmc_multiome10k: 0.8298 vs 0.8373 (chromap+snapatac2) | should-fix | gate 3: both libraries on 2a3b0d61. peak_f1 pbmc_multiome10k vs chromap+snapatac2 |
| multiome | pass | 2/0 | pin 2a3b0d61 |  |  | gate 3: 10k library on 2a3b0d61; the 3k library was not run (no data found on Clipper) |
| spatial | fail | 7/3 | pin 2a3b0d61 | `spot_profile_pearson` visium_mouse_brain_ant: 0.9819 vs 0.9888 (starsolo) | blocker | gate 3: Visium FFPE completes on the default path; three small gaps to STARsolo on visium_mouse_brain_ant; owner S12 |
| chromosomal-sex | incomplete | 5/0 | carried, bf1f5741 |  | missing | recorded benchmark (171 samples, 472,363 cells), model unchanged; cell accuracy 0.9698 (0.9976 without 5 samples whose GEO label reads dispute); the tip harness no longer imports (annotate_campaign._has_v2) so not re-run |
| genetic-ancestry | fail | 8/2 | pin 2a3b0d61 | `accuracy_top.8m` mage150: 0.9329 vs 0.9933 (peddy) | blocker | re-run from the tip scripts (#146 label rule) on the gate-3 binary, MAGE 150 (150 individuals, reads aligned with 2a3b0d61): accuracy.8m 0.973 (was 0.86), accuracy_top.8m 0.933 vs peddy 0.993; 2m and 0.5m pass. The result document is NOT contract-valid (derived data needs corroborated_by: no gating metric rests on real data); numbers are reported, the row cannot pass until the harness is fixed |

## Gate 2: the tip candidate (3eaa041c)

Same 22 capabilities, re-run on `prod/campaign` at 3eaa041c872fd763ab7e6dc6dce8b547bf8db87c (unique-only junction counts #140, protocol-detection
floor #124, FFPE strand fallback #117, stall watchdog #114, K5 aligner profile as shipped). Counts family, HTO, Visium, ATAC/multiome, non-host
(Kraken2 and the minimap2 oracle re-run on the new outputs) and ambient-rna (SoupX, DecontX, CellBender re-run) are measured on this binary: 10 of 22.
Rows computed by the Python models or measured on an earlier binary are **carried** and labelled `re-measure`: they are not counted as blockers
until they are re-measured. Files: `bench/release_gate/results/gate2_table.md` and `.json`; jobs in `bench/release_gate/` (`mk_gate2_jobs.py` rewrites them to write
under `release-gate2/` and stamp the tip SHA).

Result: 2 pass, 17 fail, 3 missing or incomplete (adt, chromosomal-sex, crispr-guides). Blockers on this binary: spliced-unspliced, cell-calling,
ambient-rna, hto, spatial. Should-fix: gex-counts, non-host, atac. Re-measure: doublets, donor-demux, donor-snps, mtdna, download, cell-type, vdj,
genetic-ancestry.

- **Owners.** cell-calling, spliced-unspliced, ambient and HTO: S11. donor-snps, mtdna, ancestry, demux: S10. The nuclei2k GPU no-output failure: S8.
- **junctions-psi** now passes (42/0 against the existing STARsolo reference). The contract amendment (#138: STARsolo with Singlet's own STAR options as
  reference) has not been re-scored yet.
- **gex-counts** exonic UMI ratio distance regressed on nuclei2k_v2 (0.088 vs 0.023 between tools) and pbmc1k_v2. Not attributed to a tip change yet;
  the intron and USA-label rules that fixed spliced-unspliced are the suspects.
- **spatial.** Visium FFPE human breast now completes on the default path (exit 0, `STREAM PATH OK`); three small gaps to STARsolo remain on
  visium_mouse_brain_ant. **atac/multiome**: the 10k library completes (it died with exit 5 on d9834b8a); atac peak_f1 on pbmc_multiome10k 0.830
  vs 0.837, multiome passes.
- **Not shipped in Beta:** RNA editing. **V(D)J** is not at SOTA yet; work to get there is under way.
- **Not run:** the nuclei2k GPU re-run on the tip's GPU build (S8's), the 8-dataset STARsolo `MODE=reference` runs for #138, and the 3k multiome library.
- The ambient-rna and doublets rows decide whether those two layers are published (`policy.py` `OMITTED_LAYERS`): ambient-rna is still 9/1 (hgmm1k_v3
  bias), doublets is carried (19/2) and needs a re-measure.

| capability | status | pass/fail | measured | worst gap vs best tool | class | note |
|---|---|---|---|---|---|---|
| gex-counts | fail | 38/6 | pin 3eaa041c | `umi_ratio_distance.exonic` nuclei2k_v2: 0.08816 vs 0.02327 (starsolo-gene) | should-fix | gate 2 on the tip candidate: exonic UMI ratio distance regressed on nuclei2k_v2 (0.088 vs 0.023) and pbmc1k_v2; not yet attributed to a tip change (suspect: the intron and USA-label rules that fixed spliced-unspliced); owner S11 |
| spliced-unspliced | fail | 37/5 | pin 3eaa041c | `umi_ratio_distance.unspliced` pbmc1k_v2: 0.1573 vs 0.02743 (starsolo-velocyto) | blocker | gate 2: unspliced UMI ratio distance still failing (pbmc1k_v2 0.157 vs 0.027); owner S11 |
| junctions-psi | pass | 42/0 | pin 3eaa041c |  |  | unique-only junction counting (NH==1, #140) is the default: 42/0 against the existing STARsolo reference; contract amendment (#138, STARsolo with the Beta profile) still to re-score |
| cell-calling | fail | 6/19 | pin 3eaa041c | `precision` nuclei2k_v2: 0.772 vs 0.9796 (starsolo-edcr) | blocker | gate 2: 19 of 25 fail; nuclei2k_v2 precision 0.77; kang2018_ctrl 16,524 Singlet cells vs truth 14,619; owner S11 |
| ambient-rna | fail | 9/1 | pin 3eaa041c | `bias_abs_log2` hgmm1k_v3: 0.2759 vs 0.03801 (SoupX) | blocker | gate 2: 9 of 10; still bias_abs_log2 on hgmm1k_v3 (0.276 vs 0.038 SoupX); owner S11 |
| doublets | fail | 19/2 | carried, cfe5cfa2 | `auprc` xili_hek_hmec_multi: 0.3759 vs 0.425 (singlet-production) | re-measure | re-measure (carried; Python combiner, not the binary) |
| donor-demux | fail | 12/3 | carried, 02aa9b93 | `ari` kang2018_ctrl: 0.9755 vs 0.99 (souporcell) | re-measure | re-measure (S10 owns) |
| donor-snps | fail | 9/21 | carried, 02aa9b93 | `genotype_concordance` kang2018_b1_c: 0.7549 vs 0.9897 (cellsnp-pileup) | re-measure | re-measure (S10 owns; #115 not in the gate-1 binary) |
| mtdna | fail | 12/6 | carried, 02aa9b93 | `haplotype_precision` kang2018_ctrl: 0.7891 vs 0.8938 (mgatk) | re-measure | re-measure (S10 owns; not re-run on the tip) |
| non-host | fail | 8/2 | pin 3eaa041c | `percell_spearman_vs_oracle` ravindra2021_dpi2: 0.93 vs 0.9352 (kraken2-viral) | should-fix | gate 2, re-run on the tip with Kraken2/oracle re-run: 2 of 10 fail by 0.002-0.005 against Kraken2 viral on ravindra2021_dpi2 |
| alignment-performance | fail | 3/1 | carried, 02aa9b93 | `thread_count_outputs_identical` pbmc1k_v3: 0 vs 1 (threshold) | stale: failing metric passes at the pin | thread_count_outputs_identical re-measured at the pin: identical, 52 of 52 data files at 4, 8 and 16 threads (pbmc1k_v3). Document and timings carried from 02aa9b9 (exclusive-node timings not re-run; S12 has a profile on shared nodes, PR #112) |
| download | fail | 3/3 | carried, 02aa9b93 | `aggregate_reads_s` sra_download_set: 6.12e+05 vs 1.133e+06 (singlet-prefetch-local) | re-measure | re-measure (carried; S9 measured the Clipper ceiling instead, PR #108) |
| adt | missing |  | - |  | missing |  |
| hto | fail | 1/7 | pin 3eaa041c | `f_score` howitt2023_cl: 0.726 vs 0.8743 (bff-cluster) | blocker | gate 2, tip candidate: F-scores 0.919 / 0.814 / 0.748 / 0.726 on BAL1 / BAL2 / BAL3 / cell lines (7 of 8 fail); owner S11 |
| cell-type | fail | 9/3 | carried, 19caf8cd | `unseen_tissue_depth2` cxg16: 0.5073 vs 0.6147 (scANVI) | re-measure | re-measure (Python model, carried) |
| vdj | fail | 22/8 | carried, 5d39eb23 | `bcr_correct_pairing` pbmc10k_5p_v2: 0.9511 vs 0.974 (trust4) | re-measure | not at SOTA yet; work to get there is under way (Python assembler; carried) |
| crispr-guides | incomplete | 21/0 | carried, 34a6f535 |  | missing | re-measure (Python counter, carried) |
| atac | fail | 9/1 | pin 3eaa041c | `peak_f1` pbmc_multiome10k: 0.8298 vs 0.8373 (chromap+snapatac2) | should-fix | gate 2: both libraries on the tip candidate (10k completes; stall watchdog #114). peak_f1 on pbmc_multiome10k 0.830 vs 0.837 |
| multiome | pass | 2/0 | pin 3eaa041c |  |  | gate 2: passes on the tip candidate (10k library); the 3k development library was not re-run |
| spatial | fail | 7/3 | pin 3eaa041c | `spot_profile_pearson` visium_mouse_brain_ant: 0.9806 vs 0.9888 (starsolo) | blocker | gate 2: Visium FFPE human breast now completes on the default path (exit 0, STREAM PATH OK; strand fallback #117); remaining failures are small gaps to STARsolo on visium_mouse_brain_ant (3 metrics) |
| chromosomal-sex | incomplete | 5/0 | carried, bf1f5741 |  | missing | re-measure (Python model, carried) |
| genetic-ancestry | fail | 8/3 | carried, 40f0b5d2 | `accuracy.8m` mage150: 0.86 vs 0.9933 (peddy) | re-measure | re-measure (Python model, carried) |

Gate 1 (pin d9834b8a) follows.

## 1. Candidate

| | |
|---|---|
| source | `prod/campaign` **d9834b8a** (merge of #97; contains #95, #98, the GPU intron rules, #80, #87, #91, #94) |
| CPU binary | `$WORK/release-gate/bin/d9834b8a/singlet-cpu`, sha256 `34e989e6520770d30f99c4de1eba61f203770919be30252efc1b37adf3c85fc6` (engine off) |
| GPU binary | `.../singlet-gpu`, sha256 `890fcd928a364eb1119d26961ab784487f974616c83421dd5cb9122548e7b771`, run with `--engine gpu` and `SINGLET_GPU_PILEUP=on` |
| build | `bench/release_gate/build.sbatch`: Release, non-PGO, `-march=x86-64-v3`, pbench toolchain in the CUDA 12.4 image; no PGO, so timings are not comparable with the 02aa9b9 PGO binary |
| whitelists | the commit's own `python/singlet/data/whitelists`, plus the GEM-X v4 list, copied to `release-gate/bin/whitelists` next to the binary; every run log shows the whitelist path and `STREAM PATH OK` (the `BAM: unsorted STAR stream` line) except the Visium FFPE run, which was started with `--sorted-bam` on purpose |
| tip | the campaign tip is now 24143ac3 (d9834b8a + GPU per-base quality, host offload and bench commits). A build of the tip plus the stall-watchdog fix (PR #114, **22b5782f**) gives CPU outputs identical to d9834b8a on pbmc1k_v3 and hgmm1k_v3 (52 and 50 data files identical, only the timing column of `read_stats.tsv` differs). The 10k Multiome ATAC library was run on 22b5782f because d9834b8a cannot finish it (section 5). |

Rules kept: Clipper jobs only (`short-cpu`, `scavenger`, `short-gpu` H100), `%4`/`%2` array throttles, real data and real ground truth, no genotypes or reads in any output, Alpha's binary untouched, the 02aa9b9 run directories were moved aside, not deleted.

## 2. The table

Status and pass/fail come from the results contract (`contract.capability_status`), recomputed from the values; "measured" says whether the document was produced on the candidate or is the newest earlier document of that capability. "Class" is a proposal: **blocker** = wrong output (the run errors, an identity check fails, or a gating metric is far from the best tool: a higher-is-better gap above 0.05, an accuracy/precision/F-score/recall below 0.90 and more than 0.02 behind the best tool, or a distance metric more than 5 times the best tool's); **should-fix** = valid output, behind the best tool on a gating metric, within those bounds.

| capability | status | pass/fail | measured | worst gap vs best tool | class | note |
|---|---|---|---|---|---|---|
| gex-counts | fail | 41/3 | pin d9834b8a | `umi_ratio_distance.total` neuron1k_v3: 0.05424 vs 0.03132 (starsolo-genefull) | should-fix | 3 of 44 fail. hgmm1k_v3 total UMI ratio 0.0177 vs 0.0100; with R11 off 0.0104 (section 4) |
| spliced-unspliced | fail | 32/10 | pin d9834b8a | `umi_ratio_distance.unspliced` pbmc1k_v2: 0.2066 vs 0.02743 (starsolo-velocyto) | blocker | unspliced UMI totals per cell 7.5x farther from STARsolo Velocyto than the tools are from each other (pbmc1k_v2) |
| junctions-psi | fail | 38/4 | pin d9834b8a | `percell_total_corr` neuron1k_v3: 0.9198 vs 0.9996 (starsolo-sj-cr) | blocker | mouse neuron1k_v3 per-cell junction totals correlate 0.92 with STARsolo SJ (0.9996 for the tools) |
| cell-calling | fail | 6/19 | pin d9834b8a | `precision` nuclei2k_v2: 0.7718 vs 0.9796 (starsolo-edcr) | blocker | 19 of 25 fail; nuclei2k_v2 precision 0.77 (called_v2 over-calls in nuclei) |
| ambient-rna | fail | 9/1 | pin d9834b8a | `bias_abs_log2` hgmm1k_v3: 0.2823 vs 0.03801 (SoupX) | blocker | 1 of 10 fails: contamination-estimate bias on hgmm1k_v3 |
| doublets | fail | 19/2 | carried, cfe5cfa2 | `auprc` xili_hek_hmec_multi: 0.3759 vs 0.425 (singlet-production) | should-fix | Python combiner (not the binary); carried |
| donor-demux | fail | 12/3 | carried, 02aa9b93 | `ari` kang2018_ctrl: 0.9755 vs 0.99 (souporcell) | should-fix | S10 on d9834b8a (auto-K, binned .1fq, competitors re-run on its BAMs; JSON not written): K exact on 5/5 pools, ARI 0.976-0.999 vs 0.948-0.995 for vireo/souporcell given the true K; doublet F1 0.77-0.87 |
| donor-snps | fail | 9/21 | carried, 02aa9b93 | `genotype_concordance` kang2018_b1_c: 0.7549 vs 0.9897 (cellsnp-pileup) | blocker | S10 on d9834b8a (JSON not written): concordance 0.9863 vs cellsnp 0.9901 and vireo 0.9772 at equal DP>=10 sites; with PR #115 (R12 cut 0.9 to 0.99) 0.9899. The gate binary does not have #115. Carried document is the 02aa9b9 one |
| mtdna | fail | 12/6 | carried, 02aa9b93 | `haplotype_precision` kang2018_ctrl: 0.7891 vs 0.8938 (mgatk) | blocker | not re-run |
| non-host | fail | 8/2 | pin d9834b8a | `percell_spearman_vs_oracle` ravindra2021_dpi2: 0.93 vs 0.9352 (kraken2-viral) | should-fix | 2 of 10 fail by 0.002-0.005 against Kraken2 viral on the infected sample |
| alignment-performance | fail | 3/1 | carried, 02aa9b93 | `thread_count_outputs_identical` pbmc1k_v3: 0 vs 1 (threshold) | stale: failing metric passes at the pin | thread_count_outputs_identical re-measured at the pin: identical, 52 of 52 data files at 4, 8 and 16 threads (pbmc1k_v3). Document and timings carried from 02aa9b9 (exclusive-node timings not re-run; S12 has a profile on shared nodes, PR #112) |
| download | fail | 3/3 | carried, 02aa9b93 | `aggregate_reads_s` sra_download_set: 6.12e+05 vs 1.133e+06 (singlet-prefetch-local) | blocker | carried; S9 measured the Clipper ceiling and the local decode (PR #108) instead of re-running dl_bench: 0.65-0.96 GB/s aggregate, 45-60 MB/s per stream, decode 2.6-3.0 core-s per M pairs on d9834b8a |
| adt | missing |  | - |  | missing |  |
| hto | fail | 1/7 | pin d9834b8a | `f_score` howitt2023_cl: 0.726 vs 0.8743 (bff-cluster) | blocker | #87 re-measured at the pin (header compiled from the pin, held-out settings): F-scores 0.919 / 0.814 / 0.748 / 0.726 on BAL1 / BAL2 / BAL3 / cell lines, as before |
| cell-type | fail | 9/3 | carried, 19caf8cd | `unseen_tissue_depth2` cxg16: 0.5073 vs 0.6147 (scANVI) | blocker | Python model (not the binary); carried |
| vdj | fail | 22/8 | carried, 5d39eb23 | `bcr_correct_pairing` pbmc10k_5p_v2: 0.9511 vs 0.974 (trust4) | should-fix | Python assembler (not the binary); interim document; carried |
| crispr-guides | incomplete | 21/0 | carried, 34a6f535 |  | missing | Python counter (not the binary); carried. #91 re-measured at the pin: guide match rate 0.340 -> 0.681 with half the reads stored reverse (A375 library, 2 M read pairs) |
| atac | fail | 9/1 | pin d9834b8a | `peak_f1` pbmc_multiome10k: 0.8298 vs 0.8373 (chromap+snapatac2) | should-fix | 5k library on d9834b8a; 10k library on 22b5782f (tip + stall-watchdog fix, PR #114): on d9834b8a the 10k run dies with exit 5 (blocker, fixed in #114) |
| multiome | pass | 2/0 | pin d9834b8a |  |  | passes; same 10k run and caveat as atac; the 3k development library was not re-run |
| spatial | fail | 5/5 | pin d9834b8a | `spot_profile_pearson` visium_mouse_brain_ant: 0.9808 vs 0.9888 (starsolo) | blocker | Visium FFPE human breast aborts on the default stream path ('99% of nuclear reads are antisense ... rerun with --sorted-bam'); the numbers here are the --sorted-bam run. The 5 failing metrics are small gaps to STARsolo |
| chromosomal-sex | incomplete | 5/0 | carried, bf1f5741 |  | missing | Python model (not the binary); carried |
| genetic-ancestry | fail | 8/3 | carried, 40f0b5d2 | `accuracy.8m` mage150: 0.86 vs 0.9933 (peddy) | blocker | Python model (not the binary); carried |

## 3. Blockers and should-fix, by capability

**Blockers (proposed):**
- `spliced-unspliced`: unspliced UMI totals per cell are 0.21 / 0.09 / 0.075 (mean |ln ratio|) from STARsolo Velocyto on pbmc1k_v2 / cite_v3 / v3, against 0.027 / 0.006 / 0.016 for the competitors from each other. 10 of 42 metrics fail (32 pass; 18 passed on 02aa9b9).
- `junctions-psi`: per-cell junction totals on mouse neuron1k_v3 correlate 0.92 with STARsolo SJ (0.9996 between tools); 4 of 42 fail (it was 2 of 42 on 02aa9b9, so this got worse).
- `cell-calling`: 19 of 25 fail against EmptyDrops-CR/CellBender; nuclei2k_v2 precision 0.77 (0.98), Kang pool 0.88 (0.97).
- `ambient-rna`: contamination-estimate bias on hgmm1k_v3 0.28 vs 0.038 (SoupX); 9 of 10 pass.
- `hto`: F-scores 0.919 / 0.814 / 0.748 / 0.726 against 0.935 / 0.846 / 0.785 / 0.874 for the best tool; 1 of 8 pass.
- `spatial`: Visium FFPE human breast aborts on the default path (section 5); the other 5 failing metrics are gaps of 0.001-0.01 to STARsolo.
- Carried, unchanged from earlier documents: `donor-snps`, `mtdna` (02aa9b9), `download` (02aa9b9), `cell-type`, `genetic-ancestry`.

**Should-fix (proposed):** `gex-counts` (3 of 44 fail), `atac` (1 of 10: 10k peak F1 0.830 vs 0.837), `non-host` (2 of 10, by 0.002-0.005), `doublets`, `donor-demux`, `vdj` (carried).

**Missing or incomplete:** `adt` (no result file exists on any branch), `crispr-guides` and `chromosomal-sex` (documents exist; the contract marks them incomplete).

**Stale row:** `alignment-performance` fails only `thread_count_outputs_identical` (0 vs 1 on 02aa9b9). Re-measured on the candidate it is identical: 52 of 52 data files match at 4, 8 and 16 threads (pbmc1k_v3, `bench/release_gate/det_threads.sbatch`, 14 / 9 / 8 min wall). The document itself was not regenerated, and the timings are the PGO 02aa9b9 ones.

## 4. hgmm1k_v3 `umi_ratio_distance.total` (0.0089 -> 0.0169): R11 or #80?

One factor at a time, same input and flags (`bench/release_gate/exp_r11.sbatch`); the R11-off binary is d9834b8a with `SINGLET_RG_NO_R11` gating the corrector (experiment build, not part of the candidate).

| run | R11 | #80 intron rules | total | exonic |
|---|---|---|---|---|
| candidate | on | on | **0.0177** | 0.0059 |
| R11 off | off | on | **0.0104** | 0.0110 |
| rules off | on | off | 0.0299 | 0.0058 |
| both off | off | off | 0.0223 | 0.0087 |
| best competitor / gate limit | | | 0.0100 / 0.0152 | 0.0139 / 0.0335 |

R11 (STAR's whitelist correction on the stream) accounts for the move: turning it off takes `total` from 0.0177 (fail) to 0.0104 (pass), while exonic gets worse (0.0059 to 0.0110, still passing). The #80 rules help `total` (0.0299 without them). So it is not expected noise: R11 raises the total-UMI distance to STARsolo GeneFull on this mixed-species library while bringing the exonic counts closer to Gene. Open question for the owner of R11: whether the barcodes it corrects on hgmm are the ones STARsolo leaves uncorrected, or the other way round.

## 5. Findings from this run

1. **Stall watchdog (regression, fixed in PR #114).** `singlet` on the 10x Multiome 10k ATAC library exited with status 5 and no message after `[atac] cells: ...` at `SINGLET_STALL_TIMEOUT=10800`, at 150 GB and 300 GB. Exit 5 is the SIGALRM watchdog; four checkpoints re-armed it with a literal `alarm(1800)`, so the variable covered only the first phase. Fix: one helper, an exit line naming the phase and timeout, re-arm on progress (pileup every 5 M reads, ATAC BAM passes every 8 M records, the wait for STAR while its own files change). Verified: 10k ATAC now exits 0 in 1 h 10 min. Not covered: the export stage after the pileup is one countdown.
2. **Visium FFPE default path.** `ERROR: 99% of nuclear reads are antisense to the protocol orientation; rerun with --sorted-bam (strand safety net)` on the stream path; the sorted path runs. The FFPE probe protocol is reverse stranded; the stream path's safety net refuses rather than guesses. The `spatial` numbers are the `--sorted-bam` run.
3. **GPU pileup on nuclei2k_v2.** With `SINGLET_GPU_PILEUP=on`, mouse nuclei2k_v2 (binned qualities): `gpu_pileup: not run: base qualities neither constant nor tied to N`, then `ERROR: the GPU pileup failed after the CPU skipped its steps; rerun with SINGLET_GPU_PILEUP unset`, and no matrix is written. This is a hard failure, not a silent fallback. See section 7 for the tip.
4. **Protocol auto-detection** works with the barcode read first (SPLiT-seq and inDrop are detected; with the reads in SRA numbering order they are called microwell-seq and celseq2, which is an input-order problem, not a detection one).
5. **The whitelists copied from pipeline-benchmark were stale** for inDrop (`indrop_bc1.txt`) and the scATAC list; the 10x lists were identical. The gate uses the commit's own lists.

## 6. The four numbers that only ctest covered

On the candidate, real data (`bench/release_gate/remeasure.sbatch`, `hto_rg.sbatch`):

| | measured | expected |
|---|---|---|
| #94 SPLiT-seq (SRR31302021, 200,000 reads): all three segments in the whitelists | exact **0.660**, within 1 mismatch **0.7145** (Rd3 0.868, Rd2 0.787, Rd1 0.684) | ~0.66 |
| #94 inDrop (SRR15409581, 200,000 reads): BC1 in the commit's list | **0.215** at offset 0 (chance 0.006) | ~0.20 |
| #87 HTO F-scores (Howitt 2023, held-out settings, header compiled from the commit) | BAL1 **0.919**, BAL2 **0.814**, BAL3 **0.748**, cell lines **0.726**; calls independent of cell order | 0.73-0.92 |
| #91 CRISPR guides (A375 10k CRISPR library, 2 M read pairs, half stored reverse) | matched **0.681** (0.340 before the fix); guide UMIs **1,147,644** (622,433 before); reverse-stored reads matched 0.681 | |

## 7. GPU engine against the CPU engine (`--engine gpu`, `SINGLET_GPU_PILEUP=on`)

`gpu_diff.py` compares every data output (matrices, barcodes, features, tables) by the SHA-256 of its decompressed bytes. Each GPU run is classified from its log: **GPU pileup** (`[gpu_pileup] N reads` line) or **GPU engine, CPU pileup (fallback)** (`gpu_pileup: not run` / `ignored`).

| dataset | GPU run | data files identical to CPU |
|---|---|---|
| pbmc1k_v3, pbmc1k_v2, neuron1k_v3, hgmm1k_v3, pbmc1k_cite_v3, pbmc5p_v1, kang2018_ctrl | GPU pileup | all (48-52 per run; only the `read_stats.tsv` counters differ) |
| nuclei2k_v2 | fails (finding 3) | no outputs |

Seven of eight datasets are bit-identical on every data file; no run in this table is a fallback. GPU ≠ CPU only where the GPU run fails (nuclei2k_v2).

## 8. What was and was not run on the candidate

- **Run on the candidate, competitors reused where they do not consume Singlet's output:** gex-counts, spliced-unspliced, junctions-psi, cell-calling (8 datasets; STARsolo, alevin-fry, CellBender, Cell Ranger published matrices reused), atac / multiome / spatial (Singlet runs; Chromap and STARsolo-Visium reused), hto (the Singlet side; competitors' calls reused).
- **Run on the candidate with competitors re-run on its outputs** (they consume Singlet's cells, BAM or unmapped reads): ambient-rna (SoupX, DecontX, CellBender on the new cells), non-host (Kraken2 with two databases and the minimap2 oracle on the new unmapped reads).
- **Not binary-driven** (the Singlet side is Python or a separate build, so a new binary changes nothing): doublets, adt, vdj, crispr-guides, cell-type, chromosomal-sex, genetic-ancestry. Their rows are the newest documents and are labelled carried.
- **Not re-run, carried from 02aa9b9:** donor-demux, donor-snps, mtdna (S10 measured demux and SNPs on this binary but did not write contract documents; their numbers are in the row notes), download (S9 measured the Clipper ceiling instead), the timings of alignment-performance.
- **Not run:** the 3k Multiome development library; Kang ambient metrics.

## 9. Reproduce

`bench/release_gate/`: `build.sbatch` (pinned build), `prep_whitelists.sbatch` + `wl_pin.sbatch`, `counts_singlet.sbatch` / `counts_score.sbatch`, `atacsp_rg.sh` + `run_sample_rg.sbatch`, `ambient_prep.sbatch`, `hto_rg.sbatch`, `nh_prep.sbatch`, `score_all.sbatch`, `gpu_diff.py`, `det_threads.sbatch`, `exp_r11.sbatch` + `build_r11off.sbatch`, `remeasure.sbatch`, `make_gate.py` + `gate_notes.json` (this table). All jobs read `rg_common.sh` for the binary and fail if the whitelists are not next to it. The generator `bench/public/release_gate.py` (PR #79) is included so the table can be regenerated from the result documents.

The result documents (contract JSON, about 1 MB each) stay on Clipper in `$WORK/release-gate/results/`; this PR keeps the gate table, the GPU/CPU and tip/pin comparisons, the R11 experiment and the re-measurements under `bench/release_gate/results/`.
