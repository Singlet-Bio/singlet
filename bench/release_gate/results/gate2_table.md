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
