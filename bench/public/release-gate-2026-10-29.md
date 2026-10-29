# Beta release gate: NOT releasable
1 of 22 capabilities pass. Policy: releasable only when every capability passes on real-data gating metrics; a waiver is explicit and listed.
Binary: not given; results computed with: 02aa9b938b54e0f92bba00c44c07b883a8660aa2, 19caf8cdef44512d237df2b69ea7168aeaa11232, 34a6f535839feb3bba8b9b883aed5490503b3988, 40f0b5d22d7a17a13f919aaab7424509d1852908, 5d39eb23a3ec3c414f4f7a17cee1b721fe1d6563, 5f9ba921f3523f074b2c4eed4dfe7742c3f5ac38, 88a41ffb7e4e1c8a2dc9f97a8e08958b7516a368, bf1f5741144eb845eddec94f0417931c4533e98e.

| capability | status | gating pass/fail/na | worst gap | fix owner |
|---|---|---|---|---|
| gex-counts | fail | 36/8/0 | adt_rna_spearman.total on pbmc1k_cite_v3: 0.4025 vs 0.4276 (starsolo-genefull) | pipeline-benchmark |
| spliced-unspliced | fail | 18/24/0 | gene_corr.unspliced on pbmc1k_v2: 0.8321 vs 0.9189 (starsolo-velocyto) | intron rule (fix/intron-rule-default) |
| junctions-psi | fail | 40/2/0 | percell_total_corr on neuron1k_v3: 0.9154 vs 0.9996 (starsolo-sj-cr) | pipeline-benchmark |
| cell-calling | fail | 5/20/0 | precision on kang2018_ctrl: 0.8822 vs 0.974 (starsolo-edcr) | pipeline-benchmark |
| ambient-rna | fail | 8/2/0 | bias_abs_log2 on hgmm1k_v2: 0.2019 vs 0.03161 (CellBender) | pipeline-benchmark |
| doublets | missing |  |  | doublet combiner (study/latent-qc, #43) |
| donor-demux | fail | 12/3/0 | ari on kang2018_ctrl: 0.9755 vs 0.99 (souporcell) | pipeline-benchmark |
| donor-snps | fail | 9/21/0 | genotype_concordance on kang2018_b1_c: 0.7549 vs 0.9897 (cellsnp-pileup) | base qualities in .1fq (fq-read-source) + SRA Lite verdict |
| mtdna | fail | 12/6/0 | haplotype_precision on kang2018_ctrl: 0.7891 vs 0.8938 (mgatk) | base qualities in .1fq (fq-read-source) + SRA Lite verdict |
| non-host | fail | 8/2/0 | percell_spearman_vs_oracle on ravindra2021_dpi2: 0.9334 vs 0.9354 (kraken2-viral) | pipeline-benchmark |
| alignment-performance | fail | 3/1/0 | thread_count_outputs_identical on pbmc1k_v3: 0 vs 1 (threshold) | engine-blockers / GPU engine |
| download | fail | 3/3/0 | aggregate_reads_s on sra_download_set: 6.12e+05 vs 1.133e+06 (singlet-prefetch-local) | engine-blockers |
| adt | missing |  |  | adt-modality |
| hto | fail | 2/6/0 | f_score on howitt2023_cl: 0.742 vs 0.8743 (bff-cluster) | HTO demux (adt-modality) |
| cell-type | fail | 9/3/0 | unseen_tissue_depth2 on cxg16: 0.5073 vs 0.6147 (scANVI) | ensemble-classifier |
| vdj | fail | 22/8/0 | bcr_correct_pairing on pbmc10k_5p_v2: 0.9511 vs 0.974 (trust4) | vdj-crispr |
| crispr-guides | incomplete | 21/0/0 |  | vdj-crispr |
| atac | fail | 9/1/0 | peak_f1 on pbmc_multiome10k: 0.8297 vs 0.8373 (chromap+snapatac2) | atac-multiome-spatial |
| multiome | pass | 2/0/0 |  |  |
| spatial | fail | 2/8/0 | spot_profile_pearson on visium_mouse_brain_ant: 0.9656 vs 0.9888 (starsolo) | atac-multiome-spatial |
| chromosomal-sex | incomplete | 5/0/0 |  | bench-public |
| genetic-ancestry | fail | 8/3/0 | accuracy.8m on mage150: 0.86 vs 0.9933 (peddy) | bench-public |

_Generated 2026-10-29 by `collect_results.py` + `release_gate.py` from the newest published result file of each capability (mixed commits, not a fresh re-run at one candidate). Fix owners: see the table; the doublets and adt files do not exist yet (doublets: bench/public-ambient-doublets has no results file; adt: claude/adt-modality has hto only)._
