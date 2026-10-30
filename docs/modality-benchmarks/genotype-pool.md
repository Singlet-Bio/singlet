# Genotype pool benchmark: donor demux, doublets, mtDNA

Date: 2026-10-24. Binary: `singlet-fixed-det` (the "fixed engine"). Scripts: `bench/modalities/genotype_pool/`. Raw results: `bench/modalities/genotype_pool/results/*.json`.

## Verdicts

| Modality | Verdict | One line |
|---|---|---|
| Donor demux, K given (8) | **better** than vireo | ARI 0.986 vs 0.92 |
| Donor demux, K auto | **broken** | picks 2 donors; ARI 0.18 |
| Doublets (expression score) | **worse** than scrublet | AUROC 0.83 vs 0.88; fixed 20% call rate |
| Doublets (genotype-aware) | **missing** | `prob_doublet` is hard-coded 0; vireo gets AUROC 0.95 |
| mtDNA per-cell VAF | **equivalent** where it reports | r = 0.85 vs pileup; 84% within 0.01 |
| mtDNA variant list | **worse** | 12% of calls supported by the pileup |
| mtDNA per-donor VCFs | **usable, imprecise** | all 28 donor pairs separated; precision 0.62, recall 0.52 |

## Dataset

Kang et al. 2018, GSE96583, batch 2 control lane `2.1` (SRR5398238). Eight lupus donors pooled, 10x 3' **v1** chemistry (14 bp barcode, 10 bp UMI). No v2 lane exists in this series. Singlet's encoder detects it as `10x-3p-v1`.

Truth: `GSE96583_batch2.total.tsne.df.tsv.gz` (`ind` = donor, `multiplets` = singlet / doublet / ambs, from demuxlet with genotypes). Control lane only: 14,619 cells, 12,315 singlets, 1,598 doublets among the 13,737 cells labelled singlet or doublet (11.6%). "ambs" cells are excluded from doublet scoring.

Reads: the archive holds only the submitted Cell Ranger 1.0 BAM (hg19, mapped reads, 23.8 GB). I rebuilt FASTQs from primary alignments: R1 = `CR`+`UR` (raw barcode and UMI), R2 = the read sequence. That gives 181,954,523 reads. Unmapped reads are not in the BAM, so they are lost. Singlet realigns to GRCh38-2024-A; the hg19 coordinates are not used.

## Commands

```
fetch_convert.sbatch   # parallel range download, samtools fastq -T CR,CQ,UR,UQ, singlet encode -> kang21.1fq
run_singlet.sbatch k8 8      # singlet-fixed-det kang21.1fq --pipeline --snps PANEL --n-donors 8 --tagged-bam \
run_singlet.sbatch kauto -1  #   --bam-sort-mode disk --output-format mtx --raw-matrix --threads 16   (200 GB)
ref_demux.sbatch k8          # cellsnp-lite mode 1a (-R PANEL --minMAF 0.1 --minCOUNT 20) then vireo -N 8
vireo_seeds.sbatch k8        # vireo, 5 seeds, --nInit 200
doublets.sbatch k8 VIREO     # scrublet on Singlet's counts of the called cells
ref_mt.sbatch k8             # pysam chrM pileup (mt_pileup.py); mgatk attempted
eval_demux.sbatch / mt_eval.sbatch
```

PANEL = `/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz` (1000G phase 3, AF >= 5%, 7.35 M SNPs). Same panel for Singlet and cellsnp-lite. All tools see the same cells: the 14,404 cells Singlet called (STARsolo filtered list) and the same tagged BAM (`CB`/`UB`).

Singlet K=8: 30 min wall end to end on 16 threads. Without `--bam-sort-mode disk` STAR's BAM sort was OOM-killed at 80 GB. Peak memory with disk sort was 182 GB (sacct); `summary.json` reports 18 GB.

## 1. Donor demultiplexing

Singlet donors are matched to truth donors by Hungarian matching. ARI is over all 12,315 truth singlets, with unassigned and doublet calls counted as their own labels.

| Tool | ARI | ARI (assigned only) | Accuracy (assigned) | Unassigned | Singlets called doublet | Lowest per-donor recall | Time |
|---|---|---|---|---|---|---|---|
| Singlet, `--n-donors 8` | **0.986** | 0.987 | **0.993** | 0.2% | 0% | 0.93 | in 30 min run |
| vireo -N 8 (default seed) | 0.923 | 0.957 | 0.948 | 2.0% | 1.5% | 0.00 | 11.7 min cellsnp + 0.5 min vireo |
| vireo, 5 seeds, nInit 200 | 0.919-0.920 | 0.958-0.959 | 0.949-0.950 | 2.0-2.1% | 2.1-2.2% | 0.00 | about 1 min each |
| Singlet, `--n-donors -1` | 0.179 | 0.178 | 0.393 | 0.1% | 0% | 0.00 | in 30 min run |

Vireo merges or splits donors so that one truth donor (1039) is never matched; this is stable across seeds. Singlet separates all eight (lowest recall 0.93, donor 1039).

Auto-K fails. The EM+BIC step reports BIC rising monotonically with K (3.7e6 at K=1 to 2.8e7 at K=8) and keeps K = 1 or 2 (`donor_demux.h:306-342`); the penalty `n_params * log(N)` dominates. Telling Singlet the true K works.

Doublets are not modelled in the demux. Every assignment has `prob_doublet = 0.0f` (`donor_demux.h:1095-1098`), so the label "doublet" is never produced. See section 2.

## 2. Doublets

Truth: heterotypic cross-donor doublets (demuxlet) vs singlets. Homotypic doublets are invisible here, so all precision numbers are lower bounds on real precision for expression-based tools. "At truth rate" means the top 11.6% of cells by score.

| Tool | AUROC | AUPRC | Recall at truth rate | Native calls | Native recall | Native precision |
|---|---|---|---|---|---|---|
| Singlet `doublet_scores.tsv` | 0.829 | 0.519 | 0.558 | 2,748 (20.0%) | 0.706 | 0.410 |
| scrublet (rate set to truth) | **0.881** | **0.588** | **0.618** | 1,296 (9.4%) | 0.549 | 0.677 |
| vireo genotype doublet prob (reference, not a peer) | 0.950 | 0.842 | 0.840 | 1,535 | 0.789 | 0.822 |
| UMI count alone (baseline) | 0.801 | 0.320 | 0.319 | - | - | - |

Singlet's score beats UMI count alone but trails scrublet. It always calls about 20% of cells (here 19.9%), whatever the data, and its score is heavily tied (10,502 of 14,404 cells score exactly 0), so ranking at a chosen rate is coarse. scDblFinder was not run (no R on Clipper); scrublet is the substitute.

Genotype-based doublet calls are the strongest signal in a genotyped pool and Singlet does not produce them (section 1). A fix needs a doublet term in the VB posterior (a two-donor mixture likelihood per cell) and `prob_doublet` filled in `assign_from_probs` (`donor_demux.h:1089-1099`).

## 3. mtDNA

Reference: a read-level pysam pileup of chrM in the same tagged BAM, same filters as Singlet (MAPQ >= 20, base quality >= 10). mgatk 0.9.0 installed but refuses to start without R ("cannot find R in environment"); it was not run. The chrM pileup used 2.29 M reads in 14,385 of the 14,404 cells; median 13.6 k chrM bases per cell.

**A. Per-cell VAF fidelity.** Singlet's `mt_heteroplasmy` entries (438,446, of which 398,124 have pileup coverage) vs pileup VAF using Singlet's own definition (best non-reference allele over depth):

| Metric | Value |
|---|---|
| Pearson r | 0.850 |
| Mean absolute difference | 0.0076 |
| Entries within 0.01 | 84.2% |

**B. Variant calling.** Singlet reports 2,396 variable positions (`mt_variants.tsv`). The pooled pileup supports 1,329 alt alleles at >= 3 reads and >= 1% AF. Only 285 Singlet calls (11.9%) are in that set. Of 205 strong pooled variants (>= 10 alt reads, AF >= 5%), Singlet recovers 19 (9.3%). 1,193 of its variants have >= 50 "heteroplasmic" cells, which is not credible for PBMCs and points to systematic error or multi-mapped reads (Singlet's chrM BAM has 3.03 M reads, many with MAPQ 3; whether the MAPQ filter applies to `pileup_mt_bases` is not verified). Cause not isolated.

**C. Donor separation.** Truth haplotypes: per truth donor, pseudobulk positions with depth >= 20 and AF >= 0.9 vs rCRS (10-27 variants per donor). Compared with Singlet's per-donor `mt/donorN_mt_variants.vcf` (18-30 variants), restricted to positions the pileup covers:

| Metric | Value |
|---|---|
| Mean precision vs truth haplotype | 0.62 |
| Mean recall | 0.52 |
| Donor pairs separable by mt haplotype, reference | 28 of 28 |
| Donor pairs separable, Singlet VCFs | 28 of 28 (agreement 1.0) |
| Smallest reference pair difference | 14 variants (1016 vs 1039) |

So mtDNA haplotypes do separate all eight donors consistently with the genotype truth, and Singlet's per-donor VCFs preserve that. The per-donor VCFs have no allele fraction or depth fields, which limits their use.

The per-cell heteroplasmy matrix cannot separate donors: `compute_heteroplasmy` keeps only VAF 0.01-0.95 (`mt_heteroplasmy.h:134-135`), so haplotype-defining variants (VAF near 0 or 1) are dropped by design. It also infers the reference from the pool majority (`export.h:867-869` passes no reference), not rCRS. Here the pool majority differs from rCRS at 16 well-covered positions.

## Defects and gaps found (no C++ changed)

1. **No donor outputs without `--snps`.** Demux, `snp_ad`/`snp_dp`, ASE and per-donor mt outputs all need it (`singlet.cpp:2665`; `export.h:876, 940, 949, 1224`). I listed `pipeline-benchmark/results/*/` and found no `donor/` directory, and `atlas_rerun/perf_run.sbatch` has no `--snps`. The older campaign default therefore produced no demux. I did not audit every campaign output.
2. **Auto-K broken** (section 1). Real data with unknown donor count gets K = 1-2.
3. **No doublet model in demux** (`donor_demux.h:1095-1098`).
4. **Doublet call rate is fixed near 20%.**
5. **No raw chrM allele counts in `--pipeline` mode** (`export.h:1167`); only called variants and VAFs. Re-analysis (for example with other callers) needs the BAM.
6. **Misnamed files.** `mt_heteroplasmy.mtx`, `mt_heteroplasmy_features.tsv` and `mt_heteroplasmy_barcodes.tsv` are gzip data. The top-level `donor_assignments.tsv` is a header-only stub (`export.h:1617`); the real file is `donor/donor_assignments.tsv` (all 155,019 barcodes, with `cell_status` cell / ambient). `filtered_feature_bc_matrix/matrix.mtx.gz` has 155,019 columns, not the 14,404 called cells.
7. **Memory.** Default in-memory BAM sort OOM-kills at 80 GB for 182 M reads; use `--bam-sort-mode disk`.

## Caveats

- One sample, one chemistry (v1, not v2), 8 donors. Conclusions on K selection and doublets may differ at other pool sizes.
- Truth is demuxlet's genotype-based call on the same data, not an independent array. Truth singlet and doublet labels carry their own error. Vireo's lowest donor recall of 0 reflects a cluster-to-donor mismatch, not necessarily poor SNP signal.
- Reads were reconstructed from a mapped-only BAM: unmapped reads are missing, and read count (182 M) is about twice the 90 M the archive reports; the cause was not checked.
- Reference tools used default or documented settings; vireo was not tuned beyond seeds and restarts, and no ELBO-based seed pick was made (vireo logs no ELBO). Auto-K was not run for vireo (it has no K search).
- souporcell, scDblFinder and mgatk were not run. The pysam pileup stands in for mgatk and shares Singlet's filters, so section 3A measures counting fidelity, not an independent caller.
- mtDNA sections B and C use my own thresholds (3 reads / 1% AF; depth 20 / AF 0.9); the numbers move with them.
- Runtime comparison is rough: Singlet's 30 min includes alignment and everything else, while vireo's time is pileup plus clustering only.
