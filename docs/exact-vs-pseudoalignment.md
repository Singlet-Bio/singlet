# Exact genome-wide alignment (Singlet) versus pseudoalignment (alevin-fry): what the extra cost buys

Status: **incomplete** (research beta wound down 2026-10-26). Real data and real truth only; nothing here is simulated.
Code: `bench/exact_vs_pseudo/`. Contract fragment: `bench/public/results/_exact_vs_pseudo_metrics.json` (validates against contract 1.1.0 as a draft).
Singlet: `singlet-campaign`, sha256 261d3a50ed62..., prod/campaign 02aa9b9. STARsolo 2.7.11b. alevin-fry 0.18.3 via simpleaf (piscem, splici index, USA, `cr-like`; `cr-like-em` as a second resolution).
Cell Ranger is **not run**: it is 10x's published 3.0.0 matrix (older references), with no runtime.

## Headline

1. **On the real-truth accuracy tests, Singlet does not beat alevin-fry by much, and STARsolo is the best of the runnable tools.** Singlet is better than alevin-fry on barnyard cross-species purity (0.41% vs 0.63% wrong-species UMIs) but slightly worse than STARsolo (0.33%). On chrY false positives in female-derived cells, STARsolo wins (5.2 per million human UMIs); Singlet (16.4) and alevin-fry (14.5) are alike. On CITE-seq RNA vs protein all tools tie within the confidence intervals.
2. **alevin-fry uses 1.7x (16 threads) to 1.8x (4 threads) less CPU than Singlet and about 5x faster on wall time at 16 threads, with 6x less memory.** That is the cost side of the frontier.
3. **Reach is where exactness pays.** Singlet produces junctions/PSI, SNP-based demux, mtDNA, non-host, poly(A), intergenic accounting and a BAM; alevin-fry produces count layers only (table below).
4. **Not measured** (no confirmed public matched data): long-read-validated spliced/unspliced, isoforms, junction truth; spike-ins (ERCC/SIRV). Spliced/unspliced accuracy is **not claimed**.

## 1. Accuracy against real truth

| Truth (real) | Dataset | Metric | Singlet | STARsolo | alevin-fry | alevin-fry EM | Cell Ranger (published) |
|---|---|---|---|---|---|---|---|
| Species of the cell | hgmm1k_v3 (495 human + 544 mouse pure cells) | wrong-species UMI fraction, exon+intron | 0.414% | **0.328%** | 0.633% | 0.622% | 0.345% |
| same | same | pure cells with >=1% wrong-species UMIs | 10.4% | **8.1%** | 18.1% | 16.7% | 10.0% |
| No Y chromosome (HEK293T) | hgmm1k_v3, 495 human cells | non-PAR chrY UMIs per million human UMIs | 16.4 | **5.2** | 14.5 | 10.7 | not comparable (hg19 ids) |
| Protein measured on the same cells | pbmc1k_cite_v3 | mean Spearman RNA vs CLR protein, 11 pairs (gex-counts definition) | 0.4215 | **0.4276** | 0.4149 | 0.4125 | 0.4028 |
| Combined | pooled | accuracy score (each component relative to the best tool, mean) | 0.70 | **1.00** | 0.62 | 0.66 | n/a |

The CITE row uses the definition of the published gex-counts metric (`adt_rna_spearman.total`: log1p(UMI per 10k), CLR over all antibodies, pairs every tool can score; same protein-gene map; no CD45RA/RO-to-PTPRC pairs). My first version paired CD45RA/RO with PTPRC and gave ~0.40 for all tools; that was dropped for the shared definition.
Differences on the CITE row are inside the paired bootstrap CIs (Singlet minus STARsolo about -0.006).

### Why Singlet is 0.41% and STARsolo 0.33% (like for like)

Singlet X is exon+intron, each molecule once; **no EM-rescued multimappers** are in it (`em_rescue.h` output is not written to the matrix in this build). Layers:

| Layer | Singlet | STARsolo | alevin-fry |
|---|---|---|---|
| exonic (Singlet exon, STARsolo `Gene`, alevin-fry S+A) | 0.398% | 0.358% | 0.689% |
| exon+intron (Singlet X, `GeneFull`, S+U+A) | 0.414% | 0.328% | 0.633% |
| intronic / unspliced (Singlet intron, Velocyto unspliced, alevin-fry U) | 0.474% | 0.271% | 0.428% |

Net Singlet minus STARsolo `GeneFull` = +30,950 wrong-species UMIs: about half from the **intronic layer** (+15.0k vs Velocyto unspliced), `Rn18s-rs5` (+4,968: STARsolo `GeneFull` drops it as multi-gene, Singlet's exon priority keeps it), `LINC00486` (+2,549), and a diffuse tail. That points to intronic assignment and the overlapping-gene rule, not to the exon layer. Unique-vs-rescued cannot be split from the outputs.

### alevin-fry's biggest error

88k UMIs of mouse `mt-Rnr2` in human cells (STARsolo 237): human mitochondrial 16S reads are placed on the mouse gene by k-mer matching. That single locus is 44% of alevin-fry's wrong-species UMIs.

### Where alevin-fry wins

* **LINC00486 (human, chr2) in mouse cells:** 15.5k (Singlet) and 13.0k (STARsolo) UMIs vs 0.6k (alevin-fry). The reads sit in one 5-kb window (chr2:32.915-32.920 Mb), at similar rates in human cells (23.9k molecules) and mouse cells (15.7k), with 2% mismatches in both; 90% of the mouse-cell reads do not map to the mouse genome alone. So the joint reference holds no homologous mouse region; the sequence comes from a repeat-like source in both species. Whether these molecules are real is unresolved. alevin-fry also drops them in human cells (1.2k vs 23.5k), so it avoids this by not counting them, not by knowing better. For this metric it wins.
* Memory (3.4 GB vs 23 GB) and time (below).
* On CITE-seq and chrY, alevin-fry is within noise of Singlet.

## 2. Reach (what a pseudoaligner cannot output)

| Output | Singlet | STARsolo | alevin-fry | Evidence |
|---|---|---|---|---|
| gene counts, exon+intron | yes | yes | yes | tables above |
| spliced/unspliced/ambiguous layers | yes | yes (Velocyto) | yes (USA) | no real-truth benchmark here |
| per-cell junction counts | yes | partial (SJ) | no | r=0.994 vs STARsolo for >=10 molecules (junctions-nonhost.md) |
| PSI | yes | no | no | cassette PSI r=0.996, MAE 0.008 |
| SNP counts + genotype-free donor demux | yes | partial (`--varVCF`) | no | ARI 0.986 vs vireo 0.923 (Kang, K given) |
| genotype doublets | partial | no | no | `prob_doublet` was 0 in the benchmarked build |
| mtDNA heteroplasmy | yes | no | no | per-cell VAF r=0.85; variant list weak before modality-fixes |
| non-host reads | yes | partial | no | Spearman 0.985 vs Kraken2 (SARS-CoV-2) |
| strand detection | partial | no | no | failed on 5' libraries (17% of atlas samples) |
| intergenic / antisense accounting | yes | partial | no | STAR places reads outside genes; piscem leaves them unmapped |
| poly(A) sites | yes | no | no | not externally benchmarked |
| ambient estimate | yes | no | no | |
| cell calling | yes | yes | partial | |
| BAM with CB/UB | yes | yes | no | |

Reads placed on pbmc1k_v3: STAR 96.1% (88.6% unique + 7.5% multi) vs piscem 87.4%. Placed is not correct; no truth exists for the extra reads.

## 3. Runtime (pbmc1k_v3, 66.6M pairs, same node b002, shared with other jobs; one run per cell)

| Tool | 16 threads: wall | CPU-h / 100M pairs | RSS | 4 threads: wall | CPU-h / 100M pairs |
|---|---|---|---|---|---|
| alevin-fry (`cr-like`) | 2.3 min | 0.80 | 3.4 GB | 7.7 min | 0.73 |
| alevin-fry + EM quant | +2 s | 0.81 | | +9 s | 0.75 |
| STARsolo (Gene+GeneFull+Velocyto) | 7.9 min | 1.67 | 19.4 GB | 18.2 min | 1.49 |
| Singlet incl. .1fq encode (2.2 min, 1 thread-ish; done by the download stage in the campaign) | 11.7 min | 1.34 | 22.8 GB | 19.9 min | 1.33 |
| Singlet excl. encode | 9.5 min | 1.21 | | 17.7 min | 1.22 |

CPU-h per 100M pairs is the Pareto x axis; wall time is secondary. Singlet is cheaper than STARsolo in CPU and on the CPU frontier next to alevin-fry; it is dominated on wall time at 16 threads.

### Pareto: what each tool produces for that time

| Tool | CPU-h / 100M (16t) | accuracy score | Outputs for that time |
|---|---|---|---|
| alevin-fry | 0.80 | 0.62 | USA counts, permit-list cells |
| STARsolo | 1.67 | 1.00 | Gene, GeneFull, Velocyto, (SJ if run), cell calls |
| Singlet | 1.34 | 0.70 | exon/intron/ambiguous, junctions + PSI, SNP counts, donor demux, mtDNA, non-host, cell calls, poly(A), QC counters, ambient |

## Missing or open

* Long-read-validated truth, spike-ins; one barnyard and one CITE dataset only (hgmm5k_v3 and hgmm1k_v2 were not run).
* Per-cell artifacts are not uploaded (no sha256/urls in the fragment); `reproduce.function` is not wired.
* Singlet fix leads: intronic assignment, overlapping-gene priority (`Rn18s-rs5`), chrY gametolog/repeat loci (`ENSG00000291032`: 277 UMIs vs 50 in STARsolo).
* Raw results on Clipper: `$WORK/exact-vs-pseudo/{runs,results/final,results/locus}` (WORK=/mnt/projects/debruinz_project/debruinz/singlet).
