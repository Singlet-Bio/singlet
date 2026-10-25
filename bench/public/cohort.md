# Singlet 1.0.0 public benchmark cohort

The table is `cohort.tsv`, generated from `make_cohort.py`. It has one row per dataset, with exact accessions, ground
truth, licence, size, the capabilities it scores and the tools run on it. Every URL, accession, size and cell count was
checked on 2026-10-25:
- 10x dataset pages and HEAD requests (Content-Length);
- NCBI GEO text pages;
- the ENA portal API.

A blank `n_cells` means the source does not state one.

**Version 2026-10-25b: 50 rows (45 core, 1 development, 4 extended).** Changes since the first version:
- Added on request from the capability sessions:
  - howitt2023_bal1/2/3 and howitt2023_cl (HTO with genotype truth);
  - kang2018_b1_a/b/c (held-out pools, true K = 4, 4, 8);
  - hgmm1k_v2 (held-out barnyard; hgmm_5k_v3 FASTQs return 403);
  - gsm5815984 (489M pairs, depth);
  - gse145370_tcr, plus V(D)J on pbmc5p_v1;
  - tenx_crispr_a549_5k;
  - atac_pbmc_5k_nextgem;
  - pbmc_multiome3k (tier `development`: the ATAC session tunes on it).
- New column `data_kind`.
- 10x software is not run: its published outputs are the comparator.

## Real data only

Zach's rule: every published claim rests on real data with real ground truth. A simulated or otherwise derived result
may only supplement a real-data result that shows the same effect. `contract.py` enforces it (CONTRACT.md 3.1).

`data_kind` per row:
- `real`: reads as deposited, including read caps and format conversion (Kang is rebuilt from its BAM).
- `derived`: real reads recombined or thinned.
- `simulated`: none in this cohort.

The two derived rows, and what corroborates them:
- **mage150**: real MAGE RNA-seq reads, thinned to single-cell donor depth. Corroborated by:
  - osorio2019_lcl_mix: a real pool of two 1000 Genomes individuals of different ancestry, run end to end through
    Singlet's demultiplexing and per-donor ancestry;
  - the real atlas donors with self-reported ancestry (docs/cell_annotation.md 5.2).
  The thinned results only count because those real single-cell results agree with them.
- **lcl_gse230239**: an in-silico pool of two real libraries. Corroborated by geo_sex21 (real samples, GEO sex) for sex,
  and by osorio2019_lcl_mix for ancestry.

The earlier ancestry simulation (docs/cell_annotation.md 5.1) is not used on the page.

## How datasets were chosen

1. **Popular.** We prefer datasets that the competitors' own papers and tutorials report on: the 10x PBMC 1k/10k sets,
   hgmm, Kang 2018, Stoeckius 2018, Norman 2019, Ding 2020, the Multiome PBMC 10k and Visium mouse brain. A reader can then
   check our competitor numbers against published ones.
2. **Truth that no tool produced.** Every capability needs at least one dataset whose answer does not come from a
   competitor:
   - species of each read (barnyards);
   - genotype-defined donors and doublets (Kang; hashtags confirmed by genotype);
   - known 1000 Genomes individuals for sex and ancestry;
   - a known virus for non-host;
   - guide identities;
   - curated cell types (CELLxGENE, Ding 2020).
   Agreement with Cell Ranger or STARsolo on identical reads is the second line of evidence.
3. **Every chemistry Singlet claims to read:**
   - 10x 3' v1 (Kang), v2, v3, v3.1 and GEM-X v4;
   - 10x 5' v1 and v2;
   - nuclei;
   - Multiome GEX + ATAC;
   - Visium;
   - Drop-seq, inDrops, Seq-Well, CEL-Seq2 and Smart-seq2;
   - CITE-seq, hashtags and CMO;
   - CRISPR direct capture and guide amplicons;
   - V(D)J.
4. **Fits the budget.** The core tier runs on Clipper in about 19 h at 12 concurrent 16-thread tasks (below). Datasets
   with hundreds of millions of reads are capped at a read count (`read_pairs_used_m`), as the competitors' benchmarks
   usually do, and the cap applies to every tool.
5. **Held out.** No dataset here was used to tune a Singlet parameter, with three exceptions:
   - hgmm1k_v3 and kang2018_ctrl informed the ambient, doublet and demux fixes (`docs/modality-benchmarks/`). Results on
     those two must say so in `caveats`, or be confirmed on hgmm10k_v3 and stoeckius2018_hashing, which were not used.
   - The cell-type guide's six development studies are excluded from cxg16.
   - geo_sex21 was the validation set of the sex model and was not used to fit it. Its parameters are fixed by
     construction: the XIST share Q = 0.97 / 0.03.

## Datasets by group

| group | datasets (core; * = extended tier) | ground truth |
|---|---|---|
| 10x 3' | pbmc1k_v2, pbmc1k_v3, pbmc5k_v4_d1 | identical-read comparison with STARsolo, alevin-fry, Cell Ranger |
| 10x 5' | pbmc5p_v1 (first 60M pairs), pbmc10k_5p_v2 (GEX + TCR + BCR + ADT) | as above; Cell Ranger vdj |
| nuclei | nuclei2k_v2, mouse_brain_nuclei5k_v31 | as above; Cell Ranger 7 counts introns too |
| Multiome | pbmc_multiome10k | Cell Ranger ARC fragments, peaks, joint calls |
| CITE-seq / hashing | pbmc1k_cite_v3, pbmc10k_cite_v3, stoeckius2018_hashing, tenx_cellplex_pbmc10k* | hashtag = donor, confirmed by genotype; protein markers |
| genotype pool | kang2018_ctrl | demuxlet calls with genotypes |
| barnyard | hgmm1k_v3, hgmm10k_v3, macosko2015_dropseq_mix | species of each read |
| CRISPR | tenx_crispr_a375_10k, norman2019_gem1, replogle2022_k562_gg41* | published guide identities, target knock-down |
| spatial | visium_mouse_brain_ant | Space Ranger, anatomy |
| other protocols | ding2020_pbmc1_{smartseq2, dropseq, indrops, seqwell, celseq2, 10xv3}, segerstolpe2016_pancreas* | one PBMC sample, seven methods, curated cell types |
| mouse | neuron1k_v3, heart1k_v3, (nuclei above), tabula_muris_kidney_facs* | Cell Ranger; Tabula Muris CL labels |
| curated cell types | cxg16 (16 GEO studies, 66 samples) | CELLxGENE curator labels transferred by barcode |
| non-host | ravindra2021_mock, ravindra2021_dpi2 | SARS-CoV-2 infected vs mock |
| sex / ancestry | geo_sex21 (19 studies), lcl_gse230239, osorio2019_lcl_mix, mage150 | GEO sex; 1000 Genomes panel (population, super-population, sex) |

## Capability coverage

| capability | n | datasets |
|---|---|---|
| `gex-counts` | 25 | every 10x, Drop-seq, inDrops, Seq-Well, CEL-Seq2, Smart-seq2, Visium and mouse row |
| `spliced-unspliced` | 6 | pbmc1k_v2, pbmc1k_v3, pbmc5k_v4_d1, nuclei2k_v2, mouse_brain_nuclei5k_v31, neuron1k_v3 |
| `junctions-psi` | 4 | pbmc1k_v3, pbmc5k_v4_d1, nuclei2k_v2, ding2020_pbmc1_smartseq2 (full length) |
| `cell-calling` | 23 | 10x rows, barnyards (truth), hashing (truth), Ding protocols |
| `ambient-rna` | 8 | barnyards (truth), PBMC, nuclei, heart |
| `doublets` | 6 | hgmm1k/10k, Drop-seq mix (species truth), kang2018_ctrl (genotype truth), stoeckius (hashtag truth), cellplex* |
| `donor-demux` | 5 | kang2018_ctrl (8 donors), stoeckius, lcl_gse230239 (in-silico pool), osorio2019_lcl_mix, cellplex* |
| `donor-snps` | 3 | lcl_gse230239 and osorio2019_lcl_mix (1000 Genomes genotypes are the truth), kang2018_ctrl |
| `mtdna` | 3 | kang2018_ctrl, lcl_gse230239, osorio2019_lcl_mix (distinct donors' haplogroups) |
| `non-host` | 3 | ravindra mock / dpi2, pbmc1k_v3 (clean) |
| `alignment-performance` | 4 | pbmc1k_v2, pbmc1k_v3, pbmc5k_v4_d1, pbmc5p_v1; every run also reports resources |
| `adt` / `hto` | 4 / 2 | CITE rows, 5' rows / stoeckius, cellplex* |
| `cell-type` | 13 | cxg16, Ding protocols, CITE 10k, mouse rows |
| `vdj` | 1 | pbmc10k_5p_v2. The V(D)J session may add vdj_v1_hs_pbmc2 t/b |
| `crispr-guides` | 3 | 10x A375, Norman, Replogle* |
| `atac`, `multiome`, `spatial` | 1 each | pbmc_multiome10k; visium_mouse_brain_ant. The ATAC/spatial session may add rows |
| `chromosomal-sex` | 8 | geo_sex21 (GEO metadata), LCLs and MAGE (1000 Genomes sex), kang, multiome, cxg16, pbmc5k_v4_d1 |
| `genetic-ancestry` | 4 | mage150 (26 populations), lcl_gse230239, osorio2019_lcl_mix, kang2018_ctrl |

Thin rows (vdj, atac, multiome, spatial with one dataset each) are owned by the V(D)J/CRISPR and ATAC/spatial
sessions. Add rows in `make_cohort.py` and regenerate.

## Budget (core tier, `python bench/public/cohort_plan.py --markdown`)

| dataset | read pairs used (M) | FASTQ GB | compute task-h | longest task h |
|---|---|---|---|---|
| pbmc1k_v2 | 77 | 6.4 | 1.6 | 0.8 |
| pbmc1k_v3 | 66.6 | 5.5 | 1.5 | 0.7 |
| pbmc5k_v4_d1 | 223 | 19.1 | 3.8 | 2.0 |
| pbmc5p_v1 | 60 | 42.9 | 1.4 | 0.7 |
| pbmc10k_5p_v2 | 300 | 63.1 | 6.3 | 2.6 |
| nuclei2k_v2 | 153 | 12.7 | 2.7 | 1.4 |
| mouse_brain_nuclei5k_v31 | 205 | 19.8 | 3.5 | 1.8 |
| pbmc_multiome10k | 600 | 95.6 | 9.9 | 5.0 |
| pbmc1k_cite_v3 | 56 | 4.7 | 2.3 | 0.6 |
| pbmc10k_cite_v3 | 345 | 28.6 | 6.5 | 3.0 |
| stoeckius2018_hashing | 386 | 97.7 | 3.8 | 1.3 |
| kang2018_ctrl | 182 | 23.8 | 3.7 | 0.7 |
| hgmm1k_v3 | 77 | 4.9 | 4.1 | 0.8 |
| hgmm10k_v3 | 675 | 56.1 | 12.9 | 5.6 |
| macosko2015_dropseq_mix | 268 | 17.5 | 3.3 | 1.3 |
| tenx_crispr_a375_10k | 760 | 63.7 | 11.6 | 6.3 |
| norman2019_gem1 | 540 | 24.0 | 8.0 | 4.5 |
| visium_mouse_brain_ant | 370 | 31.0 | 5.6 | 3.2 |
| ding2020_pbmc1 (6 protocols) | 2,584 | 99.4 | 21.9 | 2.2 |
| neuron1k_v3, heart1k_v3 | 163 | 13.5 | 3.3 | 0.9 |
| cxg16 (66 samples) | | | 59.4 | 0.9 |
| ravindra2021 mock + dpi2 | 100 | 9.0 | 2.6 | 0.5 |
| geo_sex21 (atlas counts) | | | 8.4 | 0.4 |
| lcl_gse230239 | 650 | 24 | 4.6 | 2.1 |
| osorio2019_lcl_mix | 150 | 22 | 3.6 | 0.6 |
| mage150 (8M pairs x 150) | 1,200 | 362 (about 130 streamed) | 30.0 | 0.2 |
| **total, 33 datasets** | | **1,147** | **226** (~3,600 core-h) | |

**Updated budget (50 rows, 10x software not run): 45 core datasets, 1,255 GB, 215 task-hours, about 18 h at 12 concurrent
tasks.** The table below is the first version's; `python bench/public/cohort_plan.py --markdown` prints the current one.

**Wall clock (first version): about 19 h at 12 concurrent 16-thread tasks** (longest-first list scheduling), plus about 1 h for the
largest download. The extended tier adds 4 datasets and 26 task-hours (21 h in total).

The cost model is in `cohort_plan.py`. Its coefficients are the pipeline-benchmark `time -v` measurements, rounded up.
10x software (Cell Ranger, ARC, ATAC, vdj, Space Ranger) is not run. The outputs 10x published with each dataset are
the comparator, with their version and source URL recorded (Zach, 2026-10-25). The largest single tasks (hgmm10k, A375 CRISPR,
multiome) take under 6.5 h, so they need the `cpu` partition or a read cap to fit `short-cpu`'s 4 h limit.

## Known gaps and decisions

- **Visium human lymph node** has no public reads ("to maintain donor anonymity"). Zach: pick another Visium dataset
  with reads. visium_ffpe_human_breast (Space Ranger 1.3, FFPE probe chemistry, 5.3 GB) is the only human Visium set
  10x serves with FASTQs; it joins visium_mouse_brain_ant.
- **Whitelists (found by the ATAC session).** Singlet's bundled `737K-cratac-v1.txt` is byte-identical to the 3' v2 list
  `737K-august-2016.txt`. The real scATAC list was rebuilt from 12 public Cell Ranger ATAC runs (736,795 barcodes). The
  ARC ATAC-to-GEX pair table was rebuilt from 7 public ARC runs. Both are owned by the ATAC session and must replace the
  bundled files before atac/multiome results are final.
- **No public pooled scRNA-seq of many 1000 Genomes individuals exists.** Ancestry therefore uses three sources:
  - **MAGE** (Taylor et al. 2024): bulk RNA-seq of 731 1000 Genomes LCLs from all 26 populations, of which we use 150 (30
    per super-population, sex-balanced). Reads are thinned to single-cell donor depths.
  - **Two scRNA-seq LCL studies:** HG00362, HG01277, GM12878 and GM18502. They test the full pipeline: demultiplexing,
    per-donor depth tables and the ancestry call.
  - **The earlier simulation** (`docs/cell_annotation.md` 5.1).
  MAGE note: the HG00237 / NA11919 run swap is corrected in ENA (SRR37907958/9), and both individuals are excluded.
- **Leakage control for ancestry.** The 150 MAGE individuals and the 4 LCL individuals are removed from the allele
  frequency table Singlet uses, and from somalier's labelled 1000 Genomes background. peddy's built-in background cannot be
  changed, so peddy keeps its advantage. That is conservative for Singlet.
- **Stoeckius 2018** deposited no genotype calls. Truth is the hashtag, confirmed per cell by genotype demultiplexing
  (consensus of vireo and souporcell, both independent of Singlet). Its MixCellLines RNA run is empty in SRA and is not
  used.
- **Kang 2018** is deposited only as a Cell Ranger 1.x BAM (hg19). FASTQs are rebuilt from CR/UR tags, so unmapped reads
  are lost, the same for every tool. GEO does not state the chemistry; Singlet's encoder detects 10x 3' v1.
- **Replogle 2022:** that one SRA "lane" is one GEM group is unverified. The extended tier keeps it until the paper's
  metadata confirms it.
- **Licences.** 10x datasets are CC BY 4.0, and so are CELLxGENE and the Replogle figshare data. GEO, SRA and ArrayExpress
  datasets state no licence: they are public, and we cite the paper. Singlet outputs are CC0.
