# Non-host transcripts: which viral gene or ORF does a read come from

Status: Phase 1 prototype and benchmark (no engine change). Branch `bench/nonhost-transcripts`, scripts in `bench/expression_sota/nonhost/`,
numbers in `bench/expression_sota/nonhost/results/compare_dpi2.json`. Clipper outputs under `$WORK/expression-sota/nonhost/`
(`WORK=/mnt/projects/debruinz_project/debruinz/singlet`).

## Question

Singlet's non-host screen (k-mer classification of STAR-unmapped reads, `nonhost_per_cell.tsv`, `nonhost_summary.json`) answers *which species*.
It does not say which viral gene or ORF a read comes from, and the species FASTA needed for secondary alignment is missing
(`missing_reference_fasta` in `nonhost_summary.json`; for the dpi2 run it lists `species_8877.fna`), so a confirmed SARS-CoV-2 gets no alignment
or coverage. Can a second STAR step on the reads the screen keeps give a per-cell viral gene table that matches a combined-genome STARsolo
reference, and what does Kraken2 give for the same cells?

## Data and references

* Sample: ravindra2021 GSE166766 HBEC SARS-CoV-2 dpi2, 10x 3' v3, 50 M read pairs
  (`modality-bench/sj_nonhost/data/dpi2/R{1,2}.fastq.gz`). Singlet run: release-gate `perf/nonhost/gse166766_hbec_dpi2/singlet/` (11,567 called cells).
* Viral reference, built in job 794702 from NCBI (`GCF_009858895.2`, NC_045512.2, 29,903 nt, md5 `a95c1bc5...`): GFF3 genes ORF1ab, S, ORF3a, E, M, ORF6, ORF7a, ORF7b,
  ORF8, N, ORF10, plus the GFF3 5'UTR (1-265) and 3'UTR (29675-29903) as features. TRS-L core ACGAAC at 70-75 (leader = nt 1-75); TRS-B cores
  found as the nearest upstream ACGAAC of each ORF: S 21556, ORF3a 25385, E 26237, M 26473, ORF6 27041, ORF7a 27388, ORF8 27888, N 28260
  (`ref/viral/viral.trs.json`). ORF7b and ORF10 have no TRS-B in the table.
* Combined genome (job 794705): GRCh38-2024-A `genome.fa` + NC_045512.2, GENCODE v44 GTF + viral GTF, STAR 2.7.11b, sjdbOverhang 100,
  SAindexNbases 14. 28 GB index, 32:39 wall on 16 cores, peak 37 GB.

## Reference ("truth-like") counts: STARsolo on the full reads (job 794714)

STARsolo 2.7.11b, settings of `starsolo_public.sbatch` MODE=cr (Gene + GeneFull, EmptyDrops_CR, `1MM_multi_Nbase_pseudocounts`, `MultiGeneUMI_CR`,
`1MM_CR`, CellRanger4 clipping, score >= 30) on the combined genome with the viral genes as extra genes. 5:00 wall, 33 GB, 16 threads. 12,506 cells
called by EmptyDrops (11,552 of Singlet's 11,567 called cells are among them). All comparisons below are over Singlet's 11,567 called cells.

Reference viral UMIs in called cells, **Gene** feature (the one used for every comparison; GeneFull gives 117,317 instead of 94,589):

| gene | UMIs | cells | | gene | UMIs | cells |
|---|---|---|---|---|---|---|
| 3UTR | 38,128 | 1,885 | | ORF1ab | 2,381 | 790 |
| ORF10 | 32,892 | 1,155 | | S | 1,053 | 401 |
| N | 17,675 | 1,123 | | M | 1,028 | 277 |
| ORF7a | 519 | 226 | | ORF8 | 274 | 176 |
| ORF6 | 239 | 134 | | 5UTR | 212 | 118 |
| ORF3a | 176 | 124 | | ORF7b / E | 8 / 4 | 8 / 4 |

Gene-level counts are not the whole viral signal. Any viral-genome alignment with a corrected barcode and UMI gives 264,676 UMIs in 5,397 cells;
only 35.7 % of those UMIs are assigned by `Gene` (44.3 % by GeneFull) because reads that span two annotated features (N / ORF10 / 3'UTR, 24 nt gap
between N and ORF10) are dropped as multi-gene or intergenic. 93.8 % of all viral alignments start at or after nt 29,000.
**The 3' library sees the 3'-coterminal block, not individual ORFs**: 3UTR + ORF10 + N hold 94 % of the gene-assigned UMIs, and the "ORF10"
count is a position inside that block, not evidence of ORF10 expression. Counts for S, ORF3a, M, ORF6/7a/8 (each 0.1 to 2 %) come from reads
inside those ORFs (genomic RNA, internal priming) and cannot be attributed to a particular sgRNA.

### sgRNA identity from leader-TRS junction reads (reported separately)

`sgrna_junctions.py` takes primary STAR alignments to NC_045512.2 and calls a read sgRNA evidence when a CIGAR N gap of >= 500 nt has its donor within +-12 of the end of
the TRS-L core (nt 75) and its acceptor within [core start - 10, core end + 10] of a body TRS; any other acceptor is `noncanonical`. A second class,
soft-clipped reads that start at a body TRS and whose clipped 12 to 25 nt equal the leader tail, produced **0** reads. STAR did the junction alignment
directly (no chimeric mode needed: the gap is within STAR's default window).

Reference (called cells): **147 UMIs in 96 cells**, from 179 (barcode, sgRNA) rows over all barcodes. By sgRNA (UMIs): ORF6 51, N 42, M 14,
ORF3a 13, ORF7a 12, noncanonical 9, ORF8 3, E 2, S 1. The reads are leader-first: the alignment ends 20 to 35 nt past the TRS-B (median end
M 26,507, ORF6 27,072, N 28,294), i.e. these are the 5' ends of sgRNAs captured at the template-switch side, not 3' reads. That is the only per-cell evidence in this
data that distinguishes ORFs, and it is thin: 147 UMIs against 94,589 gene UMIs, 3 % of viral cells. Their relative abundances are not an
expression estimate (capture favours short transcripts: S, 4.3 kb, has 1 UMI). Not verified against an orthogonal assay, and some of these reads
may be RT template-switch artefacts rather than sgRNAs.

## Singlet-side prototype (job 794717)

1. Take `nonhost_reads.fq` of the existing run (689,660 cDNA reads kept by the screen: viral + fungal hits; written when a kingdom fraction exceeds 1 %)
   and pair each with its barcode mate from `star_Unmapped.out.mate2` by read name (`prep_kept_reads.py`; 100 % paired).
2. STARsolo against the **viral-only** genome (556 KB index) with the viral GTF; barcodes corrected against Singlet's called-cell list (1 mismatch,
   unique, `--soloCBmatchWLtype 1MM`), UMI dedup `1MM_CR`, `MultiGeneUMI_CR`, same clipping and score filter. Runtime: 2:35 for both variants below,
   2 GB peak RSS, 8 threads.
3. Variant `all_unmapped`: the same on every STAR-unmapped read (1,780,120), to measure what the k-mer screen leaves out.
4. Same `sgrna_junctions.py` on the prototype alignments.

### Results vs the reference (11,567 called cells, `Gene`)

| | kept by screen | all unmapped |
|---|---|---|
| viral UMIs recovered, sum min(ref, proto) / ref | 0.9991 | 0.9996 |
| total proto UMIs / ref UMIs | 0.9997 | 1.0008 |
| cells with >= 1 viral UMI (ref 3,293) | 3,301; recall 0.9979 | 3,320; recall 0.9991 |
| (cell, gene) pairs recall / precision (ref 6,421 pairs) | 0.9981 / 0.9966 | 0.9992 / 0.9938 |
| calls absent from reference: cells / UMI fraction | 0.45 % / 0.06 % | 0.9 % / 0.11 % |
| per-cell total viral UMI, Spearman / Pearson | 0.9962 / 1.000 | 0.9944 / 1.000 |
| per-cell (cell x gene table) Spearman | 0.9954 | 0.9938 |

Per-gene per-cell Spearman (kept): ORF1ab 0.998, S 1.000, ORF3a 1.000, M 0.998, ORF6 0.996, ORF7a 1.000, ORF8 1.000, N 0.997, ORF10 0.998,
3UTR 0.995, 5UTR 1.000; Pearson >= 0.9988 for every gene (E, ORF7b: 4 and 8 UMIs, identical). Full table in `results/compare_dpi2.json`.
The screen discards 547 of 553,582 STAR-alignable viral reads (0.10 %).

The agreement is high because both arms use the same aligner and UMI rules; what the check shows is that (a) the k-mer screen keeps essentially all viral reads
that STAR can place, and (b) aligning only the screened reads to the viral genome loses nothing against a host+viral genome. It does *not* test host/viral
competition on the reads the screen discards, nor sensitivity on reads STAR cannot align.

sgRNA concordance (prototype kept vs reference): 149 vs 147 UMIs, 98 vs 96 cells, (cell, sgRNA) pair recall 0.992 and precision 0.977, UMI overlap 0.993.
Differences: noncanonical +2, N +1, M -1 UMIs.

### Kraken2 on the same cells (viral DB k2_viral 2024-06-05, gate run, 1.8 s, 1.5 GB)

I reused the gate's read-level run on the same 1,780,120 STAR-unmapped reads (`kraken2_viral/out.txt.gz`, `percell.tsv.gz`) rather than rerunning an identical job.

* No gene or ORF output: the finest label is the taxon (species 694009 for SARS-CoV-2 related). Per-gene, per-ORF and sgRNA information: none.
* Read level: 553,246 reads called SARS-related vs 553,582 STAR-aligned; 552,305 shared (Kraken recall of STAR-aligned 0.9977; STAR recall of Kraken SARS reads 0.9983).
* Cell level against the alignment-level reference (264,676 UMIs, 5,397 cells): Kraken SARS-related 265,002 UMIs, 5,397 cells, cell recall 0.9956, precision 0.9956,
  per-cell Spearman 0.9944. Singlet's own species table: 263,657 UMIs, Spearman 0.9938. Prototype: UMI ratio 0.9998, Spearman 0.998. minimap2 oracle (gate): 234,740 UMIs, recall 0.896, Spearman 0.934.
* Kraken2 all-viral (any viral taxon, including HERV-K, BeAn virus) gives 9,265 cells, precision 0.58, Spearman 0.62: the extra cells are non-SARS taxa.
* Against the gene-level reference total (not the alignment-level one) all taxon-level tools look worse (Spearman 0.50 to 0.75, UMI ratio about 2.5 to 3): that gap is the
  64 % of viral UMIs that have no unique gene, not a classification error.

So at the species level Singlet's screen is as sensitive as Kraken2 on this sample. The new information is only the gene table.

## Design for the engine

**Output.** Register `nonhost_gene_counts` next to `nonhost_per_cell`: a sparse cell x feature table (`.1pz` plus `mtx` export) where features are
`<species_id>|<gene_id>` with a features file giving species, kingdom, gene name, coordinates, strand, and a `class` column (`gene`, `utr`, `terminal_block`). A second table
`nonhost_sgrna_junctions` (barcode, species, sgRNA, evidence, reads, umis) is emitted only when the genome carries a leader/TRS annotation (coronaviruses).
Per-sample summary additions in `nonhost_summary.json`: aligned reads per species, fraction assigned to a gene, coverage histogram (1 kb bins) per confirmed species.

**Step.** After EM confirmation, for each confirmed species (EM-confirmed only, as `per_cell_species` already is): take the classified reads that
`write_classified` already selects (same `nh_selected` records, so the table lines up with `nonhost_per_cell`), carry the barcode and UMI of the mate in the
record (the screen currently keeps only mate 1; the prototype joined by name from `Unmapped.out.mate2`), align with an in-process or child STAR to a small
per-run genome built from the confirmed species' FASTA + GTF, and count by the engine's existing barcode rule (R11 / called cells) and `1MM_CR` UMI collapse. Fixes the missing
`species_<id>.fna` defect at the same time: the bundle must ship the FASTA that the index entry points to.

**Reference bundle.** `nonhost/<species_id>.fna` + `.gtf` (+ `.trs.json` for coronaviruses), a per-species STAR index built on demand and cached (556 KB for SARS-CoV-2;
about 1 s to build). Sizes: not measured for the full viral/bacterial set; the existing screen indices already cost +25 GB RSS and
4.6 to 5.4 min per sample, whereas the extra step measured here costs 2 GB and under 3 minutes for 1.8 M reads (estimate, about 1 min per million reads including start-up and sorting; scale with the capped 20 M
reads: an extrapolation, not measured). Annotated bacterial genomes (RefSeq, GFF) fit the same format; most bacteria without gene models should get
a genome-only count (`contig` features) rather than none.

**Per-cell feature choice.** Because 94 % of 3' viral signal is the 3'-coterminal block, ship a `terminal_block` feature (last annotated gene to the genome end, here N + ORF10 + 3'UTR, nt 28,274-29,903) in
addition to the GFF genes, and count reads overlapping *only* that block there, so the 64 % of UMIs lost to boundary-spanning reads is recovered at block level. Gene-level columns are
then a diagnostic, not an expression estimate, and the table must say so (flag `coterminal_ambiguous`).

## What could go wrong

* **Multi-species reads and host homology.** The screen's reads are those STAR could not place on the host; a read that maps to both host and virus (HERV, BeAn/HERV-K hits seen in the species list) will be absent from the viral gene table or counted by the
  screen only. Aligning to a viral-only genome never lets host win; the combined-genome reference does. On dpi2 this changed nothing (UMI ratio 0.9997), but for endogenous-like
  viruses it will inflate. Mitigation: report both the screen-only and host-competed counts when the host genome is available, and require EM-confirmed species.
* **3' coterminal ambiguity.** nested sgRNAs share the 3' end; per-gene counts for S, ORF3a, M, ORF6 to 8 here are 0.1 to 2 % of viral UMIs and are not per-sgRNA expression.
  Only junction reads separate sgRNAs (147 UMIs here), they come from the 5' ends of short sgRNAs, and 5'/long-read data would be needed for quantification.
  Do not present gene columns as sgRNA abundance.
* **Annotation edge cases.** Reads spanning adjacent features are dropped by `Gene`; ORF10 and 3'UTR numbers are positional; overlapping genes (ORF7a/7b, ORF3a/E) are lost. The GFF gene set
  is the NCBI one, not curated sgRNA models.
* **Bacterial and fungal references.** Without gene models only contig-level counts are possible; fungal hits in the dpi2 run (125,515 reads) were never aligned to anything and the viral-only STAR step reports them
  as unaligned, which is a useful false-positive check but not a gene table.
* **Memory/time at scale.** Per-species indices must be built once and cached; building one for every confirmed species per sample is cheap for viruses but a multi-species gut sample could confirm hundreds.
  Cap species per sample (for example top 10 by EM abundance).
* **Barcode handling.** The prototype corrects barcodes against the called-cell list; ambient viral RNA in non-called barcodes is intentionally ignored, as in `nonhost_per_cell`.
* **Readout dependence.** The sgRNA junction reads show up in this 3' v3 library only as 5'-end captures; other chemistries (5' kits, long reads) will see more, and their donor/acceptor windows
  (+-12 / [-10, +10]) were set by hand for NC_045512.2 and not tuned.

## Not done / not verified

* No engine change; the prototype is scripts around STAR. The in-engine implementation, memory/runtime numbers for the 20 M read cap and for bacterial genomes are extrapolations from this single sample.
* One infected sample (dpi2) and one virus. No mock/PBMC run of the gene table (false-positive rate on uninfected cells is untested; the gate's mock run is the place to do it). No orthogonal sgRNA validation.
* Kraken2 was reused from the gate, not rerun; the database is the viral-only one (no host-homology false-positive control).
* Phase 2 note: an HIV-infected-cell 10x dataset with the authors' viral reference would test the same step on a retrovirus (spliced transcripts, LTR 5'/3' ambiguity). None identified or processed here.

## Jobs

| job | step | result |
|---|---|---|
| 794702 | viral reference + viral STAR index (NCBI download) | COMPLETED 1 s |
| 794705 | combined host+viral genome | COMPLETED 32:39, 37 GB |
| 794714 | reference STARsolo on full dpi2 | COMPLETED 5:05, 33 GB |
| 794717 | Singlet-side prototype (kept + all unmapped) | COMPLETED 2:35, 2 GB |
| 794789 | comparison, sgRNA tables | COMPLETED 0:27 (final `compare_dpi2.json`) |

Failed first attempts (fixed): 794699 (script upload), 794709 and 794715 (CB/UB tags need sorted BAM; BAM sort RAM). Job scripts: `build_viral.sbatch`, `build_combined.sbatch`
(the `--dependency` lines carry the job ids of this run), `ref_starsolo.sbatch`, `proto_singlet.sbatch`, `compare.sbatch`.
