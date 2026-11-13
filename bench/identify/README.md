# singlet identify: development panel

`singlet identify SOURCE` reads the first reads of a dataset and reports the library family, protocol tag, an action and the ranked
alternatives. This folder holds the development half of the truth panel, the scripts that run it on the cluster and the scored results.
Real public runs only. The held-out candidates are listed and not scored.

## Files

- `panel_dev.tsv`: 76 runs with family, protocol, species, label source and label tier. Tier T2 is a depositor statement confirmed by a second
  source, D is the sample name of a design-controlled benchmark (Mereu 2020, Ding 2020), V is a vendor dataset name, and T2 on the
  feature-barcode and Visium rows is the GEO sample title. Rows whose only label came from a catalog read-call are not included.
- `sources_dev.tsv`: what `singlet identify --list` reads (id, source, hint, genome index). `$DATA` and `$REF` are cluster directories.
- `make_panel.py`, `build.sbatch`, `run_panel.sbatch`, `score.py`: build the panel, build the binary, run the batch, score it.
- `results_dev/summary.tsv`, `results_dev/results.md`: the scored run (binary built from the commit named in the PR).
- `heldout_candidates.tsv`: 15 runs set aside for the held-out half. They were not run through the tool and were not used to set any threshold.

## Head reader, by source type

| source | route | measured |
|---|---|---|
| SRA accession | archive spot access with a row limit (`SraReader`, libncbi-vdb fetches blocks as the cursor advances) | 200,000 spots in a median of 3.8 s, at most 84 s, over 69 runs; transfer size is not exposed by the reader |
| ENA FASTQ over HTTPS | ranged GET piped through an in-process gzip reader, closed at the spot limit | 200,000 spots in 17 to 44 s, 8 to 33 MB, on 4 runs |
| local FASTQ | same reader on files | 0.2 to 0.4 s |
| BAM | samtools view, CR and UR tags then the read | implemented; not run on a real BAM in this phase |
| staged `.1fq` | the existing reader | implemented; not run in this phase |
| submitted FASTQ or BAM on NCBI `sra-pub-src-*` | no head route yet | ENA lists no submitted file for the runs tried; an earlier Python preflight found the S3 range route returned nothing for all of its 77 runs |

SRA spot access keeps every segment of a spot, including index reads, so it shows the real layout. ENA FASTQ drops technical reads for some
runs. Accessions therefore default to the archive route and `--route ena` is the validator for the endpoint.

## Layers

A layout, B barcode lists (8 lists, both orientations for the ATAC lists) and well-barcode windows, C insert content, D alignment of the head's
cDNA reads with the bundled STAR (`--genome-dir`; 50,000 reads, 13 s median including the index load), E usefulness with its components.
All thresholds are in `Thresholds` in `include/singlet/fq/identify.h`, each with the measurement that set it.

# Results

Scored 76 of 76 panel rows (0 without a result).

## Confusion matrix (rows: label family, columns: verdict)

| label \ verdict | atac | bulk_rna | droplet_3p | droplet_5p | droplet_other | feature_barcode | multiome_atac | multiome_gex | plate_barcoded | smartseq | unknown | visium | total | correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atac | 1 |  |  |  |  |  |  |  |  |  |  |  | 1 | 1 |
| bulk_rna |  | 3 |  |  |  |  |  |  |  |  |  |  | 3 | 3 |
| droplet_3p |  | 15 | 11 |  |  |  |  | 1 |  |  |  |  | 27 | 11 |
| droplet_5p |  |  |  | 3 |  |  |  |  |  |  |  |  | 3 | 3 |
| droplet_other |  | 3 |  |  | 5 |  |  |  |  |  | 1 |  | 9 | 5 |
| feature_barcode |  |  |  |  |  | 6 |  |  |  |  |  |  | 6 | 6 |
| multiome_atac |  |  |  |  |  |  | 2 |  |  |  |  |  | 2 | 2 |
| multiome_gex |  |  |  |  |  | 1 |  | 6 |  |  |  |  | 7 | 6 |
| plate_barcoded |  |  |  |  | 4 |  |  |  | 5 |  |  |  | 9 | 5 |
| smartseq |  |  |  |  |  |  |  |  |  | 4 |  |  | 4 | 4 |
| visium |  |  |  |  |  |  |  |  |  |  |  | 5 | 5 | 5 |

Family correct: 51 of 76. 10x family and protocol correct: 20 of 37; among the 21 10x rows whose deposit holds a barcode read (a non-index segment of 40 bases or fewer): 18.
Protocol tag matches inside correct non-10x families: bulk_rna tag right 3, droplet_other tag differs 4, droplet_other tag right 1, feature_barcode tag differs 6, plate_barcoded tag differs 3, plate_barcoded tag right 2, smartseq tag differs 4.
Runs labeled ATAC, multiome ATAC, feature barcode, bulk or genomic that received `process`: 0.

## Actions by label family

| label family | process | classify_only | defer | refuse |
|---|---|---|---|---|
| atac | 0 | 1 | 0 | 0 |
| bulk_rna | 0 | 0 | 0 | 3 |
| droplet_3p | 12 | 0 | 0 | 15 |
| droplet_5p | 3 | 0 | 0 | 0 |
| droplet_other | 3 | 0 | 2 | 4 |
| feature_barcode | 0 | 6 | 0 | 0 |
| multiome_atac | 0 | 2 | 0 | 0 |
| multiome_gex | 6 | 1 | 0 | 0 |
| plate_barcoded | 0 | 0 | 9 | 0 |
| smartseq | 0 | 0 | 4 | 0 |
| visium | 5 | 0 | 0 | 0 |

## Misses

- `GSM5472074` label droplet_3p/10x-3p-v2; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 33948729 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 33948729 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: none | layout 76 | spots 200000 route sra err 
- `GSM5534708` label multiome_gex/10x-arc-gex; verdict feature_barcode/10x-3p-v3 (classify_only). feature_barcode/10x-3p-v3 support 0.933603: feature-barcode evidence: R2 top-100 15-mers hold 97% of reads; droplet_3p/10x-3p-v3 support 0.294822: barcode list v3 hit by 0.98274 of reads at segment 0 / 3'/5' call from sense fraction 0.829818 of mapped reads | lists>5%: v3 0.98, v1 0.06, v4_3p 0.16 | layout 28,91,8 | spots 200000 route sra err 
- `GSM5704176` label droplet_3p/10x-3p-v3; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 26179244 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 26179244 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: none | layout 37 | spots 200000 route sra err 
- `GSM5769275` label droplet_3p/10x-3p-v2; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 11126540 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 11126540 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.05 | layout 98,98 | spots 200000 route sra err 
- `GSM6564300` label droplet_3p/10x-3p-v4; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 67436356 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 67436356 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.09, v4_3p 0.25 | layout 651 | spots 200000 route sra err 
- `GSM6564302` label droplet_3p/10x-3p-v4; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 98349656 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 98349656 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.19, v4_3p 0.15 | layout 685 | spots 200000 route sra err 
- `GSM6564303` label droplet_3p/10x-3p-v4; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 87800959 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 87800959 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.19, v4_3p 0.14 | layout 661 | spots 200000 route sra err 
- `GSM6564304` label droplet_3p/10x-3p-v4; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 72828041 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 72828041 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.19, v4_3p 0.14 | layout 851 | spots 200000 route sra err 
- `GSM8158578` label droplet_3p/10x-3p-v3; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 30838245 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 30838245 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.05 | layout 150,150 | spots 200000 route sra err 
- `GSM8618582` label droplet_3p/10x-3p-v4; verdict multiome_gex/10x-arc-gex (process). multiome_gex/10x-arc-gex support 0.93504: barcode list arc hit by 0.93504 of reads at segment 2 | lists>5%: v3 0.10, arc 0.94, v1 0.06, v4_3p 0.09 | layout 10,10,28,90 | spots 200000 route sra err 
- `GSM8754140` label droplet_3p/10x-3p-v3; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 24452889 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 24452889 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.06 | layout 150,150 | spots 200000 route sra err 
- `GSM9287649` label droplet_3p/10x-3p-v3; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 23672386 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 23672386 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: none | layout 50 | spots 200000 route sra err 
- `SRS5091285` label droplet_3p/10x-3p-v1; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 331634696 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 331634696 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: none | layout 82 | spots 200000 route sra err 
- `SRS5091287` label droplet_3p/10x-3p-v1; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 314073780 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 314073780 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: none | layout 82 | spots 200000 route sra err 
- `SRS5091328` label droplet_3p/10x-3p-v1; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 108208338 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 108208338 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.10 | layout 98 | spots 200000 route sra err 
- `SRS5091333` label droplet_3p/10x-3p-v1; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 84987307 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 84987307 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.08 | layout 98 | spots 200000 route sra err 
- `SRS5091482` label droplet_3p/10x-3p-v1; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 305211030 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 305211030 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: none | layout 82 | spots 200000 route sra err 
- `mereu_CELseq2_HCAP1_4` label plate_barcoded/celseq2-u6b8; verdict droplet_other/barcoded-unspecified (defer). droplet_other/barcoded-unspecified support 0.627395: two disjoint barcode windows (offsets 11 and 22, 11 and 8 bases) each hold few sequences: split-pool bead barcodes | lists>5%: v4_3p 0.06 | layout 30,75 | spots 200000 route sra err 
- `mereu_CELseq2_HCAP3_1` label plate_barcoded/celseq2-u6b8; verdict droplet_other/barcoded-unspecified (defer). droplet_other/barcoded-unspecified support 0.752118: two disjoint barcode windows (offsets 11 and 23, 12 and 7 bases) each hold few sequences: split-pool bead barcodes; plate_barcoded/celseq2 support 0.35: protocol detector: celseq2 (low confidence) | lists>5%: none | layout 30,75 | spots 200000 route sra err 
- `mereu_MARSseq_5569AA` label plate_barcoded/marsseq2; verdict droplet_other/barcoded-unspecified (defer). droplet_other/barcoded-unspecified support 0.851546: the first 12 bases of segment 0 repeat across reads (0.946162 of reads): a barcode set larger than a plate; plate_barcoded/celseq2 support 0.35: protocol detector: celseq2 (low confidence) | lists>5%: v1 0.05 | layout 52,15 | spots 200000 route sra err 
- `mereu_MARSseq_5923AA` label plate_barcoded/marsseq2; verdict droplet_other/barcoded-unspecified (defer). droplet_other/barcoded-unspecified support 0.852557: the first 12 bases of segment 0 repeat across reads (0.947286 of reads): a barcode set larger than a plate; plate_barcoded/celseq2 support 0.35: protocol detector: celseq2 (low confidence) | lists>5%: v1 0.07, v4_3p 0.11 | layout 52,15 | spots 200000 route sra err 
- `ding_Mixture2.SeqWell_BC9` label droplet_other/seqwell; verdict unknown/none (refuse). droplet_other/barcoded-unspecified support 0.296785: the first 12 bases of segment 0 repeat across reads (0.659523 of reads): a barcode set larger than a plate / uniquely mapped share 0.47008 is below 0.5 | lists>5%: v1 0.06 | layout 50,50,10 | spots 200000 route sra err 
- `ding_PBMC2.SeqWell_S3_2` label droplet_other/seqwell; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.31: no cell barcode, long reads; run holds 55121988 spots (full-length single-cell and bulk libraries look alike in reads) / uniquely mapped share 0.30638 is below 0.5; smartseq/smartseq2-like support 0.19: no cell barcode, long reads; run holds 55121988 spots (full-length single-cell and bulk libraries look alike in reads) / uniquely mapped share 0.30638 is below 0.5 | lists>5%: v1 0.06 | layout 50,50,10 | spots 200000 route sra err 
- `ding_PBMC1.July2018-SeqWell_V1R1` label droplet_other/seqwell; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 63763136 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 63763136 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.06 | layout 31,50,8 | spots 200000 route sra err 
- `ding_Cortex2.DroNcSeq_3A` label droplet_other/dronc-seq; verdict bulk_rna/bulk (refuse). bulk_rna/bulk support 0.62: no cell barcode, long reads; run holds 45348872 spots (full-length single-cell and bulk libraries look alike in reads); smartseq/smartseq2-like support 0.38: no cell barcode, long reads; run holds 45348872 spots (full-length single-cell and bulk libraries look alike in reads) | lists>5%: v1 0.06 | layout 31,50,8 | spots 200000 route sra err 

## Reading the misses

Of the 25 rows that are not correct, 14 are 10x libraries whose deposited run has no barcode read (one long segment, or two equal-length
mates), so the library type cannot be read from it. All 14 are refused and none is processed. Two more are label conflicts:
a run labeled GEM-X 3' v4 whose barcodes hit the multiome GEX list at 94%, and a run labeled multiome GEX whose reads are an amplicon library
(the top 100 15-mers hold 97% of reads). One run (GSM5704176) is a single 37-base segment with no list hit. The remaining eight are protocol
families the first pass does not separate: four plate libraries (two CEL-Seq2, two MARS-seq) are called split-pool droplet beads with an
unspecified layout (deferred), and Seq-Well and DroNc-seq runs with two long reads show no repeated barcode start and fall to the bulk guard
(refused) or unknown. Without mapping evidence one of the three 5' libraries was called 3' on the shared 737K list; with it all three are right.

## Not done in this phase

V(D)J enrichment, Visium probe sets and 10x Flex (no probe set is staged; Visium FFPE reads are called Visium through the spot list),
a gene k-mer index route for layer D (the cluster has no built index), run-level duplication for the usefulness score, and the GEM-X 5' list
(the 3' GEM-X list is staged and hit one run in this panel).
