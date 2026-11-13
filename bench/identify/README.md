# singlet identify: development panel

`singlet identify SOURCE` reads the first reads of a dataset and reports the library family, protocol tag, an action and the ranked
alternatives. This folder holds both halves of the truth panel (development, held-out), the scripts that run them on the cluster and the scored results.
Real public runs only. The held-out half was scored once, with the frozen code (git tag `identify-frozen`).

## Files

- `panel_dev.tsv`: 76 runs with family, protocol, species, label source and label tier. Tier T2 is a depositor statement confirmed by a second
  source, D is the sample name of a design-controlled benchmark (Mereu 2020, Ding 2020), V is a vendor dataset name, and T2 on the
  feature-barcode and Visium rows is the GEO sample title. Rows whose only label came from a catalog read-call are not included.
- `sources_dev.tsv`: what `singlet identify --list` reads (id, source, hint, genome index). `$DATA` and `$REF` are cluster directories.
- `make_panel.py`, `build.sbatch`, `run_panel.sbatch`, `score.py`: build the panel, build the binary, run the batch, score it.
- `results_dev/summary.tsv`, `results_dev/results.md`: the scored run (binary built from the commit named in the PR).
- `heldout_candidates.tsv`, `build_heldout_panel.py`, `panel_heldout.tsv`, `sources_heldout.tsv`, `results_heldout/`: the held-out half (47 runs from studies outside the development panel), its scoring files and its single scored run. `pass1_job_out_of_memory/` holds a first attempt in which 16 mapping steps were killed by the 24 GB job limit; it was rerun unchanged (same binary, 64 GB) and only the rerun is scored.
- `endpoints_nobarcode.tsv`, `endpoints_heldout.tsv`, `endpoint_survey.py`: metadata-only survey of other endpoints for runs whose deposit has no barcode read.
- `results_species/`, `species_filters_check.tsv`, `species_check.sbatch`: species block on real runs and the sha256 check of the filter files.
- `corpus_share.py`, `corpus_share_result.txt`: share of 10x-labeled Beta corpus runs without a 26 to 28 base read.

## Head reader, by source type

| source | route | measured |
|---|---|---|
| SRA accession | archive spot access with a row limit (`SraReader`, libncbi-vdb fetches blocks as the cursor advances) | 200,000 spots in a median of 3.8 s, at most 84 s, over 69 runs; transfer size is not exposed by the reader |
| ENA FASTQ over HTTPS | ranged GET piped through an in-process gzip reader, closed at the spot limit | 200,000 spots in 17 to 44 s, 8 to 33 MB, on 4 runs |
| local FASTQ | same reader on files | 0.2 to 0.4 s |
| BAM | samtools view, CR and UR tags then the read | a real Cell Ranger BAM (200,000 records in 1.0 s): barcode list v3 hit by 0.97 of reads, called 10x 3' v3 |
| staged `.1fq` | the existing reader | a real staged file (200,000 spots in 0.25 s); the file holds a 36 + 2 base layout and is called not-sequence, refused |
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

## Wider endpoint survey of the 14 development runs without a barcode read

`endpoints_nobarcode.tsv` holds one line per run and endpoint (SDL original-format objects, ENA submitted files and BAM, GEO supplementary files on the GSM and GSE, sibling runs), collected by metadata lookups only. Result: five of the 14 (the five SRS rows, 10x v1 CAR-T libraries) have an original-format BAM, listed by SDL as type `TenX` with a direct S3 link and by ENA as a submitted BAM with index. The other nine list original FASTQ files whose names are `bulkRNA`, `Bulk_RNAseq`, nanopore `pass_*.fastq`, `B_shear`, `Plko-2_1.fq` and `T_iMK_*`; they need rehydration from cold storage and look like bulk or long-read libraries that the panel label (10x) does not describe. No sibling run holds a separate barcode read, and no GEO supplementary listing offers a BAM or FASTQ for the four GSM records that could be checked (the SRS rows carry no GSM alias in ENA).

### What `identify` needs to read the head of an original-format BAM (not implemented)

Route: SDL `retrieve` (or the ENA `submitted_ftp` URL) gives an HTTPS object that accepts range requests.
1. Fetch the first 32 MB by ranged GET (200,000 records of a Cell Ranger BAM are about 25 MB compressed), cut the buffer at the last complete BGZF block, and hand it to the existing `bam:` head reader as a local file in node-local scratch. This costs one small function and no new parser.
2. Cell Ranger BAMs are coordinate sorted, so the first records all come from one chromosome start. Barcode evidence is unaffected. Layer D (end bias, species, strand) needs a spread sample: take four to eight ranged GETs of 4 MB each at evenly spaced offsets, resynchronize on the BGZF block magic (1f 8b 08 04 with the BC subfield) and concatenate the records.
3. Barcode read: the head reader takes `CR` and `UR` (raw barcode and UMI). These BAMs must be checked once for those tags; the CART files are labeled "split_output_search_scrubbed" and may carry only `CB` and `UB`, in which case the corrected tags serve for the list hit and the barcode read length is taken from the tag. Reads with flag 16 are reverse complemented before they enter the head.
4. The endpoint record gains the object type and name, the bytes fetched and the tag set found. A `barcode_read_missing` verdict then names the BAM as the endpoint to use.
Cost: about a day, mostly for the multi-offset sampler and a test on a truncated BAM.

## Phase 2 (record version 2)

JSON record version 2 adds `protocol_tag` (the family ids of `validation/routes.yaml`, one mapping table in `identify.h`; null when the verdict has no id, with `protocol_tag_alternatives` where reads cannot separate two ids, for example Drop-seq and Seq-Well), a `sub_call` object (3'/5', Visium probe, V(D)J; the inDrop version, BD bead and Parse kit are null with the reason "not implemented"), a `species` block (`--species-filters DIR`, optional `--species-hint`), `hints[]` and `policy_source`. `--route-status FILE` makes `process` require a route with status `validated`. `--hint single_cell=1` (archive metadata, never read evidence) turns the refusal of a run without barcode structure into a defer.

Thresholds are in `Thresholds` in `include/singlet/fq/identify.h`, each with its measurement. `even_end_share` stays at 0.15: at 0.25 bulk fell from 3 of 3 to 2 of 3 and four more 10x runs without a barcode read were called bulk.

### Development panel (76 runs, frozen binary, hints applied)

Family correct 51 (phase 1: 51). 10x 3'/5' family precision 12 of 12, recall 12 of 30 (12 of 21 without the nine rows whose original files look like bulk, nanopore or sheared DNA, column `label_doubt`). Version precision 8 of 8, recall 8 of 24 (8 of 15). No ATAC, multiome ATAC, feature-barcode, bulk or genomic run received `process`; no other-protocol run was called 10x 3'/5'; no single-cell run was refused with the hint (seven were refused without it). Of the 14 runs without a barcode read, none is named from reads (7 held as `rna_no_barcode`, 3 bulk, 2 unknown, 2 genomic DNA, all deferred with the hint): 3' end bias does not separate them from bulk or Smart-seq. Five of the 14 have an original-format Cell Ranger-style BAM (`endpoints_nobarcode.tsv`); the other nine have original files named bulk, nanopore or sheared DNA.

### Held-out half (47 runs, scored once)

10x 3'/5' family precision 7 of 7, recall 7 of 12; version precision 4 of 4, recall 4 of 7. No false positive from the other 35 rows (look-alikes included: V(D)J, Flex, Visium, multiome, feature barcodes, bulk). The five not called: four deposits without a barcode read (one 98 or 99 base read) and one 5' run whose reads are 86 percent immunoglobulin or T-cell receptor genes (called V(D)J, which is probably right). `results_heldout/results.md` lists every miss with its evidence.

### Species block on real runs (`results_species/`)

Human 10x: `CLOSE_PAIR` human or chimpanzee (declared close group), 0.35 against 0.27 hit rate. Mouse Visium: `CALL` mouse, 0.81. Human tumor xenograft in mouse: `MIXTURE`, minor species mouse with 22 percent of the species-specific reads. The action stays the family's policy in all three. 14 to 31 s and 0.6 to 0.7 GB with all 38 filters, 7 to 16 s with the hint. All 36 filter files with an index checksum match the index (`species_filters_check.tsv`); human and mouse live in the reference directory and have no index checksum.

### Corpus

Of 53,486 Beta corpus runs labeled 10x by text, 21,637 (40 percent) have no read of 26 to 28 bases in their SRA layout; 50 percent of 25,260 samples have no such run, and 41 percent of read pairs. By labeled chemistry the run share is 31 percent (v2), 39 percent (v3), 45 percent (v1), 54 percent (v4), 42 percent (version unstated). `corpus_share_result.txt` has the table.

# BAM head route, ambiguous policy (branch identify/bam-head)

## What the BAM head route reads

For a run whose archive record lists an original-format BAM (ENA `submitted_ftp`, or the SRA original-format object from the SDL locate service), `--endpoints` now reads the BAM by HTTP range and never downloads it. A source of the form `bam:URL`, or a URL that ends in `.bam`, takes the same route directly. Requests: the first 4 MB (header and first records), three 1 MB pieces at a quarter, a half and three quarters of the file, and the last 1 MB, at most one request every 0.3 s. Parser: `include/singlet/fq/identify_bam.h` (BGZF blocks checked against their CRC, BAM header, records, tags; pieces from inside the file are resynchronized on a block and then on a run of three valid records).

The record gains a `bam` block: header (sort order, `@PG` and `@CO` lines, Cell Ranger version, chemistry and reference when the command line is stored), the pieces read, the share of records carrying each of CR, CY, UR, UY, CB, UB, CQ, UQ, GX, TX, AN, RE, tag lengths, the list rates of the barcode tag against each barcode list, the strand evidence, whether the file is coordinate sorted, whether unmapped reads are present, and a statement of limits. The verdict comes from the same classifier on spots made of the barcode tag plus UMI tag (segment 0) and the read in its original orientation (segment 1). `barcode_source` is `bam_tags` when the tags supplied the barcode and `reads` otherwise. A run whose own reads hold no barcode read (verdict `barcode_read_missing`, `rna_no_barcode`, `bulk_rna`, `smartseq`, `unknown`, `genomic_dna` or `ambiguous`) and that has a BAM with a list hit of at least 0.30 on its barcode tag is re-classified from the BAM; the endpoint entry names the URL, bytes and seconds.

3'/5' evidence: the TX and AN tags of a Cell Ranger BAM give the read strand against the transcript without an annotation (sense share of reads that carry TX or AN). Where they are absent the call falls back to the motif evidence in the cDNA read and to mapping when `--genome-dir` is given.

## Limits of a coordinate-sorted head

Every Cell Ranger BAM is coordinate sorted, so its first megabytes hold the first bases of the first chromosome. Used, because they do not depend on the genomic region: tag presence and lengths, the list rate of the raw CR, and the sense share from TX/AN, each pooled over the head and the spread pieces. Not used: reads per cell, number of cells, duplicate rate, MAPQ and multi-mapper shares, chrM and exonic shares. Unmapped reads of a sorted file lie after the last chromosome, so only the last piece can show them; without it the status is `unknown`, never `not_seen`.

## What the engine needs to process a sample from such a BAM (read from source, nothing built)

Existing route: `singlet download FILE.bam` or a URL ending in `.bam` (`src/pipeline/singlet.cpp`) resolves to the spec `bam:FILE`, which `SpotSource` (`include/singlet/fq/spot_source.h`) reads through `samtools view -F 0x900`. What it already does:

- Reads the raw tags: CR with UR, and their qualities CY and UY (CQ and UQ in Cell Ranger 1.x BAMs), as segment 1; the read as segment 2 with its qualities. Phred values come from the tag strings.
- Restores the sequenced orientation: records with flag 0x10 are reverse complemented, with qualities reversed.
- Keeps one spot per read: secondary and supplementary alignments (0x900) are dropped, so multi-mappers count once through their primary record. Unmapped records are kept when the file has them.
- Records with no CR or UR, or with a quality string of different length, are skipped and counted (`bam_skipped`).

What is missing or costly:

1. Corrected tags only. A BAM with CB and UB but no CR and UR (the identify route accepts it) gives zero spots, because `read_bam` requires CR, UR, CY and UY. Fix: fall back to CB (strip the `-1` suffix) and UB with a constant quality, and flag the sample as list-corrected. About half a day.
2. Naming and resolution. `download` treats only names ending `.bam` as BAMs; SDL objects end `.bam.1`. The endpoint URL that identify finds is not passed on; the campaign wiring has to hand the URL over. About half a day.
3. Whole-file download. The BAM is fetched whole by curl into the staging folder (9 to 28 GB per sample on the panel runs) before samtools reads it. Acceptable on Clipper node-local scratch, but it doubles the network cost of a sample compared with a FASTQ pair of the same reads, and the campaign is network bound. Streaming through `samtools view URL` or a ranged reader would remove the staging copy; two days, optional.
4. Passes over the file. `SpotSource::open` counts spots with a full `samtools view` pass, and `seek` backwards reopens and re-reads from the start; the encoder opens several readers (probe, late probe, samplers). On a BAM of 25 GB each is a full decode of a few hundred million records. Fix: take the count from `samtools idxstats` (ENA ships the .bai) plus the unmapped count, and give the late probe a virtual-offset seek. One to two days.
5. Coordinate order. Spots arrive sorted by position, not in run order. The protocol probe (first 100,000 spots) then sees one genomic region: barcode-list and orientation detection are unaffected, but any probe statistic that depends on content (chrM, rRNA, immunoglobulin share, 3' bias) is biased. Not verified in the encoder here; needs one real run of `download bam:` on a panel BAM compared with its FASTQ route. One day with the benchmark.
6. Unmapped reads. If the file has none (identify reports this per file), alignment-rate metrics cannot be computed from the BAM and the unmapped share of the sample is lost. Gene counts and cell calls from mapped reads are unaffected.
7. Runtime. `samtools` must be on PATH in the runtime that runs `download`; not checked.
8. Cell Ranger 5' BAMs: only the barcode and UMI of read 1 are in the tags, not the rest of read 1. 5' detection that uses read 1 motifs (the TSO tail after the barcode read) has nothing to read; the cDNA read motifs and the TX/AN strand remain.

Total: 3 to 5 working days from the current state, of which items 1, 2, 5 and the idxstats part of 4 (about 3 days) make the first processed sample.
