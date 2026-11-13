# Identify development panel: results

Scored 76 of 76 panel rows (0 without a result).

## Confusion matrix (rows: label family, columns: verdict)

| label \ verdict | atac | barcode_read_missing | bulk_rna | droplet_3p | droplet_5p | droplet_other | feature_barcode | genomic_dna | multiome_atac | multiome_gex | plate_barcoded | rna_no_barcode | smartseq | unknown | vdj | visium | total | correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atac | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 1 |
| bulk_rna |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  | 3 | 0 |
| droplet_3p |  | 1 | 3 | 11 |  |  |  | 2 |  | 1 |  | 7 |  | 2 |  |  | 27 | 11 |
| droplet_5p |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  | 2 |  | 3 | 1 |
| droplet_other |  |  |  |  |  | 7 |  |  |  |  | 2 |  |  |  |  |  | 9 | 7 |
| feature_barcode |  |  |  |  |  |  | 6 |  |  |  |  |  |  |  |  |  | 6 | 6 |
| multiome_atac |  |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  | 2 | 2 |
| multiome_gex |  |  |  |  |  |  | 1 |  |  | 6 |  |  |  |  |  |  | 7 | 6 |
| plate_barcoded |  |  |  |  |  | 1 |  |  |  |  | 8 |  |  |  |  |  | 9 | 8 |
| smartseq |  |  |  |  |  |  |  |  |  |  |  | 1 | 3 |  |  |  | 4 | 3 |
| visium |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 | 5 | 5 |

Family correct: 50 of 76. 10x family and protocol correct: 18 of 37; among the 21 10x rows whose deposit holds a barcode read (a non-index segment of 40 bases or fewer): 16.
Protocol tag matches inside correct non-10x families: atac tag right 1, droplet_other tag differs 3, droplet_other tag right 4, feature_barcode tag differs 3, feature_barcode tag right 3, plate_barcoded tag differs 4, plate_barcoded tag right 4, smartseq tag right 3.
Runs labeled ATAC, multiome ATAC, feature barcode, bulk or genomic that received `process`: 0.

## 10x 3' and 5' (v1 to v4), whole panel

Family: 12 of 30 labeled 10x 3'/5' rows called 10x 3'/5' (recall 0.40); 12 of 12 calls are labeled 10x 3'/5' (precision 1.00); false positives from the other 46 rows: none.
Version (rows with a stated version): 8 of 24 exact (recall 0.33); of 8 calls on those rows, 8 carry the labeled version (precision 1.00).
Not called: GSM4764773 (vdj), GSM5472074 (rna_no_barcode), GSM5704176 (barcode_read_missing), GSM5769275 (bulk_rna), GSM6564300 (rna_no_barcode), GSM6564302 (bulk_rna), GSM6564303 (rna_no_barcode), GSM6564304 (bulk_rna), GSM7719627 (vdj), GSM8158578 (rna_no_barcode), GSM8618582 (multiome_gex), GSM8754140 (rna_no_barcode), GSM9287649 (rna_no_barcode), SRS5091285 (unknown), SRS5091287 (unknown), SRS5091328 (genomic_dna), SRS5091333 (genomic_dna), SRS5091482 (rna_no_barcode).

Bars: ATAC, feature-barcode, bulk or genomic rows given `process`: 0. Single-cell rows given `refuse`: 7 (GSM5769275 as bulk_rna, GSM6564302 as bulk_rna, GSM6564304 as bulk_rna, SRS5091285 as unknown, SRS5091287 as unknown, SRS5091328 as genomic_dna, SRS5091333 as genomic_dna). Rows of other protocols called 10x 3'/5': 0 (none). Other-protocol rows given `process` (policy for a known droplet protocol): 1 (mereu_inDrop_SP021_024015).

## barcode_read_missing

1 rows. Endpoint that carries a barcode read (from the endpoints record):

- `GSM5704176`: endpoints not surveyed

## Actions by label family

| label family | process | classify_only | defer | refuse |
|---|---|---|---|---|
| atac | 0 | 1 | 0 | 0 |
| bulk_rna | 0 | 0 | 3 | 0 |
| droplet_3p | 12 | 0 | 8 | 7 |
| droplet_5p | 1 | 2 | 0 | 0 |
| droplet_other | 1 | 0 | 8 | 0 |
| feature_barcode | 0 | 6 | 0 | 0 |
| multiome_atac | 0 | 2 | 0 | 0 |
| multiome_gex | 6 | 1 | 0 | 0 |
| plate_barcoded | 0 | 0 | 9 | 0 |
| smartseq | 0 | 0 | 4 | 0 |
| visium | 5 | 0 | 0 | 0 |

## Misses

- `GSM4764773` label droplet_5p/10x-5p; verdict vdj/tcr (classify_only). vdj/10x-5p-v2 support 0.902443: 0.919356 of gene-assigned reads fall on immunoglobulin or T-cell receptor genes; droplet_5p/10x-5p-v2 support 0.284982: barcode list v2 hit by 0.94994 of reads at segment 1 / 3'/5' call from sense fraction 0.0391111 of mapped reads | lists>5%: v3 0.08, v2 0.95, v1 0.06, v4_3p 0.18, v4_5p 0.06 | layout 8,26,91 | spots 200000 route sra err 
- `GSM5472074` label droplet_3p/10x-3p-v2; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.214663, of the 5' end 0.069045 (even coverage would give 0.050957), sense share 0.504764) | lists>5%: none | layout 76 | spots 200000 route sra err 
- `GSM5534708` label multiome_gex/10x-arc-gex; verdict feature_barcode/cite-adt (classify_only). feature_barcode/10x-3p-v3 support 0.933603: feature-barcode evidence: R2 top-100 15-mers hold 97% of reads; droplet_3p/10x-3p-v3 support 0.294822: barcode list v3 hit by 0.98274 of reads at segment 0 / 3'/5' call from sense fraction 0.829818 of mapped reads | lists>5%: v3 0.98, v1 0.06, v4_3p 0.16, v4_5p 0.07 | layout 28,91,8 | spots 200000 route sra err 
- `GSM5704176` label droplet_3p/10x-3p-v3; verdict barcode_read_missing/ (defer). barcode_read_missing/tag-3p support 0.85: single-cell style cDNA without its barcode read in this endpoint: reads within 1000 nt of the transcript 3' end 0.536617, of the 5' end 0.0214281 (even coverage would give 0.0573766), sense share 0.944853 / try_endpoints: the barcode read may sit in another endpoint of the run (ENA submitted files, SRA original-format objects); run with --endpoints | lists>5%: none | layout 37 | spots 200000 route sra err 
- `GSM5769275` label droplet_3p/10x-3p-v2; verdict bulk_rna/generic-negative (refuse). bulk_rna/bulk support 0.62: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.0706688, of the 5' end 0.0453503 (even coverage would give 0.0335314), sense share 0.0570336); run holds 11126540 spots; smartseq/smartseq2-like support 0.38: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.0706688, of the 5' end 0.0453503 (even coverage would give 0.0335314), sense share 0.0570336); run holds 11126540 spots | lists>5%: v1 0.05 | layout 98,98 | spots 200000 route sra err 
- `GSM6564300` label droplet_3p/10x-3p-v4; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.3: no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir) / uniquely mapped share 0.00438 is below 0.5 | lists>5%: v1 0.09, v4_3p 0.25 | layout 651 | spots 200000 route sra err 
- `GSM6564302` label droplet_3p/10x-3p-v4; verdict bulk_rna/generic-negative (refuse). bulk_rna/bulk support 0.31: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.0952381, of the 5' end 0.030303 (even coverage would give 0.027621), sense share 0.566116); run holds 98349656 spots / uniquely mapped share 0.00588 is below 0.5; smartseq/smartseq2-like support 0.19: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.0952381, of the 5' end 0.030303 (even coverage would give 0.027621), sense share 0.566116); run holds 98349656 spots / uniquely mapped share 0.00588 is below 0.5 | lists>5%: v1 0.19, v4_3p 0.15 | layout 685 | spots 200000 route sra err 
- `GSM6564303` label droplet_3p/10x-3p-v4; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.3: no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir) / uniquely mapped share 0.00482 is below 0.5 | lists>5%: v1 0.19, v4_3p 0.14 | layout 661 | spots 200000 route sra err 
- `GSM6564304` label droplet_3p/10x-3p-v4; verdict bulk_rna/generic-negative (refuse). bulk_rna/bulk support 0.31: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.111538, of the 5' end 0.0307692 (even coverage would give 0.0299626), sense share 0.470588); run holds 72828041 spots / uniquely mapped share 0.00698 is below 0.5; smartseq/smartseq2-like support 0.19: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.111538, of the 5' end 0.0307692 (even coverage would give 0.0299626), sense share 0.470588); run holds 72828041 spots / uniquely mapped share 0.00698 is below 0.5 | lists>5%: v1 0.19, v4_3p 0.14 | layout 851 | spots 200000 route sra err 
- `GSM7719627` label droplet_5p/10x-5p; verdict vdj/tcr (classify_only). vdj/10x-5p-v2 support 0.794998: 0.720742 of gene-assigned reads fall on immunoglobulin or T-cell receptor genes; droplet_5p/10x-5p-v2 support 0.251052: barcode list v2 hit by 0.83684 of reads at segment 1 / 3'/5' call from sense fraction 0.152133 of mapped reads | lists>5%: v3 0.08, v2 0.84, v4_3p 0.14, v4_5p 0.06 | layout 8,26,91 | spots 200000 route sra err 
- `GSM8158578` label droplet_3p/10x-3p-v3; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.171973, of the 5' end 0.100384 (even coverage would give 0.0547241), sense share 0.498727); droplet_other/microwell-seq support 0.35: protocol detector: microwell-seq (low confidence) | lists>5%: v1 0.05 | layout 150,150 | spots 200000 route sra err 
- `GSM8618582` label droplet_3p/10x-3p-v4; verdict multiome_gex/10x-arc-gex (process). multiome_gex/10x-arc-gex support 0.93504: barcode list arc hit by 0.93504 of reads at segment 2 | lists>5%: v3 0.10, arc 0.94, v1 0.06, v4_3p 0.09 | layout 10,10,28,90 | spots 200000 route sra err 
- `GSM8754140` label droplet_3p/10x-3p-v3; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.174149, of the 5' end 0.0374001 (even coverage would give 0.0471573), sense share 0.0124849); droplet_other/microwell-seq support 0.35: protocol detector: microwell-seq (low confidence) | lists>5%: v1 0.06 | layout 150,150 | spots 200000 route sra err 
- `GSM9287649` label droplet_3p/10x-3p-v3; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.238742, of the 5' end 0.0338501 (even coverage would give 0.0528641), sense share 0.049596) | lists>5%: none | layout 50 | spots 200000 route sra err 
- `SRS5091285` label droplet_3p/10x-3p-v1; verdict unknown/ (refuse). rna_no_barcode/undetermined support 0.09: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.0666667, of the 5' end 0.239216 (even coverage would give 0.0721552), sense share 0.977778) / uniquely mapped share 0.05102 is below 0.5 | lists>5%: none | layout 82 | spots 200000 route sra err 
- `SRS5091287` label droplet_3p/10x-3p-v1; verdict unknown/ (refuse). rna_no_barcode/undetermined support 0.09: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.0583333, of the 5' end 0.3625 (even coverage would give 0.0713681), sense share 0.955056) / uniquely mapped share 0.0488 is below 0.5 | lists>5%: none | layout 82 | spots 200000 route sra err 
- `SRS5091328` label droplet_3p/10x-3p-v1; verdict genomic_dna/generic-negative (refuse). genomic_dna/none support 0.7: reads map but few land in genes (0.000733333); rna_no_barcode/undetermined support 0.18: no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir) / gene-assigned share 0.000733333 is low for RNA | lists>5%: v1 0.10 | layout 98 | spots 200000 route sra err 
- `SRS5091333` label droplet_3p/10x-3p-v1; verdict genomic_dna/generic-negative (refuse). genomic_dna/none support 0.7: reads map but few land in genes (0.00169186); rna_no_barcode/undetermined support 0.18: no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir) / gene-assigned share 0.00169186 is low for RNA | lists>5%: v1 0.08 | layout 98 | spots 200000 route sra err 
- `SRS5091482` label droplet_3p/10x-3p-v1; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.3: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.190939, of the 5' end 0.288026 (even coverage would give 0.0614928), sense share 0.887324) / uniquely mapped share 0.0292 is below 0.5 | lists>5%: none | layout 82 | spots 200000 route sra err 
- `mereu_CELseq2_HCAP3_1` label plate_barcoded/celseq2; verdict droplet_other/ (defer). droplet_other/barcoded-unspecified support 0.752118: two disjoint barcode windows (offsets 11 and 23, 12 and 7 bases) each hold few sequences: split-pool bead barcodes; plate_barcoded/celseq2 support 0.35: protocol detector: celseq2 (low confidence) | lists>5%: none | layout 30,75 | spots 200000 route sra err 
- `mereu_ddSEQ_Sample2` label droplet_other/ddseq; verdict plate_barcoded/ (defer). plate_barcoded/plate-unknown support 0.47488: top 96 9-mers at offset 15 hold 0.950107 of reads (random expectation 0.000366211): well barcodes / uniquely mapped share 0.43208 is below 0.5; plate_barcoded/marsseq2 support 0.175: protocol detector: marsseq2 (low confidence) / uniquely mapped share 0.43208 is below 0.5 | lists>5%: v1 0.06 | layout 52,75 | spots 200000 route sra err 
- `mereu_ddSEQ_Sample3` label droplet_other/ddseq; verdict plate_barcoded/ (defer). plate_barcoded/plate-unknown support 0.964482: top 96 9-mers at offset 15 hold 0.964835 of reads (random expectation 0.000366211): well barcodes | lists>5%: v1 0.06 | layout 52,75 | spots 200000 route sra err 
- `ding_Cortex1.SM2_well` label smartseq/smartseq2; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir); genomic_dna/none support 0.15: not excluded without mapping evidence | lists>5%: v1 0.06 | layout 50,50,10,10 | spots 200000 route sra err 
- `ding_PBMC2_bulk` label bulk_rna/generic-negative; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.15978, of the 5' end 0.0431882 (even coverage would give 0.0512178), sense share 0.01158); droplet_other/microwell-seq support 0.35: protocol detector: microwell-seq (low confidence) | lists>5%: v1 0.06 | layout 75,75 | spots 200000 route sra err 
- `ding_PBMC1_bulk` label bulk_rna/generic-negative; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.16899, of the 5' end 0.0466088 (even coverage would give 0.0544387), sense share 0.00813029); droplet_other/microwell-seq support 0.35: protocol detector: microwell-seq (low confidence) | lists>5%: v1 0.06 | layout 75,75 | spots 200000 route sra err 
- `ding_Cortex2_bulk` label bulk_rna/generic-negative; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir); droplet_other/microwell-seq support 0.35: protocol detector: microwell-seq (low confidence) | lists>5%: v1 0.06 | layout 75,75 | spots 200000 route sra err 
