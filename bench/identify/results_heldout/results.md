# Identify development panel: results

Scored 47 of 47 panel rows (0 without a result).

## Confusion matrix (rows: label family, columns: verdict)

| label \ verdict | ambiguous | atac | barcode_read_missing | bulk_rna | droplet_3p | droplet_5p | droplet_other | feature_barcode | flex | multiome_atac | multiome_gex | plate_barcoded | rna_no_barcode | smartseq | smartseq3 | unknown | vdj | visium | total | correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atac |  | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 | 3 |
| bulk_rna |  |  | 2 |  |  |  |  |  |  |  |  |  | 3 |  |  |  |  |  | 5 | 0 |
| droplet_3p |  |  |  |  | 4 |  |  |  |  |  |  |  | 1 |  |  | 2 |  |  | 7 | 4 |
| droplet_5p |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  | 1 | 1 |  | 5 | 3 |
| droplet_other | 2 |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  | 1 |  |  | 4 | 1 |
| feature_barcode |  |  |  |  |  |  |  | 3 |  |  |  |  |  |  |  |  |  |  | 3 | 3 |
| flex |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  | 1 | 0 |
| multiome_atac |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  | 3 | 0 |
| multiome_gex |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  | 3 | 0 |
| plate_barcoded |  |  |  | 1 |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  | 3 | 2 |
| smartseq |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  | 1 |  |  |  | 3 | 0 |
| vdj |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  | 2 | 2 |
| visium |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  |  | 1 |  | 2 | 5 | 2 |

Family correct: 20 of 47. 10x family and protocol correct: 7 of 15; among the 3 10x rows whose deposit holds a barcode read (a non-index segment of 40 bases or fewer): 3.
Protocol tag matches inside correct non-10x families: atac tag right 3, droplet_other tag right 1, feature_barcode tag differs 2, feature_barcode tag right 1, plate_barcoded tag differs 2, vdj tag right 2.
Runs labeled ATAC, multiome ATAC, feature barcode, bulk or genomic that received `process`: 0.

## 10x 3' and 5' (v1 to v4), whole panel

Family: 7 of 12 labeled 10x 3'/5' rows called 10x 3'/5' (recall 0.58); 7 of 7 calls are labeled 10x 3'/5' (precision 1.00); false positives from the other 35 rows: none.
Version (rows with a stated version): 4 of 7 exact (recall 0.57); of 4 calls on those rows, 4 carry the labeled version (precision 1.00).
Not called: SRR7722018 (unknown), SRR9644308 (rna_no_barcode), SRR7788880 (unknown), SRR11089986 (unknown), SRR11067221 (vdj).

Bars: ATAC, feature-barcode, bulk or genomic rows given `process`: 0. Single-cell rows given `refuse`: 2 with the single_cell=1 hint applied (SRR7827205 as ambiguous, SRR13247000 as ambiguous), 2 without it. Rows of other protocols called 10x 3'/5': 0 (none). Other-protocol rows given `process` (policy for a known droplet protocol): 1 (SRR1853178).

## barcode_read_missing

2 rows. Endpoint that carries a barcode read (from the endpoints record):

- `SRR23979322`: endpoints not surveyed
- `SRR36176795`: endpoints not surveyed

## Actions by label family

| label family | process | classify_only | defer | refuse |
|---|---|---|---|---|
| atac | 0 | 3 | 0 | 0 |
| bulk_rna | 0 | 0 | 5 | 0 |
| droplet_3p | 4 | 0 | 3 | 0 |
| droplet_5p | 3 | 1 | 1 | 0 |
| droplet_other | 1 | 0 | 1 | 2 |
| feature_barcode | 0 | 3 | 0 | 0 |
| flex | 0 | 0 | 1 | 0 |
| multiome_atac | 0 | 0 | 3 | 0 |
| multiome_gex | 0 | 0 | 3 | 0 |
| plate_barcoded | 0 | 0 | 3 | 0 |
| smartseq | 0 | 0 | 3 | 0 |
| vdj | 0 | 2 | 0 | 0 |
| visium | 2 | 0 | 3 | 0 |

## Misses

- `SRR17209008` label visium/10x-visium; verdict unknown/ (defer). rna_no_barcode/undetermined support 0.09: no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir) / uniquely mapped share 0.1244 is below 0.5 | lists>5%: v1 0.08 | layout 90 | spots 200000 route sra err 
- `SRR27536879` label visium/10x-visium; verdict droplet_other/ (defer). droplet_other/barcoded-unspecified support 0.407004: the first 12 bases of segment 0 repeat across reads (0.904454 of reads): a barcode set larger than a plate / uniquely mapped share 0.00466 is below 0.5 | lists>5%: v1 0.07, v4_3p 0.06 | layout 28,90 | spots 200000 route sra err 
- `SRR29202305` label visium/10x-visium; verdict droplet_other/ (defer). droplet_other/barcoded-unspecified support 0.426148: the first 12 bases of segment 2 repeat across reads (0.946995 of reads): a barcode set larger than a plate / uniquely mapped share 0.0013 is below 0.5 | lists>5%: v1 0.06, v4_3p 0.06 | layout 10,10,28,151 | spots 200000 route sra err 
- `SRR29625964` label smartseq/smartseq3; verdict smartseq3/smartseq3 (defer). smartseq3/smartseq3 support 1: Smart-seq3 5' tag at the read start in 0.30236 of reads; droplet_other/microwell-seq support 0.35: protocol detector: microwell-seq (low confidence) | lists>5%: v1 0.06 | layout 151,151 | spots 200000 route sra err 
- `SRR27301081` label smartseq/smartseq2; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.171104, of the 5' end 0.0617944 (even coverage would give 0.0581963), sense share 0.489296); droplet_other/microwell-seq support 0.35: protocol detector: microwell-seq (low confidence) | lists>5%: v1 0.07 | layout 150,151 | spots 200000 route sra err 
- `SRR18911820` label smartseq/smartseq2; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.230777, of the 5' end 0.132099 (even coverage would give 0.0804957), sense share 0.50576) | lists>5%: v1 0.07 | layout 75 | spots 200000 route sra err 
- `SRR29506390` label bulk_rna/generic-negative; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.16444, of the 5' end 0.0408707 (even coverage would give 0.0523313), sense share 0.0128751); droplet_other/dropseq support 0.35: protocol detector: dropseq (low confidence) | lists>5%: v1 0.06 | layout 51,51 | spots 200000 route sra err 
- `SRR4340391` label bulk_rna/generic-negative; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.1911, of the 5' end 0.0486599 (even coverage would give 0.0464603), sense share 0.00668584) | lists>5%: v1 0.05 | layout 100,0 | spots 200000 route sra err 
- `SRR14848344` label bulk_rna/generic-negative; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.6: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.248375, of the 5' end 0.0468953 (even coverage would give 0.0577931), sense share 0.0104569); droplet_other/microwell-seq support 0.35: protocol detector: microwell-seq (low confidence) | lists>5%: none | layout 101,101 | spots 200000 route sra err 
- `SRR23979322` label bulk_rna/generic-negative; verdict barcode_read_missing/ (defer). barcode_read_missing/tag-3p support 0.85: single-cell style cDNA without its barcode read in this endpoint: reads within 1000 nt of the transcript 3' end 0.434341, of the 5' end 0.0399356 (even coverage would give 0.0600672), sense share 0.971696 / try_endpoints: the barcode read may sit in another endpoint of the run (ENA submitted files, SRA original-format objects); run with --endpoints | lists>5%: v1 0.05 | layout 101 | spots 200000 route sra err 
- `SRR36176795` label bulk_rna/generic-negative; verdict barcode_read_missing/ (defer). barcode_read_missing/tag-3p support 0.85: single-cell style cDNA without its barcode read in this endpoint: reads within 1000 nt of the transcript 3' end 0.389639, of the 5' end 0.0122494 (even coverage would give 0.0409127), sense share 0.993633 / try_endpoints: the barcode read may sit in another endpoint of the run (ENA submitted files, SRA original-format objects); run with --endpoints | lists>5%: v1 0.06 | layout 90 | spots 200000 route sra err 
- `SRR6750041` label droplet_other/splitseq; verdict unknown/ (defer). combinatorial/splitseq support 0.1275: protocol detector: splitseq (high confidence) / uniquely mapped share 0.00516 is below 0.5 | lists>5%: v1 0.06 | layout 66,94 | spots 200000 route sra err 
- `SRR7827205` label droplet_other/sci-rna-seq3; verdict ambiguous/ (refuse). plate_barcoded/plate-unknown support 0.473018: top 96 6-mers at offset 8 hold 0.96874 of reads (random expectation 0.0234375): well barcodes / uniquely mapped share 0.48556 is below 0.5; combinatorial/sci-rna-seq3 support 0.325: protocol detector: sci-rna-seq3 (medium confidence) / uniquely mapped share 0.48556 is below 0.5 | lists>5%: v1 0.05 | layout 36,56 | spots 200000 route sra err 
- `SRR21226943` label plate_barcoded/scrbseq; verdict bulk_rna/generic-negative (defer). bulk_rna/bulk support 0.62: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.146792, of the 5' end 0.0227075 (even coverage would give 0.0271417), sense share 0.924388); run holds 329289350 spots; smartseq/smartseq2-like support 0.38: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.146792, of the 5' end 0.0227075 (even coverage would give 0.0271417), sense share 0.924388); run holds 329289350 spots | lists>5%: v1 0.06 | layout 8,15,69 | spots 200000 route sra err 
- `SRR13247000` label droplet_other/dropseq; verdict ambiguous/ (refuse). droplet_other/dropseq support 0.325: protocol detector: dropseq (medium confidence) / uniquely mapped share 0.48306 is below 0.5; plate_barcoded/plate-unknown support 0.316491: top 96 12-mers at offset 0 hold 0.632985 of reads (random expectation 5.72205e-06): well barcodes / uniquely mapped share 0.48306 is below 0.5 | lists>5%: v1 0.06, v4_3p 0.06 | layout 20,45 | spots 200000 route sra err 
- `SRR18070426` label multiome_gex/10x-arc-gex; verdict unknown/ (defer). rna_no_barcode/undetermined support 0.09: no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir) / uniquely mapped share 0.07772 is below 0.5 | lists>5%: none | layout 90 | spots 200000 route sra err 
- `SRR18070427` label multiome_gex/10x-arc-gex; verdict unknown/ (defer). rna_no_barcode/undetermined support 0.09: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.208451, of the 5' end 0.126761 (even coverage would give 0.0477502), sense share 0.570694) / uniquely mapped share 0.04016 is below 0.5 | lists>5%: atac 0.08 | layout 90 | spots 200000 route sra err 
- `SRR18070428` label multiome_gex/10x-arc-gex; verdict unknown/ (defer). rna_no_barcode/undetermined support 0.09: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.0825, of the 5' end 0.15625 (even coverage would give 0.0535537), sense share 0.838828) / uniquely mapped share 0.0747 is below 0.5 | lists>5%: v1 0.05 | layout 90 | spots 200000 route sra err 
- `SRR18070627` label multiome_atac/10x-arc-atac; verdict unknown/ (defer). droplet_other/barcoded-unspecified support 0.100737: the first 12 bases of segment 1 repeat across reads (0.746198 of reads): a barcode set larger than a plate / uniquely mapped share 0.11134 is below 0.5 | lists>5%: v4_3p 0.10 | layout 51,51 | spots 200000 route sra err 
- `SRR18070626` label multiome_atac/10x-arc-atac; verdict unknown/ (defer). droplet_other/barcoded-unspecified support 0.0866281: the first 12 bases of segment 0 repeat across reads (0.64169 of reads): a barcode set larger than a plate / uniquely mapped share 0.1015 is below 0.5 | lists>5%: v4_3p 0.08 | layout 51,51 | spots 200000 route sra err 
- `SRR18070625` label multiome_atac/10x-arc-atac; verdict unknown/ (defer). bulk_rna/bulk support 0.093: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.0364372, of the 5' end 0.0688259 (even coverage would give 0.0343364), sense share 0.47416); run holds 226716562 spots / uniquely mapped share 0.11896 is below 0.5; smartseq/smartseq2-like support 0.057: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.0364372, of the 5' end 0.0688259 (even coverage would give 0.0343364), sense share 0.47416); run holds 226716562 spots / uniquely mapped share 0.11896 is below 0.5 | lists>5%: v4_3p 0.11 | layout 51,51 | spots 200000 route sra err 
- `SRR7722018` label droplet_3p/10x-3p-v2; verdict unknown/ (defer). rna_no_barcode/undetermined support 0.09: no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir) / uniquely mapped share 0.04134 is below 0.5 | lists>5%: none | layout 99 | spots 200000 route sra err 
- `SRR9644308` label droplet_3p/10x-3p-v2; verdict rna_no_barcode/ (defer). rna_no_barcode/undetermined support 0.3: no cell barcode; coverage is neither even nor a tag end pile-up (reads within 1000 nt of the transcript 3' end 0.161737, of the 5' end 0.515537 (even coverage would give 0.0692636), sense share 0.941303) / uniquely mapped share 0.31026 is below 0.5 | lists>5%: none | layout 98 | spots 200000 route sra err 
- `SRR7788880` label droplet_3p/10x-3p-v2; verdict unknown/ (defer). rna_no_barcode/undetermined support 0.09: no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir) / uniquely mapped share 0.02512 is below 0.5 | lists>5%: v1 0.05 | layout 98 | spots 200000 route sra err 
- `SRR11089986` label droplet_5p/10x-5p; verdict unknown/ (defer). bulk_rna/bulk support 0.093: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.0598504, of the 5' end 0.0972569 (even coverage would give 0.0966708), sense share 0.15736); run holds 535938230 spots / uniquely mapped share 0.36358 is below 0.5; smartseq/smartseq2-like support 0.057: no cell barcode and even coverage along transcripts (reads within 1000 nt of the transcript 3' end 0.0598504, of the 5' end 0.0972569 (even coverage would give 0.0966708), sense share 0.15736); run holds 535938230 spots / uniquely mapped share 0.36358 is below 0.5 | lists>5%: none | layout 98 | spots 200000 route sra err 
- `SRR11067221` label droplet_5p/10x-5p; verdict vdj/tcr (classify_only). vdj/10x-5p-v2 support 0.906129: 0.860629 of gene-assigned reads fall on immunoglobulin or T-cell receptor genes; droplet_5p/10x-5p-v2 support 0.286146: barcode list v2 hit by 0.95382 of reads at segment 0 / 3'/5' call from sense fraction 0.117931 of mapped reads | lists>5%: v3 0.13, v2 0.95, v1 0.05, v4_3p 0.18, v4_5p 0.07 | layout 151,151 | spots 200000 route sra err 
- `SRR31116722` label flex/10x-flex; verdict plate_barcoded/ (defer). plate_barcoded/plate-unknown support 0.46848: top 1536 9-mers at offset 0 hold 0.942483 of reads (random expectation 0.00585938): well barcodes / uniquely mapped share 0.00016 is below 0.5 | lists>5%: v3 0.12, v1 0.06, atac 0.07, v4_3p 0.10, arc_atac 0.16 | layout 28,90 | spots 200000 route sra err 
