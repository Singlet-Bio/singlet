# Detection recall: diagnosis of the five held-out misses and development re-run

Binary `e1b37756` (identify/recall-bar, on identify/bam-head), Clipper short-cpu, jobs 816456 (build), 816457 (development panel, 76 rows, `--endpoints`, frozen sources and hints), 816458 (10 development BAM libraries, `--endpoints`, no hint). The held-out panel was not run.

## Causes of the five held-out misses (from the stored records)
| Run | Record | Cause |
|---|---|---|
| SRR7722018, SRR7788880, SRR11089986 | single 98 or 99 base read, `unknown` | the deposit holds only the cDNA read; the barcode is in the original Cell Ranger BAM (SDL object, NCBI S3) |
| SRR9644308 | `rna_no_barcode` | same, barcode only in the original BAM |
| SRR11067221 | `vdj`, classify only | V(D)J-enriched library (86% IG/TR genes) labeled 5' gene expression; the reads contradict the label, so it is outside the registered denominator and the verdict is right |

The four BAM rows are called 3' v2, `process`, by the BAM head route of PR 264 (barcode tags, list hit 0.92-0.97, sense 0.78-1.00). This PR makes the route fast: the NCBI S3 copy is read first and the ENA copy is skipped when S3 answered (3.3-3.9 s per run in this job, against 92-225 s over ENA FTP).

## Development panel, before and after
| Measure | Frozen (dev9) | This branch |
|---|---|---|
| 10x 3'/5' family, precision | 12 of 12 | 17 of 17 |
| 10x 3'/5' family, recall | 12 of 30 (12 of 21 without the 9 `label_doubt` rows) | 17 of 30 (17 of 21) |
| Rows changed | | 5 of 76, all the CAR-T runs, `unknown`/`genomic_dna`/`rna_no_barcode` to 10x 3' v2 `process` |
| Non-10x run called 10x, ATAC/feature-barcode/bulk processed | 0 | 0 |

Version: the five CAR-T runs are labeled 3' v1 and read as 3' v2 (16-base barcodes on the v2 list), so by label they are version misses; the label is wrong. The Kang 2018 files read as v1 (5 of 5).

## Development cases by miss pattern
- BAM-only barcode: `sources_bam_dev.tsv` (5 CAR-T, 5 Kang), results in `bam_dev/`.
- V(D)J labeled 5': panel rows GSM4764773 and GSM7719627, `vdj` classify only, unchanged.
- No barcode anywhere: GSM5704176 (single 37-base read, no other endpoint) and the nine `label_doubt` rows stay `barcode_read_missing` or deferred, unchanged.

## Open
- Held-out confirmation is needed: the four BAM runs were run once with the BAM route after the held-out score. S2 of the rerun plan should score them in a fresh round.
- Called runs from a BAM cannot be processed yet (engine route 3 to 5 days); recall here is detection only.
