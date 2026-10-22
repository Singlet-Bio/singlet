### Layout of the bundles (what the members of each sample are)

| members per sample | samples | ids |
|---|---|---|
| 10 | 17 | h_v3_a, h_v3_b, h_v3_c, h_v2_a, h_v2_b, h_v2_c, h_arc, h_bd, h_celseq2, h_mars, h_split, h_indrop, h_quartz, bn_v3_a, bn_v3_b, bn_v2, bn_dropseq |
| 34 | 7 | z_v3_a, z_v3_b, z_v2, mq_v2, mq_v3, r_v2, r_v3 |
| 35 | 3 | m_dropseq, m_mars, m_celseq2 |
| 36 | 2 | h_dropseq, h_microwell |
| 40 | 1 | h_visium |
| 44 | 6 | m_v3_a, m_v3_b, m_v3_c, m_v2_a, m_v2_b, m_arc |
| 46 | 1 | m_visium |

New-layout samples (34-46 members) carry extra text members; mean compressed bytes per sample where present:

| member | samples with it | mean bytes |
|---|---|---|
| ambient_profile.tsv | 20 | 3,581,858 |
| star_Solo.out/Gene/raw/barcodes.tsv | 7 | 5,454,772 |
| star_Solo.out/Gene/raw/matrix.mtx | 7 | 19,124,532 |
| star_Solo.out/Gene/filtered/matrix.mtx | 7 | 11,271,604 |
| cell_qc_metrics.tsv | 20 | 1,211,659 |
| auto_barcodes.tsv | 20 | 475,921 |
| gene_expression.tsv | 20 | 811,306 |
| star_SJ.out.tab | 20 | 1,024,288 |
| splice_events.tsv | 20 | 242,382 |

Across the 37 samples: exon+intron .1pz are 654 MB of 1387 MB of sample members (47.1%).


### (d) Row-name invariance per (reference build, member)

| build | member | distinct lists | rows | samples per list | note |
|---|---|---|---|---|---|
| GRCm39-2024-A | exon | 1 | 386028 | 11 |  |
| GRCm39-2024-A | intron | 1 | 307730 | 11 |  |
| GRCh38-2024-A | exon | 1 | 310797 | 17 | plus an empty-matrix stub (0 rows, x_stub) that has no names |
| GRCh38-2024-A | intron | 1 | 272191 | 17 | plus an empty-matrix stub (0 rows, x_stub) that has no names |
| GRCh38-GRCm39-barnyard | exon | 1 | 696825 | 4 |  |
| GRCh38-GRCm39-barnyard | intron | 1 | 579921 | 4 |  |
| Mmul_10 | exon | 1 | 244671 | 2 |  |
| Mmul_10 | intron | 1 | 209239 | 2 |  |
| mRatBN7.2 | exon | 1 | 243603 | 2 |  |
| mRatBN7.2 | intron | 1 | 213041 | 2 |  |
| GRCz11 | exon | 1 | 272316 | 3 |  |
| GRCz11 | intron | 1 | 239796 | 3 |  |

### (f) Barcodes: whitelist membership by protocol (exon member; fractions of columns)

| protocol | samples | barcode length | columns | in 3M-feb-2018 (v3) | in 737K-aug-2016 (v2/5') | in either | worst sample (in either) |
|---|---|---|---|---|---|---|---|
| 10xv3 | 14 | 16 | 1128182 | 1.0000 | 0.0152 | 1.0000 | L_a (1.0000) |
| dropseq | 3 | 12/16 | 703504 | 0.0005 | 0.0001 | 0.0005 | bn_dropseq (0.0000) |
| 10xv2 | 9 | 16 | 703802 | 0.0763 | 1.0000 | 1.0000 | bn_v2 (1.0000) |
| 10x-arc-gex | 2 | 16 | 142605 | 0.0059 | 0.0012 | 0.0071 | h_arc (0.0069) |
| bd-rhapsody | 1 | 27 | 60922 | 0.0000 | 0.0000 | 0.0000 | h_bd (0.0000) |
| celseq2 | 2 | 6 | 8192 | 0.0000 | 0.0000 | 0.0000 | h_celseq2 (0.0000) |
| indrop | 1 | 18 | 95760 | 0.0000 | 0.0000 | 0.0000 | h_indrop (0.0000) |
| marsseq2 | 2 | 7 | 25794 | 0.0000 | 0.0000 | 0.0000 | h_mars (0.0000) |
| microwell-seq | 1 | 18 | 117485 | 0.0000 | 0.0000 | 0.0000 | h_microwell (0.0000) |
| quartzseq2 | 1 | 15 | 692824 | 0.0000 | 0.0000 | 0.0000 | h_quartz (0.0000) |
| splitseq | 1 | 24 | 85694 | 0.0000 | 0.0000 | 0.0000 | h_split (0.0000) |
| 10x-visium | 2 | 16 | 9982 | 0.0014 | 0.0000 | 0.0014 | h_visium (0.0014) |

Cross-sample reuse (fraction of a sample's barcodes that also occur in another sample of the same protocol in this corpus), only protocols with >= 2 samples:

| protocol | samples | mean fraction | min | max |
|---|---|---|---|---|
| 10xv3 | 14 | 0.251 | 0.246 | 0.254 |
| 10xv2 | 9 | 0.593 | 0.582 | 0.626 |
| 10x-arc-gex | 2 | 0.096 | 0.062 | 0.130 |
| 10x-visium | 2 | 1.000 | 1.000 | 1.000 |

### E4 derivability claims on the new samples (pass / compared)

| claim | item | compared | equal | samples | failing |
|---|---|---|---|---|---|
| colsums | exon_counts | 39 | 39 | 39 |  |
| colsums_frame_bytes_zstd155 | exon_counts | 39 | 24 | 39 | bn_v2:exon_counts, bn_v3_a:exon_counts, bn_v3_b:exon_counts, h_bd:exon_counts, h_microwell:exon_counts, m_arc:exon_counts, m_v2_b:exon_counts, m_v3_a:exon_counts, m_v3_b:exon_counts, m_v3_c:exon_counts, mq_v2:exon_counts, mq_v3:exon_counts |
| ptr | exon_counts | 39 | 39 | 39 |  |
| perm | exon_counts | 39 | 39 | 39 |  |
| colsums | intron_counts | 39 | 39 | 39 |  |
| colsums_frame_bytes_zstd155 | intron_counts | 39 | 19 | 39 | bn_dropseq:intron_counts, bn_v2:intron_counts, bn_v3_b:intron_counts, h_arc:intron_counts, h_bd:intron_counts, h_indrop:intron_counts, h_mars:intron_counts, h_microwell:intron_counts, m_arc:intron_counts, m_v2_a:intron_counts, m_v2_b:intron_counts, m_v3_b:intron_counts |
| ptr | intron_counts | 39 | 39 | 39 |  |
| perm | intron_counts | 39 | 39 | 39 |  |
| colsums | sj_counts | 39 | 39 | 39 |  |
| colsums_frame_bytes_zstd155 | sj_counts | 39 | 23 | 39 | L_b:sj_counts, bn_v2:sj_counts, bn_v3_b:sj_counts, h_bd:sj_counts, h_microwell:sj_counts, m_v2_b:sj_counts, m_v3_a:sj_counts, m_v3_b:sj_counts, m_v3_c:sj_counts, m_visium:sj_counts, mq_v2:sj_counts, mq_v3:sj_counts |
| ptr | sj_counts | 39 | 39 | 39 |  |
| perm | sj_counts | 39 | 39 | 39 |  |
| colsums | splice_psi | 35 | 35 | 35 |  |
| colsums_frame_bytes_zstd155 | splice_psi | 35 | 31 | 35 | h_bd:splice_psi, h_microwell:splice_psi, m_arc:splice_psi, m_v2_b:splice_psi |
| ptr | splice_psi | 35 | 35 | 35 |  |
| perm | splice_psi | 35 | 35 | 35 |  |
| psi_structure |  | 35 | 35 | 35 |  |
| psi_values_rint |  | 35 | 8 | 35 | bn_dropseq:splice_psi, bn_v2:splice_psi, bn_v3_a:splice_psi, bn_v3_b:splice_psi, h_arc:splice_psi, h_bd:splice_psi, h_dropseq:splice_psi, h_indrop:splice_psi, h_mars:splice_psi, h_microwell:splice_psi, h_visium:splice_psi, m_arc:splice_psi |
| psi_values_trunc |  | 35 | 30 | 35 | h_celseq2:splice_psi, h_quartz:splice_psi, h_v2_a:splice_psi, h_v2_b:splice_psi, h_v3_a:splice_psi |
| cell_calls_file_regen |  | 39 | 35 | 39 | h_bd:cell_calls.tsv, h_indrop:cell_calls.tsv, h_microwell:cell_calls.tsv, m_dropseq:cell_calls.tsv |
| cell_calls_log10 |  | 202625 | 202625 | 39 |  |
| summary_field |  | 663 | 631 | 39 | h_bd:estimated_cells, h_bd:fraction_reads_in_cells, h_bd:mean_reads_per_cell, h_bd:median_genes_per_cell, h_bd:median_umi_per_cell, h_bd:median_umis_per_cell, h_bd:n_cells_called, h_bd:total_genes_detected, h_indrop:estimated_cells, h_indrop:fraction_reads_in_cells, h_indrop:mean_reads_per_cell, h_indrop:median_genes_per_cell |
| pileup_stats_field |  | 195 | 135 | 39 | L_a:exon_hits, L_a:intron_hits, L_a:umi_unique, h_dropseq:exon_hits, h_dropseq:intron_hits, h_dropseq:umi_unique, h_visium:exon_hits, h_visium:intron_hits, h_visium:umi_unique, m_arc:exon_hits, m_arc:intron_hits, m_arc:umi_unique |
| provenance_field |  | 117 | 113 | 39 | h_bd:output.cells, h_indrop:output.cells, h_microwell:output.cells, m_dropseq:output.cells |
| psi_stub |  | 4 | 0 | 4 | L_b:splice_psi, h_v2_c:splice_psi, h_v3_b:splice_psi, h_v3_c:splice_psi |

### (4) Called cells vs the other columns (exon and intron), nnz share

| protocol | member | samples | columns | called | frac of columns called | frac of nnz in called (mean / min / max) | nnz per called col | nnz per other col |
|---|---|---|---|---|---|---|---|---|
| 10x-arc-gex | exon | 2 | 142605 | 11682 | 0.082 | 0.746 / 0.623 / 0.870 | 224 | 7.5 |
| 10x-arc-gex | intron | 2 | 142605 | 11682 | 0.082 | 0.883 / 0.853 / 0.913 | 532 | 5.9 |
| 10x-visium | exon | 2 | 9982 | 7851 | 0.787 | 0.942 / 0.938 / 0.945 | 468 | 112.2 |
| 10x-visium | intron | 2 | 9982 | 7851 | 0.787 | 0.923 / 0.898 / 0.949 | 104 | 29.6 |
| 10xv2 | exon | 9 | 703802 | 36746 | 0.052 | 0.662 / 0.291 / 0.887 | 990 | 26.6 |
| 10xv2 | intron | 9 | 703802 | 36746 | 0.052 | 0.738 / 0.446 / 0.909 | 279 | 4.4 |
| 10xv3 | exon | 14 | 1128182 | 130733 | 0.116 | 0.637 / 0.287 / 0.905 | 456 | 31.6 |
| 10xv3 | intron | 14 | 1128182 | 130733 | 0.116 | 0.732 / 0.313 / 0.916 | 241 | 9.5 |
| bd-rhapsody | exon | 1 | 60922 | 4541 | 0.075 | 0.710 / 0.710 / 0.710 | 766 | 25.2 |
| bd-rhapsody | intron | 1 | 60922 | 4541 | 0.075 | 0.521 / 0.521 / 0.521 | 83 | 6.1 |
| celseq2 | exon | 2 | 8192 | 1000 | 0.122 | 0.263 / 0.236 / 0.290 | 536 | 198.5 |
| celseq2 | intron | 2 | 8192 | 1000 | 0.122 | 0.280 / 0.244 / 0.316 | 770 | 239.5 |
| dropseq | exon | 3 | 703504 | 2674 | 0.004 | 0.161 / 0.016 / 0.286 | 1305 | 23.2 |
| dropseq | intron | 3 | 703504 | 2674 | 0.004 | 0.214 / 0.017 / 0.326 | 217 | 2.1 |
| indrop | exon | 1 | 95760 | 1572 | 0.016 | 0.227 / 0.227 / 0.227 | 301 | 17.1 |
| indrop | intron | 1 | 95760 | 1572 | 0.016 | 0.368 / 0.368 / 0.368 | 141 | 4.0 |
| marsseq2 | exon | 2 | 25794 | 1000 | 0.039 | 0.210 / 0.127 / 0.293 | 207 | 37.1 |
| marsseq2 | intron | 2 | 25794 | 1000 | 0.039 | 0.212 / 0.124 / 0.299 | 841 | 238.2 |
| microwell-seq | exon | 1 | 117485 | 3902 | 0.033 | 0.165 / 0.165 / 0.165 | 518 | 89.9 |
| microwell-seq | intron | 1 | 117485 | 3902 | 0.033 | 0.161 / 0.161 / 0.161 | 30 | 5.3 |
| quartzseq2 | exon | 1 | 692824 | 500 | 0.001 | 0.018 / 0.018 / 0.018 | 235 | 9.4 |
| quartzseq2 | intron | 1 | 692824 | 500 | 0.001 | 0.016 / 0.016 / 0.016 | 16 | 0.7 |
| splitseq | exon | 1 | 85694 | 424 | 0.005 | 0.133 / 0.133 / 0.133 | 50 | 1.6 |
| splitseq | intron | 1 | 85694 | 424 | 0.005 | 0.170 / 0.170 / 0.170 | 251 | 6.1 |
