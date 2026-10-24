# idea3 codec numbers (payload only; names/colnames excluded for every codec; ref = zstd-3 VOCSC chunk payload + ptr + colsums)

| sample | member | codec | nnz | ref_bpn | payload_bits_per_nnz | vs_ref_% | K | P | C | S | side | dec_cpu_Mnnz_s | enc_cpu_ms | dec_cpu_ms_min | matrix_ok | model_MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical | exon_counts.1pz | idea3_gene_then_interval | 2569228 | 6.898 | 5.812 | -15.745 | 0.032 | 3.766 | 1.112 | 0.844 | 0.048 | 0.64 | 6516.0 | 4031.0 | 1 | 0.0 |
| pbmc_v2_typical | exon_counts.1pz | idea3_interval_direct | 2569228 | 6.898 | 5.842 | -15.315 | 0.032 | 4.548 | 1.083 | 0.0 | 0.167 | 0.47 | 7531.0 | 5438.0 | 1 | 0.0 |
| pbmc_v2_typical | intron_counts.1pz | idea3_gene_then_interval | 1202661 | 9.27 | 8.404 | -9.342 | 0.058 | 5.489 | 0.411 | 2.338 | 0.097 | 0.51 | 3609.0 | 2359.0 | 1 | 0.0 |
| pbmc_v2_typical | intron_counts.1pz | idea3_interval_direct | 1202661 | 9.27 | 8.486 | -8.456 | 0.059 | 7.752 | 0.242 | 0.0 | 0.421 | 0.41 | 4531.0 | 2953.0 | 1 | 0.0 |
| tumor_3p | exon_counts.1pz | idea3_gene_then_interval | 245117 | 13.005 | 10.419 | -19.889 | 0.473 | 7.36 | 0.433 | 1.779 | 0.365 | 0.28 | 2188.0 | 891.0 | 1 | 0.0 |
| tumor_3p | exon_counts.1pz | idea3_interval_direct | 245117 | 13.005 | 10.612 | -18.401 | 0.475 | 8.681 | 0.36 | 0.0 | 1.085 | 0.21 | 2672.0 | 1156.0 | 1 | 0.0 |
| tumor_3p | intron_counts.1pz | idea3_gene_then_interval | 107364 | 16.442 | 13.302 | -19.099 | 0.919 | 9.053 | 0.341 | 2.237 | 0.743 | 0.17 | 1797.0 | 625.0 | 1 | 0.0 |
| tumor_3p | intron_counts.1pz | idea3_interval_direct | 107364 | 16.442 | 13.538 | -17.664 | 0.921 | 10.351 | 0.316 | 0.0 | 1.939 | 0.14 | 1969.0 | 766.0 | 1 | 0.0 |
| pbmc_v3_lowgenes | exon_counts.1pz | idea3_gene_then_interval | 1759828 | 7.551 | 5.542 | -26.6 | 0.133 | 2.289 | 1.355 | 1.732 | 0.023 | 0.54 | 5078.0 | 3250.0 | 1 | 0.0 |
| pbmc_v3_lowgenes | exon_counts.1pz | idea3_interval_direct | 1759828 | 7.551 | 5.586 | -26.02 | 0.138 | 3.973 | 1.396 | 0.0 | 0.066 | 0.5 | 5141.0 | 3500.0 | 1 | 0.0 |
| pbmc_v3_lowgenes | intron_counts.1pz | idea3_gene_then_interval | 1076297 | 9.077 | 7.24 | -20.233 | 0.177 | 2.706 | 1.29 | 3.004 | 0.054 | 0.42 | 4062.0 | 2562.0 | 1 | 0.0 |
| pbmc_v3_lowgenes | intron_counts.1pz | idea3_interval_direct | 1076297 | 9.077 | 7.098 | -21.798 | 0.185 | 5.551 | 1.206 | 0.0 | 0.145 | 0.42 | 4047.0 | 2562.0 | 1 | 0.0 |
| pbmc_v2_shallow | exon_counts.1pz | idea3_gene_then_interval | 5304754 | 8.908 | 7.026 | -21.126 | 0.086 | 5.752 | 0.413 | 0.741 | 0.025 | 0.58 | 11594.0 | 9203.0 | 1 | 0.0 |
| pbmc_v2_shallow | exon_counts.1pz | idea3_interval_direct | 5304754 | 8.908 | 7.043 | -20.936 | 0.086 | 6.498 | 0.359 | 0.0 | 0.088 | 0.43 | 16188.0 | 12344.0 | 1 | 0.0 |
| pbmc_v2_shallow | intron_counts.1pz | idea3_gene_then_interval | 1170031 | 14.691 | 12.187 | -17.045 | 0.357 | 9.337 | 0.063 | 2.32 | 0.102 | 0.4 | 4438.0 | 2953.0 | 1 | 0.0 |
| pbmc_v2_shallow | intron_counts.1pz | idea3_interval_direct | 1170031 | 14.691 | 12.272 | -16.464 | 0.358 | 11.426 | 0.04 | 0.0 | 0.438 | 0.27 | 6125.0 | 4375.0 | 1 | 0.0 |
| brain_sn_v3_deep | exon_counts.1pz | idea3_gene_then_interval | 6349311 | 9.529 | 8.12 | -14.783 | 0.072 | 5.794 | 0.545 | 1.663 | 0.036 | 0.54 | 15312.0 | 11688.0 | 1 | 0.0 |
| brain_sn_v3_deep | exon_counts.1pz | idea3_interval_direct | 6349311 | 9.529 | 8.144 | -14.534 | 0.072 | 7.484 | 0.424 | 0.0 | 0.153 | 0.4 | 19453.0 | 16047.0 | 1 | 0.0 |
| brain_sn_v3_deep | intron_counts.1pz | idea3_gene_then_interval | 6678099 | 8.413 | 7.292 | -13.323 | 0.064 | 3.814 | 0.78 | 2.594 | 0.029 | 0.53 | 15359.0 | 12547.0 | 1 | 0.0 |
| brain_sn_v3_deep | intron_counts.1pz | idea3_interval_direct | 6678099 | 8.413 | 7.326 | -12.918 | 0.065 | 6.507 | 0.602 | 0.0 | 0.14 | 0.51 | 16641.0 | 13109.0 | 1 | 0.0 |
| brain_sn_v3_intronic | exon_counts.1pz | idea3_gene_then_interval | 1380834 | 12.162 | 9.609 | -20.992 | 0.232 | 7.656 | 0.43 | 1.137 | 0.144 | 0.37 | 5422.0 | 3734.0 | 1 | 0.0 |
| brain_sn_v3_intronic | exon_counts.1pz | idea3_interval_direct | 1380834 | 12.162 | 9.669 | -20.495 | 0.232 | 8.636 | 0.394 | 0.0 | 0.396 | 0.31 | 6719.0 | 4453.0 | 1 | 0.0 |
| brain_sn_v3_intronic | intron_counts.1pz | idea3_gene_then_interval | 8144777 | 8.513 | 7.503 | -11.866 | 0.051 | 4.473 | 0.655 | 2.29 | 0.024 | 0.53 | 18516.0 | 15344.0 | 1 | 0.0 |
| brain_sn_v3_intronic | intron_counts.1pz | idea3_interval_direct | 8144777 | 8.513 | 7.538 | -11.456 | 0.052 | 6.908 | 0.463 | 0.0 | 0.104 | 0.51 | 19641.0 | 16000.0 | 1 | 0.0 |
| tcell_5p | exon_counts.1pz | idea3_gene_then_interval | 10529513 | 6.998 | 6.009 | -14.127 | 0.029 | 3.452 | 1.007 | 1.498 | 0.013 | 0.69 | 18031.0 | 15297.0 | 1 | 0.0 |
| tcell_5p | exon_counts.1pz | idea3_interval_direct | 10529513 | 6.998 | 6.038 | -13.724 | 0.03 | 5.01 | 0.913 | 0.0 | 0.073 | 0.55 | 22766.0 | 19047.0 | 1 | 0.0 |
| tcell_5p | intron_counts.1pz | idea3_gene_then_interval | 1098311 | 11.844 | 10.018 | -15.415 | 0.196 | 7.171 | 0.25 | 2.285 | 0.106 | 0.46 | 3812.0 | 2375.0 | 1 | 0.0 |
| tcell_5p | intron_counts.1pz | idea3_interval_direct | 1098311 | 11.844 | 10.124 | -14.519 | 0.197 | 9.246 | 0.166 | 0.0 | 0.505 | 0.34 | 5312.0 | 3234.0 | 1 | 0.0 |
| tumor_3p | exon_counts.1pz | idea3_gene_then_interval_oos | 245117 | 13.005 | 14.365 | 10.452 | 0.473 | 10.828 | 0.483 | 2.572 | 0.0 | 0.33 | 2094.0 | 750.0 | 1 | 2.202 |
| tumor_3p | exon_counts.1pz | idea3_gti_oos_adapt | 245117 | 13.005 | 13.579 | 4.412 | 0.473 | 10.828 | 0.483 | 1.786 | 0.0 | 0.3 | 2266.0 | 812.0 | 1 | 2.202 |
| tumor_3p | intron_counts.1pz | idea3_gene_then_interval_oos | 107364 | 16.442 | 16.455 | 0.078 | 0.919 | 12.23 | 0.387 | 2.911 | 0.0 | 0.2 | 1734.0 | 531.0 | 1 | 1.936 |
| tumor_3p | intron_counts.1pz | idea3_gti_oos_adapt | 107364 | 16.442 | 15.694 | -4.552 | 0.919 | 12.23 | 0.387 | 2.149 | 0.0 | 0.18 | 1828.0 | 594.0 | 1 | 1.936 |
| pbmc_v2_typical | exon_counts.1pz | idea3_gene_then_interval_oos | 2569228 | 6.898 | 8.078 | 17.101 | 0.032 | 4.424 | 1.214 | 2.398 | 0.0 | 0.69 | 5719.0 | 3750.0 | 1 | 2.202 |
| pbmc_v2_typical | exon_counts.1pz | idea3_gti_oos_adapt | 2569228 | 6.898 | 6.518 | -5.52 | 0.032 | 4.424 | 1.214 | 0.837 | 0.0 | 0.69 | 5578.0 | 3750.0 | 1 | 2.202 |
| pbmc_v2_typical | intron_counts.1pz | idea3_gene_then_interval_oos | 1202661 | 9.27 | 9.019 | -2.703 | 0.058 | 6.142 | 0.439 | 2.37 | 0.0 | 0.59 | 3547.0 | 2047.0 | 1 | 1.936 |
| pbmc_v2_typical | intron_counts.1pz | idea3_gti_oos_adapt | 1202661 | 9.27 | 8.939 | -3.572 | 0.058 | 6.142 | 0.439 | 2.289 | 0.0 | 0.5 | 3562.0 | 2391.0 | 1 | 1.936 |
| pbmc_v3_lowgenes | exon_counts.1pz | idea3_gene_then_interval_oos | 1759828 | 7.551 | 11.078 | 46.705 | 0.133 | 6.612 | 1.415 | 2.909 | 0.0 | 0.52 | 4766.0 | 3406.0 | 1 | 2.202 |
| pbmc_v3_lowgenes | exon_counts.1pz | idea3_gti_oos_adapt | 1759828 | 7.551 | 9.9 | 31.114 | 0.133 | 6.612 | 1.415 | 1.732 | 0.0 | 0.52 | 4953.0 | 3375.0 | 1 | 2.202 |
| brain_sn_v3_deep | exon_counts.1pz | idea3_gene_then_interval_oos | 6349311 | 9.529 | 9.304 | -2.366 | 0.072 | 6.796 | 0.576 | 1.849 | 0.0 | 0.56 | 13750.0 | 11328.0 | 1 | 4.223 |
| brain_sn_v3_deep | exon_counts.1pz | idea3_gti_oos_adapt | 6349311 | 9.529 | 9.117 | -4.327 | 0.072 | 6.796 | 0.576 | 1.662 | 0.0 | 0.52 | 14578.0 | 12156.0 | 1 | 4.223 |
| tcell_5p | exon_counts.1pz | idea3_gene_then_interval_oos | 10529513 | 6.998 | 7.174 | 2.521 | 0.029 | 3.856 | 1.075 | 2.204 | 0.0 | 0.72 | 17406.0 | 14688.0 | 1 | 2.202 |
| tcell_5p | exon_counts.1pz | idea3_gti_oos_adapt | 10529513 | 6.998 | 6.465 | -7.611 | 0.029 | 3.856 | 1.075 | 1.495 | 0.0 | 0.69 | 17875.0 | 15297.0 | 1 | 2.202 |
| brain_sn_v3_intronic | intron_counts.1pz | idea3_gene_then_interval_oos | 8144777 | 8.513 | 7.774 | -8.691 | 0.051 | 4.703 | 0.68 | 2.33 | 0.0 | 0.54 | 17859.0 | 15078.0 | 1 | 2.915 |
| brain_sn_v3_intronic | intron_counts.1pz | idea3_gti_oos_adapt | 8144777 | 8.513 | 7.724 | -9.27 | 0.051 | 4.703 | 0.68 | 2.281 | 0.0 | 0.5 | 18750.0 | 16234.0 | 1 | 2.915 |


# joint exon + intron

| sample | exon_payload_B | intron_indep_B | intron_nocond_B | intron_cond_B | saved_B | saved_bits_per_intron_nnz | saved_pct_of_intron | saved_pct_of_joint | both_ok |
|---|---|---|---|---|---|---|---|---|---|
| brain_sn_v3_deep | 6444923 | 6087009 | 6090177 | 6020286 | 66723 | 0.08 | 1.096 | 0.532 | 1 |
| brain_sn_v3_intronic | 1658498 | 7638972 | 7641924 | 7592799 | 46173 | 0.045 | 0.604 | 0.497 | 1 |
| pbmc_v2_shallow | 4659209 | 1782392 | 1784984 | 1759668 | 22724 | 0.155 | 1.275 | 0.353 | 1 |
| pbmc_v2_typical | 1866646 | 1263361 | 1265809 | 1258166 | 5195 | 0.035 | 0.411 | 0.166 | 1 |
| pbmc_v3_lowgenes | 1219227 | 974113 | 976849 | 968942 | 5171 | 0.038 | 0.531 | 0.236 | 1 |
| tcell_5p | 7909592 | 1375379 | 1378043 | 1321251 | 54128 | 0.394 | 3.935 | 0.583 | 1 |
| tumor_3p | 319222 | 178521 | 180753 | 167812 | 10709 | 0.798 | 5.999 | 2.152 | 1 |
