### C1. Payload bits per interval nonzero (self-described tables, no shipped model; payload only)

| sample | matrix | nnz | today b/nnz | gene->interval b/nnz | vs today % | k | which gene | how many | which interval | side | interval-direct b/nnz | dec Mnnz/s (cpu) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical | exon | 2569228 | 6.9 | 5.81 | -15.7 | 0.03 | 3.77 | 1.11 | 0.84 | 0.05 | 5.84 | 0.64 |
| pbmc_v2_typical | intron | 1202661 | 9.27 | 8.4 | -9.3 | 0.06 | 5.49 | 0.41 | 2.34 | 0.1 | 8.49 | 0.51 |
| pbmc_v2_shallow | exon | 5304754 | 8.91 | 7.03 | -21.1 | 0.09 | 5.75 | 0.41 | 0.74 | 0.02 | 7.04 | 0.58 |
| pbmc_v2_shallow | intron | 1170031 | 14.69 | 12.19 | -17.0 | 0.36 | 9.34 | 0.06 | 2.32 | 0.1 | 12.27 | 0.4 |
| pbmc_v3_lowgenes | exon | 1759828 | 7.55 | 5.54 | -26.6 | 0.13 | 2.29 | 1.35 | 1.73 | 0.02 | 5.59 | 0.54 |
| pbmc_v3_lowgenes | intron | 1076297 | 9.08 | 7.24 | -20.2 | 0.18 | 2.71 | 1.29 | 3.0 | 0.05 | 7.1 | 0.42 |
| tcell_5p | exon | 10529513 | 7.0 | 6.01 | -14.1 | 0.03 | 3.45 | 1.01 | 1.5 | 0.01 | 6.04 | 0.69 |
| tcell_5p | intron | 1098311 | 11.84 | 10.02 | -15.4 | 0.2 | 7.17 | 0.25 | 2.29 | 0.11 | 10.12 | 0.46 |
| tumor_3p | exon | 245117 | 13.01 | 10.42 | -19.9 | 0.47 | 7.36 | 0.43 | 1.78 | 0.36 | 10.61 | 0.28 |
| tumor_3p | intron | 107364 | 16.44 | 13.3 | -19.1 | 0.92 | 9.05 | 0.34 | 2.24 | 0.74 | 13.54 | 0.17 |
| brain_sn_v3_deep | exon | 6349311 | 9.53 | 8.12 | -14.8 | 0.07 | 5.79 | 0.55 | 1.66 | 0.04 | 8.14 | 0.54 |
| brain_sn_v3_deep | intron | 6678099 | 8.41 | 7.29 | -13.3 | 0.06 | 3.81 | 0.78 | 2.59 | 0.03 | 7.33 | 0.53 |
| brain_sn_v3_intronic | exon | 1380834 | 12.16 | 9.61 | -21.0 | 0.23 | 7.66 | 0.43 | 1.14 | 0.14 | 9.67 | 0.37 |
| brain_sn_v3_intronic | intron | 8144777 | 8.51 | 7.5 | -11.9 | 0.05 | 4.47 | 0.66 | 2.29 | 0.02 | 7.54 | 0.53 |


### C2. Out-of-sample (leave-one-study-out pooled tables) vs self-described, bits per interval nonzero

| sample | matrix | today b/nnz | self-described | oos static split | oos + adaptive split | oos static vs today % | model MB |
|---|---|---|---|---|---|---|---|
| tumor_3p | exon | 13.01 | 10.42 | 14.36 | 13.58 | 10.5 | 2.2 |
| tumor_3p | intron | 16.44 | 13.3 | 16.46 | 15.69 | 0.1 | 1.94 |
| pbmc_v2_typical | exon | 6.9 | 5.81 | 8.08 | 6.52 | 17.1 | 2.2 |
| pbmc_v2_typical | intron | 9.27 | 8.4 | 9.02 | 8.94 | -2.7 | 1.94 |
| pbmc_v3_lowgenes | exon | 7.55 | 5.54 | 11.08 | 9.9 | 46.7 | 2.2 |
| brain_sn_v3_deep | exon | 9.53 | 8.12 | 9.3 | 9.12 | -2.4 | 4.22 |
| tcell_5p | exon | 7.0 | 6.01 | 7.17 | 6.47 | 2.5 | 2.2 |
| brain_sn_v3_intronic | intron | 8.51 | 7.5 | 7.77 | 7.72 | -8.7 | 2.92 |


### C3. Intron coded conditionally on the decoded exon matrix (payload bytes; saved = independent - conditional)

| sample | exon_B | intron_indep_B | intron_cond_B | saved_B | saved_bits_per_intron_nnz | saved_pct_of_intron | saved_pct_of_joint | both_ok |
|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical | 1866646 | 1263361 | 1258166 | 5195 | 0.035 | 0.411 | 0.166 | 1 |
| pbmc_v2_shallow | 4659209 | 1782392 | 1759668 | 22724 | 0.155 | 1.275 | 0.353 | 1 |
| pbmc_v3_lowgenes | 1219227 | 974113 | 968942 | 5171 | 0.038 | 0.531 | 0.236 | 1 |
| tcell_5p | 7909592 | 1375379 | 1321251 | 54128 | 0.394 | 3.935 | 0.583 | 1 |
| tumor_3p | 319222 | 178521 | 167812 | 10709 | 0.798 | 5.999 | 2.152 | 1 |
| brain_sn_v3_deep | 6444923 | 6087009 | 6020286 | 66723 | 0.08 | 1.096 | 0.532 | 1 |
| brain_sn_v3_intronic | 1658498 | 7638972 | 7592799 | 46173 | 0.045 | 0.604 | 0.497 | 1 |
