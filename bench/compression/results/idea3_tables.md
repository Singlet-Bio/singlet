### A1. Gene / interval structure

intervals per gene: mean / median / p90 / max over the reference; "UMI-wtd" = mean intervals of the gene a random UMI falls in; "1-ivl %" = share of UMIs in single-interval genes (no split to code); "c=1 %" = share of UMIs in (cell, gene) pairs with exactly one UMI.

| sample | kind | rows m | genes G | ivl/gene | med | p90 | max | UMI-wtd | 1-ivl % | nnz_int | nnz_gene | nnz_int/nnz_gene | UMI/nnz_int | c=1 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical | exon | 310797 | 38606 | 8.1 | 4 | 19 | 368 | 6.5 | 12.5 | 2569228 | 2387570 | 1.076 | 1.82 | 36.8 |
| pbmc_v2_typical | intron | 272191 | 33959 | 8.0 | 4 | 20 | 367 | 18.6 | 2.3 | 1202661 | 1141927 | 1.053 | 1.05 | 82.7 |
| pbmc_v2_shallow | exon | 310797 | 38606 | 8.1 | 4 | 19 | 368 | 6.7 | 9.5 | 5304754 | 5205822 | 1.019 | 1.10 | 80.6 |
| pbmc_v2_shallow | intron | 272191 | 33959 | 8.0 | 4 | 20 | 367 | 18.5 | 2.2 | 1170031 | 1166049 | 1.003 | 1.00 | 98.5 |
| pbmc_v3_lowgenes | exon | 310797 | 38606 | 8.1 | 4 | 19 | 368 | 8.7 | 4.7 | 1759828 | 1364137 | 1.290 | 2.42 | 20.1 |
| pbmc_v3_lowgenes | intron | 272191 | 33959 | 8.0 | 4 | 20 | 367 | 19.2 | 2.7 | 1076297 | 795012 | 1.354 | 1.48 | 29.7 |
| tcell_5p | exon | 310797 | 38606 | 8.1 | 4 | 19 | 368 | 7.8 | 7.6 | 10529513 | 9187001 | 1.146 | 1.66 | 38.8 |
| tcell_5p | intron | 272191 | 33959 | 8.0 | 4 | 20 | 367 | 15.5 | 5.1 | 1098311 | 1076780 | 1.020 | 1.04 | 90.2 |
| tumor_3p | exon | 310797 | 38606 | 8.1 | 4 | 19 | 368 | 10.2 | 11.9 | 245117 | 239885 | 1.022 | 1.11 | 81.4 |
| tumor_3p | intron | 272191 | 33959 | 8.0 | 4 | 20 | 367 | 14.7 | 7.8 | 107364 | 106134 | 1.012 | 1.11 | 85.0 |
| brain_sn_v3_deep | exon | 386028 | 78298 | 4.9 | 2 | 12 | 345 | 10.7 | 30.4 | 6349311 | 6043037 | 1.051 | 1.43 | 59.0 |
| brain_sn_v3_deep | intron | 307730 | 53217 | 5.8 | 3 | 14 | 344 | 19.9 | 1.1 | 6678099 | 5274547 | 1.266 | 1.29 | 47.6 |
| brain_sn_v3_intronic | exon | 386028 | 78298 | 4.9 | 2 | 12 | 345 | 9.9 | 42.8 | 1380834 | 1368893 | 1.009 | 1.45 | 61.8 |
| brain_sn_v3_intronic | intron | 307730 | 53217 | 5.8 | 3 | 14 | 344 | 20.0 | 1.5 | 8144777 | 6868869 | 1.186 | 1.18 | 59.4 |

### A2. Cost of the interval choice, bits per UMI (H(interval | gene))

in = in-sample static per-gene interval table (optimistic: table not paid for); in|c = also conditioned on the (cell,gene) UMI count bucket {1, 2, 3-4, 5+}; c=1 / c>=5 = the per-bucket values; multiset = in-sample static but coding the split as an exact multiset (what a codec has to transmit; smaller than iid because the order of UMIs within a gene is not information); pool = static table pooled over all other studies of the same build; best1 = the single best other sample (oracle choice); length = zero-shot prior proportional to interval length; adapt = adaptive Dirichlet-multinomial from a flat prior (self-contained: no table transmitted, no model); adapt+best1 = adaptive from the best1 prior. Last column: adapt in bits per interval nonzero.

| sample | kind | in | in|c | c=1 | c>=5 | in multiset | pool | best1 | best1 id | length | adapt | adapt+best1 | adapt b/nnz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical | exon | 0.512 | 0.508 | 0.74 | 0.31 | 0.413 | 1.460 | 0.599 | h_c | 1.616 | 0.462 | 0.439 | 0.841 |
| pbmc_v2_typical | intron | 2.050 | 2.019 | 2.04 | 0.94 | 1.994 | 2.310 | 2.239 | h_c | 2.895 | 2.203 | 2.114 | 2.315 |
| pbmc_v2_shallow | exon | 0.648 | 0.647 | 0.70 | 0.21 | 0.628 | 1.492 | 0.695 | h_c | 1.691 | 0.671 | 0.651 | 0.741 |
| pbmc_v2_shallow | intron | 2.065 | 2.056 | 2.06 | 1.65 | 2.062 | 2.347 | 2.281 | h_c | 2.891 | 2.285 | 2.196 | 2.294 |
| pbmc_v3_lowgenes | exon | 1.256 | 1.251 | 1.39 | 1.14 | 0.705 | 1.821 | 1.367 | h_c | 2.023 | 0.714 | 0.713 | 1.731 |
| pbmc_v3_lowgenes | intron | 2.416 | 2.394 | 2.22 | 2.58 | 1.994 | 2.632 | 2.615 | h_c | 2.966 | 2.033 | 2.029 | 3.006 |
| tcell_5p | exon | 1.077 | 1.072 | 1.49 | 0.63 | 0.871 | 1.538 | 1.125 | h_b | 2.388 | 0.894 | 0.888 | 1.486 |
| tcell_5p | intron | 1.938 | 1.916 | 1.99 | 0.43 | 1.917 | 2.322 | 2.203 | h_b | 3.093 | 2.146 | 2.085 | 2.229 |
| tumor_3p | exon | 1.146 | 1.123 | 1.26 | 0.24 | 1.120 | 2.341 | 2.342 | h_pbmc_a | 2.003 | 1.547 | 1.651 | 1.724 |
| tumor_3p | intron | 1.210 | 1.175 | 1.25 | 0.71 | 1.193 | 2.656 | 2.421 | h_d | 2.847 | 1.960 | 1.930 | 2.171 |
| brain_sn_v3_deep | exon | 1.150 | 1.137 | 1.54 | 0.11 | 1.099 | 1.345 | 1.297 | m_brain_b | 1.697 | 1.152 | 1.147 | 1.645 |
| brain_sn_v3_deep | intron | 2.344 | 2.325 | 2.35 | 2.26 | 1.935 | 2.635 | 2.487 | m_brain_b | 2.759 | 1.991 | 1.984 | 2.572 |
| brain_sn_v3_intronic | exon | 0.659 | 0.654 | 1.00 | 0.00 | 0.653 | 0.948 | 0.784 | m_brain_d | 1.315 | 0.784 | 0.736 | 1.136 |
| brain_sn_v3_intronic | intron | 2.159 | 2.148 | 2.07 | 2.18 | 1.892 | 2.248 | 2.226 | m_brain_d | 2.619 | 1.940 | 1.925 | 2.280 |

### A3. Interval-level vs gene-level coding, bits per interval nonzero (cross-entropies, no coder overhead)

ref = today's .1pz matrix chunk payload (zstd-3 VOCSC, Track A accounting). "int" = per-row Bernoulli presence x depth-bucket + per-row count model, i.e. the direct interval-level model; "gene" = the same model at gene level + the adaptive interval split. in-sample(cls) = per-row rates quantised to 64 rate classes, class x bucket tables transmitted (side column counts them, and the per-gene class list is in "side"); oos(pool) = per-feature x bucket tables pooled over the other studies, shrunk (m=30) to the class rate; oos(best1) = class-only tables from the single best other sample (class list from that sample, nothing transmitted). The gene-level columns P / C / S / side split the in-sample total into presence / counts / split / side info.

| sample | kind | ref | int in | gene in | P | C | S | side | int oos pool | gene oos pool | int oos best1 | gene oos best1 | best1 id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical | exon | 6.80 | 5.91 | 5.88 | 3.78 | 1.17 | 0.84 | 0.05 | 7.90 | 6.56 | 6.29 | 6.21 | h_c |
| pbmc_v2_typical | intron | 9.10 | 8.50 | 8.41 | 5.51 | 0.41 | 2.32 | 0.11 | 9.34 | 8.98 | 8.71 | 8.69 | h_pbmc_b |
| pbmc_v2_shallow | exon | 8.69 | 7.08 | 7.07 | 5.79 | 0.42 | 0.74 | 0.03 | 10.92 | 9.75 | 7.75 | 7.67 | h_c |
| pbmc_v2_shallow | intron | 13.88 | 12.46 | 12.35 | 9.52 | 0.06 | 2.29 | 0.11 | 15.88 | 14.03 | 12.74 | 12.73 | h_pbmc_b |
| pbmc_v3_lowgenes | exon | 7.16 | 5.70 | 5.70 | 2.35 | 1.45 | 1.73 | 0.02 | 11.82 | 10.03 | 10.74 | 9.52 | h_pbmc_b |
| pbmc_v3_lowgenes | intron | 8.55 | 7.19 | 7.34 | 2.76 | 1.30 | 3.01 | 0.06 | 12.79 | 11.07 | 12.07 | 10.90 | h_c |
| tcell_5p | exon | 6.92 | 6.09 | 6.05 | 3.47 | 1.05 | 1.49 | 0.01 | 7.14 | 6.49 | 6.57 | 6.43 | h_b |
| tcell_5p | intron | 11.33 | 10.22 | 10.06 | 7.23 | 0.25 | 2.23 | 0.12 | 11.72 | 11.05 | 10.70 | 10.49 | h_b |
| tumor_3p | exon | 11.91 | 10.80 | 10.56 | 7.51 | 0.40 | 1.72 | 0.41 | 15.19 | 13.65 | 13.79 | 11.60 | h_d |
| tumor_3p | intron | 14.31 | 13.92 | 13.64 | 9.35 | 0.28 | 2.17 | 0.83 | 17.70 | 15.93 | 16.38 | 15.05 | h_d |
| brain_sn_v3_deep | exon | 9.35 | 8.19 | 8.15 | 5.81 | 0.57 | 1.65 | 0.04 | 9.91 | 9.14 | 9.30 | 9.00 | m_brain_b |
| brain_sn_v3_deep | intron | 8.25 | 7.37 | 7.31 | 3.84 | 0.80 | 2.57 | 0.03 | 8.50 | 7.85 | 8.35 | 7.83 | m_brain_a |
| brain_sn_v3_intronic | exon | 11.55 | 9.78 | 9.72 | 7.72 | 0.47 | 1.14 | 0.15 | 10.86 | 10.47 | 9.95 | 9.90 | m_brain_d |
| brain_sn_v3_intronic | intron | 8.38 | 7.57 | 7.53 | 4.50 | 0.68 | 2.28 | 0.03 | 8.01 | 7.77 | 7.73 | 7.69 | m_brain_d |

### A4. Where the bits go in the factorised model (in-sample, bits per interval nonzero)

which gene = gene presence; which interval = adaptive split; how many = gene count given presence. "split %" = share of the total that the interval choice costs; "static split %" = the same with an in-sample static table (lower bound). last column = extra presence bits the flat interval-level model pays over the gene-level presence (the price of treating every interval as an independent feature).

| sample | kind | which gene | which interval | how many | split % | static split % | int-gene presence |
|---|---|---|---|---|---|---|---|
| pbmc_v2_typical | exon | 3.78 | 0.84 | 1.17 | 14.4 | 12.9 | 0.78 |
| pbmc_v2_typical | intron | 5.51 | 2.32 | 0.41 | 27.9 | 25.2 | 2.26 |
| pbmc_v2_shallow | exon | 5.79 | 0.74 | 0.42 | 10.5 | 9.8 | 0.75 |
| pbmc_v2_shallow | intron | 9.52 | 2.29 | 0.06 | 18.8 | 16.9 | 2.09 |
| pbmc_v3_lowgenes | exon | 2.35 | 1.73 | 1.45 | 30.5 | 30.1 | 1.69 |
| pbmc_v3_lowgenes | intron | 2.76 | 3.01 | 1.30 | 41.3 | 40.5 | 2.85 |
| tcell_5p | exon | 3.47 | 1.49 | 1.05 | 24.6 | 24.0 | 1.56 |
| tcell_5p | intron | 7.23 | 2.23 | 0.25 | 22.4 | 20.0 | 2.08 |
| tumor_3p | exon | 7.51 | 1.72 | 0.40 | 17.0 | 12.3 | 1.33 |
| tumor_3p | intron | 9.35 | 2.17 | 0.28 | 16.9 | 10.3 | 1.31 |
| brain_sn_v3_deep | exon | 5.81 | 1.65 | 0.57 | 20.3 | 19.4 | 1.69 |
| brain_sn_v3_deep | intron | 3.84 | 2.57 | 0.80 | 35.3 | 34.3 | 2.69 |
| brain_sn_v3_intronic | exon | 7.72 | 1.14 | 0.47 | 11.9 | 9.9 | 0.98 |
| brain_sn_v3_intronic | intron | 4.50 | 2.28 | 0.68 | 30.4 | 29.6 | 2.44 |

### B1. Exon -> intron information at gene level (per cell), 2-fold cross-validated (cells by parity)

All costs in bits per intron gene-nonzero. A = P(intron present | gene rate class, intron depth bucket); +e = also conditioned on the exon count bucket of the same gene in the same cell {0, 1, 2-3, 4+}; +cE = also on the exon depth class of the cell; MI = raw mutual information I(exon bucket; intron presence) without conditioning (bits per intron gene-nnz); P(I|e) = P(intron present | e bucket); cnt = count given presence; k = bits for the per-cell intron gene count, alone vs residual against round(ratio x k_exon); "w/ exon %" = share of intron gene-nonzeros whose gene has exon UMIs in the same cell.

| sample | nnz_gene_I | A | A+e | A+cE | A+cE+e | gain e|A | gain e|A+cE | MI raw | P(I|e=0/1/2-3/4+) | cnt | cnt+e | k alone | k|kE | corr | w/ exon % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical | 1141927 | 5.800 | 5.748 | 6.236 | 6.186 | 0.052 | 0.050 | 0.309 | 0.003 / 0.058 / 0.050 / 0.071 | 0.435 | 0.433 | 5.37 | 5.87 | 0.96 | 11.1 |
| pbmc_v2_shallow | 1166049 | 9.605 | 9.431 | 9.837 | 9.665 | 0.174 | 0.173 | 0.329 | 0.000 / 0.019 / 0.017 / 0.026 | 0.060 | 0.061 | 4.46 | 4.31 | 0.80 | 7.6 |
| pbmc_v3_lowgenes | 795012 | 3.732 | 3.651 | 3.900 | 3.821 | 0.080 | 0.079 | 2.867 | 0.000 / 0.194 / 0.289 / 0.383 | 1.746 | 1.744 | 3.34 | 3.19 | 0.92 | 38.4 |
| tcell_5p | 1076780 | 7.377 | 6.955 | 7.940 | 7.517 | 0.422 | 0.424 | 1.716 | 0.001 / 0.044 / 0.051 / 0.061 | 0.254 | 0.252 | 3.74 | 3.32 | 0.95 | 36.8 |
| tumor_3p | 106134 | 9.470 | 8.507 | 12.917 | 11.950 | 0.963 | 0.968 | 1.693 | 0.000 / 0.107 / 0.113 / 0.083 | 0.287 | 0.288 | 4.31 | 3.23 | 0.86 | 21.2 |
| brain_sn_v3_deep | 5274547 | 4.858 | 4.753 | 4.926 | 4.820 | 0.106 | 0.106 | 1.012 | 0.001 / 0.143 / 0.339 / 0.458 | 0.988 | 0.983 | 5.51 | 5.51 | 0.90 | 16.8 |
| brain_sn_v3_intronic | 6868869 | 5.331 | 5.274 | 5.406 | 5.350 | 0.057 | 0.057 | 0.227 | 0.002 / 0.233 / 0.402 / 0.296 | 0.796 | 0.795 | 5.58 | 7.08 | 0.93 | 3.8 |


