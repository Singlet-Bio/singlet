### Table 1. Payload bits/nnz (names excluded), out of sample unless marked ins

| file | nnz (M) | VOCSC zstd-3 chunks | s1 order-0 | s2 f x depth | s3 + counts | s3r rank-1 | s3 ins | s3r ins | model CE (py, s3) | coder - CE | oracle f x d (file, presence only) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 2.57 | 6.85 | 9.18 | 7.75 | 7.50 | 7.50 | 6.87 | 6.88 | 7.505 | -0.010 | 4.45 |
| pbmc_v2_typical intron | 1.20 | 9.18 | 10.94 | 8.88 | 8.85 | 8.85 | 8.59 | 8.59 | 8.846 | 0.002 | 7.54 |
| pbmc_v2_shallow exon | 5.30 | 8.81 | 9.14 | 8.73 | 8.63 | 8.65 | 7.87 | 7.91 | 8.631 | -0.001 | 6.51 |
| pbmc_v2_shallow intron | 1.17 | 14.33 | 13.55 | 12.77 | 12.72 | 12.72 | 12.38 | 12.38 | 12.716 | 0.003 | 11.38 |
| pbmc_v3_lowgenes exon | 1.76 | 7.35 | 13.33 | 11.92 | 12.31 | 12.18 | 10.13 | 10.08 | 12.515 | -0.206 | 4.00 |
| pbmc_v3_lowgenes intron | 1.08 | 8.80 | 15.15 | 12.40 | 12.80 | 12.80 | 10.39 | 10.38 | 12.865 | -0.064 | 5.46 |
| tcell_5p exon | 10.53 | 6.96 | 10.06 | 7.56 | 7.33 | 7.33 | 6.67 | 6.68 | 7.340 | -0.012 | 4.97 |
| tcell_5p intron | 1.10 | 11.59 | 13.30 | 11.02 | 10.97 | 10.99 | 10.57 | 10.57 | 10.971 | 0.003 | 9.03 |
| tumor_3p exon | 0.25 | 12.48 | 18.57 | 14.83 | 14.76 | 14.79 | 14.03 | 14.01 | 14.751 | 0.006 | 8.22 |
| tumor_3p intron | 0.11 | 15.43 | 21.15 | 16.05 | 16.09 | 16.16 | 15.10 | 15.10 | 16.082 | 0.010 | 9.64 |
| brain_sn_v3_deep exon | 6.35 | 9.44 | 11.80 | 9.55 | 9.39 | 9.44 | 8.46 | 8.51 | 9.406 | -0.012 | 7.35 |
| brain_sn_v3_deep intron | 6.68 | 8.34 | 11.39 | 8.39 | 8.25 | 8.26 | 7.76 | 7.76 | 8.252 | 0.002 | 6.36 |
| brain_sn_v3_intronic exon | 1.38 | 11.86 | 12.31 | 10.94 | 10.71 | 10.80 | 10.66 | 10.78 | 10.724 | -0.011 | 8.39 |
| brain_sn_v3_intronic intron | 8.14 | 8.45 | 9.89 | 7.81 | 7.67 | 7.67 | 7.59 | 7.58 | 7.664 | 0.002 | 6.77 |

### Table 2. Presence split and quantisation (bits/nnz, out of sample, s3 / s3r models)

| file | float cloglog presence | dense 8-bit presence | rank-1 presence | counts (s3) | bucket side info | coder escapes+rounding vs CE (s3) |
|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 6.186 | 6.185 | 6.191 | 1.299 | 0.021 | -0.13% |
| pbmc_v2_typical intron | 8.535 | 8.534 | 8.534 | 0.277 | 0.036 | +0.03% |
| pbmc_v2_shallow exon | 8.085 | 8.087 | 8.109 | 0.511 | 0.033 | -0.01% |
| pbmc_v2_shallow intron | 12.431 | 12.430 | 12.431 | 0.099 | 0.188 | +0.02% |
| pbmc_v3_lowgenes exon | 10.077 | 10.073 | 9.930 | 2.333 | 0.109 | -1.65% |
| pbmc_v3_lowgenes intron | 10.637 | 10.636 | 10.635 | 2.086 | 0.142 | -0.50% |
| tcell_5p exon | 6.241 | 6.240 | 6.242 | 1.084 | 0.016 | -0.16% |
| tcell_5p intron | 10.632 | 10.624 | 10.631 | 0.208 | 0.139 | +0.03% |
| tumor_3p exon | 14.003 | 13.968 | 13.980 | 0.476 | 0.306 | +0.04% |
| tumor_3p intron | 15.084 | 15.045 | 15.088 | 0.409 | 0.627 | +0.06% |
| brain_sn_v3_deep exon | 8.873 | 8.860 | 8.904 | 0.515 | 0.031 | -0.13% |
| brain_sn_v3_deep intron | 7.530 | 7.526 | 7.527 | 0.692 | 0.034 | +0.02% |
| brain_sn_v3_intronic exon | 10.032 | 10.025 | 10.112 | 0.561 | 0.138 | -0.11% |
| brain_sn_v3_intronic intron | 7.155 | 7.154 | 7.154 | 0.487 | 0.023 | +0.03% |

### Table 3. Decode speed of the stage-4 decoder (single thread, 4 lanes, min of 3)

| file | nnz (M) | s4 dense: wall Mnnz/s | s4 dense: core-cyc/nnz (norm) | s4 dense: Mnnz/s @3 GHz norm | s4r rank-1: wall Mnnz/s | s4r: core-cyc/nnz (norm) | s4r: Mnnz/s @3 GHz norm | payload bits/nnz s4r |
|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 2.57 | 0.39 | 366 | 8.2 | 0.53 | 279 | 10.8 | 7.502 |
| pbmc_v2_typical intron | 1.20 | 0.36 | 457 | 6.6 | 0.58 | 276 | 10.9 | 8.851 |
| pbmc_v2_shallow exon | 5.30 | 0.37 | 363 | 8.3 | 1.04 | 257 | 11.7 | 8.653 |
| pbmc_v2_shallow intron | 1.17 | 2.57 | 488 | 6.1 | 0.42 | 307 | 9.8 | 12.723 |
| pbmc_v3_lowgenes exon | 1.76 | 3.81 | 329 | 9.1 | 0.62 | 237 | 12.7 | 12.178 |
| pbmc_v3_lowgenes intron | 1.08 | 0.49 | 334 | 9.0 | 6.28 | 215 | 13.9 | 12.803 |
| tcell_5p exon | 10.53 | 3.30 | 366 | 8.2 | 0.66 | 271 | 11.1 | 7.331 |
| tcell_5p intron | 1.10 | 3.16 | 398 | 7.5 | 5.16 | 260 | 11.5 | 10.989 |
| tumor_3p exon | 0.25 | 0.31 | 682 | 4.4 | 3.56 | 285 | 10.5 | 14.797 |
| tumor_3p intron | 0.11 | 0.29 | 862 | 3.5 | 0.52 | 387 | 7.7 | 16.174 |
| brain_sn_v3_deep exon | 6.35 | 0.78 | 523 | 5.7 | 2.00 | 304 | 9.9 | 9.439 |
| brain_sn_v3_deep intron | 6.68 | 2.35 | 443 | 6.8 | 3.56 | 285 | 10.5 | 8.257 |
| brain_sn_v3_intronic exon | 1.38 | 1.84 | 524 | 5.7 | 3.97 | 284 | 10.6 | 10.806 |
| brain_sn_v3_intronic intron | 8.14 | 2.61 | 445 | 6.7 | 3.77 | 302 | 9.9 | 7.668 |

### Table 4. H1 per file (H1: within 0.2 bits/nnz of the per-feature x depth cross-entropy at >= 20 Mnnz/s; out-of-sample models)

| file | CE of shipped dense model (s3, py) | stage-4 dense coder | delta | CE rank-1 (s3r, py) | stage-4 rank-1 coder | delta | size clause | dense Mnnz/s (3 GHz-norm) | rank-1 Mnnz/s (3 GHz-norm) | speed clause |
|---|---|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 7.505 | 7.496 | -0.009 | 7.511 | 7.502 | -0.009 | met | 8.2 | 10.8 | NOT met |
| pbmc_v2_typical intron | 8.846 | 8.849 | +0.004 | 8.846 | 8.851 | +0.005 | met | 6.6 | 10.9 | NOT met |
| pbmc_v2_shallow exon | 8.631 | 8.630 | -0.001 | 8.654 | 8.653 | -0.001 | met | 8.3 | 11.7 | NOT met |
| pbmc_v2_shallow intron | 12.716 | 12.720 | +0.004 | 12.718 | 12.723 | +0.005 | met | 6.1 | 9.8 | NOT met |
| pbmc_v3_lowgenes exon | 12.515 | 12.310 | -0.206 | 12.371 | 12.178 | -0.194 | met | 9.1 | 12.7 | NOT met |
| pbmc_v3_lowgenes intron | 12.865 | 12.802 | -0.063 | 12.864 | 12.803 | -0.061 | met | 9.0 | 13.9 | NOT met |
| tcell_5p exon | 7.340 | 7.328 | -0.012 | 7.342 | 7.331 | -0.011 | met | 8.2 | 11.1 | NOT met |
| tcell_5p intron | 10.971 | 10.976 | +0.004 | 10.978 | 10.989 | +0.011 | met | 7.5 | 11.5 | NOT met |
| tumor_3p exon | 14.751 | 14.762 | +0.012 | 14.762 | 14.797 | +0.035 | met | 4.4 | 10.5 | NOT met |
| tumor_3p intron | 16.082 | 16.106 | +0.024 | 16.125 | 16.174 | +0.049 | met | 3.5 | 7.7 | NOT met |
| brain_sn_v3_deep exon | 9.406 | 9.394 | -0.012 | 9.449 | 9.439 | -0.010 | met | 5.7 | 9.9 | NOT met |
| brain_sn_v3_deep intron | 8.252 | 8.254 | +0.002 | 8.253 | 8.257 | +0.003 | met | 6.8 | 10.5 | NOT met |
| brain_sn_v3_intronic exon | 10.724 | 10.714 | -0.010 | 10.812 | 10.806 | -0.006 | met | 5.7 | 10.6 | NOT met |
| brain_sn_v3_intronic intron | 7.664 | 7.667 | +0.002 | 7.665 | 7.668 | +0.003 | met | 6.7 | 9.9 | NOT met |

### Table 5. Per-file model selection from a library of single-sample rank-1 models (all from OTHER studies), out of sample

| file | VOCSC zstd-3 chunks | s3r pooled (leave-study-out) | s5r selected | chosen model | gain vs pooled | vs VOCSC | stage-4 speed (3 GHz-norm Mnnz/s) |
|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 6.85 | 7.50 | 6.35 | only_h_c | -1.15 | -0.50 | 11.0 |
| pbmc_v2_typical intron | 9.18 | 8.85 | 8.69 | only_h_c | -0.16 | -0.49 | 10.5 |
| pbmc_v2_shallow exon | 8.81 | 8.65 | 7.86 | only_h_c | -0.79 | -0.95 | 11.0 |
| pbmc_v2_shallow intron | 14.33 | 12.72 | 12.56 | only_h_c | -0.16 | -1.76 | 8.8 |
| pbmc_v3_lowgenes exon | 7.35 | 12.18 | 11.61 | only_h_c | -0.57 | +4.26 | 13.0 |
| pbmc_v3_lowgenes intron | 8.80 | 12.80 | 12.80 | GSE244263 | +0.00 | +4.00 | 12.6 |
| tcell_5p exon | 6.96 | 7.33 | 6.57 | only_h_b | -0.76 | -0.39 | 10.7 |
| tcell_5p intron | 11.59 | 10.99 | 10.50 | only_h_b | -0.49 | -1.09 | 8.8 |
| tumor_3p exon | 12.48 | 14.79 | 14.72 | only_h_pbmc_a | -0.07 | +2.24 | 9.6 |
| tumor_3p intron | 15.43 | 16.16 | 16.19 | only_h_b | +0.03 | +0.76 | 8.0 |
| brain_sn_v3_deep exon | 9.44 | 9.44 | 9.37 | only_m_brain_a | -0.07 | -0.07 | 10.1 |
| brain_sn_v3_deep intron | 8.34 | 8.26 | 8.11 | only_m_brain_a | -0.15 | -0.22 | 10.7 |
| brain_sn_v3_intronic exon | 11.86 | 10.80 | 10.41 | only_m_brain_d | -0.39 | -1.44 | 10.4 |
| brain_sn_v3_intronic intron | 8.45 | 7.67 | 7.67 | GSE285936 | +0.00 | -0.78 | 11.0 |

### Table 0 (summary). Payload bits/nnz (out of sample), model size, coder-vs-CE gap and speed per file

| file | VOCSC zstd-3 | s1 order-0 | s2 + depth | s3 + counts | s5r per-file model | coder - CE (s4/s4r, worst) | s4r Mnnz/s @3 GHz | s4 dense Mnnz/s |
|---|---|---|---|---|---|---|---|---|
| pbmc_v2_typical exon | 6.85 | 9.18 | 7.75 | 7.50 | 6.35 | -0.009 | 10.8 | 8.2 |
| pbmc_v2_typical intron | 9.18 | 10.94 | 8.88 | 8.85 | 8.69 | +0.005 | 10.9 | 6.6 |
| pbmc_v2_shallow exon | 8.81 | 9.14 | 8.73 | 8.63 | 7.86 | -0.001 | 11.7 | 8.3 |
| pbmc_v2_shallow intron | 14.33 | 13.55 | 12.77 | 12.72 | 12.56 | +0.005 | 9.8 | 6.1 |
| pbmc_v3_lowgenes exon | 7.35 | 13.33 | 11.92 | 12.31 | 11.61 | -0.194 | 12.7 | 9.1 |
| pbmc_v3_lowgenes intron | 8.80 | 15.15 | 12.40 | 12.80 | 12.80 | -0.061 | 13.9 | 9.0 |
| tcell_5p exon | 6.96 | 10.06 | 7.56 | 7.33 | 6.57 | -0.011 | 11.1 | 8.2 |
| tcell_5p intron | 11.59 | 13.30 | 11.02 | 10.97 | 10.50 | +0.011 | 11.5 | 7.5 |
| tumor_3p exon | 12.48 | 18.57 | 14.83 | 14.76 | 14.72 | +0.035 | 10.5 | 4.4 |
| tumor_3p intron | 15.43 | 21.15 | 16.05 | 16.09 | 16.19 | +0.049 | 7.7 | 3.5 |
| brain_sn_v3_deep exon | 9.44 | 11.80 | 9.55 | 9.39 | 9.37 | -0.010 | 9.9 | 5.7 |
| brain_sn_v3_deep intron | 8.34 | 11.39 | 8.39 | 8.25 | 8.11 | +0.003 | 10.5 | 6.8 |
| brain_sn_v3_intronic exon | 11.86 | 12.31 | 10.94 | 10.71 | 10.41 | -0.006 | 10.6 | 5.7 |
| brain_sn_v3_intronic intron | 8.45 | 9.89 | 7.81 | 7.67 | 7.67 | +0.003 | 9.9 | 6.7 |
