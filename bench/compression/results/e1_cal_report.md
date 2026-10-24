# E1 extension: calibrated own-half predictor (L3cal), recalibrated baseline (L1cal) and mixtures

Bits per held-out nonzero (b/nnzB); hyper-parameters cross-fitted over cells. Gains are vs the main run's L1 unless a column says vs L1cal. L1cal = a-free recalibration by expression level (static model repair); L2cal = neighbours on L1cal; L3cal = learned A->B map; mix = (1-lam) L3cal + lam L2cal (see tools/e1_cal.py).

| file | cells | L1 | L1cal gain | L2 gain | L2cal gain | L3 (global alpha) gain | L3cal gain | mix gain | own-half = L3cal-L1cal | nbr = L2cal-L1cal | mix-L1cal | mix-L1cal bits/cell | mix-L1cal b/nnz_cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| brain_sn_v3_deep/exon interval | 3288 | 7.182 | +0.017 | +0.066 | +0.075 | +0.000 | +0.102 | +0.122 | +0.085 | +0.058 | +0.105 | +55.2 | +0.055 |
| brain_sn_v3_deep/intron interval | 3278 | 6.673 | +0.004 | +0.100 | +0.101 | +0.007 | -inf | +0.136 | -inf | +0.097 | +0.132 | +104.0 | +0.073 |
| brain_sn_v3_intronic/exon interval | 5000 | 9.122 | +0.034 | -0.018 | +0.010 | +0.002 | +0.048 | +0.048 | +0.014 | -0.024 | +0.014 | +0.7 | +0.007 |
| pbmc_v2_shallow/exon interval | 5000 | 7.181 | +0.020 | -0.005 | +0.013 | -0.000 | +0.022 | +0.022 | +0.002 | -0.007 | +0.002 | +0.1 | +0.001 |
| pbmc_v2_typical/exon interval | 5000 | 5.684 | -0.000 | +0.070 | +0.064 | +0.023 | +0.053 | +0.090 | +0.054 | +0.065 | +0.090 | +23.6 | +0.053 |
| pbmc_v2_typical/intron interval | 5000 | 8.776 | +0.013 | +0.005 | +0.016 | +0.015 | +0.034 | +0.036 | +0.022 | +0.003 | +0.023 | +2.5 | +0.012 |
| pbmc_v3_lowgenes/exon interval | 5000 | 5.329 | -0.015 | +0.154 | +0.154 | +0.066 | +0.129 | +0.193 | +0.144 | +0.169 | +0.207 | +18.2 | +0.126 |
| pbmc_v3_lowgenes/intron interval | 5000 | 6.870 | -0.008 | +0.155 | +0.146 | +0.410 | +0.457 | +0.471 | +0.464 | +0.154 | +0.479 | +31.7 | +0.282 |
| tcell_5p/exon interval | 5000 | 6.059 | +0.013 | +0.099 | +0.113 | +0.034 | +0.115 | +0.151 | +0.101 | +0.100 | +0.138 | +60.6 | +0.079 |
| tumor_3p/exon interval | 839 | 8.874 | +0.178 | +0.127 | +0.213 | +0.066 | +0.293 | +0.295 | +0.115 | +0.035 | +0.117 | +4.5 | +0.062 |
| brain_sn_v3_deep/exon gene | 2000 | 5.726 | -0.000 | +0.109 | +0.112 | +0.014 | -inf | +0.129 | -inf | +0.112 | +0.129 | +64.0 | +0.071 |
| pbmc_v2_typical/exon gene | 2000 | 5.267 | -0.004 | +0.088 | +0.091 | +0.037 | +0.064 | +0.113 | +0.068 | +0.095 | +0.117 | +28.9 | +0.071 |
| pbmc_v3_lowgenes/exon gene | 2000 | 4.642 | -0.023 | +0.235 | +0.229 | +0.118 | +0.176 | +0.249 | +0.198 | +0.252 | +0.271 | +19.0 | +0.173 |
| tcell_5p/exon gene | 2000 | 5.217 | -0.013 | +0.154 | +0.153 | +0.061 | +0.126 | +0.175 | +0.139 | +0.165 | +0.187 | +74.0 | +0.112 |

brain_sn_v3_deep__exon__interval: marginal rho_m by u=N_A q bin: 1e-07:1.00, 1e-06:1.00, 1e-05:1.00, 1e-04:1.13, 1e-03:0.88, 1e-02:0.99, 1e-01:1.04, 1e+00:1.10, 1e+01:1.26
brain_sn_v3_deep__exon__interval: rho(a=0) by u bin: 1e-07:0.95, 1e-06:0.95, 1e-05:0.95, 1e-04:1.13, 1e-03:0.88, 1e-02:0.98, 1e-01:0.97, 1e+00:0.56, 1e+01:0.08
brain_sn_v3_deep__intron__interval: marginal rho_m by u=N_A q bin: 1e-07:1.00, 1e-06:1.00, 1e-05:4.66, 1e-04:1.24, 1e-03:0.91, 1e-02:0.99, 1e-01:1.01, 1e+00:1.03, 1e+01:0.81
brain_sn_v3_deep__intron__interval: rho(a=0) by u bin: 1e-07:0.90, 1e-06:0.90, 1e-05:4.62, 1e-04:1.23, 1e-03:0.91, 1e-02:0.97, 1e-01:0.89, 1e+00:0.55, 1e+01:0.04
brain_sn_v3_intronic__exon__interval: marginal rho_m by u=N_A q bin: 1e-07:1.00, 1e-06:1.00, 1e-05:57.45, 1e-04:0.64, 1e-03:1.00, 1e-02:1.04, 1e-01:1.00, 1e+00:1.11, 1e+01:1.04
brain_sn_v3_intronic__exon__interval: rho(a=0) by u bin: 1e-07:0.96, 1e-06:0.96, 1e-05:57.53, 1e-04:0.64, 1e-03:0.99, 1e-02:1.03, 1e-01:0.95, 1e+00:0.93, 1e+01:0.96
pbmc_v2_shallow__exon__interval: marginal rho_m by u=N_A q bin: 1e-07:1.00, 1e-06:1.00, 1e-05:0.73, 1e-04:0.92, 1e-03:1.00, 1e-02:1.01, 1e-01:1.01, 1e+00:1.03, 1e+01:0.68
pbmc_v2_shallow__exon__interval: rho(a=0) by u bin: 1e-07:0.99, 1e-06:0.99, 1e-05:0.73, 1e-04:0.92, 1e-03:1.00, 1e-02:1.01, 1e-01:0.99, 1e+00:0.84, 1e+01:0.99
pbmc_v2_typical__exon__interval: marginal rho_m by u=N_A q bin: 1e-07:0.99, 1e-06:0.99, 1e-05:0.99, 1e-04:1.07, 1e-03:0.79, 1e-02:1.00, 1e-01:1.03, 1e+00:1.03, 1e+01:1.00
pbmc_v2_typical__exon__interval: rho(a=0) by u bin: 1e-07:0.91, 1e-06:0.91, 1e-05:0.91, 1e-04:1.07, 1e-03:0.79, 1e-02:0.98, 1e-01:0.97, 1e+00:0.84, 1e+01:0.15
pbmc_v2_typical__intron__interval: marginal rho_m by u=N_A q bin: 1e-07:1.00, 1e-06:1.00, 1e-05:18.53, 1e-04:0.78, 1e-03:0.98, 1e-02:1.05, 1e-01:1.05, 1e+00:0.85, 1e+01:1.00
pbmc_v2_typical__intron__interval: rho(a=0) by u bin: 1e-07:0.98, 1e-06:0.98, 1e-05:18.53, 1e-04:0.77, 1e-03:0.97, 1e-02:1.02, 1e-01:0.98, 1e+00:0.39, 1e+01:0.98
pbmc_v3_lowgenes__exon__interval: marginal rho_m by u=N_A q bin: 1e-07:0.98, 1e-06:0.98, 1e-05:0.98, 1e-04:1.08, 1e-03:0.62, 1e-02:0.97, 1e-01:1.00, 1e+00:1.01, 1e+01:1.01
pbmc_v3_lowgenes__exon__interval: rho(a=0) by u bin: 1e-07:0.83, 1e-06:0.83, 1e-05:0.83, 1e-04:1.04, 1e-03:0.59, 1e-02:0.92, 1e-01:0.91, 1e+00:0.65, 1e+01:0.15
pbmc_v3_lowgenes__intron__interval: marginal rho_m by u=N_A q bin: 1e-07:0.99, 1e-06:0.99, 1e-05:17.80, 1e-04:0.64, 1e-03:0.89, 1e-02:1.00, 1e-01:0.99, 1e+00:1.08, 1e+01:1.03
pbmc_v3_lowgenes__intron__interval: rho(a=0) by u bin: 1e-07:0.74, 1e-06:0.74, 1e-05:17.40, 1e-04:0.60, 1e-03:0.77, 1e-02:0.82, 1e-01:0.71, 1e+00:0.53, 1e+01:0.74
tcell_5p__exon__interval: marginal rho_m by u=N_A q bin: 1e-07:0.99, 1e-06:0.99, 1e-05:587.00, 1e-04:0.48, 1e-03:0.90, 1e-02:0.99, 1e-01:1.02, 1e+00:1.02, 1e+01:0.98
tcell_5p__exon__interval: rho(a=0) by u bin: 1e-07:0.90, 1e-06:0.90, 1e-05:594.44, 1e-04:0.48, 1e-03:0.90, 1e-02:0.97, 1e-01:0.93, 1e+00:0.77, 1e+01:0.12
tumor_3p__exon__interval: marginal rho_m by u=N_A q bin: 1e-07:0.99, 1e-06:0.99, 1e-05:0.99, 1e-04:8.11, 1e-03:0.52, 1e-02:1.58, 1e-01:1.65, 1e+00:1.58, 1e+01:0.99
tumor_3p__exon__interval: rho(a=0) by u bin: 1e-07:0.88, 1e-06:0.88, 1e-05:0.88, 1e-04:8.01, 1e-03:0.51, 1e-02:1.50, 1e-01:1.30, 1e+00:0.51, 1e+01:0.88
brain_sn_v3_deep__exon__gene: marginal rho_m by u=N_A q bin: 1e-07:0.99, 1e-06:0.99, 1e-05:0.99, 1e-04:4.32, 1e-03:0.68, 1e-02:0.97, 1e-01:1.01, 1e+00:1.05, 1e+01:1.10
brain_sn_v3_deep__exon__gene: rho(a=0) by u bin: 1e-07:0.93, 1e-06:0.93, 1e-05:0.93, 1e-04:4.29, 1e-03:0.67, 1e-02:0.95, 1e-01:0.96, 1e+00:0.67, 1e+01:0.10
pbmc_v2_typical__exon__gene: marginal rho_m by u=N_A q bin: 1e-07:0.99, 1e-06:0.99, 1e-05:0.99, 1e-04:1.32, 1e-03:0.89, 1e-02:0.98, 1e-01:1.01, 1e+00:1.01, 1e+01:0.99
pbmc_v2_typical__exon__gene: rho(a=0) by u bin: 1e-07:0.92, 1e-06:0.92, 1e-05:0.92, 1e-04:1.31, 1e-03:0.88, 1e-02:0.96, 1e-01:0.95, 1e+00:0.82, 1e+01:0.10
pbmc_v3_lowgenes__exon__gene: marginal rho_m by u=N_A q bin: 1e-07:0.99, 1e-06:0.99, 1e-05:0.99, 1e-04:0.47, 1e-03:0.52, 1e-02:0.97, 1e-01:0.99, 1e+00:1.00, 1e+01:1.01
pbmc_v3_lowgenes__exon__gene: rho(a=0) by u bin: 1e-07:0.81, 1e-06:0.81, 1e-05:0.81, 1e-04:0.45, 1e-03:0.49, 1e-02:0.90, 1e-01:0.88, 1e+00:0.62, 1e+01:0.22
tcell_5p__exon__gene: marginal rho_m by u=N_A q bin: 1e-07:0.99, 1e-06:0.99, 1e-05:181.89, 1e-04:1.49, 1e-03:0.80, 1e-02:0.97, 1e-01:1.00, 1e+00:1.01, 1e+01:0.99
tcell_5p__exon__gene: rho(a=0) by u bin: 1e-07:0.88, 1e-06:0.88, 1e-05:181.51, 1e-04:1.47, 1e-03:0.79, 1e-02:0.95, 1e-01:0.93, 1e+00:0.69, 1e+01:0.11