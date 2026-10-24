# E1 learning curve: gain of the held-out part B given a context part A holding a fraction f of the UMIs

Gains vs L1 on B (cross-fitted). b/nnzB = per held-out nonzero; bits/cell = per cell; I(A;B) is symmetric in A and B for the true process, so compare f and 1-f at equal bits/cell. mean N_B/N = 1-f.

| file | f (context share) | cells | L1 b/nnzB | L2 gain | L3 seq gain | L3 best gain | L3dm best gain | L3 best bits/cell | L2 bits/cell | seq(L3seq) bits/cell |
|---|---|---|---|---|---|---|---|---|---|---|
| brain_sn_v3_deep/exon | 0.25 | 2500 | 6.787 | +0.055 | +0.000 | +0.055 | +0.055 | +42.7 | +42.7 | +0.0 |
| brain_sn_v3_deep/exon | 0.50 | 3288 | 7.182 | +0.066 | +0.000 | +0.066 | +0.066 | +34.9 | +34.9 | +0.0 |
| brain_sn_v3_deep/exon | 0.75 | 2500 | 7.903 | +0.075 | +0.000 | +0.075 | +0.075 | +20.5 | +20.5 | +0.0 |
| pbmc_v2_typical/exon | 0.25 | 2500 | 5.585 | +0.076 | +0.026 | +0.075 | +0.074 | +26.6 | +27.0 | +9.3 |
| pbmc_v2_typical/exon | 0.50 | 5000 | 5.684 | +0.070 | +0.023 | +0.068 | +0.067 | +17.8 | +18.3 | +6.0 |
| pbmc_v2_typical/exon | 0.75 | 2500 | 5.941 | +0.058 | +0.026 | +0.056 | +0.056 | +8.5 | +8.7 | +3.9 |
| pbmc_v3_lowgenes/exon | 0.25 | 2500 | 5.274 | +0.126 | +0.041 | +0.146 | +0.185 | +17.3 | +14.9 | +4.8 |
| pbmc_v3_lowgenes/exon | 0.50 | 5000 | 5.329 | +0.154 | +0.066 | +0.182 | +0.200 | +16.0 | +13.5 | +5.8 |
| pbmc_v3_lowgenes/exon | 0.75 | 2500 | 5.477 | +0.141 | +0.079 | +0.179 | +0.186 | +9.4 | +7.5 | +4.2 |