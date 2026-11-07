## double_count

| dataset | cells | per-cell overcount, median | per-cell overcount, p10-p90 | total overcount | total overcount, 16 workers | exon / intron matrix ratio old:new | same-gene exon+intron doubles | cross-worker multi-mapper molecules | gene overcount, median (>=50 UMI) | gene overcount, p99 | cells called old / new |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PBMC 3' v3 | 1136 | 9.1% | 6.8% to 11.8% | 9.1% | 9.2% | 1.121 / 1.027 | 2.2% | 10.5% | 1.3% | 51.4% | 1136 / 1136 |
| PBMC 3' v2 | 983 | 18.4% | 14.3% to 22.9% | 18.8% | 20.0% | 1.208 / 1.121 | 4.6% | 17.0% | 1.7% | 74.7% | 983 / 983 |
| PBMC 5' v1 | 6627 | 3.9% | 1.1% to 6.9% | 4.0% | 2.0% | 1.206 / 0.943 | 9.3% | 4.8% | 2.8% | 63.1% | 6627 / 6627 |
| PBMC CITE 3' v3 | 583 | 7.3% | 5.7% to 9.6% | 7.5% | 7.8% | 1.093 / 1.039 | 2.7% | 8.1% | 2.1% | 46.3% | 583 / 583 |
| Mouse neurons 3' v3 | 1102 | 5.9% | 5.0% to 6.7% | 5.9% | 6.1% | 1.033 / 1.111 | 2.1% | 2.7% | 1.7% | 50.8% | 1102 / 1102 |
| Mouse brain nuclei 3' v2 | 2188 | 5.5% | 4.2% to 6.5% | 5.5% | 5.4% | 1.051 / 1.057 | 3.9% | 1.1% | 3.3% | 63.6% | 2188 / 2188 |
| Human+mouse barnyard 3' v3 | 880 | 2.8% | 2.5% to 3.2% | 2.7% | 2.7% | 1.021 / 1.052 | 1.1% | 4.6% | 0.8% | 32.8% | 880 / 880 |
| Visium GSM7090362 | 58 | 2.3% | 1.8% to 2.8% | 2.3% | - | 1.004 / 1.422 | 1.4% | - | 0.0% | 14.5% | 57 / 58 |
| Visium GSM7412962 | 97 | 2.1% | 1.8% to 2.3% | 2.1% | - | 1.004 / 1.291 | 0.9% | - | 0.0% | 9.3% | 96 / 97 |
| Visium GSM7596594 | 59 | 1.6% | 1.4% to 1.8% | 1.6% | - | 1.002 / 1.920 | 0.6% | - | 0.0% | 11.4% | 56 / 59 |
| Visium GSM7983360 | 124 | 2.1% | 1.8% to 2.4% | 2.1% | - | 1.004 / 1.433 | 0.8% | - | 0.0% | 12.1% | 125 / 124 |

## worker_invariance

| dataset | engine | total UMIs, 16 / 4 workers | matrix entries that differ | sum of |differences| |
|---|---|---|---|---|
| PBMC 3' v3 | Singlet (atlas engine, 16 workers) | 1.0008 | 58378 | 150118 |
| PBMC 3' v3 | Singlet (fixed) | 1.0000 | 0 | 0 |
| PBMC 3' v2 | Singlet (atlas engine, 16 workers) | 1.0091 | 41400 | 138034 |
| PBMC 3' v2 | Singlet (fixed) | 1.0000 | 0 | 0 |
| PBMC 5' v1 | Singlet (atlas engine, 16 workers) | 0.9807 | 33939 | 93833 |
| PBMC 5' v1 | Singlet (fixed) | 1.0000 | 0 | 0 |
| PBMC CITE 3' v3 | Singlet (atlas engine, 16 workers) | 1.0037 | 33663 | 78229 |
| PBMC CITE 3' v3 | Singlet (fixed) | 1.0000 | 0 | 0 |
| Mouse neurons 3' v3 | Singlet (atlas engine, 16 workers) | 1.0017 | 48122 | 83343 |
| Mouse neurons 3' v3 | Singlet (fixed) | 1.0000 | 0 | 0 |
| Mouse brain nuclei 3' v2 | Singlet (atlas engine, 16 workers) | 0.9993 | 29106 | 30912 |
| Mouse brain nuclei 3' v2 | Singlet (fixed) | 1.0000 | 0 | 0 |
| Human+mouse barnyard 3' v3 | Singlet (atlas engine, 16 workers) | 0.9999 | 32129 | 49351 |
| Human+mouse barnyard 3' v3 | Singlet (fixed) | 1.0000 | 0 | 0 |

## suf

| dataset | tool | spliced | unspliced | ambiguous |
|---|---|---|---|---|
| PBMC 3' v3 | Singlet (atlas engine, 16 workers) | 69.4% | 30.6% | 0.0% |
| PBMC 3' v3 | Singlet (fixed) | 66.1% | 32.6% | 1.3% |
| PBMC 3' v3 | Singlet (atlas engine) | 69.3% | 30.7% | 0.0% |
| PBMC 3' v3 | Singlet (fixed, 4 workers) | 66.1% | 32.6% | 1.3% |
| PBMC 3' v3 | STARsolo Velocyto | 55.4% | 34.6% | 9.9% |
| PBMC 3' v3 | alevin-fry (USA) | 21.5% | 31.7% | 46.8% |
| PBMC 3' v2 | Singlet (atlas engine, 16 workers) | 79.0% | 21.0% | 0.0% |
| PBMC 3' v2 | Singlet (fixed) | 72.8% | 22.8% | 4.4% |
| PBMC 3' v2 | Singlet (atlas engine) | 78.5% | 21.5% | 0.0% |
| PBMC 3' v2 | Singlet (fixed, 4 workers) | 72.8% | 22.8% | 4.4% |
| PBMC 3' v2 | STARsolo Velocyto | 66.6% | 23.2% | 10.2% |
| PBMC 3' v2 | alevin-fry (USA) | 35.6% | 19.9% | 44.5% |
| PBMC 5' v1 | Singlet (atlas engine, 16 workers) | 43.9% | 56.1% | 0.0% |
| PBMC 5' v1 | Singlet (fixed) | 34.9% | 62.9% | 2.2% |
| PBMC 5' v1 | Singlet (atlas engine) | 43.0% | 57.0% | 0.0% |
| PBMC 5' v1 | Singlet (fixed, 4 workers) | 34.9% | 62.9% | 2.2% |
| PBMC 5' v1 | singlet-fixed-rev | 88.9% | 10.3% | 0.8% |
| PBMC 5' v1 | STARsolo Velocyto | 80.3% | 9.8% | 9.9% |
| PBMC 5' v1 | alevin-fry (USA) | 1.8% | 74.1% | 24.1% |
| PBMC 5' v1 | alevinfry-fw | 46.5% | 11.5% | 42.1% |
| PBMC CITE 3' v3 | Singlet (atlas engine, 16 workers) | 67.6% | 32.4% | 0.0% |
| PBMC CITE 3' v3 | Singlet (fixed) | 64.8% | 33.7% | 1.5% |
| PBMC CITE 3' v3 | Singlet (atlas engine) | 67.4% | 32.6% | 0.0% |
| PBMC CITE 3' v3 | Singlet (fixed, 4 workers) | 64.8% | 33.7% | 1.5% |
| PBMC CITE 3' v3 | STARsolo Velocyto | 53.9% | 36.2% | 9.9% |
| PBMC CITE 3' v3 | alevin-fry (USA) | 20.9% | 33.2% | 45.9% |
| Mouse neurons 3' v3 | Singlet (atlas engine, 16 workers) | 65.1% | 34.9% | 0.0% |
| Mouse neurons 3' v3 | Singlet (fixed) | 65.5% | 33.3% | 1.1% |
| Mouse neurons 3' v3 | Singlet (atlas engine) | 65.0% | 35.0% | 0.0% |
| Mouse neurons 3' v3 | Singlet (fixed, 4 workers) | 65.5% | 33.3% | 1.1% |
| Mouse neurons 3' v3 | STARsolo Velocyto | 57.9% | 35.5% | 6.6% |
| Mouse neurons 3' v3 | alevin-fry (USA) | 14.8% | 34.7% | 50.6% |
| Mouse brain nuclei 3' v2 | Singlet (atlas engine, 16 workers) | 37.4% | 62.6% | 0.0% |
| Mouse brain nuclei 3' v2 | Singlet (fixed) | 34.5% | 62.6% | 2.9% |
| Mouse brain nuclei 3' v2 | Singlet (atlas engine) | 37.3% | 62.7% | 0.0% |
| Mouse brain nuclei 3' v2 | Singlet (fixed, 4 workers) | 34.5% | 62.6% | 2.9% |
| Mouse brain nuclei 3' v2 | STARsolo Velocyto | 25.5% | 67.9% | 6.6% |
| Mouse brain nuclei 3' v2 | alevin-fry (USA) | 3.2% | 65.1% | 31.7% |
| Human+mouse barnyard 3' v3 | Singlet (atlas engine, 16 workers) | 78.3% | 21.7% | 0.0% |
| Human+mouse barnyard 3' v3 | Singlet (fixed) | 78.5% | 21.3% | 0.2% |
| Human+mouse barnyard 3' v3 | Singlet (atlas engine) | 78.2% | 21.8% | 0.0% |
| Human+mouse barnyard 3' v3 | Singlet (fixed, 4 workers) | 78.5% | 21.3% | 0.2% |
| Human+mouse barnyard 3' v3 | STARsolo Velocyto | 69.8% | 21.9% | 8.4% |
| Human+mouse barnyard 3' v3 | alevin-fry (USA) | 18.9% | 20.9% | 60.2% |

## concordance

| dataset | comparison | what | cells | per-cell total ratio (median) | per-cell total r (log) | per-cell gene r (log, median) | pseudobulk rho | genes/cell ratio |
|---|---|---|---|---|---|---|---|---|
| PBMC 3' v3 | singlet-fixed[X] vs starsolo_GeneFull[X] | total (exon+intron) | 1130 | 1.032 | 1.000 | 0.978 | 0.942 | 1.050 |
| PBMC 3' v3 | singlet-baseline-t4[X] vs starsolo_GeneFull[X] | total (exon+intron), atlas config | 1130 | 1.127 | 0.999 | 0.976 | 0.939 | 1.038 |
| PBMC 3' v3 | singlet-baseline[X] vs starsolo_GeneFull[X] | total (exon+intron), 16 workers | 1130 | 1.128 | 0.999 | 0.976 | 0.939 | 1.035 |
| PBMC 3' v3 | alevinfry[X] vs starsolo_GeneFull[X] | total (S+U+A) | 1130 | 1.059 | 1.000 | 0.977 | 0.943 | 0.998 |
| PBMC 3' v3 | singlet-fixed[exon] vs starsolo_Gene[X] | exonic (S+A) | 1130 | 0.996 | 1.000 | 0.993 | 0.988 | 1.005 |
| PBMC 3' v3 | singlet-baseline-t4[exon] vs starsolo_Gene[X] | exonic, atlas config | 1130 | 1.127 | 0.999 | 0.986 | 0.984 | 1.033 |
| PBMC 3' v3 | cellranger_published[X] vs starsolo_Gene[X] | exonic (CR published) | 1130 | 0.962 | 1.000 | 0.982 | 0.976 | 0.955 |
| PBMC 3' v3 | singlet-fixed[exon] vs cellranger_published[X] | exonic vs CR | 1130 | 1.036 | 1.000 | 0.977 | 0.969 | 1.051 |
| PBMC 3' v3 | singlet-fixed[spliced] vs starsolo_Velocyto[spliced] | spliced | 1130 | 1.254 | 0.998 | 0.930 | 0.902 | 1.231 |
| PBMC 3' v3 | singlet-fixed[unspliced] vs starsolo_Velocyto[unspliced] | unspliced | 1130 | 0.987 | 0.999 | 0.951 | 0.935 | 1.084 |
| PBMC 3' v3 | singlet-fixed[ambiguous] vs starsolo_Velocyto[ambiguous] | ambiguous | 1130 | 0.137 | 0.902 | 0.090 | 0.366 | 0.275 |
| PBMC 3' v3 | singlet-fixed[spliced] vs alevinfry[spliced] | spliced | 1130 | 2.892 | 0.838 | 0.651 | 0.749 | 4.117 |
| PBMC 3' v3 | singlet-fixed[unspliced] vs alevinfry[unspliced] | unspliced | 1130 | 1.004 | 0.992 | 0.936 | 0.942 | 1.071 |
| PBMC 3' v3 | singlet-fixed[X] vs alevinfry[X] | total | 1130 | 0.974 | 1.000 | 0.971 | 0.963 | 1.051 |
| PBMC 3' v2 | singlet-fixed[X] vs starsolo_GeneFull[X] | total (exon+intron) | 982 | 1.081 | 1.000 | 0.959 | 0.932 | 1.118 |
| PBMC 3' v2 | singlet-baseline-t4[X] vs starsolo_GeneFull[X] | total (exon+intron), atlas config | 982 | 1.279 | 0.997 | 0.960 | 0.933 | 1.085 |
| PBMC 3' v2 | singlet-baseline[X] vs starsolo_GeneFull[X] | total (exon+intron), 16 workers | 982 | 1.292 | 0.997 | 0.960 | 0.933 | 1.079 |
| PBMC 3' v2 | alevinfry[X] vs starsolo_GeneFull[X] | total (S+U+A) | 982 | 1.176 | 0.999 | 0.965 | 0.936 | 1.001 |
| PBMC 3' v2 | singlet-fixed[exon] vs starsolo_Gene[X] | exonic (S+A) | 982 | 1.018 | 1.000 | 0.988 | 0.980 | 1.029 |
| PBMC 3' v2 | singlet-baseline-t4[exon] vs starsolo_Gene[X] | exonic, atlas config | 982 | 1.231 | 0.997 | 0.982 | 0.976 | 1.051 |
| PBMC 3' v2 | cellranger_published[X] vs starsolo_Gene[X] | exonic (CR published) | 982 | 0.973 | 1.000 | 0.985 | 0.973 | 0.964 |
| PBMC 3' v2 | singlet-fixed[exon] vs cellranger_published[X] | exonic vs CR | 982 | 1.045 | 1.000 | 0.978 | 0.962 | 1.064 |
| PBMC 3' v2 | singlet-fixed[spliced] vs starsolo_Velocyto[spliced] | spliced | 982 | 1.204 | 0.996 | 0.932 | 0.881 | 1.206 |
| PBMC 3' v2 | singlet-fixed[unspliced] vs starsolo_Velocyto[unspliced] | unspliced | 982 | 1.077 | 0.990 | 0.844 | 0.868 | 1.258 |
| PBMC 3' v2 | singlet-fixed[ambiguous] vs starsolo_Velocyto[ambiguous] | ambiguous | 982 | 0.478 | 0.867 | 0.065 | 0.116 | 0.688 |
| PBMC 3' v2 | singlet-fixed[spliced] vs alevinfry[spliced] | spliced | 982 | 1.910 | 0.938 | 0.711 | 0.642 | 3.249 |
| PBMC 3' v2 | singlet-fixed[unspliced] vs alevinfry[unspliced] | unspliced | 982 | 1.049 | 0.992 | 0.819 | 0.882 | 1.179 |
| PBMC 3' v2 | singlet-fixed[X] vs alevinfry[X] | total | 982 | 0.918 | 0.999 | 0.945 | 0.935 | 1.116 |
| PBMC 5' v1 | singlet-fixed[X] vs starsolo_GeneFull[X] | total (exon+intron) | 6605 | 0.186 | 0.969 | 0.038 | 0.570 | 0.234 |
| PBMC 5' v1 | singlet-baseline-t4[X] vs starsolo_GeneFull[X] | total (exon+intron), atlas config | 6605 | 0.193 | 0.967 | 0.039 | 0.570 | 0.223 |
| PBMC 5' v1 | singlet-baseline[X] vs starsolo_GeneFull[X] | total (exon+intron), 16 workers | 6605 | 0.189 | 0.966 | 0.039 | 0.570 | 0.221 |
| PBMC 5' v1 | alevinfry[X] vs starsolo_GeneFull[X] | total (S+U+A) | 6605 | 0.164 | 0.959 | 0.041 | 0.575 | 0.209 |
| PBMC 5' v1 | singlet-fixed[exon] vs starsolo_Gene[X] | exonic (S+A) | 6605 | 0.073 | 0.910 | 0.037 | 0.507 | 0.101 |
| PBMC 5' v1 | singlet-baseline-t4[exon] vs starsolo_Gene[X] | exonic, atlas config | 6605 | 0.088 | 0.928 | 0.036 | 0.496 | 0.120 |
| PBMC 5' v1 | cellranger_published[X] vs starsolo_Gene[X] | exonic (CR published) | 6605 | 1.974 | 0.998 | 0.891 | 0.980 | 1.602 |
| PBMC 5' v1 | singlet-fixed[exon] vs cellranger_published[X] | exonic vs CR | 6605 | 0.029 | 0.911 | 0.044 | 0.551 | 0.057 |
| PBMC 5' v1 | singlet-fixed[spliced] vs starsolo_Velocyto[spliced] | spliced | 6605 | 0.083 | 0.889 | 0.036 | 0.469 | 0.111 |
| PBMC 5' v1 | singlet-fixed[unspliced] vs starsolo_Velocyto[unspliced] | unspliced | 6605 | 1.463 | 0.547 | 0.031 | 0.554 | 0.939 |
| PBMC 5' v1 | singlet-fixed[ambiguous] vs starsolo_Velocyto[ambiguous] | ambiguous | 6605 | 0.041 | 0.769 | -0.001 | -0.075 | 0.050 |
| PBMC 5' v1 | singlet-fixed[spliced] vs alevinfry[spliced] | spliced | 6605 | 31.833 | 0.639 | 0.159 | 0.434 | 15.273 |
| PBMC 5' v1 | singlet-fixed[unspliced] vs alevinfry[unspliced] | unspliced | 6605 | 0.962 | 0.988 | 0.752 | 0.911 | 1.024 |
| PBMC 5' v1 | singlet-fixed[X] vs alevinfry[X] | total | 6605 | 1.133 | 0.993 | 0.795 | 0.934 | 1.115 |
| PBMC 5' v1 | alevinfry-fw[X] vs starsolo_GeneFull[X] | total (S+U+A), expected-ori fw | 6605 | 1.046 | 0.998 | 0.984 | 0.954 | 1.013 |
| PBMC 5' v1 | singlet-fixed-rev[X] vs starsolo_GeneFull[X] | total, forced --reverse-strand | 6605 | 1.025 | 1.000 | 0.987 | 0.959 | 1.018 |
| PBMC 5' v1 | singlet-fixed-rev[exon] vs starsolo_Gene[X] | exonic, forced --reverse-strand | 6605 | 0.990 | 1.000 | 0.994 | 0.992 | 1.000 |
| PBMC 5' v1 | singlet-fixed-rev[unspliced] vs starsolo_Velocyto[unspliced] | unspliced, forced --reverse-strand | 6605 | 1.112 | 0.986 | 0.900 | 0.950 | 1.083 |
| PBMC CITE 3' v3 | singlet-fixed[X] vs starsolo_GeneFull[X] | total (exon+intron) | 583 | 1.020 | 1.000 | 0.983 | 0.948 | 1.035 |
| PBMC CITE 3' v3 | singlet-baseline-t4[X] vs starsolo_GeneFull[X] | total (exon+intron), atlas config | 583 | 1.095 | 0.999 | 0.979 | 0.943 | 1.029 |
| PBMC CITE 3' v3 | singlet-baseline[X] vs starsolo_GeneFull[X] | total (exon+intron), 16 workers | 583 | 1.098 | 0.999 | 0.979 | 0.943 | 1.027 |
| PBMC CITE 3' v3 | alevinfry[X] vs starsolo_GeneFull[X] | total (S+U+A) | 583 | 1.071 | 1.000 | 0.978 | 0.946 | 1.002 |
| PBMC CITE 3' v3 | singlet-fixed[exon] vs starsolo_Gene[X] | exonic (S+A) | 583 | 0.991 | 1.000 | 0.994 | 0.991 | 0.998 |
| PBMC CITE 3' v3 | singlet-baseline-t4[exon] vs starsolo_Gene[X] | exonic, atlas config | 583 | 1.087 | 0.999 | 0.986 | 0.987 | 1.030 |
| PBMC CITE 3' v3 | cellranger_published[X] vs starsolo_Gene[X] | exonic (CR published) | 583 | 0.959 | 1.000 | 0.982 | 0.975 | 0.958 |
| PBMC CITE 3' v3 | singlet-fixed[exon] vs cellranger_published[X] | exonic vs CR | 583 | 1.032 | 1.000 | 0.980 | 0.972 | 1.040 |
| PBMC CITE 3' v3 | singlet-fixed[spliced] vs starsolo_Velocyto[spliced] | spliced | 583 | 1.240 | 0.998 | 0.932 | 0.910 | 1.201 |
| PBMC CITE 3' v3 | singlet-fixed[unspliced] vs starsolo_Velocyto[unspliced] | unspliced | 583 | 0.957 | 0.999 | 0.960 | 0.941 | 1.061 |
| PBMC CITE 3' v3 | singlet-fixed[ambiguous] vs starsolo_Velocyto[ambiguous] | ambiguous | 583 | 0.153 | 0.927 | 0.109 | 0.311 | 0.317 |
| PBMC CITE 3' v3 | singlet-fixed[spliced] vs alevinfry[spliced] | spliced | 583 | 3.080 | 0.899 | 0.661 | 0.750 | 3.591 |
| PBMC CITE 3' v3 | singlet-fixed[unspliced] vs alevinfry[unspliced] | unspliced | 583 | 0.965 | 0.999 | 0.945 | 0.943 | 1.042 |
| PBMC CITE 3' v3 | singlet-fixed[X] vs alevinfry[X] | total | 583 | 0.950 | 1.000 | 0.975 | 0.965 | 1.032 |
| Mouse neurons 3' v3 | singlet-fixed[X] vs starsolo_GeneFull[X] | total (exon+intron) | 1096 | 1.041 | 1.000 | 0.980 | 0.840 | 1.053 |
| Mouse neurons 3' v3 | singlet-baseline-t4[X] vs starsolo_GeneFull[X] | total (exon+intron), atlas config | 1096 | 1.102 | 0.999 | 0.971 | 0.831 | 1.067 |
| Mouse neurons 3' v3 | singlet-baseline[X] vs starsolo_GeneFull[X] | total (exon+intron), 16 workers | 1096 | 1.104 | 0.999 | 0.971 | 0.831 | 1.067 |
| Mouse neurons 3' v3 | alevinfry[X] vs starsolo_GeneFull[X] | total (S+U+A) | 1096 | 1.032 | 1.000 | 0.969 | 0.841 | 1.038 |
| Mouse neurons 3' v3 | singlet-fixed[exon] vs starsolo_Gene[X] | exonic (S+A) | 1096 | 0.981 | 1.000 | 0.989 | 0.963 | 1.005 |
| Mouse neurons 3' v3 | singlet-baseline-t4[exon] vs starsolo_Gene[X] | exonic, atlas config | 1096 | 1.014 | 1.000 | 0.984 | 0.958 | 1.026 |
| Mouse neurons 3' v3 | cellranger_published[X] vs starsolo_Gene[X] | exonic (CR published) | 1096 | 1.080 | 0.999 | 0.980 | 0.984 | 0.993 |
| Mouse neurons 3' v3 | singlet-fixed[exon] vs cellranger_published[X] | exonic vs CR | 1096 | 0.902 | 0.998 | 0.972 | 0.979 | 1.006 |
| Mouse neurons 3' v3 | singlet-fixed[spliced] vs starsolo_Velocyto[spliced] | spliced | 1096 | 1.200 | 0.997 | 0.934 | 0.826 | 1.194 |
| Mouse neurons 3' v3 | singlet-fixed[unspliced] vs starsolo_Velocyto[unspliced] | unspliced | 1096 | 0.983 | 0.984 | 0.982 | 0.928 | 1.016 |
| Mouse neurons 3' v3 | singlet-fixed[ambiguous] vs starsolo_Velocyto[ambiguous] | ambiguous | 1096 | 0.175 | 0.720 | 0.102 | 0.297 | 0.271 |
| Mouse neurons 3' v3 | singlet-fixed[spliced] vs alevinfry[spliced] | spliced | 1096 | 4.801 | 0.941 | 0.581 | 0.740 | 4.307 |
| Mouse neurons 3' v3 | singlet-fixed[unspliced] vs alevinfry[unspliced] | unspliced | 1096 | 0.973 | 0.985 | 0.977 | 0.917 | 0.991 |
| Mouse neurons 3' v3 | singlet-fixed[X] vs alevinfry[X] | total | 1096 | 1.011 | 1.000 | 0.978 | 0.923 | 1.015 |
| Mouse brain nuclei 3' v2 | singlet-fixed[X] vs starsolo_GeneFull[X] | total (exon+intron) | 2077 | 1.102 | 0.999 | 0.948 | 0.835 | 1.130 |
| Mouse brain nuclei 3' v2 | singlet-baseline-t4[X] vs starsolo_GeneFull[X] | total (exon+intron), atlas config | 2077 | 1.163 | 0.999 | 0.943 | 0.825 | 1.129 |
| Mouse brain nuclei 3' v2 | singlet-baseline[X] vs starsolo_GeneFull[X] | total (exon+intron), 16 workers | 2077 | 1.162 | 0.999 | 0.944 | 0.823 | 1.127 |
| Mouse brain nuclei 3' v2 | alevinfry[X] vs starsolo_GeneFull[X] | total (S+U+A) | 2077 | 1.071 | 1.000 | 0.961 | 0.839 | 1.018 |
| Mouse brain nuclei 3' v2 | singlet-fixed[exon] vs starsolo_Gene[X] | exonic (S+A) | 2077 | 1.023 | 1.000 | 0.957 | 0.949 | 1.028 |
| Mouse brain nuclei 3' v2 | singlet-baseline-t4[exon] vs starsolo_Gene[X] | exonic, atlas config | 2077 | 1.077 | 0.999 | 0.941 | 0.942 | 1.074 |
| Mouse brain nuclei 3' v2 | singlet-fixed[spliced] vs starsolo_Velocyto[spliced] | spliced | 2077 | 1.552 | 0.987 | 0.806 | 0.785 | 1.515 |
| Mouse brain nuclei 3' v2 | singlet-fixed[unspliced] vs starsolo_Velocyto[unspliced] | unspliced | 2077 | 1.045 | 0.997 | 0.930 | 0.862 | 1.097 |
| Mouse brain nuclei 3' v2 | singlet-fixed[ambiguous] vs starsolo_Velocyto[ambiguous] | ambiguous | 2077 | 0.491 | 0.931 | 0.062 | 0.290 | 0.638 |
| Mouse brain nuclei 3' v2 | singlet-fixed[spliced] vs alevinfry[spliced] | spliced | 2077 | 11.273 | 0.894 | 0.291 | 0.595 | 9.054 |
| Mouse brain nuclei 3' v2 | singlet-fixed[unspliced] vs alevinfry[unspliced] | unspliced | 2077 | 0.988 | 0.988 | 0.936 | 0.862 | 1.050 |
| Mouse brain nuclei 3' v2 | singlet-fixed[X] vs alevinfry[X] | total | 2077 | 1.027 | 1.000 | 0.943 | 0.894 | 1.108 |
| Human+mouse barnyard 3' v3 | singlet-fixed[X] vs starsolo_GeneFull[X] | total (exon+intron) | 880 | 1.023 | 1.000 | 0.984 | 0.916 | 1.033 |
| Human+mouse barnyard 3' v3 | singlet-baseline-t4[X] vs starsolo_GeneFull[X] | total (exon+intron), atlas config | 880 | 1.053 | 1.000 | 0.977 | 0.907 | 1.046 |
| Human+mouse barnyard 3' v3 | singlet-baseline[X] vs starsolo_GeneFull[X] | total (exon+intron), 16 workers | 880 | 1.053 | 1.000 | 0.977 | 0.907 | 1.046 |
| Human+mouse barnyard 3' v3 | alevinfry[X] vs starsolo_GeneFull[X] | total (S+U+A) | 880 | 0.973 | 0.999 | 0.965 | 0.902 | 1.018 |
| Human+mouse barnyard 3' v3 | singlet-fixed[exon] vs starsolo_Gene[X] | exonic (S+A) | 880 | 0.967 | 1.000 | 0.989 | 0.984 | 1.000 |
| Human+mouse barnyard 3' v3 | singlet-baseline-t4[exon] vs starsolo_Gene[X] | exonic, atlas config | 880 | 0.985 | 0.999 | 0.985 | 0.980 | 1.016 |
| Human+mouse barnyard 3' v3 | cellranger_published[X] vs starsolo_Gene[X] | exonic (CR published) | 880 | - | - | - | - | 1.000 |
| Human+mouse barnyard 3' v3 | singlet-fixed[exon] vs cellranger_published[X] | exonic vs CR | 880 | - | - | - | - | 1.000 |
| Human+mouse barnyard 3' v3 | singlet-fixed[spliced] vs starsolo_Velocyto[spliced] | spliced | 880 | 1.164 | 0.998 | 0.943 | 0.895 | 1.184 |
| Human+mouse barnyard 3' v3 | singlet-fixed[unspliced] vs starsolo_Velocyto[unspliced] | unspliced | 880 | 1.028 | 0.999 | 0.980 | 0.956 | 1.041 |
| Human+mouse barnyard 3' v3 | singlet-fixed[ambiguous] vs starsolo_Velocyto[ambiguous] | ambiguous | 880 | 0.023 | 0.948 | 0.051 | 0.180 | 0.050 |
| Human+mouse barnyard 3' v3 | singlet-fixed[spliced] vs alevinfry[spliced] | spliced | 880 | 5.367 | 0.965 | 0.592 | 0.765 | 4.927 |
| Human+mouse barnyard 3' v3 | singlet-fixed[unspliced] vs alevinfry[unspliced] | unspliced | 880 | 1.031 | 0.998 | 0.964 | 0.935 | 1.037 |
| Human+mouse barnyard 3' v3 | singlet-fixed[X] vs alevinfry[X] | total | 880 | 1.048 | 0.998 | 0.971 | 0.951 | 1.015 |

## cell_calls

| dataset | Singlet (fixed) | Singlet (atlas) | STARsolo EmptyDrops_CR | knee CR2.2 on STARsolo GeneFull | knee CR2.2 on Singlet (fixed) | Cell Ranger (published) | J(Singlet fixed, STARsolo) | J(Singlet fixed, Cell Ranger) | J(STARsolo, Cell Ranger) | J(knee CR2.2, Cell Ranger) | J(Singlet atlas, fixed) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PBMC 3' v3 | 1136 | 1136 | 1223 | 1152 | 1152 | 1222 | 0.927 | 0.920 | 0.978 | 0.932 | 1.000 |
| PBMC 3' v2 | 983 | 983 | 1016 | 992 | 990 | 996 | 0.968 | 0.985 | 0.978 | 0.986 | 1.000 |
| PBMC 5' v1 | 6627 | 6627 | 7740 | 6871 | 6841 | 8258 | 0.851 | 0.801 | 0.937 | 0.832 | 1.000 |
| PBMC CITE 3' v3 | 583 | 583 | 721 | 627 | 626 | 713 | 0.809 | 0.818 | 0.989 | 0.879 | 1.000 |
| Mouse neurons 3' v3 | 1102 | 1102 | 1263 | 1135 | 1136 | 1301 | 0.864 | 0.844 | 0.947 | 0.871 | 1.000 |
| Mouse brain nuclei 3' v2 | 2188 | 2188 | 2151 | 2016 | 1995 | - | 0.918 | - | - | - | 1.000 |
| Human+mouse barnyard 3' v3 | 880 | 880 | 1041 | 900 | 900 | 1046 | 0.845 | 0.841 | 0.995 | 0.860 | 1.000 |

## clusters

| dataset | tool | ARI vs STARsolo GeneFull clusters | top-50 marker Jaccard (same labels) | ARI vs ADT clusters |
|---|---|---|---|---|
| PBMC 3' v3 | Singlet (fixed) | 0.833 | 0.926 | - |
| PBMC 3' v3 | Singlet (atlas engine) | 0.809 | 0.895 | - |
| PBMC 3' v3 | alevin-fry (USA) | 0.866 | 0.901 | - |
| PBMC 3' v3 | STARsolo Gene | 0.767 | 0.511 | - |
| PBMC 3' v3 | Cell Ranger (10x published) | 0.780 | 0.455 | - |
| PBMC 3' v2 | Singlet (fixed) | 0.789 | 0.901 | - |
| PBMC 3' v2 | Singlet (atlas engine) | 0.712 | 0.893 | - |
| PBMC 3' v2 | alevin-fry (USA) | 0.825 | 0.888 | - |
| PBMC 3' v2 | STARsolo Gene | 0.613 | 0.604 | - |
| PBMC 3' v2 | Cell Ranger (10x published) | 0.612 | 0.549 | - |
| PBMC 5' v1 | Singlet (fixed) | 0.446 | 0.016 | - |
| PBMC 5' v1 | Singlet (atlas engine) | 0.437 | 0.016 | - |
| PBMC 5' v1 | alevin-fry (USA) | 0.379 | 0.019 | - |
| PBMC 5' v1 | STARsolo Gene | 0.519 | 0.803 | - |
| PBMC 5' v1 | Cell Ranger (10x published) | 0.655 | 0.670 | - |
| PBMC CITE 3' v3 | Singlet (fixed) | 0.698 | 0.918 | 0.559 |
| PBMC CITE 3' v3 | Singlet (atlas engine) | 0.724 | 0.903 | 0.471 |
| PBMC CITE 3' v3 | alevin-fry (USA) | 0.720 | 0.891 | 0.509 |
| PBMC CITE 3' v3 | STARsolo Gene | 0.537 | 0.499 | 0.572 |
| PBMC CITE 3' v3 | Cell Ranger (10x published) | 0.529 | 0.449 | 0.573 |
| PBMC CITE 3' v3 | STARsolo GeneFull | 1.000 | 1.000 | 0.440 |
| Mouse neurons 3' v3 | Singlet (fixed) | 0.680 | 0.911 | - |
| Mouse neurons 3' v3 | Singlet (atlas engine) | 0.739 | 0.879 | - |
| Mouse neurons 3' v3 | alevin-fry (USA) | 0.670 | 0.910 | - |
| Mouse neurons 3' v3 | STARsolo Gene | 0.658 | 0.520 | - |
| Mouse neurons 3' v3 | Cell Ranger (10x published) | 0.645 | 0.484 | - |
| Mouse brain nuclei 3' v2 | Singlet (fixed) | 0.562 | 0.894 | - |
| Mouse brain nuclei 3' v2 | Singlet (atlas engine) | 0.613 | 0.879 | - |
| Mouse brain nuclei 3' v2 | alevin-fry (USA) | 0.613 | 0.887 | - |
| Mouse brain nuclei 3' v2 | STARsolo Gene | 0.274 | 0.263 | - |
| Human+mouse barnyard 3' v3 | Singlet (fixed) | 0.608 | 0.885 | - |
| Human+mouse barnyard 3' v3 | Singlet (atlas engine) | 0.578 | 0.866 | - |
| Human+mouse barnyard 3' v3 | alevin-fry (USA) | 0.508 | 0.804 | - |
| Human+mouse barnyard 3' v3 | STARsolo Gene | 0.348 | 0.531 | - |
| Human+mouse barnyard 3' v3 | Cell Ranger (10x published) | 0.390 | 0.000 | - |

## barnyard

| tool | cells | human | mouse | mixed (doublet) | doublet rate | cross-species UMIs in singlets (median) | (mean) |
|---|---|---|---|---|---|---|---|
| Singlet (atlas engine, 16 workers) | 880 | 421 | 458 | 1 | 0.1% | 0.55% | 0.49% |
| Singlet (fixed) | 880 | 421 | 458 | 1 | 0.1% | 0.59% | 0.52% |
| Singlet (atlas engine) | 880 | 421 | 458 | 1 | 0.1% | 0.55% | 0.50% |
| Singlet (fixed, 4 workers) | 880 | 421 | 458 | 1 | 0.1% | 0.59% | 0.52% |
| STARsolo Gene | 1049 | 499 | 545 | 5 | 0.5% | 0.47% | 0.68% |
| STARsolo GeneFull | 1041 | 491 | 545 | 5 | 0.5% | 0.51% | 0.59% |
| STARsolo Velocyto | 1049 | 499 | 544 | 6 | 0.6% | 0.46% | 0.57% |
| alevin-fry (USA) | 880 | 421 | 458 | 1 | 0.1% | 0.57% | 0.66% |
| Cell Ranger (10x published) | 1046 | 495 | 544 | 7 | 0.7% | 0.45% | 0.63% |

## resources

| dataset | tool | wall (min) | + .1fq encode (min) | CPU (h) | peak RSS (GB) |
|---|---|---|---|---|---|
| PBMC 3' v3 | Singlet (atlas engine, 16 workers) | 9.5 | 2.5 | 1.06 | 20.1 |
| PBMC 3' v3 | Singlet (fixed) | 9.4 | 2.6 | 1.06 | 20.1 |
| PBMC 3' v3 | Singlet (atlas engine) | 16.9 | 2.1 | 0.92 | 17.7 |
| PBMC 3' v3 | Singlet (fixed, 4 workers) | 17.0 | 2.1 | 0.92 | 17.7 |
| PBMC 3' v3 | STARsolo | 8.5 | - | 1.09 | 19.4 |
| PBMC 3' v3 | alevin-fry (USA) | 1.9 | - | 0.44 | 3.4 |
| PBMC 3' v2 | Singlet (atlas engine, 16 workers) | 9.5 | 2.4 | 1.17 | 19.5 |
| PBMC 3' v2 | Singlet (fixed) | 9.6 | 2.4 | 1.17 | 19.5 |
| PBMC 3' v2 | Singlet (atlas engine) | 21.4 | 2.4 | 1.16 | 17.4 |
| PBMC 3' v2 | Singlet (fixed, 4 workers) | 23.5 | 2.4 | 1.29 | 17.4 |
| PBMC 3' v2 | STARsolo | 9.5 | - | 1.40 | 18.7 |
| PBMC 3' v2 | alevin-fry (USA) | 2.0 | - | 0.49 | 3.4 |
| PBMC 5' v1 | Singlet (atlas engine, 16 workers) | 9.9 | 1.7 | 0.87 | 19.5 |
| PBMC 5' v1 | Singlet (fixed) | 9.8 | 1.8 | 0.88 | 19.5 |
| PBMC 5' v1 | Singlet (atlas engine) | 15.1 | 1.7 | 0.74 | 17.3 |
| PBMC 5' v1 | Singlet (fixed, 4 workers) | 14.9 | 1.7 | 0.74 | 17.4 |
| PBMC 5' v1 | STARsolo | 7.3 | - | 0.92 | 18.7 |
| PBMC 5' v1 | alevin-fry (USA) | 1.0 | - | 0.20 | 3.8 |
| PBMC CITE 3' v3 | Singlet (atlas engine, 16 workers) | 5.6 | 1.5 | 0.66 | 20.1 |
| PBMC CITE 3' v3 | Singlet (fixed) | 5.7 | 1.5 | 0.66 | 20.1 |
| PBMC CITE 3' v3 | Singlet (atlas engine) | 12.1 | 1.5 | 0.66 | 17.8 |
| PBMC CITE 3' v3 | Singlet (fixed, 4 workers) | 11.3 | 1.3 | 0.62 | 17.8 |
| PBMC CITE 3' v3 | STARsolo | 6.2 | - | 0.82 | 19.4 |
| PBMC CITE 3' v3 | alevin-fry (USA) | 1.5 | - | 0.35 | 3.4 |
| Mouse neurons 3' v3 | Singlet (atlas engine, 16 workers) | 9.6 | 2.8 | 0.76 | 29.2 |
| Mouse neurons 3' v3 | Singlet (fixed) | 9.8 | 2.8 | 0.77 | 29.2 |
| Mouse neurons 3' v3 | Singlet (atlas engine) | 19.1 | 3.4 | 0.88 | 26.6 |
| Mouse neurons 3' v3 | Singlet (fixed, 4 workers) | 19.0 | 3.4 | 0.90 | 26.6 |
| Mouse neurons 3' v3 | STARsolo | 8.0 | - | 0.81 | 28.4 |
| Mouse neurons 3' v3 | alevin-fry (USA) | 1.5 | - | 0.35 | 2.7 |
| Mouse brain nuclei 3' v2 | Singlet (atlas engine, 16 workers) | 13.0 | 4.1 | 1.61 | 28.5 |
| Mouse brain nuclei 3' v2 | Singlet (fixed) | 12.6 | 4.1 | 1.61 | 28.6 |
| Mouse brain nuclei 3' v2 | Singlet (atlas engine) | 28.7 | 4.1 | 1.60 | 26.2 |
| Mouse brain nuclei 3' v2 | Singlet (fixed, 4 workers) | 33.5 | 5.0 | 1.90 | 26.2 |
| Mouse brain nuclei 3' v2 | STARsolo | 10.6 | - | 1.61 | 27.4 |
| Mouse brain nuclei 3' v2 | alevin-fry (USA) | 8.7 | - | 0.59 | 2.6 |
| Human+mouse barnyard 3' v3 | Singlet (atlas engine, 16 workers) | 9.4 | 2.3 | 0.75 | 57.2 |
| Human+mouse barnyard 3' v3 | Singlet (fixed) | 9.4 | 2.3 | 0.74 | 57.2 |
| Human+mouse barnyard 3' v3 | Singlet (atlas engine) | 13.3 | 2.0 | 0.63 | 54.5 |
| Human+mouse barnyard 3' v3 | Singlet (fixed, 4 workers) | 14.1 | 2.0 | 0.64 | 54.5 |
| Human+mouse barnyard 3' v3 | STARsolo | 8.6 | - | 0.72 | 56.3 |
| Human+mouse barnyard 3' v3 | alevin-fry (USA) | 1.7 | - | 0.37 | 5.9 |
