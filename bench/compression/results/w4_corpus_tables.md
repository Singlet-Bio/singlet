files with ref+a rows: 77; a_names_perm_model matrix roundtrip OK on 77
files with b_rans16 rows: 69; roundtrip OK on 69; unsupported on 8

POOLED all files: today 679.8 MB; a_names_perm_model 316.2 MB = 0.465 (2.15x) [original corpus 0.468 = 2.14x]
POOLED combined projection (A names + A row counts + B payload): 282.3 MB = 0.415 (2.41x)
POOLED b_rans16 payload vs today's payload (chunks+ptr+chunk table), n=69 files: -14.8%; incl. row-count side stream: -12.0% [original corpus -15.1%]
  exon: n=34 payload -15.6%
  intron: n=35 payload -13.0%

### (a) container ratio = a_names_perm_model bytes / today's file bytes, by protocol (original corpus pooled 0.468)

| protocol | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| 10x-arc-gex | 4 | 2 | 0.429 | 0.320 | 0.537 | h_arc/intron (0.537) | 0.434 |
| 10x-visium | 4 | 2 | 0.176 | 0.083 | 0.362 | m_visium/exon (0.362) | 0.207 |
| 10xv2 | 18 | 9 | 0.430 | 0.187 | 0.691 | z_v2/exon (0.691) | 0.459 |
| 10xv3 | 28 | 14 | 0.487 | 0.076 | 0.738 | L_b/exon (0.738) | 0.518 |
| bd-rhapsody | 2 | 1 | 0.431 | 0.298 | 0.565 | h_bd/exon (0.565) | 0.470 |
| celseq2 | 4 | 2 | 0.186 | 0.088 | 0.318 | h_celseq2/intron (0.318) | 0.188 |
| dropseq | 5 | 3 | 0.358 | 0.101 | 0.730 | m_dropseq/exon (0.730) | 0.429 |
| indrop | 2 | 1 | 0.323 | 0.252 | 0.394 | h_indrop/exon (0.394) | 0.335 |
| marsseq2 | 4 | 2 | 0.260 | 0.012 | 0.688 | h_mars/intron (0.688) | 0.364 |
| microwell-seq | 2 | 1 | 0.523 | 0.305 | 0.740 | h_microwell/exon (0.740) | 0.633 |
| quartzseq2 | 2 | 1 | 0.560 | 0.437 | 0.683 | h_quartz/exon (0.683) | 0.602 |
| splitseq | 2 | 1 | 0.195 | 0.129 | 0.262 | h_split/intron (0.262) | 0.197 |
| **all** | 77 | 39 | 0.408 | 0.012 | 0.740 | | 0.465 |

### (a) container ratio = a_names_perm_model bytes / today's file bytes, by species (original corpus pooled 0.468)

| species | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| barnyard | 8 | 4 | 0.331 | 0.101 | 0.457 | bn_v3_a/intron (0.457) | 0.351 |
| human | 33 | 17 | 0.417 | 0.083 | 0.740 | h_microwell/exon (0.740) | 0.490 |
| macaque | 4 | 2 | 0.607 | 0.451 | 0.718 | mq_v3/exon (0.718) | 0.634 |
| mouse | 22 | 11 | 0.342 | 0.012 | 0.730 | m_dropseq/exon (0.730) | 0.427 |
| rat | 4 | 2 | 0.455 | 0.210 | 0.641 | r_v2/exon (0.641) | 0.501 |
| zebrafish | 6 | 3 | 0.542 | 0.316 | 0.709 | z_v3_b/exon (0.709) | 0.585 |
| **all** | 77 | 39 | 0.408 | 0.012 | 0.740 | | 0.465 |

### (b) b_rans16 payload vs today's payload, by protocol (negative = smaller; original corpus -15.1% pooled, -11..-24% per file)

| protocol | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| 10x-arc-gex | 4 | 2 | -0.145 | -0.238 | -0.086 | h_arc/intron (-0.086) | -0.142 |
| 10x-visium | 4 | 2 | -0.169 | -0.215 | -0.117 | m_visium/exon (-0.117) | -0.138 |
| 10xv2 | 16 | 8 | -0.151 | -0.205 | -0.109 | r_v2/intron (-0.109) | -0.151 |
| 10xv3 | 24 | 12 | -0.151 | -0.325 | -0.085 | h_v3_a/intron (-0.085) | -0.140 |
| bd-rhapsody | 2 | 1 | -0.157 | -0.180 | -0.133 | h_bd/exon (-0.133) | -0.143 |
| celseq2 | 4 | 2 | -0.138 | -0.160 | -0.118 | h_celseq2/intron (-0.118) | -0.129 |
| dropseq | 3 | 2 | -0.230 | -0.359 | -0.137 | m_dropseq/exon (-0.137) | -0.153 |
| indrop | 2 | 1 | -0.233 | -0.281 | -0.185 | h_indrop/intron (-0.185) | -0.255 |
| marsseq2 | 4 | 2 | -0.151 | -0.273 | -0.069 | h_mars/intron (-0.069) | -0.080 |
| microwell-seq | 2 | 1 | -0.191 | -0.234 | -0.149 | h_microwell/exon (-0.149) | -0.157 |
| quartzseq2 | 2 | 1 | -0.250 | -0.286 | -0.214 | h_quartz/exon (-0.214) | -0.223 |
| splitseq | 2 | 1 | -0.246 | -0.322 | -0.170 | h_split/intron (-0.170) | -0.207 |
| **all** | 69 | 35 | -0.164 | -0.359 | -0.069 | | -0.148 |

### (b) b_rans16 payload vs today's payload, by species (negative = smaller; original corpus -15.1% pooled, -11..-24% per file)

| species | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| human | 33 | 17 | -0.163 | -0.359 | -0.069 | h_mars/intron (-0.069) | -0.145 |
| macaque | 4 | 2 | -0.139 | -0.159 | -0.128 | mq_v3/intron (-0.128) | -0.136 |
| mouse | 22 | 11 | -0.165 | -0.325 | -0.091 | m_v3_a/intron (-0.091) | -0.145 |
| rat | 4 | 2 | -0.176 | -0.228 | -0.109 | r_v2/intron (-0.109) | -0.158 |
| zebrafish | 6 | 3 | -0.171 | -0.205 | -0.139 | z_v3_a/intron (-0.139) | -0.172 |
| **all** | 69 | 35 | -0.164 | -0.359 | -0.069 | | -0.148 |

### (c) derivable share of today's file bytes = (names + permutation + ptr + colsums) / file, by protocol (original corpus: 77.9% per sample)

| protocol | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| 10x-arc-gex | 4 | 2 | 0.622 | 0.503 | 0.725 | h_arc/intron (0.503) | 0.617 |
| 10x-visium | 4 | 2 | 0.837 | 0.652 | 0.927 | m_visium/exon (0.652) | 0.807 |
| 10xv2 | 18 | 9 | 0.607 | 0.334 | 0.840 | z_v2/exon (0.334) | 0.575 |
| 10xv3 | 28 | 14 | 0.550 | 0.280 | 0.967 | L_b/exon (0.280) | 0.514 |
| bd-rhapsody | 2 | 1 | 0.622 | 0.476 | 0.768 | h_bd/exon (0.476) | 0.580 |
| celseq2 | 4 | 2 | 0.824 | 0.693 | 0.924 | h_celseq2/intron (0.693) | 0.822 |
| dropseq | 5 | 3 | 0.731 | 0.319 | 0.918 | m_dropseq/exon (0.319) | 0.640 |
| indrop | 2 | 1 | 0.743 | 0.660 | 0.825 | h_indrop/exon (0.660) | 0.729 |
| marsseq2 | 4 | 2 | 0.756 | 0.333 | 0.994 | h_mars/intron (0.333) | 0.653 |
| microwell-seq | 2 | 1 | 0.536 | 0.290 | 0.782 | h_microwell/exon (0.290) | 0.412 |
| quartzseq2 | 2 | 1 | 0.666 | 0.468 | 0.864 | h_quartz/exon (0.468) | 0.599 |
| splitseq | 2 | 1 | 0.874 | 0.809 | 0.939 | h_split/intron (0.809) | 0.873 |
| **all** | 77 | 39 | 0.636 | 0.280 | 0.994 | | 0.578 |

### (c) derivable share of today's file bytes = (names + permutation + ptr + colsums) / file, by species (original corpus: 77.9% per sample)

| species | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| barnyard | 8 | 4 | 0.690 | 0.568 | 0.915 | bn_v3_a/intron (0.568) | 0.670 |
| human | 33 | 17 | 0.640 | 0.280 | 0.939 | L_b/exon (0.280) | 0.569 |
| macaque | 4 | 2 | 0.431 | 0.310 | 0.598 | mq_v3/exon (0.310) | 0.402 |
| mouse | 22 | 11 | 0.694 | 0.319 | 0.994 | m_dropseq/exon (0.319) | 0.610 |
| rat | 4 | 2 | 0.594 | 0.393 | 0.854 | r_v2/exon (0.393) | 0.544 |
| zebrafish | 6 | 3 | 0.501 | 0.319 | 0.741 | z_v3_b/exon (0.319) | 0.454 |
| **all** | 77 | 39 | 0.636 | 0.280 | 0.994 | | 0.578 |

### (c2) names (row + column) share of today's file bytes, by protocol

| protocol | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| 10x-arc-gex | 4 | 2 | 0.497 | 0.403 | 0.579 | h_arc/exon (0.579) | 0.493 |
| 10x-visium | 4 | 2 | 0.671 | 0.522 | 0.742 | h_visium/exon (0.742) | 0.646 |
| 10xv2 | 18 | 9 | 0.485 | 0.263 | 0.671 | m_v2_a/intron (0.671) | 0.459 |
| 10xv3 | 28 | 14 | 0.439 | 0.220 | 0.783 | L_a/intron (0.783) | 0.411 |
| bd-rhapsody | 2 | 1 | 0.497 | 0.379 | 0.616 | h_bd/intron (0.616) | 0.463 |
| celseq2 | 4 | 2 | 0.659 | 0.552 | 0.742 | m_celseq2/intron (0.742) | 0.657 |
| dropseq | 5 | 3 | 0.594 | 0.254 | 0.768 | h_dropseq/intron (0.768) | 0.518 |
| indrop | 2 | 1 | 0.596 | 0.526 | 0.666 | h_indrop/intron (0.666) | 0.584 |
| marsseq2 | 4 | 2 | 0.604 | 0.262 | 0.801 | m_mars/intron (0.801) | 0.521 |
| microwell-seq | 2 | 1 | 0.429 | 0.229 | 0.629 | h_microwell/intron (0.629) | 0.328 |
| quartzseq2 | 2 | 1 | 0.557 | 0.382 | 0.732 | h_quartz/intron (0.732) | 0.497 |
| splitseq | 2 | 1 | 0.705 | 0.653 | 0.758 | h_split/exon (0.758) | 0.704 |
| **all** | 77 | 39 | 0.510 | 0.220 | 0.801 | | 0.463 |

### (c2) names (row + column) share of today's file bytes, by species

| species | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| barnyard | 8 | 4 | 0.555 | 0.454 | 0.735 | bn_dropseq/intron (0.735) | 0.538 |
| human | 33 | 17 | 0.514 | 0.220 | 0.768 | h_dropseq/intron (0.768) | 0.456 |
| macaque | 4 | 2 | 0.341 | 0.244 | 0.476 | mq_v2/intron (0.476) | 0.318 |
| mouse | 22 | 11 | 0.558 | 0.254 | 0.801 | m_mars/intron (0.801) | 0.489 |
| rat | 4 | 2 | 0.474 | 0.312 | 0.684 | r_v3/intron (0.684) | 0.434 |
| zebrafish | 6 | 3 | 0.398 | 0.251 | 0.593 | z_v2/intron (0.593) | 0.360 |
| **all** | 77 | 39 | 0.510 | 0.220 | 0.801 | | 0.463 |

### Total file ratio, combined projection (vocab id + whitelist/inline barcodes + row counts + b_rans16 payload) / today, by protocol (original corpus about 0.40)

| protocol | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| 10x-arc-gex | 4 | 2 | 0.380 | 0.290 | 0.496 | h_arc/intron (0.496) | 0.384 |
| 10x-visium | 4 | 2 | 0.153 | 0.069 | 0.321 | m_visium/exon (0.321) | 0.181 |
| 10xv2 | 18 | 9 | 0.381 | 0.164 | 0.579 | z_v2/exon (0.579) | 0.408 |
| 10xv3 | 28 | 14 | 0.436 | 0.069 | 0.636 | L_b/exon (0.636) | 0.465 |
| bd-rhapsody | 2 | 1 | 0.381 | 0.263 | 0.499 | h_bd/exon (0.499) | 0.415 |
| celseq2 | 4 | 2 | 0.164 | 0.077 | 0.283 | h_celseq2/intron (0.283) | 0.166 |
| dropseq | 5 | 3 | 0.329 | 0.101 | 0.643 | m_dropseq/exon (0.643) | 0.390 |
| indrop | 2 | 1 | 0.267 | 0.227 | 0.306 | h_indrop/exon (0.306) | 0.273 |
| marsseq2 | 4 | 2 | 0.239 | 0.011 | 0.642 | h_mars/intron (0.642) | 0.337 |
| microwell-seq | 2 | 1 | 0.452 | 0.263 | 0.641 | h_microwell/exon (0.641) | 0.547 |
| quartzseq2 | 2 | 1 | 0.499 | 0.414 | 0.584 | h_quartz/exon (0.584) | 0.528 |
| splitseq | 2 | 1 | 0.175 | 0.114 | 0.236 | h_split/intron (0.236) | 0.176 |
| **all** | 77 | 39 | 0.363 | 0.011 | 0.643 | | 0.415 |

### Total file ratio, combined projection (vocab id + whitelist/inline barcodes + row counts + b_rans16 payload) / today, by species (original corpus about 0.40)

| species | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |
|---|---|---|---|---|---|---|---|
| barnyard | 8 | 4 | 0.331 | 0.101 | 0.457 | bn_v3_a/intron (0.457) | 0.351 |
| human | 33 | 17 | 0.370 | 0.069 | 0.642 | h_mars/intron (0.642) | 0.433 |
| macaque | 4 | 2 | 0.534 | 0.394 | 0.631 | mq_v3/exon (0.631) | 0.558 |
| mouse | 22 | 11 | 0.301 | 0.011 | 0.643 | m_dropseq/exon (0.643) | 0.375 |
| rat | 4 | 2 | 0.395 | 0.185 | 0.559 | r_v2/exon (0.559) | 0.435 |
| zebrafish | 6 | 3 | 0.463 | 0.270 | 0.587 | z_v3_b/exon (0.587) | 0.496 |
| **all** | 77 | 39 | 0.363 | 0.011 | 0.643 | | 0.415 |

### per-file detail

| sample | member | protocol | build | m | n | nnz | file MB | derivable % | names % | container ratio | b payload vs today | combined ratio | A row/bc mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_a | exon | 10xv3 | GRCm39-2024-A | 386028 | 67271 | 429257 | 6.39 | 88.6 | 71.3 | 0.150 | -23.6% | 0.127 | 1/2 |
| L_a | intron | 10xv3 | GRCm39-2024-A | 307730 | 67271 | 74209 | 4.76 | 96.7 | 78.3 | 0.076 | -32.5% | 0.069 | 1/2 |
| L_b | exon | 10xv3 | GRCh38-2024-A | 310797 | 70726 | 12784194 | 16.84 | 28.0 | 22.0 | 0.738 | -14.6% | 0.636 | 1/2 |
| L_b | intron | 10xv3 | GRCh38-2024-A | 272191 | 70726 | 3784856 | 9.61 | 43.8 | 34.6 | 0.593 | -13.3% | 0.525 | 1/2 |
| bn_dropseq | exon | dropseq | GRCh38-GRCm39-barnyard | 696825 | 25840 | 4120360 | 12.61 | 77.3 | 62.2 | 0.238 | n/a | 0.238 | 1/0 |
| bn_dropseq | intron | dropseq | GRCh38-GRCm39-barnyard | 579921 | 25840 | 668236 | 8.84 | 91.5 | 73.5 | 0.101 | n/a | 0.101 | 1/0 |
| bn_v2 | exon | 10xv2 | GRCh38-GRCm39-barnyard | 696825 | 83275 | 7389676 | 16.49 | 61.5 | 49.4 | 0.404 | n/a | 0.404 | 1/2 |
| bn_v2 | intron | 10xv2 | GRCh38-GRCm39-barnyard | 579921 | 83275 | 2675308 | 11.64 | 72.8 | 58.5 | 0.299 | n/a | 0.299 | 1/2 |
| bn_v3_a | exon | 10xv3 | GRCh38-GRCm39-barnyard | 696825 | 85824 | 4892188 | 16.65 | 61.3 | 49.2 | 0.408 | n/a | 0.408 | 1/2 |
| bn_v3_a | intron | 10xv3 | GRCh38-GRCm39-barnyard | 579921 | 85824 | 6138866 | 15.08 | 56.8 | 45.4 | 0.457 | n/a | 0.457 | 1/2 |
| bn_v3_b | exon | 10xv3 | GRCh38-GRCm39-barnyard | 696825 | 75366 | 8031264 | 16.97 | 59.7 | 47.9 | 0.423 | n/a | 0.423 | 1/2 |
| bn_v3_b | intron | 10xv3 | GRCh38-GRCm39-barnyard | 579921 | 75366 | 3279094 | 11.84 | 71.3 | 57.4 | 0.316 | n/a | 0.316 | 1/2 |
| h_arc | exon | 10x-arc-gex | GRCh38-2024-A | 310797 | 46324 | 1500040 | 6.14 | 72.5 | 57.9 | 0.320 | -12.2% | 0.290 | 1/1 |
| h_arc | intron | 10x-arc-gex | GRCh38-2024-A | 272191 | 46324 | 4173348 | 7.90 | 50.3 | 40.3 | 0.537 | -8.6% | 0.496 | 1/1 |
| h_bd | exon | bd-rhapsody | GRCh38-2024-A | 310797 | 60922 | 4900306 | 9.81 | 47.6 | 37.9 | 0.565 | -13.3% | 0.499 | 1/0 |
| h_bd | intron | bd-rhapsody | GRCh38-2024-A | 272191 | 60922 | 722361 | 5.42 | 76.8 | 61.6 | 0.298 | -18.0% | 0.263 | 1/0 |
| h_celseq2 | exon | celseq2 | GRCh38-2024-A | 310797 | 4096 | 1333900 | 5.31 | 79.1 | 63.0 | 0.215 | -12.5% | 0.190 | 1/0 |
| h_celseq2 | intron | celseq2 | GRCh38-2024-A | 272191 | 4096 | 2242285 | 5.38 | 69.3 | 55.2 | 0.318 | -11.8% | 0.283 | 1/0 |
| h_dropseq | intron | dropseq | GRCh38-2024-A | 272191 | 435682 | 213808 | 6.79 | 91.8 | 76.8 | 0.320 | -35.9% | 0.303 | 1/0 |
| h_indrop | exon | indrop | GRCh38-2024-A | 310797 | 95760 | 2079158 | 7.25 | 66.0 | 52.6 | 0.394 | -28.1% | 0.306 | 1/0 |
| h_indrop | intron | indrop | GRCh38-2024-A | 272191 | 95760 | 599993 | 5.16 | 82.5 | 66.6 | 0.252 | -18.5% | 0.227 | 1/0 |
| h_mars | exon | marsseq2 | GRCh38-2024-A | 310797 | 16383 | 741536 | 5.52 | 76.9 | 61.0 | 0.257 | -12.1% | 0.231 | 1/0 |
| h_mars | intron | marsseq2 | GRCh38-2024-A | 272191 | 16383 | 6732602 | 11.43 | 33.3 | 26.2 | 0.688 | -6.9% | 0.642 | 1/0 |
| h_microwell | exon | microwell-seq | GRCh38-2024-A | 310797 | 117485 | 12227148 | 17.15 | 29.0 | 22.9 | 0.740 | -14.9% | 0.641 | 1/0 |
| h_microwell | intron | microwell-seq | GRCh38-2024-A | 272191 | 117485 | 718911 | 5.65 | 78.2 | 62.9 | 0.305 | -23.4% | 0.263 | 1/0 |
| h_quartz | exon | quartzseq2 | GRCh38-2024-A | 310797 | 692824 | 6597852 | 18.15 | 46.8 | 38.2 | 0.683 | -21.4% | 0.584 | 1/0 |
| h_quartz | intron | quartzseq2 | GRCh38-2024-A | 272191 | 692824 | 509641 | 8.96 | 86.4 | 73.2 | 0.437 | -28.6% | 0.414 | 1/0 |
| h_split | exon | splitseq | GRCh38-2024-A | 310797 | 85694 | 159444 | 5.00 | 93.9 | 75.8 | 0.129 | -32.2% | 0.114 | 1/0 |
| h_split | intron | splitseq | GRCh38-2024-A | 272191 | 85694 | 627825 | 5.23 | 80.9 | 65.3 | 0.262 | -17.0% | 0.236 | 1/0 |
| h_v2_a | exon | 10xv2 | GRCh38-2024-A | 310797 | 87935 | 3077129 | 7.82 | 60.6 | 48.2 | 0.432 | -18.6% | 0.365 | 1/2 |
| h_v2_a | intron | 10xv2 | GRCh38-2024-A | 272191 | 87935 | 1014814 | 5.39 | 78.2 | 62.9 | 0.273 | -12.3% | 0.252 | 1/2 |
| h_v2_b | exon | 10xv2 | GRCh38-2024-A | 310797 | 76221 | 4413066 | 9.38 | 49.9 | 39.5 | 0.533 | -14.6% | 0.464 | 1/2 |
| h_v2_b | intron | 10xv2 | GRCh38-2024-A | 272191 | 76221 | 774980 | 5.23 | 79.5 | 63.7 | 0.255 | -14.1% | 0.232 | 1/2 |
| h_v2_c | exon | 10xv2 | GRCh38-2024-A | 310797 | 24807 | 5775529 | 9.44 | 46.4 | 36.6 | 0.554 | -14.4% | 0.479 | 1/2 |
| h_v2_c | intron | 10xv2 | GRCh38-2024-A | 272191 | 24807 | 489204 | 4.60 | 84.0 | 67.0 | 0.187 | -16.4% | 0.164 | 1/2 |
| h_v3_a | exon | 10xv3 | GRCh38-2024-A | 310797 | 98824 | 3979688 | 8.56 | 56.4 | 45.0 | 0.481 | -12.4% | 0.433 | 1/2 |
| h_v3_a | intron | 10xv3 | GRCh38-2024-A | 272191 | 98824 | 3552487 | 7.64 | 56.7 | 45.5 | 0.484 | -8.5% | 0.453 | 1/2 |
| h_v3_b | exon | 10xv3 | GRCh38-2024-A | 310797 | 77618 | 7798444 | 10.86 | 43.4 | 34.4 | 0.597 | -13.0% | 0.527 | 1/2 |
| h_v3_b | intron | 10xv3 | GRCh38-2024-A | 272191 | 77618 | 2562317 | 6.86 | 61.2 | 49.0 | 0.435 | -10.2% | 0.400 | 1/2 |
| h_v3_c | exon | 10xv3 | GRCh38-2024-A | 310797 | 77373 | 7690782 | 11.75 | 40.2 | 31.8 | 0.626 | -13.8% | 0.547 | 1/2 |
| h_v3_c | intron | 10xv3 | GRCh38-2024-A | 272191 | 77373 | 5350939 | 9.92 | 42.6 | 33.9 | 0.608 | -9.2% | 0.560 | 1/2 |
| h_visium | exon | 10x-visium | GRCh38-2024-A | 310797 | 4991 | 301870 | 4.53 | 92.7 | 74.2 | 0.083 | -21.5% | 0.069 | 1/1 |
| h_visium | intron | 10x-visium | GRCh38-2024-A | 272191 | 4991 | 520401 | 4.30 | 86.7 | 69.4 | 0.149 | -16.2% | 0.128 | 1/1 |
| m_arc | exon | 10x-arc-gex | GRCm39-2024-A | 386028 | 96281 | 2101051 | 8.74 | 67.2 | 53.6 | 0.383 | -23.8% | 0.311 | 1/1 |
| m_arc | intron | 10x-arc-gex | GRCm39-2024-A | 307730 | 96281 | 2819316 | 8.20 | 58.6 | 47.0 | 0.474 | -13.4% | 0.424 | 1/1 |
| m_celseq2 | exon | celseq2 | GRCm39-2024-A | 386028 | 4096 | 629605 | 5.94 | 88.7 | 71.1 | 0.122 | -14.7% | 0.107 | 1/0 |
| m_celseq2 | intron | celseq2 | GRCm39-2024-A | 307730 | 4096 | 250260 | 4.57 | 92.4 | 74.2 | 0.088 | -16.0% | 0.077 | 1/0 |
| m_dropseq | exon | dropseq | GRCm39-2024-A | 386028 | 241982 | 12620291 | 20.85 | 31.9 | 25.4 | 0.730 | -13.7% | 0.643 | 1/0 |
| m_dropseq | intron | dropseq | GRCm39-2024-A | 307730 | 241982 | 1205421 | 7.54 | 73.0 | 59.2 | 0.402 | -19.3% | 0.360 | 1/0 |
| m_mars | exon | marsseq2 | GRCm39-2024-A | 386028 | 9411 | 384352 | 5.69 | 92.9 | 74.5 | 0.081 | -14.0% | 0.072 | 1/0 |
| m_mars | intron | marsseq2 | GRCm39-2024-A | 307730 | 9411 | 14833 | 4.26 | 99.4 | 80.1 | 0.012 | -27.3% | 0.011 | 1/0 |
| m_v2_a | exon | 10xv2 | GRCm39-2024-A | 386028 | 81247 | 3903581 | 9.27 | 62.0 | 49.7 | 0.411 | -17.1% | 0.351 | 1/2 |
| m_v2_a | intron | 10xv2 | GRCm39-2024-A | 307730 | 81247 | 906856 | 5.63 | 83.0 | 67.1 | 0.219 | -14.8% | 0.199 | 1/2 |
| m_v2_b | exon | 10xv2 | GRCm39-2024-A | 386028 | 95392 | 7533447 | 12.40 | 47.2 | 37.7 | 0.556 | -13.9% | 0.486 | 1/2 |
| m_v2_b | intron | 10xv2 | GRCm39-2024-A | 307730 | 95392 | 2097474 | 7.14 | 66.9 | 53.9 | 0.376 | -13.7% | 0.337 | 1/2 |
| m_v3_a | exon | 10xv3 | GRCm39-2024-A | 386028 | 73747 | 4544589 | 9.75 | 58.9 | 47.1 | 0.444 | -12.7% | 0.396 | 1/2 |
| m_v3_a | intron | 10xv3 | GRCm39-2024-A | 307730 | 73747 | 2419391 | 6.94 | 67.2 | 54.2 | 0.373 | -9.1% | 0.347 | 1/2 |
| m_v3_b | exon | 10xv3 | GRCm39-2024-A | 386028 | 79736 | 9784223 | 15.82 | 36.7 | 29.2 | 0.655 | -14.4% | 0.567 | 1/2 |
| m_v3_b | intron | 10xv3 | GRCm39-2024-A | 307730 | 79736 | 2990373 | 8.05 | 58.8 | 47.1 | 0.454 | -12.1% | 0.409 | 1/2 |
| m_v3_c | exon | 10xv3 | GRCm39-2024-A | 386028 | 78712 | 9241054 | 14.79 | 39.2 | 31.2 | 0.631 | -14.6% | 0.546 | 1/2 |
| m_v3_c | intron | 10xv3 | GRCm39-2024-A | 307730 | 78712 | 2287781 | 7.64 | 61.8 | 49.6 | 0.422 | -13.3% | 0.378 | 1/2 |
| m_visium | exon | 10x-visium | GRCm39-2024-A | 386028 | 4991 | 3610087 | 8.13 | 65.2 | 52.2 | 0.362 | -11.7% | 0.321 | 1/1 |
| m_visium | intron | 10x-visium | GRCm39-2024-A | 307730 | 4991 | 355403 | 4.70 | 90.4 | 72.6 | 0.111 | -18.2% | 0.094 | 1/1 |
| mq_v2 | exon | 10xv2 | Mmul_10 | 244671 | 74351 | 6536918 | 9.53 | 38.6 | 30.5 | 0.644 | -13.6% | 0.566 | 1/2 |
| mq_v2 | intron | 10xv2 | Mmul_10 | 209239 | 74351 | 1776687 | 5.36 | 59.8 | 47.6 | 0.451 | -15.9% | 0.394 | 1/2 |
| mq_v3 | exon | 10xv3 | Mmul_10 | 244671 | 81323 | 8455875 | 12.16 | 31.0 | 24.4 | 0.718 | -13.3% | 0.631 | 1/2 |
| mq_v3 | intron | 10xv3 | Mmul_10 | 209239 | 81323 | 3471705 | 7.68 | 43.0 | 34.0 | 0.613 | -12.8% | 0.547 | 1/2 |
| r_v2 | exon | 10xv2 | mRatBN7.2 | 243603 | 93227 | 6707987 | 9.53 | 39.3 | 31.2 | 0.641 | -14.2% | 0.559 | 1/2 |
| r_v2 | intron | 10xv2 | mRatBN7.2 | 213041 | 93227 | 2531804 | 6.06 | 55.5 | 44.5 | 0.499 | -10.9% | 0.455 | 1/2 |
| r_v3 | exon | 10xv3 | mRatBN7.2 | 243603 | 74304 | 2096414 | 6.40 | 57.2 | 45.4 | 0.470 | -22.7% | 0.380 | 1/2 |
| r_v3 | intron | 10xv3 | mRatBN7.2 | 213041 | 74304 | 334866 | 3.83 | 85.4 | 68.4 | 0.210 | -22.8% | 0.185 | 1/2 |
| z_v2 | exon | 10xv2 | GRCz11 | 272316 | 87347 | 8765227 | 12.44 | 33.4 | 26.3 | 0.691 | -17.5% | 0.579 | 1/2 |
| z_v2 | intron | 10xv2 | GRCz11 | 239796 | 87347 | 957140 | 5.06 | 74.1 | 59.3 | 0.316 | -20.5% | 0.270 | 1/2 |
| z_v3_a | exon | 10xv3 | GRCz11 | 272316 | 98959 | 3329249 | 8.24 | 51.5 | 40.8 | 0.530 | -17.9% | 0.450 | 1/2 |
| z_v3_a | intron | 10xv3 | GRCz11 | 239796 | 98959 | 2330634 | 6.99 | 55.3 | 44.1 | 0.500 | -13.9% | 0.446 | 1/2 |
| z_v3_b | exon | 10xv3 | GRCz11 | 272316 | 88399 | 8087519 | 13.19 | 31.9 | 25.1 | 0.709 | -18.5% | 0.587 | 1/2 |
| z_v3_b | intron | 10xv3 | GRCz11 | 239796 | 88399 | 2438187 | 7.01 | 54.3 | 43.2 | 0.505 | -14.1% | 0.448 | 1/2 |
