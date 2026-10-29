## Table B1. GSE145370 bundle, today vs w3_total_v1 (bytes)

| member kind | today | w3_total (measured) | treatment |
|---|---|---|---|
| exon_counts.1pz (14) | 67,060,711 | 6,485,170 | w3_total_v1, bit-exact matrix on all 14 |
| intron_counts.1pz (14) | 58,705,745 | 5,266,761 | w3_total_v1, bit-exact matrix on all 14 |
| sj_counts.1pz (14) | 31,810,097 | 31,810,097 | kept as today: not recoded (E5 not run) |
| vdj_gene_usage.1pz (14) | 4,254,148 | 4,254,148 | kept as today: not recoded |
| mt_heteroplasmy.1pz (14) | 2,114 | 2,114 | kept as today (0x0 stubs) |
| splice_psi.1pz (14) | 24,575,390 | 0 | derived from sj_counts (E4: 14/14 byte-identical); not rebuilt here |
| cell_calls.tsv (14) | 313,060 | 3,180 | called-set stream (round trip verified); the TSV regenerates from it + exon column sums (E4) |
| JSON members | 324,471 | 324,471 | kept as today (E4 lists derivable numeric fields, not removed here) |
| zip framing (central directory + local headers) | 20,614 | 18,514 | measured: 129 entries now, 143 before |
| **total** | **187,066,350** | **48,164,455** | ratio 0.2575 = 3.88x smaller; file `corpus_w3_total/GSE145370.w3.zip` |

Share of today's bundle: exon_counts.1pz 35.8%, intron_counts.1pz 31.4%, sj_counts.1pz 17.0%, splice_psi.1pz 13.1%, vdj_gene_usage.1pz 2.3%, cell_calls.tsv 0.2%

New bundle: exon+intron 11,751,931 (24.4%), sj+vdj+mt unchanged 36,066,359 (74.9%), everything else 346,165

## Table B2. per sample (bytes)

| sample | called | exon today | exon w3 | intron today | intron w3 | calls stream |
|---|---|---|---|---|---|---|
| GSM4317423 | 2,722 | 5,027,865 | 639,858 | 4,307,655 | 449,121 | 233 |
| GSM4317424 | 1,727 | 4,639,118 | 337,236 | 4,018,334 | 226,721 | 183 |
| GSM4317425 | 1,207 | 4,726,504 | 402,983 | 4,130,724 | 313,880 | 151 |
| GSM4317426 | 2,928 | 5,454,394 | 975,281 | 5,069,379 | 1,083,533 | 211 |
| GSM4317427 | 3,576 | 4,952,111 | 566,994 | 4,271,075 | 411,030 | 289 |
| GSM4317428 | 2,473 | 4,736,261 | 437,280 | 3,997,696 | 232,202 | 405 |
| GSM4317429 | 3,820 | 4,704,028 | 392,898 | 4,040,778 | 247,389 | 316 |
| GSM4317430 | 3,732 | 4,568,381 | 296,282 | 3,991,858 | 221,051 | 203 |
| GSM4317431 | 3,371 | 4,504,589 | 245,565 | 3,947,508 | 187,476 | 174 |
| GSM4317432 | 4,289 | 4,983,448 | 615,731 | 4,378,531 | 528,116 | 276 |
| GSM4317433 | 2,028 | 4,913,525 | 567,004 | 4,485,853 | 629,456 | 165 |
| GSM4317434 | 2,106 | 4,621,610 | 343,801 | 4,041,051 | 259,800 | 140 |
| GSM4317435 | 2,977 | 4,648,199 | 359,875 | 4,047,987 | 267,133 | 210 |
| GSM4317436 | 3,658 | 4,580,678 | 304,382 | 3,977,316 | 209,853 | 224 |

## Table B3. Same bundle with Track A's container treatment applied to sj_counts and vdj (PROJECTION, not a measured codec)

sj_counts 31,810,097 -> about 14,347,675 (chunks 13,172,766 + other 48,920 + row counts 555,130 + study junction vocabulary subset coding 570,859); vdj_gene_usage 4,254,148 -> about 1,780,740. New bundle would be about 28,228,625 B (these use Track A's results/track_a_bundle_whatif.md.txt numbers, payload unchanged).
