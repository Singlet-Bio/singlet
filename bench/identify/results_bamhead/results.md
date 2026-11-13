# BAM head route on real files (binary built from ea5c5011, job 813904, `--endpoints`, no hint, no genome index)

Files are the originals the archive mirrors; each row read 5 ranged pieces (head 4 MB, three 1 MB interior pieces, last 1 MB), 8.4 MB at most. "Rate" is the exact and one-mismatch hit rate of raw CR against the best 10x list. Seconds are for the endpoint that supplied the record (ENA FTP unless stated); the same BAM on the NCBI S3 copy took 3.8 s (SRR9644308), so the S3 link is the quick one. The final build keeps the quicker of two copies (job 813971, binary from 1bd0e21d, three rows): the head then came from the NCBI S3 copy every time, in 3.6 to 3.9 s, against 92 to 175 s for ENA FTP of the same BAM.

| run | label in panel | tags | CR/UR length | best list, exact / 1-mm | sense (TX/AN) | call | unmapped reads | bytes | seconds |
|---|---|---|---|---|---|---|---|---|---|
| SRS5091285 (SRR9873585) | 10x 3' v1 | CR+UR, CB+UB | 16/10 | v2 0.959 / 0.979 | 0.98 | 3' v2, process | not seen (tail read) | 8.4 M | 171 |
| SRS5091287 (SRR9873595) | 10x 3' v1 | CR+UR | 16/10 | v2 0.932 / 0.952 | 0.99 | 3' v2, process | not seen | 8.4 M | 179 |
| SRS5091328 (SRR9873661) | 10x 3' v1 | CR+UR | 16/10 | v2 0.956 / 0.966 | 0.97 | 3' v2, process | not seen | 8.4 M | 143 |
| SRS5091333 (SRR9873660) | 10x 3' v1 | CR+UR | 16/10 | v2 0.939 / 0.950 | 1.00 | 3' v2, process | not seen | 8.4 M | 179 |
| SRS5091482 (SRR9873514) | 10x 3' v1 | CR+UR | 16/10 | v2 0.957 / 0.976 | 1.00 | 3' v2, process | not seen | 8.4 M | 150 |
| SRR7722018 | held-out no-barcode | CR+UR | 16/10 | v2 0.922 / 0.972 | 0.78 | 3' v2, process | present | 8.4 M | 205 |
| SRR9644308 | held-out no-barcode | CR+UR | 16/10 | v2 0.972 / 0.981 | 1.00 | 3' v2, process | present | 8.4 M | 3.8 (S3) |
| SRR7788880 | held-out no-barcode | CR+UR | 16/10 | v2 0.971 / 0.986 | 1.00 | 3' v2, process | present | 8.4 M | 225 |
| SRR11089986 | held-out no-barcode | CR+UR | 16/10 | v2 0.953 / 0.968 | 0.99 | 3' v2, process | present | 8.4 M | 171 |
| SRR17209008 | held-out no-barcode | CR+UR | 16/12 | visium 0.940 / 0.963 | 0.97 | Visium, process | unknown (tail piece not read) | 5.2 M | 85 |
| Kang A (SRR5398235) | Kang 2018 | CR+UR | 14/10 | v1 0.927 / 0.986 | 1.00 | 3' v1, process | present | 7.3 M | 203 |
| Kang B (SRR5398236) | Kang 2018 | CR+UR | 14/10 | v1 0.919 / 0.983 | 1.00 | 3' v1, process | present | 7.3 M | 146 |
| Kang C (SRR5398237) | Kang 2018 | CR+UR | 14/10 | v1 0.921 / 0.983 | 1.00 | 3' v1, process | present | 8.4 M | 325 |
| Kang 2.1 (SRR5398238) | Kang 2018 | CR+UR | 14/10 | v1 0.844 / 0.959 | 1.00 | 3' v1, process | present | 8.4 M | 193 |
| Kang 2.2 (SRR5398239) | Kang 2018 | CR+UR | 14/10 | v1 0.725 / 0.909 | 1.00 | 3' v1, process | present | 8.4 M | 203 |

Findings

- All 15 files give a barcode source (`bam_tags`) and a 10x call. None carries a Cell Ranger version line. The header of SRS5091285 (the only one inspected in detail) shows STAR 2.5.1b run in a cloud workflow (genome directory `/cromwell_root/genome_dir/star`); its records carry CR, CY, UR, UY, CB, UB, GX, GN, TX, AN and RE tags.
- The five panel rows labeled 10x 3' v1 read as 3' v2 from the BAM: 16-base CR and 10-base UR, 93 to 96 percent of raw CR on the 737K-august-2016 list (5 percent on the v1 list for SRS5091285, the one inspected). The label looks wrong; the original panel label came from a text statement and was never checked against a barcode read. Kang 2018 reads as v1 (14-base CR, v1 list).
- Coordinate sorting matters for one statistic in practice. With the first version, three of the ten rows were called feature-barcode because 56 to 97 percent of read-2 15-mers fell in the top 100 (reads piled on a few loci). The final build does not use the two diversity statistics for a sorted BAM; the rate and tag statistics are unaffected.
- SRR17209008 returned the head and one more piece (5.2 MB; the other requests gave nothing); the unmapped-reads status is therefore `unknown`, as designed.
- Time is dominated by the endpoint: ENA FTP 85 to 325 s for 5 requests, NCBI S3 under 4 s.

Recommendation: try the NCBI S3 link from the SDL locate service first and ENA `submitted_ftp` only when SDL lists no BAM or the S3 request fails. Today `--endpoints` still reads both copies, so the ENA read (90 to 175 s) is paid even when S3 answered; skipping it once S3 succeeds is a two-line change not made here.

Development-panel regression (job 813970, 76 rows, same sources and hints as the frozen run, new binary): 0 rows differ in verdict, protocol, protocol_tag, action or ambiguous_with (`compare_summary.py`). The development panel has no ambiguous verdict and no BAM source, so neither intended change touches it.
