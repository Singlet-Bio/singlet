# Protocol detection panel

`singlet download` reads the head of a run, calls its chemistry and stops runs it cannot process. This directory holds the libraries and
scripts that test those calls against what is known about each library, and the end-to-end check that a routed call runs.

## What the gate does with a non-10x call

In auto mode (no `--protocol`) the gate used to reject every non-10x call at MEDIUM or higher confidence. It now routes the call to the
spec that was detected when the probe reads carry evidence beyond read length (`route_evidence` in `include/singlet/fq/protocol.h`):

| Evidence | Measured on | Needed |
|---|---|---|
| barcode reuse | the first `bc_len` bases of the barcode segment (`bc_len` of 8 or more) | 0.20 of reads on a barcode that occurs twice or more |
| linker | a fixed linker of 10 nt or more at its offset (+-2) | 0.50 of reads |
| whitelist excess | share of barcode segments in the list beyond what a random sequence would hit | 0.30 |

A spec with neither linker nor list (Drop-seq, Seq-Well) also needs the barcode read to be the spec length, up to 4 nt longer, or to run on
into poly(T) after the UMI. A barcode list that loads and does not hold the barcodes vetoes the call. CEL-Seq2, MARS-seq and STRT-seq have
6 to 7 nt barcodes, no linker and no list, so no read statistic can confirm them: the gate rejects those calls and names the missing evidence.
`--reject-non10x` restores the earlier rule.

A call below MEDIUM, or a 10x call whose barcodes are in the 10x list at under 10%, is retried on the reads as given without whitelists and
on the swapped reads (barcode read stored second). The first MEDIUM or higher non-10x call with evidence replaces it, and the run then goes on
exactly as `--protocol TAG` (with `--swap-reads on` for swapped reads) would run it. A retry is adopted only with evidence, so no library that
passed before is newly rejected.

Short barcode lists catch random sequence at about n(1+3L)/4^L when one mismatch is allowed (microwell-seq: 96 six-mers, 45%). Where that
chance level is 2% or more, the whitelist rate is now the share beyond chance. All 10x lists have chance levels near zero and are untouched.

Libraries of two assays without a spec are named and stopped at the gate (`reason=unsupported_assay_<name>`): 10x Chromium Flex (the probe
backbone `ACGCGGTTAGCACGTA` at R2 offset 50) and Singleron GEXSCOPE (the linkers `ATCCACGTGCTTGAGA` at R1 offset 8 and `TCAGCATGCGGCTACG` at
offset 32). Both were read off public libraries (`read_geometry.py`).

## Assay type of libraries without a barcode

When no read carries a barcode the encoder stops the run and now logs the assay type the reads show:

- short-fragment libraries (ATAC and kin): at least 20% of read pairs overlap by 20 nt or more (R2 reverse-complemented against R1, one
  mismatch). The Tn5 adapter alone was too weak: it read 2.6% on a bulk ATAC library at 2x50 and 13 to 19% at 2x75 and 2x150, against a
  cut of 15% on the wrong strand. Nextera-tagmented RNA libraries carry the same adapter.
- bisulfite: the smallest C or G fraction over both reads below 5%.
- bulk RNA and plate-based full-length RNA (Smart-seq2) are not separable by read geometry; both read as no-barcode RNA.
- a 10x-chemistry library whose R2 is a tag from a small panel (the 100 most common first 15-mers hold at least half the reads) is a
  feature-barcode (ADT, hashtag) library. Gene-expression libraries measured 0.05 to 0.31 and feature libraries 0.91 to 1.0.
- V(D)J libraries cannot be told from 5' gene-expression libraries by reads (0.26 to 0.44 on the same statistic); the call stays 10x 5'.

## Files

| File | Purpose |
|---|---|
| `panel_design.tsv` | libraries whose protocol is known by design: the multi-protocol benchmark of Mereu et al. 2020 (GSE133549), CITE-seq, bulk RNA-seq, bulk ATAC-seq, bisulfite-seq, V(D)J |
| `panel_pick.py` | adds corpus libraries whose protocol GEO states (weaker truth), minus an exclusion list |
| `run_panel.sbatch`, `score_panel.py`, `score.sbatch` | head of each library (HTTP range or local FASTQ) through `singlet download --r1 --r2`; confusion table |
| `read_geometry.py` | read lengths, fixed regions, barcode reuse, tag concentration and adapter rates of a library head |
| `run_route_e2e.sbatch`, `e2e.tsv` | routing exit test: the same reads staged and counted in auto mode and with the protocol forced |
| `run_b20_compare.sbatch`, `compare_b20.py` | calls of two binaries on the same libraries (10x calls must not change) |
| `build.sbatch` | CPU build and the protocol unit tests (`tests/cpp/test_protocol_route.cpp`) |
