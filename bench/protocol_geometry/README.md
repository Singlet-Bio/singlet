# Protocol geometry checks

`wl_scan.py` measures where a whitelist really sits in a barcode read (exact-hit fraction at every offset) and the
fraction of reads whose segments are all whitelist-exact or within one mismatch for a stated geometry.
Run on Clipper (`debug-cpu`, 200 000 reads, pure Python) over the 1c harness reads in
`$WORK/data/barcode-layer/{splitseq_parse,indrop,bd}`.

## Results (2026-10-29)

| Protocol (sample) | Before | After | Cause / change |
|---|---|---|---|
| SPLiT-seq (GSE281535 / SRR31302021, 200 000 reads of R2) | all 3 segments exact 0.0000 (Rd3 0.868, Rd2 0.0001, Rd1 0.0007); 1MM 0.0001 | exact 0.6600, 1MM 0.7145 (Rd3 0.868, Rd2 0.787, Rd1 0.684) | The ligation linkers are 30 nt, not 15: Rd2 starts at 48 and Rd1 at 78 (STAR's recipe), not 33 and 56 |
| inDrop (SRR15409581, 200 000 reads of R2) | 0.0000 (fixed 10-nt BC1 window against 8-mer-only list) | 0.1985 (fixed window, 10-nt BC1 only); 0.7879 with a W1-anchored BC1 (not implemented, see below) | `indrop_bc1.txt` was the 8-nt prefix of the published table, in table orientation. The reads carry the reverse complement (as the indrops pipeline's `*_revcomp_list`), BC1 is 8-11 nt (96 of each). Replaced with revcomp of the published `gel_barcode1_list.txt`; `indrop_bc2.txt` was already revcomp of `gel_barcode2_list.txt` |
| BD Rhapsody (1.39 M-read sample, 100 000 reads of R1) | n/a, unchanged | exact 0.5169, 1MM 0.5854 | Geometry confirmed: peaks at exactly 0 / 21 / 43 (CLS1 0.72, CLS2 0.61, CLS3 0.57); no better offset. Low rate is the data (one-base shifts hold 2-4 %) |
