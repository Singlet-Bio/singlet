# Hashtag demultiplexing (`HtoDemux`)

`include/singlet/pileup/hto_demux.h`: hashedDrops fold-change rule (ambient per tag from the called cells, top-two and second-vs-ambient
log2 folds, MAD outliers) plus a doublet veto from per-tag two-component Gaussian mixtures. Cells only; deterministic; independent of
barcode order (cells are processed in the lexicographic order of their count vectors); `bench/adt/hto_lodo.py` is the numpy reference
and the header matches it call for call.

Selection: `bench/adt/hto_lodo.py` picks the setting (120 fixed ones) by leave-one-dataset-out on Howitt et al. 2023 (genotype truth),
criterion = mean of (F - genetic doublets called singlets) over the training datasets. `bench/adt/results/hto_lodo_selection.json` has
the held-out choices; the defaults of `HtoDemuxParams` are the setting chosen on all four datasets.

Capability `hto` (contract 1.1.0): `bench/adt/results/hto.json`, built by `hto_public_run.py` + `public_hto.py`. Singlet values are the
held-out ones. 1 of 8 gating metrics passes: with hashtag counts alone F-score and doublets-called-singlet trade off, and no setting
of this family (or of leave-one-dataset-out supervised classifiers tried) beats the best competitor on both.
