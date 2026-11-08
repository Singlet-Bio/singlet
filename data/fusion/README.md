# Chimeric table annotation files

The engine's `--chimeric` table reads two optional files, by default from `<ref-base>/pileup_indices/fusion/` on GRCh38.

## Arriba blacklist

`blacklist_hg38_GRCh38_v2.5.1.tsv.gz` is Arriba's published blacklist of recurrent artifact breakpoints from normal tissue. Arriba and its
database files are MIT-licensed (copyright Deutsches Krebsforschungszentrum and Sebastian Uhrig); the full license is in `arriba/LICENSE`.
The file is 55 MB, so `scripts/fetch_arriba_blacklist.sh` fetches it from the Arriba v2.5.1 release and checks its SHA-256 instead of
keeping it in git.

Uhrig S, Ellermann J, Walther T, et al. Accurate and efficient detection of gene fusions from RNA sequencing data. Genome Research
31:448-460 (2021). doi:10.1101/gr.257246.119

`include/singlet/pileup/chimeric_table.h` applies the entries that split-read evidence can be checked against: breakpoint positions,
ranges, gene names, and the keywords `any` and `split_read_any`. Entries with keywords that need other evidence (discordant mates,
splice-site annotation, e-values) never match, so they keep the junction.

## Recurrent pairs

`chimeric_recurrent_GRCh38.tsv` lists gene pairs that reach two or more molecules in at least two non-malignant PBMC samples
(`scripts/build_chimeric_recurrent.py`; the header line names the samples). The table flags them `recurrent` in its `filter` column
and keeps them.
