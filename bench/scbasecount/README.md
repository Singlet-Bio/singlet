# Singlet vs scBaseCount

Code for [docs/scbasecount-comparison.md](../../docs/scbasecount-comparison.md). The jobs run on Clipper: the scripts are put into
`$WORK/jobs/sbc` and the data goes to `$WORK/data/scbasecount/`. The figure scripts run anywhere on the small tables pulled back.

1. `fetch_meta.sbatch` downloads three inputs:
   - scBaseCount 2026-01-12 sample metadata for all organisms;
   - singlet.bio `/api/gsm`;
   - Singlet's processing catalog (GSM → SRX/SRR, status, SRA read counts).
2. `coverage.sbatch` joins the two atlases by SRX. For scBaseCount SRX that Singlet's catalog does not know, it asks NCBI
   `esummary` whether they are GEO samples.
3. `sbc_obs_summary.sbatch` gives per-SRX cell counts and median UMIs/genes from scBaseCount's cell table, by HTTP range reads.
4. `compare.sbatch` is an array over `picks.tsv`. For each sample it reads:
   - Singlet's bundle members (with `bench/empty_droplets/fetch.py`);
   - scBaseCount's `GeneFull_Ex50pAS`, `Gene` and `Velocyto` h5ad;
   - the authors' GEO matrix, when deposited.

   It writes `<GSM>.json` metrics and `<GSM>.cells.parquet` per-cell totals.
5. `explore.sbatch` lists the bundle members and matrix shapes for the studies in `verify_empty.txt` (the empty-bundle spot checks).
6. The figure scripts:
   - `figs_coverage.py` → figs 1–5 and `coverage_tables.md`;
   - `figs_samples.py` → figs 6–8 and `samples.tsv`;
   - `figs_atlaswide.py` → fig 9 and `atlaswide.json`.

`empty_bundle_suspects.tsv` lists the atlas studies whose published bundle holds under 150 bytes per catalog cell. Every one
checked has empty count matrices.
