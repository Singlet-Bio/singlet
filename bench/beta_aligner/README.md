# Beta aligner profile (CellRanger4 clipping + junction guard + mismatch cap)

`known_protocols().cr4` (include/singlet/fq/protocol.h) is true for 10x 3' v2/v3 only. For those runs singlet.cpp adds
`--clipAdapterType CellRanger4 --alignSJoverhangMin 8 --outFilterIntronMotifs RemoveNoncanonicalUnannotated
--outFilterMismatchNoverLmax 0.05` (config K5 of singlet-agents bench/ground-truth) after a probe: the first 200 000
reads are mapped with and without the profile (no BAM, same solo arguments); the profile stays on only if its unique
rate is not more than 2 points below the plain run's. `--no-cr4` opts out, `--cr4-probe-reads N` sets the probe size
(0 = no probe, no profile). The decision is in `<out>/cr4_probe.json` and the run provenance (`aligner_profile`).
Off (with the reason logged) for every other protocol, 5' clipping (`--clip5pNbases`, TSO adapter), complex/concatenated
barcodes, ATAC, inverted reads (cDNA not mate 1), `--star-extra --clipAdapterType`.
The probe works in every read-source mode: chunk (in-process source), fifo (it decodes the first reads of the .1fq itself, leaving the fifos for
the main stream) and file.

- `gate_result_K5w.md`, `gate_manifest_K5w.json`: frozen ground-truth gate verdict (M1-M7 vs `star_prod_campaign@1`, PASS)
  and the re-pinned manifest, from singlet-agents `bench/ground-truth` (PR Singlet-Bio/singlet-agents#4).
- `identity.sbatch`, `compare_bam.py`: production-path identity check (below).
