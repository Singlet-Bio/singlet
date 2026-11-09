# Protocol detection on GSE230239 (HG00362): 10x 3' v3 encoded as split-seq

`singlet-gpu encode --reads R1 R2` auto-detected the LCL library HG00362 (SRR24236333, 10x Chromium 3' v3 per GEO) as split-seq with
confidence 2. It then trimmed R1 from 101 to 66 nt, so no read kept a usable barcode. The engine later ran out of memory at 421 GB on
that input. Probe job 804315 (`jobs/protocol_probe.sbatch`, `protocol_probe.py`) measured what the detector would have seen and reran
the encoder on the first 400k pairs.

## What the reads show

Both reads are 101 nt. R1 has the v3 layout: high-entropy bases at 0-15 (barcode) and 16-27 (UMI), then poly(T) from position 28.
T makes up 75% of position 28 and stays above 40% through position 60. Measured on the first 5000 pairs:

| Candidate | Whitelist entries | Hit rate | UMI entropy | poly(T) after BC+UMI | Score (whitelist branch) |
|---|---|---|---|---|---|
| 10x-3p-v3 | 3,686,400 | 0.955 exact | 1.98 | 0.71 | 0.877, HIGH |
| 10x-3p-v2 | 737,280 | 0.014 (Hamming-1) | 1.98 | 0.65 | 0.207 |
| splitseq (round-3 list) | 96 | 0.046 (Hamming-1) | 2.00 | 0.32 | 0.225 |

With the whitelists loaded, detect_protocol scores v3 first (both candidates are over-sequenced and both have whitelists, so the lower
protocol id goes first). It reaches HIGH and exits early, so split-seq is never scored.

## Why the encoder chose split-seq anyway

`cmd_encode` (src/pipeline/singlet.cpp, the FASTQ entry point) takes whitelist directories only from `--whitelist-dir` or
`<binary dir>/../whitelists`. Unlike `cmd_download`, it does not read `$SINGLET_WHITELIST_DIR`. A binary outside a release layout
therefore runs detection with no whitelists, and every whitelist protocol falls to the scoring branch that has no whitelist. In that
branch read length and UMI entropy decide, and a 101-nt R1 suits split-seq's 66-nt geometry better than 10x's 28. `cmd_download`
carries a comment describing the same failure on the SRA path ("a 151-nt R1 reads as split-seq") and a fix for it; the FASTQ path
never received that fix. The engine prints no warning in this case.

The same binary with `--protocol 10x-3p-v3` trims R1 to 28, turns on poly(A) trimming and reports confidence 5.

## Campaign exposure and proposed fixes

Any FASTQ-encoded sample whose binary lacks `../whitelists` is detected without whitelists. The SRA download path and the pre-gate already
honor `$SINGLET_WHITELIST_DIR` and refuse to call a sample non-10x when the whitelist is missing.

1. Give `cmd_encode` the `$SINGLET_WHITELIST_DIR` fallback that `cmd_download` has.
2. When no whitelist directory resolves, make auto-detection fail loudly, or at least cap it at LOW confidence so the over-sequenced-R1
   trim cannot fire. Today it silently commits to a whitelist protocol scored without its whitelist.
3. Add a launch oracle: encode a small subset of each touched chemistry from FASTQ and from SRA, with and without the environment
   variable, and require the detected tag to equal the GEO chemistry.
