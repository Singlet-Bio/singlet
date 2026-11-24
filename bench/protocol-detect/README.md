# Protocol detection truth panel and reference detector

Regression fixtures for the engine's protocol detection. Every new protocol rule lands in the engine together with the panel rows
that exercise it.

- `panel.tsv`: 223 SRA runs and BAMs with verified protocols, split into `dev.tsv` (114) and `heldout.tsv` (109). The held-out half
  was registered by sha256 before any tuning (`REGISTRATION.md`, with the scoring rules and the freeze hashes).
- `protocol_detect.py`: the reference detector (Python, numpy). It scans every read slot at every offset and on both strands
  against the 10x lists and the short-word lists of other platforms, adds motif and read-structure evidence, and returns ranked
  candidates with a decision (clear, trial, unsupported, none). `trial_map.py` maps about 1 M reads under close candidates with
  the engine and picks by the usable share. Both run in the Beta campaign today (singlet-agents `hpc/clipper/campaign/`).
- `run_panel.sbatch`, `trial.sbatch`, `score.py`: run the panel on Clipper and score it. `results/`: the development and held-out
  scores of 2026-11-23.

Measured on the held-out half (frozen detector, with trial mapping): family recall 0.881, family precision 0.881, version recall
0.955. Bars: 0.98, 0.98, 0.95.

## Port into the engine

v1.4.5 carries the first rule, the 10x whitelist guard in `detect_protocol` (a non-10x winner never stands against a 10x list with
at least 40 % exact hits, in either read; 1.4.3 read only one read and 1.4.4 did not compile). The rest of the reference detector moves into the engine as `singlet identify --rank` and library code
in `include/singlet/fq/`, in this order, each with its panel rows as tests:

1. Every-slot, every-offset, both-strand list scan with ranked candidates and evidence (replaces the length-first candidate filter).
2. Encoder support for a barcode at an offset or on the reverse strand (three campaign samples wait on it).
3. 3'/5' from the template-switch tail after barcode+UMI and from cDNA read-through, and trial mapping as `singlet trial`.
4. Tag-library, MARS-seq and plate-protocol rules; Flex and CytAssist probe lists (not yet on Clipper); a V(D)J signal from
   constant-region reads after alignment.
5. Post-run diagnosis (`singlet diagnose`) with the four outcomes and the requeue request.
6. One protocol per library: runs of one SRA experiment are detected together and encoded with the same protocol, so a sample's
   runs always pool. Six campaign samples failed with "RunSet: ... differs from the first run in its protocol/R1 length" because
   per-run detection split one library (2026-11-24); a run of another library type filed under the experiment (a screen's barcode
   amplicon) is reported for exclusion instead of being pooled.
7. Visium end to end (D-154): the `10x-visium` route encodes and counts, but without the slide's tissue positions every spot fails
   QC. Fetch the spatial files (tissue_positions, scale factors) from the GEO supplement with the reads, and route Visium from
   detection. GSM8675021 (fetal Visium, two runs, one called 10x v3) waits in quarantine on this item.
