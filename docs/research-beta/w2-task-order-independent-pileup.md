# W2 engine task: make the pileup's output independent of BAM record order

Owner: W2 (`include/singlet/pileup/pileup_engine.h` and export). Requested by W3b (BAM-stream study, `ws-w3b-bam-stream.md`, updates (b)-(d) of 2026-10-28). Base: `prod/campaign` at `c4a1be38` or later; the streaming flag itself is in draft PR #63 (`--stream-unsorted`, `PileupConfig::cb_from_cr`) and is not a prerequisite for any item here.

## Why

The streaming prototype saves ~19% of wall time but is not byte-reproducible, and the sorted path is reproducible only because STAR's coordinate sort fixes the order in which a handful of first-seen rules see their evidence. Measured on GSM4006843 (67 M reads, 3' v2), five streaming runs at 4/8/16 threads and three sorted runs: every count tally (`exon_hits`, `intron_hits`, `sj_hits`, `snp_hits`, `secondary_reads`, `multigene_reads`, `wrong_strand`) and the exon/intron/gene matrices are already identical in all runs. Six rules are not order-free. Closing them makes the serial `run()`, the parallel `run_parallel()` and the stream agree byte for byte, which is criterion (b) of the BAM-stream protocol, and it removes a class of silent irreproducibility from the sorted path too (junction order and rRNA sampling already differ between sorted runs at different thread counts).

## The six rules and their canonical replacements

Line numbers are for `pileup_engine.h` at `db696122` (PR #63 head); they are close in `c4a1be38`.

### 1. Junction feature numbering (bytes only)

- Now: `extract_splice_junctions` (~2403) and `extract_splice_junctions_parallel` (~2827) assign `sj_idx = sj_names_.size()` on first sight, so `sj_counts.1pz`, `sj_features.tsv.gz`, `splice_events.tsv`, `splice_psi.1pz` differ in row order between any two runs whose discovery order differs (stream timing; parallel worker count). Contents verified identical: 170,281 junctions, 0 rows differing in `n_molecules`/`n_cells`.
- Replace: after the pass (and after `merge_splice_junctions` in the parallel path), build a permutation sorting `sj_names_` by (tid, donor, acceptor); remap `sj_acc_` rows and `sj_names_` before export. `build_splice_events` then sees a canonical list. Keep the hash key `(tid<<32|donor) ^ acceptor*K` for lookup; only the emitted index changes.
- Cost: one sort of ~2e5 names and one pass over the SJ COO. Negligible.

### 2. rRNA sampling by position

- Now: `if (b->core.l_qseq >= 21 && stats.mapped_reads % 1000 == 0)` (~604 serial, ~2938 parallel) samples every 1000th mapped read; the slice differs with order (19/23/22/20 hits of ~73.5k across runs).
- Replace: sample when `hash_qname(bam_get_qname(b)) % 1000 == 0` (the engine already has `hash_qname`). Same expected rate, order-free, no periodicity. In the parallel path drop the per-worker counter for the same test. `rrna_reads_sampled` then also becomes exactly reproducible.

### 3. Strand probe from the first 10,000 informative reads

- Now: `probe_strand_file` samples 64 windows over the 4 busiest nuclear chromosomes when an index exists (`kStrandProbeTarget 20000`); a stream falls back to the inline probe (`kStrandInlineProbeReads 10000`, ~862), which decides on whatever reads come first, and reads before the decision keep the starting orientation. Measured: 18,835 informative / 5.6% antisense (indexed) vs 10,000 / 2.2% or 1.9% (inline, T4/T16); same decision here, but a library near the 0.4-0.6 band can decide differently by mode, and the pre-decision reads are counted in the starting orientation only.
- Replace, in order of preference:
  a. Orientation from the protocol table (`known_protocols()`: 10x 3'/5', Drop-seq, inDrops, SMART-seq2 are all known), probe only to confirm, and log a warning on disagreement instead of flipping. The probe then has no effect on counts.
  b. If a data-driven flip must stay: buffer the pre-decision reads' candidate exon/intron assignments for both orientations (bounded by `kStrandInlineProbeReads`) and commit after the decision. Order still picks the sample but no read is counted in the wrong orientation.

### 4. EM-rescue candidate genes = first two seen

- Now: `MultiHitEntry::first_gene` / `second_gene` (~534, ~652-658, ~979-985) are the first two distinct non-intergenic genes encountered across a multimapper's alignments. In a sorted BAM that is the leftmost pair on the genome; in the stream it is STAR's primary-then-secondaries order; across modes the EM rescue (`em_rescue::AmbigRead gene_a/gene_b`) sees different pairs. Suspected source of the 0.002-0.12% molecule gap between sorted and streaming (not yet proven to be the whole gap).
- Replace: keep the candidate set (small fixed array, e.g. up to 4 gene indices, else mark unresolvable) and pass it to EM; or, if EM stays pairwise, choose the two smallest gene indices. Either is a pure function of the alignment set.

### 5. SNP allele = first read of the molecule

- Now: `pileup_aligned_bases` (~2511-2518, parallel ~2905-2911): `umi_new = snp_umi_dedup_.insert(bc, snp, umi)`; on the first read of a (barcode, SNP, UMI) DP += 1 and AD += 1 iff *that read's* base is ALT. Later reads of the molecule are ignored, so when reads of one molecule disagree at the SNP the arrival order decides ALT vs REF. Measured: `snp_dp.1pz` identical in all eight runs, `snp_hits` identical, `snp_ad.1pz` and the donor VCF/coverage differ across thread counts in the stream and between sorted and stream; identical across sorted thread counts only because STAR's sort breaks coordinate ties by read index.
- Replace: per (barcode, SNP, UMI) keep base counts (4 x uint8 or a packed 16-bit pair REF/ALT/other) in a hash map keyed like `UmiDedup`, and emit DP/AD at finalize with majority base; ties go to REF (conservative) and `other` never counts as ALT. This is also more accurate than first-read-wins. Memory: one entry per covered molecule-SNP pair, same order as `snp_umi_dedup_` today (it replaces that set). Apply the same rule to `pileup_mt_bases` if it has a first-read branch (not checked).

### 6. Candidate barcode list from a BAM scan

- Now (sorted path, `singlet.cpp`): after sorting, the CB-tag scan refreshes `auto_barcodes` (barcodes with >= 5 primary CB records, ordered by count, ties by first-seen). The stream has no scan, so its candidate universe is the whitelist-matched set seen during the pass, and EmptyDrops' ambient model differs (alpha 33,506 vs 36,219; 13,098 vs 13,302 cells).
- Replace: count CR per barcode in the existing `.1fq` pass-1 (`[1fq-pass1] Scanned N blocks`; barcodes are whitelist-exact there), apply the same >= 5 rule, order by count with lexicographic ties. Same list in every mode, no BAM pass, and the tie rule is canonical. Falls back to the current scan when input is a BAM rather than a `.1fq`.

## Acceptance test (the proof)

Reruns at different thread counts only sample STAR's timing; the direct test is to feed the *same* alignments in different orders:

1. One STAR run with `--outSAMtype BAM Unsorted` on a test sample (GSM4006843 is fine: 67 M reads, ~10 min at 16 threads), kept in job `/tmp`.
2. Three orderings of that BAM: as written; `samtools collate` (groups by name, shuffles groups); a coordinate sort. All three go through `run()` with the same `--barcodes` list (and once more with auto barcodes for rule 6).
3. Required: `out_equiv.py` reports 0 differences on every output except `read_stats.tsv` wall times and `provenance` timings; in particular `sj_*`, `splice_*`, `rrna_report.json`, `private/snp_ad.1pz`, `private/donor/*`, `auto_barcodes.tsv`, `summary.json` cells.
4. Then the same three orderings through `run_parallel()` on the sorted copy at 4 and 16 workers must equal the `run()` result. Any remaining difference is a region-boundary effect in the parallel path, not an order effect, and is reported separately.
5. Finally sorted (default) vs `--stream-unsorted` on the same `.1fq` must be byte-identical. That closes criterion (b) and lets W7 skip the per-difference acceptance.

Recipe and comparison script: `bench/bam-stream/equiv_modes.sbatch` and `jobs/deep-perf/out_equiv.py` (sha256 per file, JSON key by key, timings ignored). Add `read_stats.tsv:wall_seconds` to its ignore list.

## Out of scope here, noted

- Threaded floating-point drift in the nonhost EM (`nonhost_em_abundance.tsv`, 6th decimal, occasionally a read count) is pre-existing and shared by all paths; fixed partition + ordered reduction if W7 wants it byte-stable.
- The remaining sorted-vs-stream `sj_hits` gap (+0.05%, 1,418 junctions) has no identified cause yet; the acceptance test's step 5 will either close it with rules 1-6 or isolate it as an input difference (CB from CR vs STAR CB).
- A barcode-sharded streaming pileup (`hash(CB) % W` workers, exact per-worker dedup and `mm_buffer`, no merge) is the performance follow-on once the rules above are in: at 16 threads the serial stream pileup (~261k reads/s) already matches STAR's output rate.

## Size

Rules 1, 2, 4 are a few dozen lines each. Rule 5 is a new per-molecule base-count store plus a finalize step (~150 lines) and the main behaviour change (majority instead of first read; expect small AD shifts everywhere, all in the direction of fewer error-driven ALT calls). Rule 3a is a config-table change plus logging. Rule 6 touches `singlet.cpp`'s `.1fq` pass and the barcode refresh. The test is one sbatch (~1 node-hour per sample). Estimate: one week including the acceptance runs.
