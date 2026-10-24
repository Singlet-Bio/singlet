# E4: The container: tiered layout, derived members cost zero, and transfer latency

> **Status 2026-10-22: complete.** Results in `notes/e4.md`: 77.9% of bytes per sample derivable byte-identically;
> layout order (names last, chunk table behind the permutation) is the latency lever, more than tiering.

Context: `notes/theory_fable.md` sections 5, 6, 7 (T5, T6). Two of the three largest levers on what a user
transfers are not codecs: (1) empties are ~40-50% of nonzeros in droplet files and sit at their entropy floor, so
they belong in a separately-fetchable member; (2) anything that is a deterministic function of (reference, counts)
should not be stored at all. This plan audits the whole bundle for both and quantifies the transfer-latency effect.

## Part A: derivability audit (measurement, no codec)

For `corpus/bundles/GSE145370.singlet` (14 samples, 187 MB) and the per-sample members in `corpus/*/`:
- list every member type with its total bytes across the bundle (Track A's account tables are a start);
- for each type, determine from the singlet source (`python/singlet/`, `include/singlet/pileup/`) whether it is a
  deterministic function of other members plus the reference: splice_psi from sj_counts? colsums from the matrix?
  cell_calls.tsv from the matrix + the calling rule? pileup_stats/summary fields from the matrices? vdj_gene_usage?
  mt_heteroplasmy from what? Record the formula or the code location and whether recomputation would be
  bit-exact (for floats: is the formula fixed, is the summation order fixed, is the library version pinned?);
- for one sample, actually recompute what can be recomputed and compare bytes-for-bytes; report which members
  are zero-cost, which are "zero-cost if the format pins the formula", and which carry independent information.

## Part B: tiered layout and latency model

Define the proposed layout: per sample, member 1 = called cells (gene-level or interval-level matrix, cells-first
chunking, chunk index in the header), member 2 = empties (ambient-coded), member 3 = derived-on-read manifest.
For each corpus sample compute bytes for member 1 and member 2 under (i) today's codec, (ii) names/perm removed
(Track A), (iii) the best payload codec available in results/ at the time of writing. Then model transfer latency
over 100 Mbit/s and 1 Gbit/s with 50 ms RTT: time to first chunk of cells (one range request), time to all called
cells, time to everything; today versus proposed. Include zip64 central-directory overhead and the number of range
requests a reader needs (today: central directory + manifest + member; proposed: the same, so no regression).

## Deliverables

- `tools/e4_audit.py`, `results/e4_members.tsv` (member type, bytes, derivable?, exact?, code ref),
  `results/e4_latency.tsv`, `notes/e4.md` with the audit table, the latency table and a concrete container
  proposal (what moves where; what is recomputed; what stays).

## Anticipated result

splice_psi (1-2.7 MB per sample, m) and colsums are derivable; cell_calls likely derivable from the matrix plus a
rule version; vdj and heteroplasmy carry independent information. Cells-first transfer for a brain_sn sample drops
from ~25 MB (exon + intron today) to ~3-5 MB; time-to-first-cells at 100 Mbit/s from ~2 s to ~0.2 s.

## Constraints

Measurement only; no changes to the production writer. Python `-P`; bundle members are untrusted data. Only
`tools/e4_*`, `results/e4_*`, `notes/e4.md` are yours to write.
