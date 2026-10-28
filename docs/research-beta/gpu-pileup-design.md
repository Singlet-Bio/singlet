# GPU pileup in `singlet --engine gpu`: design (W11, 2026-10-28)

Branch `gpu/engine-gpu-stream`, stacked on PR #66. It includes `perf/bam-stream` (PR #63) unchanged as the CPU reference.
Spec for anything that combines evidence across reads: `docs/research-beta/order-free-rules.md` (bam-stream session, PR #60).
Gates:
- **(a)** every order-independent output is identical to the CPU engine's;
- **(b)** the outputs the spec defines (R1 SNP consensus, R4 rRNA hash sample, R5 strand, R6 barcode universe, R7 canonical
  order) are identical to the CPU streaming path implementing the same spec;
- every difference from `prod/campaign`'s sorted output is listed by file, size and cause.

## Why the data are per distinct sequence

STAR maps a cDNA read from its sequence alone (the exactness argument of `src/star/SeqCache.h`). So every alignment property is
a function of the read's sequence: chromosome, position, strand, CIGAR, NH, nM, MAPQ, primary flag, gene/exon/intron assignment,
junctions, SNP alleles, chrM bases, poly(A) clip, VDJ gene and rRNA k-mers.

The GPU already holds one mapping result per distinct sequence: 31.9 M for the 66.6 M reads of pbmc1k_v3, and the 0.17% of reads
with N in the R2 sequence, which STAR maps itself. What a read adds is only its barcode, its UMI and its identity:
- the qname hash, for R4;
- the read number, for grouping a read's alignments.

So the pileup becomes a join:

    per distinct sequence s (GPU, once):  alignments(s) -> features(s)
    per read r (GPU, 66.6 M):            (s(r), barcode column b(r), UMI u(r), qname hash h(r))
    outputs = reductions over reads of features(s(r)) keyed by b(r), u(r)

That is the decomposition the research GPU pileup already uses (PR #61 `bench/gpu/pileup/`: "signatures" = distinct
(sequence, alignment set)), measured identical to the CPU engine on 4 samples for its 9 outputs.

## Inputs, and where they come from on the GPU path (no BAM)

| input | source |
|---|---|
| distinct sequence id per read | the GPU dedup in `mapDistinct` (`order`, `head`): already computed, today discarded |
| alignments per distinct sequence | the M5 result blob (STAR's cache encoding: transcripts with exons, strand, nM, scores, nTr, trBest), decoded to (chr, start, strand, primary, MAPQ, CIGAR) as STAR's `alignBAM` does for single-end reads |
| reads STAR maps itself (R2 with N; host-lane failures) | STAR's own per-read pass emits records for these reads only, as an unsorted stream |
| barcode column per read | `.1fq` R1 → CR; CB = CR unless CR has N or UR has N or is a homopolymer (#63 `inline_scan_cb_from_cr`); column = index in the R6 universe |
| UMI per read | `.1fq` R1 (UR) |
| qname hash per read | the read name as STAR writes it, hashed with the engine's `hash_qname` |
| gene model, SNP panel, chrM, VDJ, rRNA k-mers | the engine's own loaded references (the research pileup's `anno.bin` layout: the interval trees as sorted arrays, so the CPU's pre-order is reproduced) |

STAR still runs its per-read pass, but only for its summary outputs: `Log.final.out`, `SJ.out.tab`, `Solo.out/Barcodes.stats`
and the unmapped reads that feed the non-host screen. These are byte-identical by construction. Its BAM output is off except for
the reads that took no GPU result. Removing the rest of that pass (cache lookups, FASTQ parsing) is a later step.

## Outputs

**Covered by the research port (identical on 4 samples):**
- exon, intron and ambiguous matrices (M6 UMI collapse);
- the junction table;
- SNP AD/DP;
- chrM base counts, total and reverse;
- chrM indel events.

**To add** (per-distinct property × per-read key; simple reductions):

| output | per distinct sequence | per read |
|---|---|---|
| `per_cell_reads`, `no_barcode`, totals | primary exists, mapped | b(r) |
| CellAlignQC (multimapped, mismatch rate) | NH, nM, aligned length of the primary | b(r) |
| strand sense / antisense (R5 confirmation) | gene strand vs read strand of the primary | b(r) valid |
| poly(A) records | terminal soft clip, A/T run, end position, gene | b(r), u(r) |
| VDJ gene usage | VDJ gene from the first/last block | b(r), u(r) |
| rRNA sample (R4) | rRNA hit of the stored sequence | h(r) % 64 == 0 |
| WL ambient profile | gene of the primary | CR in whitelist but not in the universe |
| CRISPR guides | guide match | b(r), u(r) (only when a guide reference is given) |

**Spec rules on the GPU:**
- R1: per (b, SNP, u), count REF/ALT/other reads; strict majority decides; ties go to `snp_molecules_ambiguous`.
- R4: hash sample.
- R5: protocol orientation; sense/antisense over all reads; exit code on contradiction.
- R6: the universe from pass 1.
- R7: junction rows sorted by (contig, donor, acceptor); other discovery axes by natural key.

Order-free by construction: every reduction is a sort plus segmented reduce or a commutative count. No result depends on read
order, so thread count and rerun cannot change it.

## Steps, each gated

1. **Hand-off, with STAR's own code for the alignments.** In the STAR child, right after the prefill, run STAR's
   `seqCacheRestore` + `alignBAM` once per distinct sequence: 31.9 M calls instead of 66.6 M per-read calls, on STAR's threads,
   with a placeholder name and quality. Keep each alignment's core fields: flag, chr, pos, MAPQ, CIGAR, NH, nM. They are then
   exactly STAR's by construction, with no re-implementation of `alignBAM`'s flag, MAPQ and CIGAR rules.
   **Original plan for this step:** `SeqCacheGpuPrefill` keeps the per-read distinct id and the per-distinct results (device-resident while the GPU
   is held; on the host, in shared memory for the parent). Per distinct sequence, decode its alignments into the research
   pileup's `AlnRec` layout.
   - Gate: the alignments equal those `extract.cpp` builds from the CPU engine's BAM for the same reads.
2. **Covered outputs** through the research kernels, fed from step 1 instead of the BAM.
   - Gate (a): exon, intron, ambiguous and junction counts and chrM equal to the CPU engine's.
   - Gate (b): SNP under R1 and junction order under R7 equal to the CPU streaming path, once #63 implements them.
3. **The table above.**
   - Gate (a) per output, against the CPU streaming path.
4. **Switch off** STAR's BAM output for GPU-served reads and the CPU pileup on the GPU path.
   - Gates: identity_test, the node runs, the CPU-only build unchanged.
   - Remeasure core-s per M pairs, node rate, memory and cost.

## Projected effect (from measured stages, pbmc1k_v3, core-s per M read pairs)

| | today, `--engine gpu` (sorted BAM) | + #63 stream (no sort, index, CB refresh) | + steps 1–4 |
|---|---|---|---|
| total | 16.1 (measured, job 790078) | about 14 (projected) | about 11 (projected): STAR's BAM records (1.2) and the CPU pileup (1.5) move to the GPU |

Below the 12.7 core-s per M pairs cloud break-even on a 54-core GPU node. The next lever after that is STAR's remaining per-read
pass (cache lookups, FASTQ parsing: about 3).
