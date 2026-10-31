# Long-read truth for spliced / unspliced / ambiguous (USA), PBMC-E pbmc1

Registered 2026-10-31 BEFORE any Singlet, STARsolo or alevin-fry output of the pbmc1 run was opened. Not to be edited after;
later changes go in a dated "Amendments" section at the end.

Data: PBMC-E pbmc1, dev split only (the whole PBMC-E set is "dev" in ground-truth/split/split.tsv.json). Molecules: tier T1 of
`truth.v2.parquet` (PacBio and ONT agree; `tier == 1`, `has_truth`). Annotation: GRCh38-2024-A `genes/genes.gtf` (the one Singlet uses).
Reads: ERR16108677 + ERR16108679 (pbmc1 lanes), full depth (227.6 M read pairs, no subsampling).

## Gene of a molecule
Mode of the non-empty, non-`__multi__` `true_gene` of the molecule's T1 reads (CB+UB); ties or none: molecule dropped.

## Gene model (per gene_id, version stripped)
- exon union E_g = all exons of all transcripts of the gene.
- annotated introns = gaps between consecutive exons of each transcript.
- intronic bases of g = (union of annotated introns of g) minus (exons of every gene on the same strand, itself included).

## Class of one aligned long read (primary alignments only, any MAPQ, CB/UB tagged, on the gene's chromosome)
Aligned blocks = reference-contiguous segments (M/=/X/D extend, N splits, soft clips excluded); a junction = an N gap.
- n_int = number of aligned reference bases inside intronic bases of g.
- UNSPLICED  (U): n_int >= 10. (An exon-intron boundary crossing counts only if >= 10 nt fall in the intron; shorter overhangs are
  treated as alignment jitter.)
- SPLICED    (S): not U, and at least one junction whose donor and acceptor are each within 2 nt of the donor and acceptor of one
  annotated intron of g.
- AMBIGUOUS  (A): not U and not S (all aligned bases exonic, no annotated junction).
Priority U > S > A (a partially spliced read that keeps an intron is unspliced).

## Class of a molecule
Over every primary alignment of the molecule, both platforms pooled: U if any alignment is U; else S if any is S; else A.
Per-platform classes and the maximum n_int are stored so the 10-nt threshold can be varied in sensitivity analyses (5 and 20 nt;
PacBio-only and ONT-only), but the headline is the rule above.

Two truths come from the same rule:
- TRUTH-LR (headline): the whole long read of the molecule (what the molecule is).
- TRUTH-R2: the same rule applied to the Illumina R2 placement projected from the long read (`block_starts/ends` and `junc_*` of
  `truth.v2.parquet`), molecule class by the same aggregation over the molecule's T1 R2 reads. This is what a perfect short-read
  classifier can know, since R2 covers only ~89 nt at the 3' end.

## Scoring plan (frozen)
Cells: truth cells that each tool reports in its raw output; headline on the intersection over the three tools (also per tool alone).
Genes: genes with >= 1 truth molecule.
1. Per-tool unspliced molecule count per (cell, gene): U_tool. Tools report S, U, A layers (Singlet: S = exon_counts - ambiguous_counts,
   U = intron_counts, A = ambiguous_counts; total = S+U+A).
   Pearson r on log1p(U_tool) vs log1p(U_truth), pooled over all (cell, gene) pairs with U_truth>0 or U_tool>0 in the stratum "truth covers
   >= 90% of the tool's molecules of the pair" (q = N_truth/N_tool >= 0.9), and per-cell median r over cells with >= 20 such pairs.
   The same on all pairs (no q filter) is reported as secondary.
2. Unspliced share U/(S+U+A): truth share vs each tool's share, (a) all molecules of the truth cells, (b) over (cell, gene) pairs with >= 1 truth
   molecule, (c) over the q >= 0.9 pairs, (d) molecule-matched for STARsolo (below).
3. Pairs with exactly one molecule in the truth and exactly one in the tool (N_truth = N_tool = 1): class confusion 3x3 (truth x tool) for
   every tool. For STARsolo also the per-UMI confusion of `velo_umis.tsv.gz` (CB, UMI, gene, class) joined to the truth molecules on (CB, UMI).
4. "Gene has an unspliced molecule in this cell": precision / recall / F1 of (U_tool > 0) against (U_truth > 0) in the q >= 0.9 stratum.
5. Answer to "which share is closest, and is Singlet's lower share signal loss, a classification choice, or correct": decided from 2, 3, 4
   with the TRUTH-LR and TRUTH-R2 side by side; the 10-nt threshold, the exclusion of intron bases covered by overlapping same-strand
   genes, and the +-2 nt junction tolerance are the classification choices whose effect on the truth is reported.
