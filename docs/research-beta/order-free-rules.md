# Order-free pileup rules

Status: spec, 2026-10-28. One shared spec for the CPU engine (`include/singlet/pileup/pileup_engine.h`, export, `src/pipeline/singlet.cpp`; branch `perf/bam-stream`, PR #63) and the W11 GPU pileup (PR #61). Supersedes the "canonical replacement" notes in `w2-task-order-independent-pileup.md` where they differ.

## Principle

Every output is a function of the **set** of alignments, never of the order in which records arrive. Where a choice exists, the rule is the one that is most defensible biologically, not the one that reproduces STAR, STARsolo or the old sorted path. A rule that is order-free but arbitrary (first read wins, lowest index wins) is acceptable only for choices that carry no biological meaning (row order of an output table).

Order already does not matter in the stream for anything decided inside one read: STAR emits all alignments of a read together and in a fixed per-read order, independent of threads. What varies across threads and reruns is the order **between** reads. So the rules below are the ones that combine evidence across reads.

Notation: a *molecule* is (cell barcode, UMI, gene) for counting and (cell barcode, UMI) for site-level evidence. `.1fq` input has no base qualities (`qual[0] == 0xff`), so no rule may depend on them.

## R1. SNP allele per molecule: consensus vote

Today: for each (barcode, SNP, UMI) the first read seen sets DP += 1 and AD += 1 iff that read carries ALT (`pileup_aligned_bases`, `snp_umi_dedup_`). When the reads of one molecule disagree at the site (sequencing error, PCR error in a late cycle, or a chimeric/barcode-collision molecule) the arrival order decides.

Rule: collect, for every (barcode, SNP, UMI), the count of reads showing REF, ALT and other bases at the site. At finalize, per molecule:

- strict majority REF: DP += 1;
- strict majority ALT: DP += 1, AD += 1;
- majority "other" (neither panel allele): no count (not evidence for either allele);
- tie between the two most frequent of REF/ALT/other: the molecule is **ambiguous** and is excluded from both AD and DP; count it in `pileup_stats.json:snp_molecules_ambiguous` (and `snp_molecules_total`).
- reads without a UMI: each read is its own molecule (one read, never a tie).

Why: a molecule is one template; its reads are PCR copies of one allele, so the consensus is the maximum-likelihood allele under any per-read error rate below 0.5, and it is the standard for UMI data (UMI-tools/fgbio consensus, cellsnp-lite and Vireo's UMI mode count one allele per UMI). A tie carries no information about the template's allele; counting it either way, or as both, biases AD/DP toward whichever allele the tie-breaker favours, which distorts allelic ratios at heterozygous sites (ASE, donor genotyping). Excluding it costs only depth, and the counter makes the loss visible.

Key: exact, never a hash: (barcode index, SNP index, UMI), the UMI packed 3 bits per base behind a length bit (UMIs over 21 bases as strings). A hashed key could merge two molecules on a collision (raised by W11; the GPU uses the exact key, CPU since `perf/bam-stream` after `7aa3339`).

Memory: one entry per covered (barcode, SNP, UMI) with three uint8 saturating counters, replacing the `snp_umi_dedup_` set (same key count). Order-free by construction (counts are commutative).

## R2. mtDNA base counts: unchanged for now, consensus as a follow-on

Today `pileup_mt_bases` counts every read base (no UMI collapse). That is order-free already. Molecule consensus (as R1, per chrM position) is the more correct estimator of heteroplasmy (it removes PCR-duplicate inflation; maegatk does this for MAESTER), but per-position-per-molecule state is ~90 entries per chrM molecule, of order 1 GB for a 2 M-chrM-read sample. Spec: implement after measuring memory against the 32 GB per-job budget; until then, per-read counts, documented as such in the mt outputs' provenance.

## R3. Ambiguous multimappers

Today: a multimapper whose alignments touch more than one gene is dropped from the count matrices (`resolve_mm` returns on `gene_ambiguous`). The EM rescue collects `AmbigRead{gene_a, gene_b}` with the first two genes seen, but `run_em_rescue` is never called: no output uses it, so the "first two genes" rule has no effect today.

Rule: keep the count matrices on uniquely assigned molecules (current behaviour, order-free). If EM rescue is enabled later, it must take each read's **full candidate gene set** (all genes of all its alignments, sorted, deduplicated; reads with more than 16 candidates are dropped as uninformative), UMI-collapse per (barcode, UMI, candidate set), run the EM over equivalence classes with a fixed iteration order (classes sorted by their gene vector, genes in index order, fixed iteration count or convergence test on sorted data), and distribute each cell's class count with largest-remainder rounding (remainders tied by smaller gene index), which preserves the integer total.

Why: dropping is conservative and unbiased for genes without paralogs; it is what STARsolo `--soloMultiMappers Unique` does. A two-gene truncation of the candidate set is never correct (it hands a 3-way ambiguous molecule to an arbitrary pair). The full set with EM is the alevin-fry/STARsolo `EM` model. Rounding per gene with a clamp (current `run_em_rescue`) can lose or add counts; largest remainder cannot.

## R4. rRNA fraction: deterministic hash sample

Today: every 1000th mapped read (`stats.mapped_reads % 1000 == 0`), so the sampled set depends on order; 16-23 hits of ~73,500 on GSM4006843.

Rule: a read is sampled iff `hash64(qname) % 64 == 0` (the engine's `hash_qname`), counted once per read (primary or first alignment only). The detector itself is unchanged.

Why: the rate is an estimate of a fraction; the right sample is a uniform one fixed by the read, not by its position in a stream. Counting all reads costs ~70 k-mer lookups per read on the serial pileup thread (several percent of pileup time with the current string-based detector), while 1/64 gives ~1.1 M reads per 70 M (standard error ~5% of a 0.03% fraction, versus ~23% today) at ~1/64 of that cost. If the detector is rewritten with packed k-mers, counting all reads becomes the rule.

## R5. Strand orientation: protocol prior, confirmed by all informative reads

Today: indexed BAM: 64 windows on 4 chromosomes; stream: the first 10,000 informative reads decide, and reads before the decision keep the starting orientation.

Rule:
1. The orientation comes from the protocol table (`known_protocols()` and the `.1fq` header, which already set `reverse_strand` per chemistry, e.g. 3' sense vs 5' antisense on the aligned cDNA read). Counting starts in that orientation and never changes mid-pass.
2. The pass accumulates sense/antisense votes over **all** informative nuclear reads (already counted: `strand_sense`, `strand_antisense`).
3. At the end: if the aggregate antisense share is above 0.6 (the existing `kStrandProbeReverse`), the protocol call was wrong for this library. The stream cannot be re-read, so the run exits with a dedicated code (`strand_contradiction`, recorded in `summary.json`) and the driver reruns the sample once with the orientation flipped (`--reverse-strand`) and records why. Between 0.4 and 0.6 the library is unstranded or mixed: keep the protocol orientation and flag `strand_unclear` in QC.

Why: the protocol is known and is right for essentially every library in the atlas, so the prior is strong; a decision taken from the first 10,000 reads of a stream is a small, position-biased sample that only adds variance, and reads counted before a flip are counted in a wrong orientation. A full-data confirmation is free (the counters exist), and a rare, explicit rerun is cheaper than buffering every read's two candidate assignments for the whole pass.

## R6. Candidate barcode universe

Today (sorted path): pass-1 list of barcodes with >= 100 raw reads, replaced after alignment by the CB-tag scan (>= 5 primary CB records, count order, ties by hash-map insertion order). Stream (b237f57): pass-1 list with >= 5 raw reads, ties by barcode.

Rule: the universe is every barcode with >= 5 reads in the complete pass-1 count of the `.1fq` (barcodes there are whitelist-exact), ordered by read count descending, ties by barcode string ascending. This is the barcode axis of every matrix and the input to cell calling and the ambient model.

Why: it is decided before alignment from the full read set, so it is order-free and needs no BAM. A barcode with fewer than 5 reads carries no usable expression and adds nothing to the ambient profile beyond sampling noise, and 5 matches the threshold the sorted refresh used, so the universe is a superset of the old one (raw reads >= mapped CB records). Counting by reads rather than mapped records keeps it independent of the aligner. Cell calling (EmptyDrops-style, `called_v2`) is then a deterministic function of this universe and the counts.

## R7. Canonical output order

- Barcode axis: the R6 order everywhere (matrices, `cell_calls.tsv`, `cell_qc_metrics.tsv`, every per-barcode table).
- Splice junctions: rows sorted by (contig index in the BAM header, donor, acceptor); `sj_names`, `sj_counts`, `sj_features`, `splice_events`, `splice_psi` all built from that order. Today rows are numbered first-seen.
- Feature assignment of a multi-mapped read: the primary alignment's gene and exon; if the primary lies outside every gene, the smallest exon index over its secondary alignments (not the first secondary seen), in the serial pass, the region workers and the cross-worker merge. Molecules already take the smallest exon/intron index over their reads.
- Any other feature axis built from first-seen discovery (poly-A sites, mt indel events, non-host taxa, VDJ genes) is sorted by its natural key (coordinate, or stable reference ID) before export.
- Floating-point reductions (EmptyDrops alpha, ambient model, demux VB, non-host EM) run over inputs in that canonical order, with a fixed partition when threaded, so results are bit-identical across thread counts.

Why: row order carries no biology; a canonical order makes identity checkable with a byte compare and lets downstream code join outputs across samples by position.

## R8. Non-host EM: canonical input and reduction order

Today: `em_deconvolve` sums E-step responsibilities in single precision over the reads in the order STAR wrote its unmapped mate file, which varies with threads; species are indexed first-seen and output ties keep input order. Result: the 6th decimal of abundances, occasionally a read count, and the EM iteration count change between runs and between sorted reruns (update (g) of ws-w3b). Above 20 M unmapped reads the screen kept the first 20 M in file order.

Rule: each read's hits become a vector of (species id, hit rate) sorted by species id; reads are sorted by that content; species are indexed in species-id order; every sum (prior, E-step, M-step, posterior statistics) runs over that order; output ties are broken by species id. Above the cap, reads are kept by read-name hash: the kept set is every read whose hash is below a threshold that halves until the set fits, so it depends only on the set of unmapped reads.

Why: an EM over a fixed set of reads has one answer; summing in a fixed order makes the floating-point result reproducible without changing the model. Sorting by content needs no read names and costs one sort of the non-host reads (a small fraction of a sample). A hash sample above the cap is uniform over reads, where "first 20 M" is biased toward whatever STAR emitted first.

## R9. UMI error correction for junction, V(D)J and SNP evidence

Today: exon and intron molecules are collapsed with a directional 1-mismatch rule (`DirectionalUmiStore`, Smith et al. 2017), but splice junctions, V(D)J gene usage and SNP evidence are deduplicated on the exact UMI. The sorted path read STAR's `UB` tag, which STARsolo writes into a coordinate-sorted BAM with its own 1-mismatch correction; the stream has only the raw `UR`. So the same alignments gave different counts by path: on GSM7901264 (5') the stream counted +3.4% junction molecules, +6.8% V(D)J and +1.1% SNP depth, all at highly expressed features (cell CGATGTAAGGTGCTAG x RPL10 junction chrX:154400626-154400701: 271 raw UMIs, 243 after STAR's correction).

Rule (revised 2026-10-29 after probe 792017, ws-w3b update i): a UMI is corrected where the molecule is defined, within (barcode, gene), using every read of that gene in the cell; a junction, V(D)J or SNP feature then counts the distinct corrected molecules among its reads.
1. During the pass keep, for each (barcode, feature, exact UMI), its read count and the gene its unique reads were assigned to (the gene of the exon/intron counts; the smallest gene index if they differ). The feature is a junction, a V(D)J gene or a panel SNP.
2. At finalize, a UMI with a gene is replaced by its root in that (barcode, gene) group of the gene-count molecule store: the same directional collapse the exon/intron counts use (UMIs sorted by read count descending, then packed UMI ascending; UMI *i* absorbs a later UMI *j* that differs at exactly one base when count_i >= 2 count_j - 1; each connected component is one molecule).
3. UMIs without a gene (multimappers, intergenic or unassigned reads, a UMI the gene store does not hold) are collapsed among themselves within the (barcode, feature) group by the same directional rule.
4. A junction or V(D)J gene emits its number of distinct molecules. For SNPs the REF/ALT/other read counts are summed over each molecule and R1's majority vote runs on it. Reads without a UMI stay one molecule each; UMIs over 21 bases are deduplicated exactly. `UB` is never read: both paths use the raw UMI and the same rule.

Why: a UMI with one sequencing or PCR error is the same template, and counting it again inflates exactly the deepest features. A molecule is a (cell, gene, UMI) entity, and its reads land on many features, so the gene-wide read counts are the evidence that tells an error copy (few reads next to a parent) from a second template (similar numbers). A single junction or SNP site holds only a few of a molecule's reads. In probe 792017 (GSM7901264, chr1 SNP sites and IG/TR loci) the two groupings disagreed on 50 UMI pairs out of about 210,000 molecules. Where per-feature merged and per-gene did not, the two UMIs had similar gene-wide support (median ratio 0.82, 96% >= 0.5), which points to two templates. Where per-gene merged and per-feature did not, the profile was parent and error copy (median ratio 0.2). The counts differ by only 0.02-0.06% (V(D)J 43,616 per feature vs 43,641 per gene; SNP 165,240 vs 165,276). The gap to prod/campaign's `UB`-based counts (44,097; 167,472) is STAR's lighter correction: `UB` differs from the raw UMI in 0.68% of reads. It is not the grouping.

### R10b. The cell clusters of the ambient fit are seeded from the data

Measured with R10 alone (`2a45dd3`, run 792059, stream vs sorted on one binary): the pools agree (profile L1 2.5e-6 on 3', 7e-4 on 5') and so does the mean contamination (0.4045 vs 0.4061; 0.1877 vs 0.1872). Per-cell rho does not: Pearson 0.79 (3') and 0.73 (5'), with mostly different cluster labels. This holds even on 5', where the two paths share 1,461 of 1,462 cells. The fit is deterministic on identical input, but its random numbers are drawn in sequence over the cell list. The PCA start vector of cell *j* is the *j*-th draw, and k-means++ picks centres by a uniform draw over the cumulative D2 of the list. One added cell therefore shifts every later draw, the clusters change, and each cell's rho follows its cluster's contamination prior.

Rule (orchestrator / Zach, 2026-10-29):
1. Every random number the fit uses for a cell comes from that cell's barcode alone: u(cell, purpose) = SplitMix64(packed barcode XOR salt(purpose)). The PCA start row of a cell is u(cell, component) - 1/2.
2. k-means++ seeding without a shared stream. In restart *r*, the first centre is the cell with the smallest u(cell, r, 0). Each further centre *k* is the cell with the smallest -ln u(cell, r, k) / D2(cell), an exponential race that selects cells with probability proportional to D2, as k-means++ requires. A cell's chance depends only on its own barcode and its distance, so adding or dropping one cell changes the seeds only if that cell would have won.
3. 10 restarts (Lloyd to convergence, at most 30 iterations each). The restart with the lowest within-cluster sum of squares is kept, ties by restart index.
4. Canonical labels: clusters are numbered in ascending order of their smallest member barcode.

Gate: per-cell rho, stream vs sorted, Pearson >= 0.99 on 3' and 5'; also reported, the same path against a run on 99% of the reads (a 1% read subsample removed). If restarts cannot reach the gate, the next step is a more stable clustering (consensus) or a shrinkage that lowers each cell's dependence on its cluster prior, whichever is more defensible. Accepting the instability is not an option.

## R11. Cell barcode correction on the stream: STAR's 1MM_multi_Nbase_pseudocounts

Today: STAR corrects a CB_UMI_Simple barcode against the whitelist only when it builds the sorted BAM. The stream has no CB tag; the pileup took the raw `CR` and kept it only when it was an exact cell-list entry with no N. Upstream of both paths, the `.1fq` reader rebuilt every whitelist miss as an all-N barcode, although the writer stores the raw barcode after the sentinel index. So STAR received all-N barcodes for misses, and its 1MM rules rescued nothing on either path: a read one base from a real barcode, or with one N in it, was lost (barcode-layer-prep.md section 10; reader.h decode, both paths).

Rule (orchestrator / Zach, 2026-10-29: those reads are real molecules; 10x's rule; order-free):
1. The `.1fq` reader restores the raw barcode of a whitelist miss.
2. Exact histogram, STAR's `cbReadCountExact`: in pass 1 of the `.1fq`, before STAR starts, count every read (duplicates by their count, unmapped reads included) whose barcode is an exact whitelist entry and whose UMI has no N and is not a homopolymer.
3. Each stream read whose UMI passes that check gets its barcode from `CR` by STAR 2.7.11b's `1MM_multi_Nbase_pseudocounts`, against the same whitelist file:
   - two or more N: rejected;
   - no N: an exact hit is taken; otherwise every single substitution (last base first, base codes XOR 1, 2, 3) that is a whitelist entry is a candidate, with the quality of the changed base;
   - one N: the four bases at the N, in order A C G T; every hit is a candidate (Nbase);
   - no candidate: rejected. One candidate: taken (pseudocount types have no exact-read requirement). Several: posterior with count = exact reads + 1, q = min(quality - 33, 33), pin = (float)(count * 10^(-q/10)), ptot and pmax float, the first candidate wins ties, accepted when ptot > 0 and pmax >= 0.975f * ptot.
   singlet writes the barcode read with quality `F` on every base, so q = 33 for every candidate and the posterior reduces to read counts.
4. The corrected barcode is then looked up in the cell universe (R6). The histogram is complete before the first read is corrected, so the result does not depend on read order. That is the order-free form of STAR's second pass.

Why: these reads come from real cells. A sequencing error in the barcode read, or one uncalled base, does not make a molecule disappear, and 10x's rule (Cell Ranger, STARsolo `1MM_multi_Nbase_pseudocounts`) assigns them where the evidence is strong (posterior >= 0.975). Using STAR's code path and float semantics exactly makes the stream's cell assignment identical to STAR's own CB on the sorted path. Each read is checked against that tag (SINGLET_TEST_CB_CHECK=1).

## R12. Donor genotype call: binomial posterior on aggregated molecule counts
**Rule.** For each (donor, covered site), with `dp` = aggregated molecule depth and `ad` = alt molecules from that donor's assigned cells (the R1/R9 molecule counts, unchanged), set `r = dp - ad`, `e = 0.01`, and the three log-likelihoods `L0 = r*ln(1-e) + ad*ln(e)`, `L1 = dp*ln(0.5)`, `L2 = ad*ln(1-e) + r*ln(e)`. With a flat prior over {0/0, 0/1, 1/1}, GT is the genotype whose posterior is >= 0.99, else `./.`. `dp == 0` -> `./.`. The VB `af_vb` stays in the output for information but no longer decides GT.

**Order-free.** The inputs are integer sums over the donor's cells (commutative), the function is pure and stateless, and ties cannot occur at posterior >= 0.99 > 0.5. The donor assignment of cells is unchanged by R12. Computation in double; the log and exp of three terms. Any threshold flip at exactly 0.99 is the only place implementations can differ, so the GPU reducer should use the same double formula (or compare in log-odds: `Lbest - ln(sum exp(L)) >= ln(0.99)`).

**GPU SNP reducer needs.** Per (donor, site): the two uint32 sums `ad` and `dp` (already in the sparse donor depth tables), then a trivial epilogue kernel that applies the formula above and writes the GT code (0, 1, 2, -1). It does not need qualities, positions or strands. Gate: identical GT codes to the CPU function for every (ad, dp) with dp <= 4096 (exhaustive table), and identical `ad`/`dp` sums to the CPU path.

**Evidence.** See above: false-alt 0.054 / 0.044 / 0.033 / 0.026 / 0.040 -> 0.0058 / 0.0059 / 0.0119 / 0.0034 / 0.0053 on five Kang pools, true alt kept 0.97-0.98 (at 0.9).

**Threshold 0.99 (genotype-sota, 2026-10-31).** At 0.9 Singlet trailed the cellsnp-lite pileup by 0.4 pp concordance at equal DP>=10 sites on four Kang pools and on Osorio 2019 (independent 1000 Genomes truth). Re-genotyping the same aggregated (ad, dp) at 0.99 closed it: concordance 0.9899 vs 0.9901, true-alt kept 0.9880 vs 0.9880, false-alt 0.64% vs 0.61% (all five pools; Osorio 0.9871 vs 0.9873). The counts are not the issue, only the cut. 0.999 reaches 0.9917 at 3.5% fewer sites. Anchor cases: (1,10), (2,10), (9,10), (0,5) are `./.` at 0.99.

Note: the `--bam-file` replay harness (the shuffled-order gate) does not apply R11, so a replay of the tee'd stream BAM has about 2% fewer barcoded reads than the stream run it came from; replay is not a reference for R11.

## R12. UMI collapse of exon / intron molecules: seed-based directional, molecule UMI = best-supported UMI

Status: implemented behind `SINGLET_UMI_COLLAPSE=seed` (off by default), measured, not made the default for the beta (2026-11-01).

Today (`DirectionalUmiStore::collapse_groups`, GPU `m6.cu`): per (barcode, gene) the UMIs are sorted by reads descending then UMI ascending, an edge joins UMIs at Hamming distance 1 when `count_i >= 2 count_j - 1`, and the connected components of those edges are the molecules. The partition is a function of the (UMI, reads) multiset, so it does not depend on read order. Two things in it are not the best choice:
1. A component can chain two well-supported UMIs 2 mismatches apart through a one-read error copy that sits one mismatch from both (A1 100 reads, A2 80 reads, B 1 read: A1->B and A2->B), merging two templates into one molecule. The directional method of Smith et al. 2017 (UMI-tools) does not: a seed claims what it reaches, and seeds do not merge.
2. The molecule's UMI is the union-find root, which depends on the order the edges were visited and is not always the best-supported UMI. It is the key of the cross-gene rule (`kUmiGeneCR`: a UMI in more than one gene of a cell keeps the gene with most reads) and of R9's gene roots, and the GPU has to replay the pairwise union in the same order to reproduce it.

Rule: rank the UMIs of a (barcode, gene) group by reads descending, UMI ascending (a total order). In rank order, a UMI that no earlier seed has claimed becomes a seed and claims every unclaimed UMI reachable from it through directional edges. A molecule is a seed with the UMIs it claimed; its UMI is the seed (the best-supported UMI of the molecule, ties to the smaller UMI), its reads, flags, diag and smallest feature indices are the union over its members. The result depends only on the (UMI, reads) multiset, and it needs no replay: the owner of a UMI is the first seed that reaches it.

Measured (build 09977a3e, CPU engine, prod/campaign + PR #165): pbmc1k_v3 molecules +0.003% (15,335,599 -> 15,336,115 unique UMIs; spliced +67, unspliced +54, ambiguous +395), pbmc1k_cite_v3 +0.002%; gate-2 counts metrics identical to 4 digits on pbmc1k_v3, cite_v3 and neuron1k_v3. The effect is real but far below anything the gates see, so the beta keeps the component rule; adopting R12 later costs one change on the CPU (`umi_directional_seeds`) and one in `m6.cu` (owner by seed instead of the pairwise replay), gated on identical molecules CPU = GPU.

## Gates

1. Byte-identical outputs at 4, 8 and 16 threads and on a rerun, including the non-host tables (`nonhost/*`), except wall-time fields (`read_stats.tsv:wall_seconds`, `provenance` timings) and fields that hold paths or the sizes of those files.
2. A shuffled-order test: the same unsorted STAR BAM fed to the serial pileup in three record orders (as written, `samtools collate`, coordinate-sorted) gives byte-identical outputs.
3. Every difference from `prod/campaign`'s sorted output is listed by file, size and cause (rule R1-R7 or input difference), in `ws-w3b-bam-stream.md`.
4. Real data only: GSM4006843 (3'), GSM7901264 (5'); the Kang pool for donor-SNP truth where it can be staged.

## Status

| rule | CPU engine (`perf/bam-stream`) | GPU (W11) |
|---|---|---|
| R1 SNP consensus | done, `8ab560d`, serial and parallel pileup | to implement |
| R2 mt consensus | follow-on (memory first) | follow-on |
| R3 multimappers | already order-free (unique only); EM collected but unused | match unique-only |
| R4 rRNA hash sample | done, `8ab560d`, both paths | to implement |
| R5 strand | done for the stream, `8ab560d` (no inline probe; aggregate reported; exit 8 on contradiction). The sorted path keeps its genome-sampled probe plus safety net, which re-reads the whole BAM | to implement |
| R6 barcode universe | done for the stream, `b237f57`; the sorted path still refreshes from the CB-tag scan | take the list from the CPU pass-1 |
| R7 canonical order | junctions done, `8ab560d`, both paths; multimapper exon `8699980`; other discovery axes still to check against the gates | to implement (match the multimapper exon rule) |
| thread/rerun gate | met for every output including `nonhost/*` on `f4a1e95` and `253c2c6`, 3' and 5' (4 vs 16 threads, rerun) | — |
| R8 non-host EM | done, `f4a1e95` (canonical input order, hash sample above the cap) | match if the GPU runs the non-host EM |
| R9 UMI correction (SJ, V(D)J, SNP) | per (cell, gene) `13d8faf` (per-feature version `253c2c6`), both paths; gates 1-2 met; vs prod: junctions within 0.03% (5') / 0.001% (3'), V(D)J -2.0% and SNP DP -1.6% on 5' (residual, see ws-w3b update h) | to implement (same rule) |
| R12 UMI collapse (seed-based directional, UMI = best-supported) | `umi_directional_seeds` behind `SINGLET_UMI_COLLAPSE=seed` (off), test `test_umi_seeds`; +0.003% molecules, not default for the beta | to implement with the default |
| R10 ambient pool (5-100 UMIs, sequence ties, cells in sequence order) | done, `2a45dd3`: pool and mean rho path-independent; R10b (data-seeded clusters, 10 restarts) in progress | match if the GPU runs the ambient correction |
| R11 stream CB correction (STAR 1MM_multi_Nbase_pseudocounts; reader restores raw barcodes) | in progress on `perf/bam-stream` | gate the GPU barcode layer against it (mm1 mode) |
| shuffled-order gate | met on `8699980` (791182) and `253c2c6` (791783), GSM4006843: 44 files identical across as-written / collate / coord replays. `f4a1e95` had failed on `exon_counts` (multimapper exon, fixed by R7) | — |
