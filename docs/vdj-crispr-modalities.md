# V(D)J and CRISPR guide libraries as Singlet modalities

Status 2026-10-26, V(D)J + CRISPR session. The branch is `claude/vdj-crispr-benchmark-721d3a`, stacked on the ADT branch
(PR #29, whose counting core it reuses). Nothing has been written to R2 or D1.

- **Code:** `python/singlet/vdj/`, `python/singlet/crispr/`.
- **Benchmark drivers and Clipper jobs:** `bench/vdj/`, `bench/crispr/`.
- **Results contract:** `bench/public/capabilities/{vdj,crispr_guides}.py`, `bench/public/results/{vdj,crispr-guides}.json`.
- **Tests:** `tests/python/test_vdj_assemble.py`, `test_crispr.py`, `test_vdj_crispr_store.py`.
- **Storage:** `docs/format-v2.md` section 14.

## Results

<!-- VDJ RESULTS -->

### CRISPR guides (`bench/public/results/crispr-guides.json`: 21 of 21 metrics pass)

Every caller below runs on Singlet's guide counts and the same called cells. The published calls are the reference.

| screen | single-call F1: Singlet / CR-style GMM / Poisson-Gaussian | guides with significant knockdown (count): Singlet / best other | usable perturbed cells: Singlet / best other |
|---|---|---|---|
| 10x A549 5K (3' v3.1, 93 guides) | **0.936** / 0.912 / 0.872 | **59** / 58 (Cell Ranger) | **3,153** / 2,709 (Cell Ranger) |
| 10x A375 10k (2 guides) | **0.998** / 0.997 / 0.773 | 1 / 1 | **4,552** / 4,369 (Cell Ranger) |
| Replogle 2022 K562, gem group 41 (CRISPRi, 4,337 + 288 guides) | **0.919** / 0.894 / 0.876 | **204** / 191 | **2,935** / 2,698 |
| Norman 2019 K562, gem group 1, held-out half (CRISPRa) | **0.904** / 0.899 / 0.902 | **90** / 88 | **3,105** / 3,088 |

**Counting matches Cell Ranger.** On A549 Singlet counts 1.012 times Cell Ranger's guide UMIs (A375: 1.008). Cell
Ranger's own per-guide thresholds, applied to Singlet's counts, reproduce 99.4% of its single-guide calls (A375: 99.9%).

**Where Singlet and Cell Ranger disagree, the expression decides.** On A549, Cell Ranger calls 255 cells multiple that
Singlet calls single.
- The guide both callers keep leaves its target at **0.25x** control expression.
- The guides only Cell Ranger adds leave theirs at **1.20x**, i.e. they are ambient molecules.

On A375, a two-guide screen, Singlet calls 9.1% of cells multiple and Cell Ranger 8.9%. These are doublets; this is the
case that shaped the caller's ambient-load rule.

**Runtime.** Counting runs on one core in Python: 14 min for A549's guide library, 1 min for Norman's. Assignment takes
under a second. Cell Ranger was not run, so there is no time for it.


## 1. What was wrong

The library-type screen (`docs/library-type-screen.md`) found 3,907 V(D)J (TCR/BCR enriched) and 616 CRISPR guide
libraries in the atlas, all processed as gene expression.

**V(D)J libraries.** V(D)J amplicons map well to the TR/IG loci, so 3,059 of these libraries pass atlas QC as if they were
GEX samples. They inflate the cell counts. The pipeline's V(D)J code does not read a V(D)J library as one:

- **`vdj_gene_usage.1pz`** (`include/singlet/pileup/vdj_counter.h`) counts GEX reads over TR/IG gene bodies. It gives no
  junctions and no clonotypes.
- **`include/singlet/pileup/vdj_assembly.h`** is an unwired k-mer CDR3 extractor.

**CRISPR guide libraries.** These have no counting path at all.

## 2. V(D)J: method (`singlet.vdj`)

### 2.1 Segment reference (`singlet.vdj.reference`, packaged as `vdj/data/vdj_{human,mouse}.tsv.gz`)

The V, D, J and C segments come from open annotations of the genome the pipeline uses: GRCh38 with GENCODE v44, and
GRCm39 with vM33. No IMGT or vendor reference is needed (IMGT data is non-commercial). Gene names are the GEX names, with
one allele per gene and no germline genotype.

- **V segments:** the spliced leader + V exons from the CDS start. The 5' UTR is cut because its length varies between
  paralogs (IGLV2-8 has 117 nt more than IGLV2-11), and R1 reads at the transcript's 5' end would vote for the longer
  annotation. The anchor is Cys104, the last in-frame TGT/TGC within 45 nt of the 3' end.
- **J segments:** the F/W-G-X-G motif where the J region's 3' end puts it, 25-45 nt before the splice donor. The motif can
  sit in any frame. GENCODE's IGLJ3 exon carries a spurious motif 100 nt upstream.
- **D and C segments:** the exons.

**Gaps in GENCODE, and how they are filled.** GENCODE misses functional genes that Cell Ranger and TRUST4 have from
IMGT. Three sources fill them:
  1. **By name.** A gene named like a V or J segment but annotated otherwise is kept when its own sequence qualifies:
     - for a V, an open frame with at most one stop codon before an in-frame Cys104;
     - for a J, the motif at the J position.

     This restores TRBV11-2 (`lncRNA` in GENCODE), TRBV21-1, TRAV35 (its reference allele has a stop) and TRAJ8.
  2. **GENCODE's alternate-haplotype annotation:** IGHV3-42, IGHV7-40, IGLV1-41.
  3. **NCBI RefSeq** GRCh38.p14 / GRCm39 (public domain, IG/TR `*_gene_segment` features): TRBV4-3, TRBV5-8, TRBV6-3,
     TRBV6-9, TRBV7-8, TRAJ15.

  With these, every V, J and C gene Cell Ranger calls in the benchmark TCR libraries is in the reference. In the BCR
  libraries only IGHV1-8 and IGHV4-30-4 are missing (54 of 2,066 PBMC 10k BCR contigs). Neither GRCh38 annotation has
  them.
- **Orphons** (IGKV1OR2-108 ...) are excluded: they lie outside the loci.
- **Equivalence classes.** V genes identical or one substitution apart up to Cys104 form one class, which is called by its
  first member (`reference.equivalence`): TRBV6-2|TRBV6-3, IGKV1-39|IGKV1D-39, IGHV1-69|IGHV1-69D and 7 more. The
  benchmark counts a class as one gene for every tool.

### 2.2 Reads to contigs (`singlet.vdj.assemble`)

1. **Orientation.** A probe of the first 100,000 pairs classifies R2, its reverse complement and the R1 insert (after
   barcode, UMI and template-switch oligo). On 10x 5' V(D)J, R1 is sense and R2 antisense. R1 inserts are used in
   long-R1 runs (150x150).
2. **Classification** (`_kernels.classify`, numba, parallel).
   - Every mate is matched with exact 15-mers. The best V, J and C segment and diagonal place Cys104 and F/W118 in the
     read.
   - A read on which two *different* genes tie (TRBC1/TRBC2, IGHG1-4) does not vote for either. Members of one
     equivalence class never tie.
   - When the reference lacks the J gene, F/W118 is found in the read: the F/W-G-X-G motif 20-60 nt upstream of the C
     region's start.
3. **Barcodes and UMIs.** These follow Cell Ranger's logic through the ADT counting code: the 1-mismatch barcode
   posterior of 0.975 and `correct_umis`. Two filters then remove molecules that are not what they seem:
   - **Read support.** A molecule needs at least 2% of the library's read-weighted median mates per molecule. A V(D)J
     library reads each molecule hundreds of times, so UMIs with a few reads are index hops, chimeras or uncorrected UMI
     errors.
   - **One chain per molecule.** Each molecule keeps only the majority chain of its mates. Mates of another chain are
     chimeras, and as contigs of their own they carried other cells' junctions (2 reads each in pbmc2 TCR, against
     ~3,800 for real contigs).
4. **Junction assembly** (`_kernels.assemble_group`, per barcode and chain). The graph is built from the 19-mers of the
   reads that reach the junction.
   - **Seeds** start at a codon-checked Cys104.
   - **Terminals** end at a codon-checked F/W118 followed by the Gly of F/W-G-X-G. A poly-T run (`...CTGTTTTTTGG`)
     otherwise passes one base early.
   - **Support is a per-UMI consensus.** At a seed, branch or terminal, each UMI votes for the alternative its reads hold
     most often. Counting the UMIs that merely contain a k-mer does not work at V(D)J depth (~100 reads per UMI): almost
     every UMI then holds some read with any given one-base error.
   - **Exploration.** Branches with >= 2 UMI votes and >= 20% of the best are explored too, which covers dual alpha
     chains and doublets.
   - **Assignment.** Each read goes to the contig sharing most of its k-mers, so an error variant gets only the reads that
     carry its error. A UMI follows the majority of its reads.
5. **Annotation.**
   - **V and C genes:** k-mer votes of all mates of the contig's UMIs, by equivalence class.
   - **J gene:** the read vote, unless another J matches the junction's germline 3' end by more than 2 nt.
   - **C-gene rule:** TRBJ2 implies TRBC2, because V-D2-J2 joining deletes TRBC1.
   - **D gene:** the longest exact match of at least 6 nt.
   - **`cdr3`:** the IMGT junction (Cys104 ... F/W118), as Cell Ranger and TRUST4 report it.
   - **Productivity:** in frame, no stop codon, and the conserved C and F/W.
6. **Contig filter.** A contig is kept when all of these hold:
   - it has >= 1 UMI over the junction;
   - its mates per UMI are >= 5% of the library's read-weighted median, and it has >= 3 mates;
   - it has >= 20% of the UMIs of the barcode's top contig of the same chain;
   - it is not within 2 substitutions of a stronger contig of the same cell and chain (an error variant).

### 2.3 Cells and clonotypes (`singlet.vdj.clonotype`)

**V(D)J cells.** These are called GEX cells with a productive kept contig. Singlet always pairs a V(D)J library with its
GEX sample (`singlet.vdj.library`).

**Chains.** A cell's chains are at most two per chain. A cell with more is flagged `excess_chains`.

**Clonotypes.**
- **T cells:** exact clonotypes, i.e. the same set of (chain, V, J, junction nt).
- **B cells:** enclone-style lineages. Two cells join when they have the same heavy V and J, heavy junctions of the same
  length within 15% Hamming distance, and the same light V/J.

## 3. CRISPR guides: method (`singlet.crispr`)

### 3.1 Counting

- **Protospacer window.** It is cut from R2 by the library's geometry: before the scaffold anchor for direct capture, or
  at a fixed offset. Cell Ranger patterns are read.
- **Geometry from the reads.** When no pattern is given, the geometry is measured from the reads (`detect_geometry`),
  including dual-guide vectors with one scaffold variant per protospacer (Replogle 2022).
- **Counting core.** Everything else is the ADT counting core (Cell Ranger feature-barcoding logic):
  - an exact or unique Hamming-1 guide match;
  - whitelist barcode correction;
  - 3' v3 feature-barcode translation, decided from the reads;
  - `correct_umis`.
- **Guide lengths.** Guides of 19 and 20 nt are matched right-aligned to the anchor.
- **Matrix and ambient profile.** The matrix covers all GEX columns. Molecules on other whitelist barcodes form the
  ambient profile.

### 3.2 Assignment

**Singlet's caller** (`call_ambient`) tests every (guide, cell) against what ambient guide molecules alone would give that
cell.

- **Ambient profile.** The profile `p` (each guide's share of the empty droplets' molecules) is measured on the library
  itself.
- **Ambient load.** A cell's ambient load is the larger of two estimates:
  - the empty droplets' load, the 95% quantile of their guide UMIs;
  - the cell's own estimate: the UMIs of all its guides but the two most abundant (real guides: one, two in a doublet or
    a dual-guide vector), scaled by their ambient share. The tested guide is never in its own background.
- **Call rule.** A guide is called when all three hold:
  - `x >= 3`;
  - `P(Poisson(b_c p_g) >= x) < 0.001 / G`;
  - `x >= 5%` of the cell's guide UMIs.

**Comparators**, run on the same counts and cells:

- **Cell Ranger's per-guide two-component GMM** on log10 UMIs, calling cells with >= 3 UMIs (reimplemented).
- **A per-guide Poisson-Gaussian mixture** fitted by EM, in the style of Replogle 2022 and Norman 2019.
- **Cell Ranger's published calls**, where 10x provides them.

### 3.3 Guide references for the benchmark

| dataset | guide reference |
|---|---|
| 10x A549 5K, A375 10k | Cell Ranger feature reference |
| Replogle 2022 K562 gem group 41 | Table S1 has no sequences, so the 4,337 protospacers were rebuilt from the sgRNA ids, which encode strand and **hg19** coordinate (`bench/crispr/prep_replogle.py`). The convention was measured on the reads: a "-" id is 19 nt at coord - 21, a "+" id the reverse complement at coord + 3. 98% of these protospacers are seen in the reads. Non-targeting guides have no coordinate and are counted as "unidentified" windows. |
| Norman 2019 gem group 1 | GEO gives identities but no guide-barcode sequences. The barcode -> identity table is learnt on a hash-chosen half of the published cells, and every caller is scored on the other half only. |

## 4. QC (pass/fail; a failing library is not attached)

V(D)J (`singlet.vdj.qc.THRESHOLDS`):

| metric | fail | warn |
|---|---|---|
| `frac_on_target` (mates hitting a V/J/C segment) | < 0.20 | < 0.50 |
| `frac_valid_barcode` | < 0.50 | < 0.75 |
| `frac_vdj_cells_called` (V(D)J cells that are called GEX cells: the pairing check) | < 0.30 | < 0.60 |
| `called_vs_uncalled` (contig rate in called vs uncalled GEX columns) | < 3 | < 10 |
| `frac_called_with_contig` | < 0.01 | < 0.05 |
| `pairing_rate` | < 0.20 | < 0.40 |
| `productive_frac` | < 0.50 | < 0.70 |
| `excess_chain_rate` | > 0.20 | > 0.10 |

CRISPR (`singlet.crispr.qc.THRESHOLDS`):

| metric | fail | warn |
|---|---|---|
| `frac_valid_barcode` | < 0.50 | < 0.75 |
| `frac_protospacer` | < 0.20 | < 0.50 |
| `frac_guide` | < 0.15 | < 0.40 |
| `guide_unexplained` (windows that match no guide) | > 0.50 | > 0.20 |
| `frac_usable_in_cells` | < 0.05 | < 0.25 |
| `cell_empty_ratio` | < 3 | < 10 |
| `frac_cells_assigned` | < 0.20 | < 0.50 |
| `multiple_rate` | - | > 0.40 |

## 5. Storage and processing atlas libraries

The layout is in `docs/format-v2.md` section 14 and follows the ADT rules: members of the GEX sample, aligned to its
`cells` table.

- **V(D)J:** `vdj_contigs.parquet` and `vdj_summary.json`.
- **CRISPR:** `crispr_counts.1pz` (pz2) and `crispr_guides.json`.
- **Per-cell labels:** feature sidecars keyed by `col`.

The entry points are `python -m singlet.vdj.library` and `python -m singlet.crispr.library`. Each takes one library plus
its GEX sample's columns and called cells, and writes the members as a sidecar `.s2`, the label sidecar and the QC
verdict.

For the atlas campaign (not run here):

1. Process the 3,907 V(D)J libraries against their GEX siblings (the screen's pairing).
2. Process the 616 guide libraries where a guide reference exists (study feature references, or a published table). A
   library without one cannot be named; this branch reports such libraries but does not count them.
