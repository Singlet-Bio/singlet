# V(D)J and CRISPR guide libraries as Singlet modalities

Status 2026-10-25, V(D)J + CRISPR session. The branch is `claude/vdj-crispr-benchmark-721d3a`, stacked on the ADT branch (PR #29).

- Code: `python/singlet/vdj/` and `python/singlet/crispr/`.
- Benchmark drivers and Clipper jobs: `bench/vdj/` and `bench/crispr/`.
- Tests: `tests/python/test_vdj_assemble.py`, `test_crispr.py` and `test_vdj_crispr_store.py`.
- Storage: `docs/format-v2.md` section 14.
- Benchmark results contract: `bench/public/results/{vdj,crispr-guides}.json`.

Nothing has been written to R2 or D1.

<!-- RESULTS -->

## 1. What was wrong

The library-type screen (`docs/library-type-screen.md`) found 3,907 V(D)J libraries (TCR/BCR enriched) and 616 CRISPR
guide libraries in the atlas, all processed as gene expression.

**V(D)J libraries.** V(D)J amplicons map well to the TR/IG loci, so 3,059 of these libraries pass atlas QC as if they were
GEX samples. They inflate the cell counts.

The pipeline has two pieces of V(D)J code, and neither reads a V(D)J library as one:

- **`vdj_gene_usage.1pz`** (`include/singlet/pileup/vdj_counter.h`). It counts GEX reads over TR/IG gene bodies, and
  gives no junctions and no clonotypes.
- **`include/singlet/pileup/vdj_assembly.h`**. It is a k-mer CDR3 extractor that is never wired into the pipeline and has
  only a unit test.

**CRISPR guide libraries.** These have no counting path. Their reads do not map to transcripts, so the screen saw them as
failed GEX samples.

## 2. V(D)J: method (`singlet.vdj`)

### 2.1 Segment reference (`singlet.vdj.reference`)

V, D, J and C segments are built from the same genome and GENCODE annotation the pipeline uses (GRCh38 / GENCODE v44,
GRCm39 / vM33), so no IMGT or vendor reference is needed and the gene names are the GEX names.

- **V segments**: the spliced leader + V exons, in the CDS frame. The anchor is Cys104, the last in-frame TGT/TGC
  within 45 nt of the 3' end.
- **J segments**: the frame whose translation holds F/W-G-X-G. The anchor is the F/W118 codon.
- **D and C segments**: the exons.

GENCODE v44 loses some functional genes, which the builder adds back in two ways:

1. **By name.** TRBV11-2 is annotated `lncRNA` and TRBV21-1 a pseudogene. A gene named like a V segment is kept when
   its own sequence is an open V: a frame with no stop codon before an in-frame Cys104, which is IMGT's ORF rule.
2. **From alternate haplotypes.** TRBV4-3, TRBV6-3 and TRBV7-8 exist only on GRCh38 alt loci. Their segments come from
   GENCODE's `chr_patch_hapl_scaff` annotation on the full genome, as in Cell Ranger's `-alts` reference.

### 2.2 Reads to contigs (`singlet.vdj.assemble`)

1. **Orientation.** A probe of the first 100,000 pairs classifies R2, its reverse complement and the R1 insert after the
   barcode, UMI and template-switch oligo. Whatever hits segments is used. On 10x 5' V(D)J, R1 is sense and R2 antisense.
   R1 inserts count only when reads are long, as in 150x150 runs.
2. **Classification** (`_kernels.classify`, numba, parallel). Every mate is matched with exact 15-mers to the segment
   index. The best V, J and C segment and diagonal follow from the hit votes, and they place Cys104 and F/W118 in the
   read. When two genes tie on a read (TRBC1/TRBC2, IGHG1-4, TRBV6-2/6-3), the read gets a tie flag and does not vote
   for either.
3. **Barcodes and UMIs.** These use the ADT counting code, which is Cell Ranger's logic. Barcodes are corrected against
   the whitelist with the 1-mismatch posterior of 0.975, and UMIs within a barcode with `correct_umis`.
4. **Junction assembly** (`_kernels.assemble_group`, per barcode and chain).
   - The k-mer graph uses 19-mers over the reads that reach the junction, and each k-mer is weighted by distinct UMIs.
   - Seeds are the k-mers starting at a codon-checked Cys104. Seeds with under 5% of the top seed's UMIs are dropped as
     error k-mers.
   - Greedy extension runs to a terminal k-mer that ends at a codon-checked F/W118. Branches with >= 2 UMIs and >= 20% of
     the best extension are explored too, which covers dual alpha chains and doublets.
   - Each read goes to the contig sharing most of its k-mers. An error variant therefore gets only the reads that carry
     its error, never those of the true junction.
   - A UMI follows the majority of its reads.
5. **Annotation.**
   - **V, J and C genes** come from the k-mer votes of all mates of the contig's UMIs. TRBJ2 implies TRBC2, because
     V-D2-J2 joining deletes TRBC1.
   - **D gene**: the longest exact match of at least 6 nt in the junction.
   - **`cdr3`**: the IMGT junction (Cys104 ... F/W118), as Cell Ranger and TRUST4 report it.
   - **Productivity**: in frame, no stop codon, and the conserved C and F/W.
   - **UMI count**: UMIs over the junction, plus the chain's other UMIs that have the same V gene. This is close to Cell
     Ranger's `umis`.
6. **Contig filter.** A contig is kept when it has >= 2 UMIs (>= 1 over the junction) and >= 10% of the UMIs of the
   barcode's top contig of the same chain.

### 2.3 Cells and clonotypes (`singlet.vdj.clonotype`)

**Chains.** A cell's chains are its kept productive contigs, at most two per chain. A cell with more is flagged
`excess_chains`.

**Pairing.** A cell is paired when it has TRA+TRB, TRG+TRD, or IGH+IGK/L.

**T-cell clonotypes.** These are exact: the same set of (chain, V, J, junction nt).

**B-cell clonotypes.** These are lineages, as in enclone. Two cells join when they have:
- the same heavy V and J;
- the same heavy junction length, with junctions within a 15% Hamming distance;
- the same light V/J, when both cells have a light chain.

## 3. CRISPR guides: method (`singlet.crispr`)

### 3.1 Counting

- **Protospacer window.** It is cut from R2 by the library geometry: before the scaffold anchor for direct capture, or
  at a fixed offset. Cell Ranger `(BC)CONST` / `^N..(BC)` patterns are read.
- **Geometry from the reads.** When no pattern is known (`detect_geometry`), the geometry is measured from the reads,
  including dual-guide vectors with one scaffold variant per protospacer (Replogle 2022).
- **Barcodes, UMIs and guides.** Everything else is the ADT counting core (Cell Ranger feature-barcoding logic):
  - an exact or unique Hamming-1 guide match;
  - whitelist barcode correction;
  - 3' v3 feature-barcode translation, decided from the reads;
  - `correct_umis`.
- **Guide lengths.** Guides of 19 and 20 nt are matched right-aligned to the anchor.
- **Matrix and ambient profile.** The matrix covers all GEX columns, called or not. Molecules on other whitelist
  barcodes form the ambient profile.

### 3.2 Assignment

**Singlet's caller** (`call_ambient`) tests every (guide, cell) against what ambient guide molecules alone would give that
cell.

- **Ambient profile.** The profile `p` is measured on the library's empty droplets.
- **Ambient load.** A cell's ambient load `b_c` is the UMIs of its non-called guides, scaled by their ambient share. It
  is estimated iteratively.
- **Call rule.** A guide is called when all three hold:
  - `x >= 3`;
  - `P(Poisson(b_c p_g) >= x) < 0.001 / G`;
  - `x >= 5%` of the cell's guide UMIs.
- **Effect.** Abundant guides need more UMIs than rare ones, and a deeply sequenced library does not turn ambient
  molecules into multiplets.

**Comparators**, run on the same matrix and the same cells:

- **Cell Ranger's per-guide two-component GMM** on log10 UMIs, calling cells with >= 3 UMIs.
- **A per-guide Poisson-Gaussian mixture** fitted by EM, in the style of Replogle 2022 and Norman 2019.

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

## 5. Storage

The layout is in `docs/format-v2.md` section 14 and follows the ADT rules: members of the GEX sample, aligned to its
`cells` table.

- **V(D)J:** `vdj_contigs.parquet` and `vdj_summary.json`.
- **CRISPR:** `crispr_counts.1pz` (pz2) and `crispr_guides.json`.
- **Per-cell labels:** clonotypes and guide calls go to feature sidecars keyed by `col`.
- **Unchanged:** the legacy `vdj_gene_usage.1pz`.
