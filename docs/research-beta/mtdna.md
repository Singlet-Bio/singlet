# Research beta handoff: mtDNA lineage (2026-10-26)

Goal: design and benchmark how Singlet stores and uses chrM data, so atlas users can reconstruct clonality and lineage from
10x scRNA-seq. Full report: `docs/mtdna-lineage.md` (design + benchmark). Code: `bench/mtdna_lineage/` (README lists the jobs).
Draft PR: Singlet-Bio/singlet#53 (branch `bench/mtdna-lineage`).

## Done
- Benchmark on MAESTER (GSE182685): paired standard 10x 3' + mtDNA-enriched libraries of the same cDNA, full depth (no read
  cap). Truth: K562/BT142 cell-line labels; 23 BPDCN712 clones (1,397 cells, authors' MAESTER calls).
- The production `singlet-modfix` ran on both 3' libraries. STARsolo aligned both MAESTER libraries. Per-cell chrM counts came
  from `mtpile.cpp`, which reproduces Singlet's private counts exactly. mgatk 0.9 / maegatk / MQuad 0.1.8 / vireoSNP 0.5.9 ran on
  the same reads. Haplogrep 3.2.2 ran on pseudobulk calls. An atlas survey covered 747 samples in 400 studies.

## Results (details and tables in docs/mtdna-lineage.md)
1. **Representation.** Private: a molecule-level tensor of depth by strand plus non-reference counts, 13 MB for a 3.5k-cell
   sample, against 200 MB today. Public: a cell x site AD/DP matrix (about 1 MB). Today's public `mt_heteroplasmy.1pz` has no
   depth and no zeros, covers 0.5-3.4% of covered cell-site pairs, and is mostly unsupported by MAESTER.
2. **Whole-genome deconvolution.** It works for donor and haplogroup identity: 99.8% per-cell line accuracy from 3', and
   haplogroups match MAESTER for all three genomes. It does not work for somatic lineage: co-clonal variants sit at different
   VAFs in the same cells (0.17 vs 0.013), and 3' molecules read 90 bp. Site-level counts are the unit.
3. **Clonal inference on 3'.**
   - Known markers: VAF r = 0.98 against MAESTER; 64% of clone cells show their own variant; specificity >= 99.3% at 25 of 26 sites.
   - Discovery from 3' alone: Singlet finds 5 of 26, mgatk's default rules 0 (3' reads are 98.6-99% single-strand per position).
   - Atlas: 49% of samples would reach this benchmark's depth at full SRA depth; 4% do at the 30 M cap used.
4. **Privacy.** Public if AF < 0.9 in every donor and gnomAD AF_hom < 0.001. This leaks no germline variants and keeps 11/13
   and 9/9 somatic truth variants. A sample-level 0.9 cut leaks 29/40 donor variants in a pool.

## Decisions received during the work (fold into the Beta design)
- Zach, via the Research Alpha session: **no mtDNA allele data in Alpha**; data.singlet.bio blocks those members.
- For Beta he wants a lineage-safe release. My recommendation, given the results:
  - Publish somatic sites only (rule R3), as per-cell AD/DP **relative to each donor's own consensus** (alt = non-consensus). The
    consensus and haplogroup are never published.
  - Also publish derived clone labels (vireoSNP or MQuad, versioned, as a cell-label sidecar) and per-sample cell-cell distance
    or tree summaries.
  - Keep the molecule tensor and the homoplasmic calls behind controlled access (restricted sidecar).
  - A residual risk to evaluate before Beta: the somatic positions themselves (recurrent hotspots) and the consensus at covered
    positions can be inferred from AD/DP. Consensus-relative encoding removes the second only if reference-allele counts are
    not published at germline-variant positions.

## Open problems
- `called_v2` overcalls the K562/BT142 mix: 126,598 cells, with 123,744 ambient barcodes at median 312 UMIs. Owner: called_v2.
- The vireoSNP K grid (10-30) was too narrow (K = 30 chosen); rerun with a wider grid, or score per clone.
- MQuad on BPDCN 3' and the BPDCN/MAESTER storage rows are unfinished (doc cells marked `@@`).
- maegatk agreement on K562 MAESTER is pending.
- The pipeline changes in `docs/mtdna-lineage.md` section 7 are not implemented: molecule-level chrM counting, called cells
  only, AD/DP output, rule R3 with a gnomAD AF_hom >= 0.001 list (1,579 SNVs), and strand tests by library type.

## Clipper state (WORK=/mnt/projects/debruinz_project/debruinz/singlet)
- Data: `$WORK/data/maester/{k562_3p,k562_maester,bpdcn_3p,bpdcn_maester,truth}`.
- Singlet runs: `$WORK/out/mtdna_lineage/{k562_3p,bpdcn_3p}`. MAESTER chrM BAMs: `$WORK/out/mtdna_lineage/{k562,bpdcn}_maester/chrM.bam`.
- Piles: `$WORK/out/mtdna_lineage/pile/<name>/mt.{read,umi1,umi3}.bin`, mgatk/maegatk outputs alongside.
- Private haplogroups and VCFs: `$WORK/out/mtdna_lineage/private_vcf/` (do not commit).
- Unsanitized results: `$WORK/jobs/mtl/results/*.json`. Environment: `$WORK/envs/mtl`.
- Jobs still running at handoff, none to resubmit during the freeze:

| Job | What | Writes |
|---|---|---|
| 787777 | `analyze bpdcn _v3`, with MQuad | `results/bpdcn_v3.json` |
| 787968 | `analyze size _all` | `results/size_all.json` |
| 787317 | K562 MAESTER pile + maegatk | `pile/k562_maester/maegatk/final` |

## How to resume (after the freeze)
1. Pull `bpdcn_v3.json` and `size_all.json`. Run `python bench/mtdna_lineage/sanitize.py` on them before committing, then fill the
   `@@` cells in `docs/mtdna-lineage.md`.
2. If 787317 produced maegatk output, rerun `analyze.sbatch lines` for the maegatk agreement row.
3. Take the schema (section 7) and the Beta lineage-safe format to Zach; record the R3 rule in the canonical CELL-ARCHITECTURE.md.
