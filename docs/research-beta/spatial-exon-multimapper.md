# Spatial / exon counts: multimapper resolution (diagnosis and plan for Thursday)

Rows: visium_mouse_brain_ant `spot_profile_pearson` 0.9806 vs 0.9888, `gene_totals_spearman` 0.9773 vs 0.9843,
`spot_genes_spearman` 0.9999 vs 1 (STARsolo is the reference). Likely also gex-counts neuron1k_v3 `umi_ratio_distance.total`
(0.054 vs 0.031).

## Mechanism
Singlet's exon count loses 90-99% of the reads of highly expressed genes with unannotated retrocopies; STARsolo Gene and
Space Ranger keep them. Over Space Ranger tissue spots (Singlet / Space Ranger / STARsolo): Eef1a1 11,076 / 118,194 / 116,549;
Gapdh 123k / 203k / 203k; Rps10 1.3k / 52.9k / 39.6k; Rpl24 0.8k / 52k / 41k; Bc1 (ENSMUSG00000115783) 17k / 2.79M / 2.78M.
250 genes (Space Ranger >= 20) are under half of Space Ranger.

- Alignments are not the cause: the STAR run embedded in Singlet's run (`star_Solo.out`) matches the standalone STARsolo on
  these genes.
- Cause (read from the code, not yet confirmed by a molecule dump): `include/singlet/pileup/pileup_engine.h`,
  `resolve_mm_parallel` and `note_gene`. For a multimapper, each alignment's gene is `exon_gene` else `intron_gene`
  (end of the primary path; the secondary path uses `determine_exon_gene_parallel`). A read with one exonic alignment in gene A
  and others inside an intron of gene B (retrocopies sit in host introns) gets two distinct genes, so `gene_ambiguous`
  (GeneFull semantics) and goes to EM or is dropped. STARsolo Gene and Space Ranger use exonic overlaps only. The serial path
  has the same logic (second `resolve_mm_parallel`-like block near line 770-830 of the file at 2a3b0d61); the GPU side is
  `k_resolve` in the GPU pileup (locate; `gpu_pileup_bridge.h` groups a multimapper's alignments).
- STARsolo GeneFull on neuron1k gives Eef1a1 13.8k, matching Singlet's total (15.2k): consistent with the mechanism.

## Evidence
- Real matrices (release-gate/runs/atacsp/visium_mouse_brain_ant vs pipeline-benchmark/runs/atacsp/starsolo_visium_mouse_brain_ant
  vs Space Ranger raw h5): replacing only the 250 genes (Singlet < 0.5 x Space Ranger, SR >= 20) in Singlet with STARsolo's rows
  gives `spot_profile_pearson` 0.9890 (STARsolo 0.9888, Singlet now 0.9808). Removing mt/NUMT genes does not close it (0.9843
  vs 0.9935).
- neuron1k_v3 (gex-counts artifact): exonic Singlet vs STARsolo Gene vs Cell Ranger: Eef1a1 14,876 / 81,086 / 83,968; Ftl1 8,901 /
  32,979 / 34,324; Ppia 18,281 / 33,946 / 37,916; Rps10 411 / 6,641; Rpl24 260 / 7,106.
- Scripts: `scripts/gate2` is for ambient; the spatial scripts (sp_abl2.py, sp_genes.py, sp_neuron.py, sp_embed.py) are in
  `$WORK/jobs/gate2/` on Clipper.

## Plan
1. Option (1), shared: the exon matrix decides by exonic evidence only (an alignment that is only intronic in another gene does not
   make the read ambiguous for the exon count); total and intron keep GeneFull behaviour. CPU and GPU `k_resolve` in one PR; tell
   S8 first, run its CPU=GPU gate.
2. Evidence before merging: before/after on the 8 gate datasets (gex-counts and spliced-unspliced rows), the 3 spatial rows, and
   neuron1k_v3 `umi_ratio_distance`. If a currently passing row regresses beyond its CI, fall back to option (2).
3. Option (2), fallback: apply the rule only for protocol `10x-visium` (zero blast radius; fixes the 3 spatial rows, leaves neuron1k).
4. First step Thursday: confirm the mechanism with a molecule dump on a small Visium subset (reads of Eef1a1 with NH > 1), then
   implement.
