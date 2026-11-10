"""Cohort rows that the donor-demux / donor-snps / mtdna benchmarks need and bench/public/cohort.tsv does not have yet.
Paste into make_cohort.py (same `add` helper) after the kang2018 rows and regenerate cohort.tsv. Not applied here: the
cohort belongs to bench-public. `python bench/public/genotype/merged_cohort.py OUT.tsv` writes a cohort with these rows
added, which is what validate was run against.
"""
G = "GRCh38-2024-A"
add(id="kang2018_stim", group="genotype-pool", tier="core", title="Kang 2018 demuxlet, batch 2 IFN-beta lane (the same 8 lupus donors)",
    species="human", reference=G, chemistry="10x-3p-v1", source="geo", gse="GSE96583", gsm="GSM2560249", srr="SRR5398239",
    n_cells=14446, read_pairs_m=172, read_pairs_used_m=172, fastq_gb=22.8,
    truth="demuxlet donor and doublet calls with genotypes (GSE96583_batch2.total.tsne.df.tsv.gz); replicate library of the donors in kang2018_ctrl",
    licence="GEO public data (cite paper)", capabilities="donor-demux,donor-snps,mtdna", tools="singlet,vireo,souporcell,cellsnp-lite,mgatk",
    why="same donors as kang2018_ctrl in a second library: the donors' reads in one lane give independent genotype and mt-haplotype truth for the other")
add(id="kang2018_b1_a", group="genotype-pool", tier="core", title="Kang 2018 demuxlet, batch 1 lane A (4 lupus donors)",
    species="human", reference=G, chemistry="10x-3p-v1", source="geo", gse="GSE96583", gsm="GSM2560245", srr="SRR5398235",
    n_cells=3639, read_pairs_m=187, read_pairs_used_m=187, fastq_gb=21.7,
    truth="demuxlet donor and doublet calls with genotypes (GSE96583_batch1.total.tsne.df.tsv.gz)",
    licence="GEO public data (cite paper)", capabilities="donor-demux,donor-snps,mtdna", tools="singlet,vireo,souporcell,cellsnp-lite,mgatk",
    why="held-out real pool with K = 4 and donors not in batch 2 (the development lane); its donors are also in lane C")
add(id="kang2018_b1_b", group="genotype-pool", tier="core", title="Kang 2018 demuxlet, batch 1 lane B (4 lupus donors)",
    species="human", reference=G, chemistry="10x-3p-v1", source="geo", gse="GSE96583", gsm="GSM2560246", srr="SRR5398236",
    n_cells=4246, read_pairs_m=165, read_pairs_used_m=165, fastq_gb=19.6,
    truth="demuxlet donor and doublet calls with genotypes (GSE96583_batch1.total.tsne.df.tsv.gz)",
    licence="GEO public data (cite paper)", capabilities="donor-demux,donor-snps,mtdna", tools="singlet,vireo,souporcell,cellsnp-lite,mgatk",
    why="held-out real pool with K = 4; the other four donors of lane C")
add(id="kang2018_b1_c", group="genotype-pool", tier="core", title="Kang 2018 demuxlet, batch 1 lane C (8 lupus donors = lanes A + B)",
    species="human", reference=G, chemistry="10x-3p-v1", source="geo", gse="GSE96583", gsm="GSM2560247", srr="SRR5398237",
    n_cells=6145, read_pairs_m=175, read_pairs_used_m=175, fastq_gb=20.8,
    truth="demuxlet donor and doublet calls with genotypes (GSE96583_batch1.total.tsne.df.tsv.gz); replicate libraries of its donors: lanes A and B",
    licence="GEO public data (cite paper)", capabilities="donor-demux,donor-snps,mtdna", tools="singlet,vireo,souporcell,cellsnp-lite,mgatk",
    why="held-out real pool with K = 8 whose donors are in lanes A and B, which gives independent genotype truth")
add(id="maester_k562_bt142_3p", group="genotype-pool", tier="core", title="MAESTER cell-line mixture K562 + BT142, standard 10x 3' library (SRR15598773)",
    species="human", reference=G, chemistry="10x-3p-v3", source="geo", gse="GSE182685", srr="SRR15598773", n_cells=2401,
    truth="the authors' cell-line labels (TenX_CellLineMix_Cells.rds: expression and mt-variant labels) and per-line variant allele fractions from the MAESTER library",
    licence="GEO public data (cite paper)", capabilities="donor-demux,mtdna", tools="singlet,vireo,souporcell,cellsnp-lite,mgatk",
    why="a real K = 2 mixture of two cell lines with high-coverage mt truth (MAESTER) for the same cells")
add(id="maester_bpdcn712_3p", group="genotype-pool", tier="core", title="MAESTER BPDCN712 bone marrow, standard 10x 3' library (SRR15598776)",
    species="human", reference=G, chemistry="10x-3p-v3", source="geo", gse="GSE182685", srr="SRR15598776", n_cells=1397,
    truth="one patient (K = 1); 23 MAESTER clones and 26 clone-defining mt variants with positive cells (authors' calls, 4.2_vois / 4.4_positive_cells)",
    licence="GEO public data (cite paper)", capabilities="donor-demux,mtdna", tools="singlet,vireo,souporcell,cellsnp-lite,mgatk",
    why="real heteroplasmy truth (clonal mt variants validated by the MAESTER library of the same cells) and a real K = 1 pool")
