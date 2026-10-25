"""Source of cohort.tsv: edit the rows here, then run  python bench/public/make_cohort.py bench/public/cohort.tsv

Every accession, URL, size and cell count was checked on 2026-10-25 against the 10x dataset pages (HEAD requests),
NCBI GEO and the ENA portal API. Blank n_cells means not stated by the source.
"""
import csv, sys
CF = "https://cf.10xgenomics.com/samples/"
S3 = "https://s3-us-west-2.amazonaws.com/10x.files/samples/"
cols = ["id", "group", "tier", "data_kind", "title", "species", "reference", "chemistry", "source", "gse", "gsm", "srr", "url",
        "n_cells", "read_pairs_m", "read_pairs_used_m", "fastq_gb", "truth", "licence", "capabilities", "tools", "why"]
G = "GRCh38-2024-A"; M = "GRCm39-2024-A"; B = "GRCh38-GRCm39-barnyard"
CORE_GEX = "singlet,starsolo,alevinfry,cellranger"
R = []
def add(**k):
    k.setdefault("data_kind", "real")  # real | derived (real reads recombined or thinned) | simulated (never core)
    R.append({c: str(k.get(c, "")) for c in cols})

# ---- 10x 3' chemistries
add(id="pbmc1k_v2", group="10x-3p", tier="core", title="10x PBMC 1k, 3' v2", species="human", reference=G, chemistry="10x-3p-v2",
    source="10x", url=CF + "cell-exp/3.0.0/pbmc_1k_v2/pbmc_1k_v2_fastqs.tar", n_cells=996, read_pairs_m=77, read_pairs_used_m=77,
    fastq_gb=6.4, truth="10x Cell Ranger matrix; tools compared on identical reads", licence="CC BY 4.0",
    capabilities="gex-counts,spliced-unspliced,cell-calling,ambient-rna,alignment-performance", tools=CORE_GEX,
    why="oldest still-common chemistry (10 bp UMI, 737K whitelist); the PBMC 1k pair is the community's standard toy set")
add(id="pbmc1k_v3", group="10x-3p", tier="core", title="10x PBMC 1k, 3' v3", species="human", reference=G, chemistry="10x-3p-v3",
    source="10x", url=CF + "cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_fastqs.tar", n_cells=1222, read_pairs_m=66.6, read_pairs_used_m=66.6,
    fastq_gb=5.5, truth="10x Cell Ranger matrix; tools compared on identical reads", licence="CC BY 4.0",
    capabilities="gex-counts,spliced-unspliced,junctions-psi,cell-calling,ambient-rna,non-host,alignment-performance", tools=CORE_GEX,
    why="the most used 3' v3 reference dataset (STARsolo, alevin-fry and kallisto papers all report on it)")
add(id="pbmc5k_v4_d1", group="10x-3p", tier="core", title="10x 5k Human PBMCs Donor 1, GEM-X 3' v4", species="human", reference=G,
    chemistry="10x-3p-v4", source="10x",
    url=CF + "cell-exp/9.0.0/5k_Human_Donor1_PBMC_3p_gem-x_Multiplex/5k_Human_Donor1_PBMC_3p_gem-x_Multiplex_fastqs.tar",
    n_cells=5709, read_pairs_m=223, read_pairs_used_m=223, fastq_gb=19.1, truth="10x Cell Ranger 9 matrix; tools on identical reads",
    licence="CC BY 4.0", capabilities="gex-counts,spliced-unspliced,junctions-psi,cell-calling,alignment-performance,chromosomal-sex",
    tools=CORE_GEX, why="current GEM-X 3' v4 chemistry (3.6M whitelist); no plain v4 PBMC set on Cell Ranger 8 exists")
# ---- 10x 5'
add(id="pbmc5p_v1", group="10x-5p", tier="core", title="10x PBMC 5' v1 + TotalSeq-C (vdj_v1_hs_pbmc2_5gex_protein), first 60M read pairs",
    species="human", reference=G, chemistry="10x-5p-v1", source="10x",
    url=CF + "cell-vdj/3.0.0/vdj_v1_hs_pbmc2_5gex_protein/vdj_v1_hs_pbmc2_5gex_protein_fastqs.tar", n_cells=7231,
    read_pairs_m=500, read_pairs_used_m=60, fastq_gb=42.9, truth="10x Cell Ranger matrix; ADT for cell type",
    licence="CC BY 4.0", capabilities="gex-counts,cell-calling,alignment-performance,adt,vdj", tools=CORE_GEX + ",cellranger-vdj,trust4",
    why="5' chemistry exposes strand handling (Singlet's 5' strand bug was found here). Its TCR/BCR libraries are the separate "
        "datasets vdj_v1_hs_pbmc2_t / _b (cell-vdj/3.0.0, 6.4 / 1.8 GB), with Cell Ranger 3.0 filtered_contig_annotations; "
        "V(D)J tuning set (caveat)")
add(id="pbmc10k_5p_v2", group="10x-5p", tier="core", title="10x PBMC 10k 5' v2 with TCR, BCR and antibodies (sc5p_v2_hs_PBMC_10k)",
    species="human", reference=G, chemistry="10x-5p-v2", source="10x",
    url=S3 + "cell-vdj/5.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_fastqs.tar", n_cells=10548, read_pairs_m=760,
    read_pairs_used_m=300, fastq_gb=63.1, truth="Cell Ranger 5 vdj clonotypes and count matrix; separate t/b/f tars exist",
    licence="CC BY 4.0", capabilities="gex-counts,cell-calling,vdj,adt",
    tools=CORE_GEX + ",cellranger-vdj,trust4,mixcr",
    why="the standard 5' v2 immune profiling reference: GEX + TCR + BCR + protein on the same cells")
# ---- nuclei
add(id="nuclei2k_v2", group="nuclei", tier="core", title="10x mouse brain nuclei 2k, 3' v2", species="mouse", reference=M,
    chemistry="10x-3p-v2", source="10x", url=CF + "cell-exp/2.1.0/nuclei_2k/nuclei_2k_fastqs.tar", n_cells=2059,
    read_pairs_m=153, read_pairs_used_m=153, fastq_gb=12.7, truth="10x Cell Ranger matrix", licence="CC BY 4.0",
    capabilities="gex-counts,spliced-unspliced,junctions-psi,cell-calling,ambient-rna", tools=CORE_GEX,
    why="intron-rich single-nucleus data, where exon-only counting loses most molecules")
add(id="mouse_brain_nuclei5k_v31", group="nuclei", tier="core", title="10x 5k adult mouse brain nuclei (CNIK), 3' v3.1",
    species="mouse", reference=M, chemistry="10x-3p-v3", source="10x",
    url=CF + "cell-exp/7.0.0/5k_mouse_brain_CNIK_3pv3/5k_mouse_brain_CNIK_3pv3_fastqs.tar", n_cells=7377, read_pairs_m=205,
    read_pairs_used_m=205, fastq_gb=19.8, truth="10x Cell Ranger 7 matrix (introns included by default)", licence="CC BY 4.0",
    capabilities="gex-counts,spliced-unspliced,cell-calling,ambient-rna,cell-type", tools=CORE_GEX,
    why="modern nuclei kit; Cell Ranger 7 counts introns by default, a like-for-like test of Singlet's exon + intron counts")
# ---- multiome
add(id="pbmc_multiome10k", group="multiome", tier="core", title="10x Multiome PBMC 10k, granulocytes sorted out (ARC 2.0)",
    species="human", reference=G, chemistry="10x-arc", source="10x",
    url=S3 + "cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_fastqs.tar", n_cells=11898,
    read_pairs_m=1100, read_pairs_used_m=600, fastq_gb=95.6, truth="Cell Ranger ARC 2.0 joint calls, fragments and peaks",
    licence="CC BY 4.0", capabilities="gex-counts,atac,multiome,cell-calling,chromosomal-sex",
    tools="singlet,cellranger-arc,archr,signac,starsolo",
    why="the reference Multiome dataset (Signac, ArchR and MultiVI tutorials); nuclei GEX + ATAC on one barcode")
# ---- CITE-seq / hashing
add(id="pbmc1k_cite_v3", group="adt", tier="core", title="10x PBMC 1k + TotalSeq-B (pbmc_1k_protein_v3)", species="human",
    reference=G, chemistry="10x-3p-v3", source="10x", url=CF + "cell-exp/3.0.0/pbmc_1k_protein_v3/pbmc_1k_protein_v3_fastqs.tar",
    n_cells=713, read_pairs_m=56, read_pairs_used_m=56, fastq_gb=4.7, truth="Cell Ranger feature-barcode counts; protein markers for cell type",
    licence="CC BY 4.0", capabilities="adt,gex-counts,cell-calling", tools=CORE_GEX + ",cite-seq-count,kite",
    why="smallest CITE-seq reference; note its TotalSeq-B sequences are reused across product generations")
add(id="pbmc10k_cite_v3", group="adt", tier="core", title="10x PBMC 10k + TotalSeq-B (pbmc_10k_protein_v3)", species="human",
    reference=G, chemistry="10x-3p-v3", source="10x", url=S3 + "cell-exp/3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_fastqs.tar",
    n_cells=7865, read_pairs_m=345, read_pairs_used_m=345, fastq_gb=28.6, truth="Cell Ranger feature-barcode counts",
    licence="CC BY 4.0", capabilities="adt,gex-counts,cell-calling,cell-type", tools=CORE_GEX + ",cite-seq-count,kite",
    why="the CITE-seq dataset Seurat's WNN and totalVI tutorials use")
add(id="stoeckius2018_hashing", group="hto", tier="core", title="Stoeckius 2018 cell hashing, PBMC 8 HTOs (RNA first 200M spots + HTO + ADT)",
    species="human", reference=G, chemistry="10x-3p-v2", source="geo", gse="GSE108313", gsm="GSM2895282;GSM2895283;GSM2895284",
    srr="SRR8281306;SRR8281307;SRR8281308", n_cells="", read_pairs_m=974, read_pairs_used_m=386, fastq_gb=97.7,
    truth="genotype demultiplexing of the same cells (donor = hashtag); authors' HTO calls (Hashtag-HTO-count.csv.gz)",
    licence="GEO public data (cite paper)", capabilities="hto,doublets,donor-demux,cell-calling",
    tools="singlet,htodemux,demuxem,gmm-demux,vireo,souporcell",
    why="the original cell hashing paper's PBMC experiment; hashtag-defined doublets give doublet truth")
add(id="tenx_cellplex_pbmc10k", group="hto", tier="extended", title="10x PBMC 10k multiplexed with 2 CellPlex CMOs, 3' v3.1",
    species="human", reference=G, chemistry="10x-3p-v3", source="10x",
    url=S3 + "cell-exp/6.0.0/SC3_v3_NextGem_DI_CellPlex_Human_PBMC_10K_Multiplex/SC3_v3_NextGem_DI_CellPlex_Human_PBMC_10K_Multiplex_fastqs.tar",
    n_cells=13446, read_pairs_m=600, read_pairs_used_m=600, fastq_gb=50.2, truth="Cell Ranger multi CMO assignments; genotype demultiplexing of the two donors",
    licence="CC BY 4.0", capabilities="hto,donor-demux,doublets", tools="singlet,cellranger,htodemux,demuxem,vireo",
    why="10x's own sample-multiplexing chemistry (CMO), the commercial successor of hashtags")
add(id="howitt2023_bal1", group="hto", tier="core", title="Howitt 2023 HTO benchmark, paediatric BAL batch 1 (8 donors, 8 TotalSeq-A hashtags, 2 captures)",
    species="human", reference=G, chemistry="hto-counts", source="zenodo", other="10.5281/zenodo.8304003 (BAL_data.zip)", n_cells=24091,
    fastq_gb=0.08, truth="vireo genetic donor per droplet, including genetic doublets; every competitor's published calls",
    licence="CC BY 4.0", capabilities="hto,doublets", tools="singlet,htodemux,multiseqdemux,hashsolo,demuxmix,gmm-demux,hasheddrops",
    why="the only public hashing benchmark with genotype truth and every competitor's calls (counts only: no reads)")
add(id="howitt2023_bal2", group="hto", tier="core", title="Howitt 2023 HTO benchmark, paediatric BAL batch 2", species="human", reference=G,
    chemistry="hto-counts", source="zenodo", other="10.5281/zenodo.8304003 (BAL_data.zip)", n_cells=48841, fastq_gb=0.08,
    truth="vireo genetic donor per droplet, including genetic doublets", licence="CC BY 4.0", capabilities="hto,doublets",
    tools="singlet,htodemux,multiseqdemux,hashsolo,demuxmix,gmm-demux,hasheddrops", why="as howitt2023_bal1")
add(id="howitt2023_bal3", group="hto", tier="core", title="Howitt 2023 HTO benchmark, paediatric BAL batch 3", species="human", reference=G,
    chemistry="hto-counts", source="zenodo", other="10.5281/zenodo.8304003 (BAL_data.zip)", n_cells=62306, fastq_gb=0.08,
    truth="vireo genetic donor per droplet, including genetic doublets", licence="CC BY 4.0", capabilities="hto,doublets",
    tools="singlet,htodemux,multiseqdemux,hashsolo,demuxmix,gmm-demux,hasheddrops", why="as howitt2023_bal1")
add(id="howitt2023_cl", group="hto", tier="core", title="Howitt 2023 HTO benchmark, 3 lung cancer cell lines with MULTI-seq lipid tags (3 captures)",
    species="human", reference=G, chemistry="hto-counts", source="zenodo", other="10.5281/zenodo.8304003 (cell_line_data.zip)",
    n_cells=45977, fastq_gb=0.006, truth="vireo genetic identity per droplet, including genetic doublets", licence="CC BY 4.0",
    capabilities="hto,doublets", tools="singlet,htodemux,multiseqdemux,hashsolo,demuxmix,gmm-demux,hasheddrops",
    why="lipid-tag multiplexing with genotype truth")
# ---- genotype pools
for lane, gsm, srr, k, m in (("a", "GSM2560245", "SRR5398235", 4, 69.3), ("b", "GSM2560246", "SRR5398236", 4, 63.5),
                             ("c", "GSM2560247", "SRR5398237", 8, 71.6)):
    add(id=f"kang2018_b1_{lane}", group="genotype-pool", tier="core", title=f"Kang 2018 demuxlet, batch 1 lane {lane.upper()} ({k} donors)",
        species="human", reference=G, chemistry="10x-3p-v1", source="geo", gse="GSE96583", gsm=gsm, srr=srr, fastq_gb=21, read_pairs_m=m, read_pairs_used_m=m,
        truth="demuxlet donor and doublet calls with genotypes (GSE96583_batch1.total.tsne.df.tsv.gz)",
        licence="GEO public data (cite paper)", capabilities="donor-demux,doublets,ambient-rna",
        tools="singlet,vireo,souporcell,scrublet,scdblfinder",
        why=f"held out from the demux/doublet fixes; true K = {k} tests donor-count selection; deposited as a BAM, reads rebuilt from it")
add(id="kang2018_ctrl", group="genotype-pool", tier="core", title="Kang 2018 demuxlet, batch 2 control lane (8 lupus donors)",
    species="human", reference=G, chemistry="10x-3p-v1", source="geo", gse="GSE96583", gsm="GSM2560248", srr="SRR5398238",
    n_cells=14619, read_pairs_m=182, read_pairs_used_m=182, fastq_gb=23.8,
    truth="demuxlet donor and doublet calls with genotypes (GSE96583_batch2.total.tsne.df.tsv.gz); cell types",
    licence="GEO public data (cite paper)", capabilities="donor-demux,doublets,mtdna,donor-snps,chromosomal-sex,genetic-ancestry",
    tools="singlet,vireo,souporcell,cellsnp-lite,scrublet,scdblfinder,mgatk",
    why="the standard genotype-pool benchmark (demuxlet, vireo, souporcell papers); deposited as a BAM, reads rebuilt from it")
# ---- barnyard
add(id="hgmm1k_v3", group="barnyard", tier="core", title="10x HEK293T + NIH3T3 1:1, 1k, 3' v3", species="human+mouse", reference=B,
    chemistry="10x-3p-v3", source="10x", url=CF + "cell-exp/3.0.0/hgmm_1k_v3/hgmm_1k_v3_fastqs.tar", n_cells=1046, read_pairs_m=77,
    read_pairs_used_m=77, fastq_gb=4.9, truth="species of every read: cross-species UMIs measure ambient RNA, mixed cells are doublets",
    licence="CC BY 4.0", capabilities="cell-calling,ambient-rna,doublets,gex-counts", tools=CORE_GEX + ",soupx,decontx,cellbender,scrublet,scdblfinder",
    why="species mixing gives read-level truth for ambient RNA and heterotypic doublets")
add(id="hgmm1k_v2", group="barnyard", tier="core", title="10x HEK293T + NIH3T3 1:1, 1k, 3' v2 (held out from the ambient/doublet fixes)",
    species="human+mouse", reference=B, chemistry="10x-3p-v2", source="10x", url=CF + "cell-exp/3.0.0/hgmm_1k_v2/hgmm_1k_v2_fastqs.tar",
    read_pairs_m=70, read_pairs_used_m=70, fastq_gb=5.8, truth="species per read", licence="CC BY 4.0",
    capabilities="cell-calling,ambient-rna,doublets", tools=CORE_GEX + ",soupx,decontx,cellbender,scrublet,scdblfinder",
    why="held-out barnyard in the older chemistry; hgmm_5k_v3 FASTQs return 403")
add(id="hgmm10k_v3", group="barnyard", tier="core", title="10x HEK293T + NIH3T3 1:1, 10k, 3' v3", species="human+mouse", reference=B,
    chemistry="10x-3p-v3", source="10x", url=S3 + "cell-exp/3.0.0/hgmm_10k_v3/hgmm_10k_v3_fastqs.tar", n_cells=11456, read_pairs_m=675,
    read_pairs_used_m=675, fastq_gb=56.1, truth="species per read; ~4% heterotypic doublets at 10k cells",
    licence="CC BY 4.0", capabilities="cell-calling,ambient-rna,doublets", tools=CORE_GEX + ",soupx,decontx,cellbender,scrublet,scdblfinder",
    why="the 1k barnyard has a single doublet; at 10k cells there are hundreds, enough to score doublet callers")
add(id="macosko2015_dropseq_mix", group="barnyard", tier="core", title="Macosko 2015 Drop-seq HEK293T + NIH3T3, ~1,000 STAMPs",
    species="human+mouse", reference=B, chemistry="dropseq", source="geo", gse="GSE63269", gsm="GSM1544798", srr="SRR1748411",
    n_cells=1020, read_pairs_m=268, read_pairs_used_m=268, fastq_gb=17.5,
    truth="species per read; authors' digital expression for human and mouse", licence="GEO public data (cite paper)",
    capabilities="gex-counts,cell-calling,ambient-rna,doublets", tools="singlet,starsolo,alevinfry,dropseq-tools",
    why="the original Drop-seq species mix; covers a protocol with no barcode whitelist")
# ---- CRISPR
add(id="tenx_crispr_a375_10k", group="crispr", tier="core", title="10x A375 + 1 non-target / 1 target sgRNA, 10k, 3' v3.1 CRISPR",
    species="human", reference=G, chemistry="10x-3p-v3", source="10x",
    url=S3 + "cell-exp/4.0.0/SC3_v3_NextGem_DI_CRISPR_10K/SC3_v3_NextGem_DI_CRISPR_10K_fastqs.tar", n_cells=11791,
    read_pairs_m=760, read_pairs_used_m=760, fastq_gb=63.7, truth="Cell Ranger 4 protospacer calls; target knock-down in expression",
    licence="CC BY 4.0", capabilities="crispr-guides,gex-counts,cell-calling", tools=CORE_GEX,
    why="10x's reference CRISPR Feature Barcode dataset (direct capture)")
add(id="norman2019_gem1", group="crispr", tier="core", title="Norman 2019 K562 CRISPRa Perturb-seq, gem group 1 (mRNA + guide library)",
    species="human", reference=G, chemistry="10x-3p-v2", source="geo", gse="GSE133344", gsm="GSM3906020;GSM3906028",
    srr="SRR9602535;SRR9602536;SRR9602551;SRR9602552", n_cells="", read_pairs_m=540, read_pairs_used_m=540, fastq_gb=24.0,
    truth="authors' guide identities (GSE133344_filtered_cell_identities.csv.gz); induced target expression",
    licence="GEO public data (cite paper)", capabilities="crispr-guides,gex-counts", tools="singlet,cellranger,starsolo",
    why="the most re-analysed Perturb-seq dataset (GEARS, CPA, scGen); amplified guide library rather than direct capture")
add(id="replogle2022_k562_lane4", group="crispr", tier="extended", title="Replogle 2022 K562 essential-scale Perturb-seq, lane 4 (mRNA + sgRNA)",
    species="human", reference=G, chemistry="10x-3p-v3", source="sra", other="PRJNA831566 KD6_seq1_essential lane_4", n_cells="",
    read_pairs_m=255, read_pairs_used_m=255, fastq_gb=15.9,
    truth="published cell-to-guide assignments (figshare+ 10.25452/figshare.plus.20029387, CC BY 4.0)",
    licence="CC BY 4.0", capabilities="crispr-guides", tools="singlet,cellranger",
    why="genome-scale Perturb-seq; lane-to-GEM-group mapping must be checked before use")
add(id="gse145370_tcr", group="vdj", tier="core", title="GSE145370 ESCC TCR libraries (4 samples), 10x 5' V(D)J",
    species="human", reference=G, chemistry="10x-5p-vdj", source="geo", gse="GSE145370", gsm="GSM4317423;GSM4317424;GSM4317428;GSM4317429",
    srr="SRR11094260;SRR11094261;SRR11094265;SRR11094266", truth="authors' Cell Ranger 3.0.1 filtered_contig_annotations (GEO supplementary)",
    licence="GEO public data (cite paper)", capabilities="vdj", tools="singlet,cellranger-vdj,trust4",
    why="held-out V(D)J with the authors' own contig calls; the atlas pilot study")
add(id="tenx_crispr_a549_5k", group="crispr", tier="core", title="10x A549 CRISPR 5k, 3' v3.1 direct capture (many guides)",
    species="human", reference=G, chemistry="10x-3p-v3", source="10x",
    url=CF + "cell-exp/6.0.0/SC3_v3_NextGem_DI_CRISPR_A549_5K_Multiplex/SC3_v3_NextGem_DI_CRISPR_A549_5K_Multiplex_fastqs.tar",
    fastq_gb=20.2, read_pairs_m=240, read_pairs_used_m=240, truth="Cell Ranger 6.0 protospacer calls; target knock-down",
    licence="CC BY 4.0", capabilities="crispr-guides,gex-counts", tools="singlet,cellranger",
    why="many guides per screen, unlike the two-guide A375 set")
# ---- spatial
add(id="visium_mouse_brain_ant", group="spatial", tier="core", title="10x Visium mouse brain, sagittal anterior section 1 (Space Ranger 1.0)",
    species="mouse", reference=M, chemistry="10x-visium", source="10x",
    url=S3 + "spatial-exp/1.0.0/V1_Mouse_Brain_Sagittal_Anterior/V1_Mouse_Brain_Sagittal_Anterior_fastqs.tar", n_cells=2695,
    read_pairs_m=370, read_pairs_used_m=370, fastq_gb=31.0, truth="Space Ranger 1.0 spot matrix and tissue positions; anatomy",
    licence="CC BY 4.0", capabilities="spatial,gex-counts", tools="singlet,spaceranger,starsolo",
    why="the Visium dataset in the Seurat, Squidpy and cell2location tutorials (the human lymph node set has no public reads)")
# ---- other protocols, one PBMC sample with author cell types (Ding 2020, PBMC1)
for pid, chem, gsm, rp, gb, why in [
    ("ding2020_pbmc1_smartseq2", "smartseq2", "GSM3836772-GSM3837155 (384 cells)", 375, 14.0, "plate-based full-length protocol without UMIs"),
    ("ding2020_pbmc1_dropseq", "dropseq", "GSM3836751;GSM3836752", 487, 17.9, "Drop-seq on the same PBMC sample"),
    ("ding2020_pbmc1_indrops", "indrop", "GSM3836766-GSM3836771 (4 runs)", 434, 16.6, "inDrops (V3 linker) on the same PBMC sample"),
    ("ding2020_pbmc1_seqwell", "seqwell", "GSM3836757;GSM3836769;GSM3836770", 236, 11.0, "Seq-Well on the same PBMC sample"),
    ("ding2020_pbmc1_celseq2", "celseq2", "GSM3836747-GSM3836750", 385, 13.8, "CEL-Seq2 plate protocol with UMIs"),
    ("ding2020_pbmc1_10xv3", "10x-3p-v3", "GSM3836762-GSM3836765", 667, 26.1, "10x v3 on the same sample, the anchor for cross-protocol comparisons"),
]:
    add(id=pid, group="protocols", tier="core", title=f"Ding 2020 PBMC1, {chem}", species="human", reference=G, chemistry=chem,
        source="geo", gse="GSE132044", gsm=gsm, srr="see ENA PRJNA545730 filereport", n_cells="", read_pairs_m=rp,
        read_pairs_used_m=rp, fastq_gb=gb, truth="authors' cell types per method (Single Cell Portal SCP424, GSE132044_pbmc_hg38_cell.tsv.gz)",
        licence="GEO public data (cite paper)", capabilities="gex-counts,cell-calling,cell-type" + (",junctions-psi" if chem == "smartseq2" else ""),
        tools="singlet,starsolo,alevinfry" if chem not in ("smartseq2",) else "singlet,star-featurecounts,salmon",
        why=f"{why}; one sample profiled by seven methods with curated labels (Ding et al. 2020)")
add(id="segerstolpe2016_pancreas", group="protocols", tier="extended", title="Segerstolpe 2016 human pancreas Smart-seq2 (3,514 cells)",
    species="human", reference=G, chemistry="smartseq2", source="arrayexpress", other="E-MTAB-5061;PRJEB15401", n_cells=3514,
    read_pairs_m=2645, read_pairs_used_m=2645, fastq_gb=101.3, truth="authors' cell types, donor and disease (E-MTAB-5061.sdrf.txt)",
    licence="ArrayExpress public data (cite paper)", capabilities="gex-counts,cell-type", tools="singlet,star-featurecounts,salmon",
    why="the standard Smart-seq2 cell-type benchmark (scmap, SingleR, scPred)")
# ---- mouse tissues
add(id="neuron1k_v3", group="mouse", tier="core", title="10x mouse E18 brain cells 1k, 3' v3", species="mouse", reference=M,
    chemistry="10x-3p-v3", source="10x", url=CF + "cell-exp/3.0.0/neuron_1k_v3/neuron_1k_v3_fastqs.tar", n_cells=1301,
    read_pairs_m=85, read_pairs_used_m=85, fastq_gb=7.1, truth="10x Cell Ranger matrix", licence="CC BY 4.0",
    capabilities="gex-counts,spliced-unspliced,cell-calling,cell-type", tools=CORE_GEX, why="standard mouse 3' v3 set")
add(id="heart1k_v3", group="mouse", tier="core", title="10x mouse E18 heart cells 1k, 3' v3", species="mouse", reference=M,
    chemistry="10x-3p-v3", source="10x", url=CF + "cell-exp/3.0.0/heart_1k_v3/heart_1k_v3_fastqs.tar", n_cells=1011,
    read_pairs_m=78, read_pairs_used_m=78, fastq_gb=6.4, truth="10x Cell Ranger matrix", licence="CC BY 4.0",
    capabilities="gex-counts,cell-calling,ambient-rna,cell-type", tools=CORE_GEX,
    why="second mouse tissue; cardiomyocyte transcripts dominate the ambient RNA")
add(id="tabula_muris_kidney_facs", group="mouse", tier="extended", title="Tabula Muris FACS (Smart-seq2) kidney", species="mouse",
    reference=M, chemistry="smartseq2", source="geo", gse="GSE109774", gsm="GSM2967051", n_cells=1730, read_pairs_m=600,
    read_pairs_used_m=600, fastq_gb=47.6, truth="Tabula Muris curated cell ontology labels (annotations_facs.csv)",
    licence="GEO public data (cite paper)", capabilities="gex-counts,cell-type", tools="singlet,star-featurecounts",
    why="mouse plate-based reference with curated Cell Ontology labels")
# ---- curated cell types from CELLxGENE (atlas studies; list in docs/cell_annotation.md section 3)
add(id="cxg16", group="curated", tier="core", title="16 GEO studies with CELLxGENE curated cell types (66 samples)", species="human+mouse",
    reference=f"{G};{M}", chemistry="mixed 10x", source="geo",
    gse="GSE125970;GSE130973;GSE131882;GSE144870;GSE157480;GSE161089;GSE164291;GSE166766;GSE184111;GSE185222;GSE185224;GSE185477;GSE201773;GSE260685;GSE270225;GSE280852",
    n_cells=332755, truth="author labels mapped to CL by CELLxGENE curators, transferred by barcode (scripts/annotate_truth_cxg.py)",
    licence="CELLxGENE CC BY 4.0; GEO public data", capabilities="cell-type,chromosomal-sex", tools="singlet,celltypist,azimuth,singler",
    why="held out from the guide's development; truth independent of every tool")
add(id="atac_pbmc_5k_nextgem", group="multiome", tier="core", title="10x scATAC PBMC 5k Next GEM (Cell Ranger ATAC 1.2)",
    species="human", reference=G, chemistry="10x-atac", source="10x",
    url=CF + "cell-atac/1.2.0/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_fastqs.tar", fastq_gb=18.8,
    truth="Cell Ranger ATAC 1.2 cell calls, fragments and peaks (published outputs)", licence="CC BY 4.0", capabilities="atac",
    tools="singlet,cellranger-atac,snapatac2,macs3", why="standalone scATAC reference")
add(id="pbmc_multiome3k", group="multiome", tier="development", title="10x Multiome PBMC 3k, granulocytes sorted out (ARC 2.0); tuning set",
    species="human", reference=G, chemistry="10x-arc", source="10x",
    url=CF + "cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_fastqs.tar", fastq_gb=22.4,
    truth="Cell Ranger ARC 2.0 published outputs", licence="CC BY 4.0", capabilities="atac,multiome", tools="singlet,cellranger-arc",
    why="development data for ATAC/Multiome; pbmc_multiome10k stays held out")
add(id="gsm5815984", group="depth", tier="core", title="GSE193672 Hs#14 testis, 10x 3', 2 runs (about 489M read pairs)",
    species="human", reference=G, chemistry="10x-3p-v3", source="geo", gse="GSE193672", gsm="GSM5815984", srr="SRR17615087;SRR17615088",
    read_pairs_m=489, read_pairs_used_m=489, truth="identical-read comparison with STARsolo", licence="GEO public data (cite paper)",
    capabilities="alignment-performance,gex-counts", tools="singlet,starsolo,alevinfry",
    why="runtime and memory at full atlas depth (the median atlas sample has about 200M pairs)")
# ---- non-host
add(id="ravindra2021_mock", group="non-host", tier="core", title="Ravindra 2021 human bronchial epithelium, mock (first 50M spots)",
    species="human", reference=G, chemistry="10x-3p-v3", source="geo", gse="GSE166766", gsm="GSM5082289", srr="SRR13711613",
    read_pairs_m=50, read_pairs_used_m=50, fastq_gb=4.5, truth="uninfected", licence="GEO public data (cite paper)",
    capabilities="non-host", tools="singlet,kraken2,minimap2-oracle", why="negative control for viral read detection")
add(id="ravindra2021_dpi2", group="non-host", tier="core", title="Ravindra 2021 bronchial epithelium, SARS-CoV-2 2 dpi (first 50M spots)",
    species="human", reference=G, chemistry="10x-3p-v3", source="geo", gse="GSE166766", gsm="GSM5082291", srr="SRR13711615",
    read_pairs_m=50, read_pairs_used_m=50, fastq_gb=4.5, truth="SARS-CoV-2 infected; minimap2 oracle per cell",
    licence="GEO public data (cite paper)", capabilities="non-host", tools="singlet,kraken2,minimap2-oracle",
    why="positive control: a known virus at known per-cell load")
# ---- sex and ancestry
add(id="geo_sex21", group="sex-ancestry", tier="core", title="19 atlas studies with donor sex in GEO metadata (179 labelled samples, 171 with usable counts)",
    species="human+mouse", reference=f"{G};{M}", chemistry="mixed 10x", source="geo",
    gse="GSE130653;GSE143381;GSE145370;GSE155224;GSE157783;GSE176417;GSE179502;GSE193416;GSE207097;GSE209832;GSE220764;GSE221494;GSE228056;GSE240975;GSE254999;GSE262873;GSE267240;GSE292067;GSE316782",
    n_cells=472363, truth="sex recorded in GEO sample characteristics", licence="GEO public data; Singlet counts CC0",
    capabilities="chromosomal-sex", tools="singlet,cellxy,xist-chry-rule",
    why="real samples across protocols and tissues with independent sex labels (docs/cell_annotation.md section 4)")
add(id="lcl_gse230239", group="sex-ancestry", tier="core", data_kind="derived", title="1000 Genomes LCLs HG00362 (FIN, female) and HG01277 (CLM, male), 10x 3' v3",
    species="human", reference=G, chemistry="10x-3p-v3", source="geo", gse="GSE230239", gsm="GSM7208744;GSM7208745",
    srr="SRR24236331;SRR24236332;SRR24236333;SRR24236334", read_pairs_m=650, read_pairs_used_m=650, fastq_gb=24,
    truth="1000 Genomes sample panel: population, super-population, sex", licence="GEO public data (cite paper)",
    capabilities="chromosomal-sex,genetic-ancestry,donor-demux,donor-snps,mtdna", tools="singlet,cellxy,somalier,peddy,cellsnp-lite,mgatk",
    why="known individuals of opposite sex and different ancestry; an in-silico mix of the two gives per-cell sex truth")
add(id="osorio2019_lcl_mix", group="sex-ancestry", tier="core", title="Osorio 2019 GM12878 (CEU) + GM18502 (YRI) 1:1 mix, 10x 3', first 150M spots",
    species="human", reference=G, chemistry="10x-3p-v2", source="geo", gse="GSE126321", gsm="GSM3596322", srr="SRR8551678",
    read_pairs_m=486, read_pairs_used_m=150, fastq_gb=22, truth="1000 Genomes panel for both individuals (female, EUR and AFR)",
    licence="GEO public data (cite paper)", capabilities="genetic-ancestry,donor-demux,donor-snps,mtdna,chromosomal-sex",
    tools="singlet,vireo,souporcell,somalier,peddy,cellsnp-lite,mgatk",
    why="a two-donor pool of known 1000 Genomes individuals: demultiplexing, then ancestry per donor, end to end")
add(id="mage150", group="sex-ancestry", tier="core", data_kind="derived", title="MAGE bulk RNA-seq of 1000 Genomes LCLs: 150 individuals, 26 populations (first 8M read pairs each)",
    species="human", reference=G, chemistry="bulk-rna-pe150", source="sra", other="PRJNA851328 (runs: bench/public/sexancestry/mage_subset.tsv)",
    n_cells="", read_pairs_m=3480, read_pairs_used_m=1200, fastq_gb=362,
    truth="1000 Genomes super-population, population and sex of each individual", licence="SRA public data (cite Taylor et al. 2024)",
    capabilities="genetic-ancestry,chromosomal-sex", tools="singlet,somalier,peddy",
    why="the only RNA-seq of 1000 Genomes individuals from every super-population; reads are thinned to single-cell donor depths")

with open(sys.argv[1], "w", newline="", encoding="utf-8") as fh:
    fh.write("# Singlet 1.0.0 public benchmark cohort. One row per dataset; cohort.md explains the choices and the budget.\n")
    w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n")
    w.writeheader()
    for row in R:
        w.writerow(row)
print(len(R), "rows")
