#!/usr/bin/env python3
"""Build the development panel (panel_dev.tsv) and the source list that `singlet identify --list` reads (sources_dev.tsv).

Rows come from three public sources with labels the depositors or vendors give:
  * GEO sample text of 10x libraries whose chemistry is stated in the deposit and confirmed by a second source (tier T2)
  * Mereu et al. 2020 (GSE133549): the protocol is in the sample name, one library per protocol
  * Ding et al. 2020 (GSE132044): protocol in the sample name (Smart-seq2 wells, bulk, Seq-Well, 10x v2/v3, DroNc-seq)
  * vendor datasets (10x Genomics) and cell-hashing / CITE-seq depositions
`$DATA` in a source stands for the Clipper data directory and is expanded by run_panel.sbatch.

usage: make_panel.py TIER_TABLE.tsv OUTDIR     (the first file lists tier-T2 rows: id, run, truth, tier, gse)
"""
import csv
import sys

T = "SRA run accession"
rows = []   # id, source, family, protocol, species, label_source, tier


# Panel labels use the family ids of validation/routes.yaml. "10x-3p" and "10x-5p" mean the version is not stated (no registry id).
REGISTRY_ID = {"bulk": "generic-negative", "dronc-seq": "dropseq", "adt": "cite-adt", "hto": "cite-hashtag", "10x-atac": "scatac", "celseq2-u6b8": "celseq2"}


def add(i, src, fam, proto, sp, label_source, tier):
    rows.append((i, src, fam, REGISTRY_ID.get(proto, proto), sp, label_source, tier))


def tenx_family(t):
    if t.startswith("10x-3p"):
        return "droplet_3p", t
    if t == "10x-5p":
        return "droplet_5p", "10x-5p"
    if t == "10x-arc-gex":
        return "multiome_gex", "10x-arc-gex"
    raise ValueError(t)


# 1. depositor-labelled 10x libraries (tier T2)
with open(sys.argv[1], newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["tier"] != "T2" or r["group"] != "consensus_T2":
            continue
        fam, proto = tenx_family(r["truth"])
        add(r["id"], r["run"], fam, proto, r["species"], "GEO sample text, confirmed by a second source", "T2")

# 2. Mereu et al. 2020, GSE133549: protocol from the sample name
mereu = [
    ("SRR9621416", "droplet_3p", "10x-3p", "10X2x5Kcell250Kreads"), ("SRR10587810", "droplet_3p", "10x-3p", "10XV3_AN4492"),
    ("SRR9621776", "droplet_3p", "10x-3p", "SingleNuclei_nextseq1"),
    ("SRR9621728", "plate_barcoded", "celseq2-u6b8", "CELseq2_HCAP1_4"), ("SRR9621733", "plate_barcoded", "celseq2-u6b8", "CELseq2_HCAP3_1"),
    ("SRR9621741", "droplet_other", "dropseq", "Dropseq_N701"), ("SRR9621742", "droplet_other", "dropseq", "Dropseq_N702"),
    ("SRR9621749", "plate_barcoded", "icell8", "ICELL8_SP021_022"),
    ("SRR9621752", "plate_barcoded", "marsseq2", "MARSseq_5569AA"), ("SRR9621758", "plate_barcoded", "marsseq2", "MARSseq_5923AA"),
    ("SRR9621763", "plate_barcoded", "quartzseq2", "QUARTZseq_P2S5"), ("SRR9621766", "plate_barcoded", "quartzseq2", "QUARTZseq_P5S8"),
    ("SRR9621768", "plate_barcoded", "scrbseq", "SCRBseq_N702"), ("SRR9621769", "plate_barcoded", "scrbseq", "SCRBseq_N703"),
    ("SRR9621775", "smartseq", "smartseq2", "SMARTseq2_allLanes"),
    ("SRR9621785", "droplet_other", "ddseq", "ddSEQ_Sample2"), ("SRR9621786", "droplet_other", "ddseq", "ddSEQ_Sample3"),
    ("SRR9621794", "droplet_other", "indrop", "inDrop_SP021_024015"),
]
for run, fam, proto, name in mereu:
    add("mereu_" + name, run, fam, proto, "Homo sapiens", "GSE133549 sample name (Mereu 2020)", "D")

# 3. Ding et al. 2020, GSE132044: protocol from the sample name
ding = [
    ("SRR9167490", "smartseq", "smartseq2", "Mixture2.SM2_103117p3_B8"), ("SRR9167495", "smartseq", "smartseq2", "Mixture2.SM2_103117p3_C1"),
    ("SRR9167561", "smartseq", "smartseq2", "Cortex1.SM2 well"),
    ("SRR9170689", "bulk_rna", "bulk", "PBMC2_bulk"), ("SRR9169420", "bulk_rna", "bulk", "PBMC1_bulk"), ("SRR9171226", "bulk_rna", "bulk", "Cortex2_bulk"),
    ("SRR9167464", "droplet_other", "seqwell", "Mixture2.SeqWell_BC9"), ("SRR9169213", "droplet_other", "seqwell", "PBMC2.SeqWell_S3_2"),
    ("SRR9169186", "droplet_other", "seqwell", "PBMC1.July2018-SeqWell_V1R1"),
    ("SRR9169188", "droplet_3p", "10x-3p-v2", "PBMC1.011719V2_10X_B"), ("SRR9169190", "droplet_3p", "10x-3p-v2", "PBMC1.011719V2_10X_D"),
    ("SRR9169193", "droplet_3p", "10x-3p-v3", "PBMC1.011719V3_10X_B"), ("SRR9169192", "droplet_3p", "10x-3p-v3", "PBMC1.011719V3_10X_A"),
    ("SRR9170887", "droplet_other", "dronc-seq", "Cortex2.DroNcSeq_3A"),
]
for run, fam, proto, name in ding:
    add("ding_" + name.replace(" ", "_"), run, fam, proto, "Homo sapiens" if "Cortex" not in name else "Mus musculus",
        "GSE132044 sample name (Ding 2020)", "D")

# 4. feature-barcode libraries: GEO sample title says HTO or ADT (cell hashing, CITE-seq)
for run, lab, gsm in [("SRR8281307", "hto", "GSM2895283"), ("SRR8787262", "hto", "GSM3687894"), ("SRR11176314", "hto", "GSM4338142"),
                      ("SRR8281308", "adt", "GSM2895284"), ("SRR5808750", "adt", "GSM2695379"), ("SRR9159911", "adt", "GSM3832736")]:
    add("feat_" + run, run, "feature_barcode", lab, "Homo sapiens", "GEO sample title (%s) names an %s library" % (gsm, lab.upper()), "T2")

# 5. Visium: GEO sample titles and 10x Genomics dataset names
for run, nm, sp in [("SRR21230112", "GSM6506116 Visium_poor_response_P7", "Homo sapiens"), ("SRR21230121", "GSM6506117 Visium_poor_response_P8", "Homo sapiens"),
                    ("SRR22816622", "GSM6859073 Visium_Tumor_liver_Mouse3", "Mus musculus")]:
    add("visium_" + run, run, "visium", "10x-visium", sp, "GEO sample title: " + nm, "T2")
D = "$DATA/atac-spatial"
add("visium_V1_Mouse_Brain_Sagittal_Anterior", "%s/V1_Mouse_Brain_Sagittal_Anterior/V1_Mouse_Brain_Sagittal_Anterior_Section_1_fastqs/V1_Mouse_Brain_Sagittal_Anterior_Section_1_S7_L001_R1_001.fastq.gz,%s/V1_Mouse_Brain_Sagittal_Anterior/V1_Mouse_Brain_Sagittal_Anterior_Section_1_fastqs/V1_Mouse_Brain_Sagittal_Anterior_Section_1_S7_L001_R2_001.fastq.gz" % (D, D),
    "visium", "10x-visium", "Mus musculus", "10x Genomics dataset name (Visium V1 mouse brain)", "V")
add("visium_FFPE_Human_Breast_Cancer", "%s/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_fastqs/Visium_FFPE_Human_Breast_Cancer_S1_L001_R1_001.fastq.gz,%s/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_fastqs/Visium_FFPE_Human_Breast_Cancer_S1_L001_R2_001.fastq.gz" % (D, D),
    "visium", "10x-visium", "Homo sapiens", "10x Genomics dataset name (Visium FFPE, probe based)", "V")

# 6. ATAC: 10x Genomics datasets (FASTQ on the cluster, label is the vendor dataset name)
def fq(path, stem, lane, reads):
    return ",".join("%s/%s_L00%d_%s_001.fastq.gz" % (path, stem, lane, r) for r in reads)
add("atac_pbmc_5k_nextgem", fq(D + "/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_fastqs", "atac_pbmc_5k_nextgem_S1", 1, ["R1", "R2", "R3", "I1"]),
    "atac", "10x-atac", "Homo sapiens", "10x Genomics dataset name (scATAC PBMC 5k Next GEM)", "V")
add("multiome_atac_pbmc_3k", fq(D + "/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k/atac", "pbmc_granulocyte_sorted_3k_S12", 1, ["R1", "R2", "R3", "I1"]),
    "multiome_atac", "10x-arc-atac", "Homo sapiens", "10x Genomics dataset name (PBMC granulocyte-depleted 3k multiome, ATAC library)", "V")
add("multiome_atac_pbmc_10k", fq(D + "/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k/atac", "pbmc_granulocyte_sorted_10k_S16", 1, ["R1", "R2", "R3", "I1"]),
    "multiome_atac", "10x-arc-atac", "Homo sapiens", "10x Genomics dataset name (PBMC granulocyte-depleted 10k multiome, ATAC library)", "V")
add("multiome_gex_pbmc_3k", fq(D + "/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k/gex", "pbmc_granulocyte_sorted_3k_S1", 3, ["R1", "R2"]),
    "multiome_gex", "10x-arc-gex", "Homo sapiens", "10x Genomics dataset name (PBMC granulocyte-depleted 3k multiome, GEX library)", "V")
add("multiome_gex_pbmc_10k", fq(D + "/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k/gex", "pbmc_granulocyte_sorted_10k_S1", 3, ["R1", "R2"]),
    "multiome_gex", "10x-arc-gex", "Homo sapiens", "10x Genomics dataset name (PBMC granulocyte-depleted 10k multiome, GEX library)", "V")



# Rows whose labeled chemistry (10x 3’ or 5’) the original deposited files do not support (names and read lengths suggest bulk, nanopore or sheared DNA). Labels are kept.
LABEL_DOUBT = {'GSM5472074': 'original file bulkRNA_*.fastq', 'GSM5769275': 'original files *_Bulk_RNAseq.fastq', 'GSM6564300': 'original file NP19.fastq (nanopore-like, 651-base reads)', 'GSM6564302': 'original file PAG*_pass_*.fastq (nanopore-like, 685-base reads)', 'GSM6564303': 'original file PAG*_pass_*.fastq (nanopore-like, 661-base reads)', 'GSM6564304': 'original file PAG*_pass_*.fastq (nanopore-like, 851-base reads)', 'GSM8158578': 'original files B_shear_*.fq (sheared DNA)', 'GSM8754140': 'original files Plko-2_*.fq', 'GSM9287649': 'original file T_iMK_*.fq'}

def sc_hint(family):
    """1 where the archive record (GEO or SRA text) states single-cell: every family except bulk and genomic"""
    return "0" if family in ("bulk_rna", "genomic_dna") else "1"


out = sys.argv[2]
with open(out + "/panel_dev.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["id", "source", "family", "protocol", "species", "label_source", "tier", "single_cell_hint", "label_doubt"])
    seen = set()
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        w.writerow(list(r) + [sc_hint(r[2]), LABEL_DOUBT.get(r[0], "")])
with open(out + "/sources_dev.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    for r in rows:
        # columns: id, source, hint (single_cell=1 from the archive record), genome for the mapping layer (GRCh38 or GRCm39 STAR index; $REF is the reference directory)
        w.writerow([r[0], r[1], "single_cell=1" if sc_hint(r[2]) == "1" else "", "$REF/GRCh38-2024-A/star_2.7.11b" if r[4] == "Homo sapiens" else "$REF/GRCm39-2024-A/star_2.7.11b"])
print(len(seen), "rows")
