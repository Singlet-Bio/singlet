#!/usr/bin/env python3
"""Build the held-out half of the truth panel (panel_heldout.tsv) and its source list (sources_heldout.tsv).

Every study here is absent from the development panel. Labels are the chemistry or protocol the depositing paper or the vendor names
(sample titles and library text in the archive for the feature-barcode, Visium, Smart-seq and bulk rows). The panel was fixed and the code
frozen before it was scored; it is scored once.

usage: make_heldout.py OUTDIR
"""
import csv
import sys

rows = []   # id, source, family, protocol, species, label_source


def add(run, fam, proto, sp, src):
    rows.append(("h_" + run, run, fam, proto, sp, src))


H, M, N = "Homo sapiens", "Mus musculus", "Macaca"
# 10x 3' v3: Ravindra 2021 (Next GEM 3'), a lymphoblastoid line study (Chromium 3' v3), two further depositions with a stated v3 kit
for r in ("SRR13711613", "SRR13711614", "SRR13711615"):
    add(r, "droplet_3p", "10x-3p-v3", H, "Ravindra 2021 (GSE166766): Chromium Next GEM 3' library text")
for r in ("SRR24236331", "SRR24236333"):
    add(r, "droplet_3p", "10x-3p-v3", H, "GSE230239 library text: Chromium Single Cell 3' v3")
for r in ("SRR17615087", "SRR24098804", "SRR25032524"):
    add(r, "droplet_3p", "10x-3p-v3", H, "GEO library text (GSE193672 and two further depositions): 3' v3 kit")
# 10x 3' v2
for r, nm, sp in (("SRR8281306", "Stoeckius 2018 cell hashing", H), ("SRR9602535", "Norman 2019 Perturb-seq", H), ("SRR8606521", "Tian 2019 5-cell-line mixture", H),
                  ("SRR8551678", "Osorio 2019 lymphoblastoid mixture", H), ("SRR10765304", "Shami 2020 macaque testis", N)):
    add(r, "droplet_3p", "10x-3p-v2", sp, nm + ": paper names Chromium 3' v2")
# 10x GemCode (3' v1): Zheng 2017
for r in ("SRR3561730", "SRR3561735", "SRR3561742", "SRR3561743", "SRR3561750"):
    add(r, "droplet_3p", "10x-3p-v1", H, "Zheng 2017: GemCode single cell 3' (v1)")
# 10x 5'
for r, nm in (("SRR28010702", "Howitt 2023"), ("SRR28010644", "Howitt 2023"), ("SRR11094242", "GSE145370 (5' GEX of a 5' V(D)J study)")):
    add(r, "droplet_5p", "10x-5p", H, nm + ": 5' chemistry named by the depositors")
# Visium: GEO sample titles (Visium) of a study absent from the development panel
for r in ("SRR21008992", "SRR21008995", "SRR21008997"):
    add(r, "visium", "10x-visium", M, "GEO sample title (GSE for PRJNA729044) names a Visium library")
# multiome ATAC: archive experiment title (10x multiome, ATAC library); the normalized SRA run is empty, the original BAM holds the reads
for r in ("SRR30632679", "SRR30632680", "SRR30632681"):
    add(r, "multiome_atac", "10x-arc-atac", H, "archive experiment title: single nucleus ATAC-seq library of 10x multiome assay (PRJNA1156550)")
# feature barcodes: sample titles say HTO or ADT (cell hashing, CITE-seq), studies absent from the development panel
for r, g in (("SRR11479076", "GSM4453842"), ("SRR10091202", "GSM4072330"), ("SRR11784377", "GSM4551241"), ("SRR21012643", "GSM6444431")):
    add(r, "feature_barcode", "hto", H, "GEO sample title (%s) names an HTO library" % g)
for r, g in (("SRR12478200", "GSM4732691"), ("SRR12018251", "GSM4617145"), ("SRR11250157", "GSM4388145")):
    add(r, "feature_barcode", "adt", H, "GEO sample title (%s) names an ADT library" % g)
# Drop-seq family, Seq-Well, SPLiT-seq, CEL-Seq2
for r, nm, sp in (("SRR1748411", "Macosko 2015 species mixture", H), ("SRR1873278", "Macosko 2015 ERCC experiment", H), ("SRR5454666", "Lake 2018 nuclei (Drop-seq family)", H)):
    add(r, "droplet_other", "dropseq", sp, nm + ": Drop-Seq")
for r in ("SRR5250839", "SRR5250841"):
    add(r, "droplet_other", "seqwell", H, "Gierahn 2017: Seq-Well")
add("SRR6750056", "combinatorial", "splitseq", H, "Rosenberg 2018: SPLiT-seq")
for r, nm in (("SRR8606532", "Tian 2019 CEL-Seq2 plate"), ("SRR7588583", "Tian 2018 CEL-Seq2 mixture")):
    add(r, "plate_barcoded", "celseq2-u6b8", H, nm)
# Smart-seq2 (Tabula Muris FACS plates, mouse)
for r in ("SRR6573422", "SRR6573426", "SRR6573440"):
    add(r, "smartseq", "smartseq2", M, "Tabula Muris 2018 (GSM2967051, FACS plates): Smart-seq2")
# bulk RNA-seq, four kits: poly(A) TruSeq paired, poly(A) single-end, QuantSeq 3' tag (two depositions)
add("SRR1039508", "bulk_rna", "bulk", H, "GSM1275862 untreated airway smooth muscle cells (GSE52778), poly(A) TruSeq paired-end")
add("SRR099223", "bulk_rna", "bulk", M, "GEO series SRP004777 mouse striatum, single-end poly(A)")
add("DRR219358", "bulk_rna", "bulk", H, "archive library text names QuantSeq (3' tag bulk), single-end")
add("DRR180592", "bulk_rna", "bulk", H, "archive library text names QuantSeq (3' tag bulk), paired-end")

out = sys.argv[1]
with open(out + "/panel_heldout.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["id", "source", "family", "protocol", "species", "label_source", "tier"])
    for r in rows:
        w.writerow(list(r) + ["H"])
with open(out + "/sources_heldout.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    for r in rows:
        genome = "$REF/GRCh38-2024-A/star_2.7.11b" if r[4] == H else ("$REF/GRCm39-2024-A/star_2.7.11b" if r[4] == M else "-")
        w.writerow([r[0], r[1], "", genome])
print(len(rows), "rows")
