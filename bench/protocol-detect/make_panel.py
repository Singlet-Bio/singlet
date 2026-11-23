"""Build the protocol-detection truth panel and split it into development and held-out halves.

    python make_panel.py DONE_JSONL            # writes panel.tsv, dev.tsv, heldout.tsv, REGISTRATION.md

Sources (all real public runs, labels from the depositor, the paper or the vendor; never from a detector call):
  src_identify_dev.tsv       76 rows of the identify development panel (engine repo bench/identify/panel_dev.tsv)
  src_identify_heldout.tsv   47 rows of the identify held-out panel (bench/identify/heldout_candidates.tsv)
  src_s2_runmaps.txt         run maps of the prelaunch S2 held-out libraries (Ding 2020 PBMC and cortex, from GEO titles)
  hard-coded rows below      Ding mixture, King tonsil GEX and BCR, schihost, GM12891, Snisor, chx10k_b, Visium sections
  DONE_JSONL                 campaign samples that shipped with mapping >= 0.85 and >= 2000 cells and whose engine call agreed
                             with the stated chemistry (tier P: label confirmed by a full run with strong mapping)

Family taxonomy (what the campaign must decide, and what reads can tell apart):
  10x_3p, 10x_5p, 10x_arc_gex, visium, visium_probe, 10x_flex, atac (10x ATAC and multiome ATAC), feature_barcode, vdj,
  other_barcoded (Drop-seq, Seq-Well, inDrop, ddSEQ, BD, microwell, sci-RNA-seq3, SPLiT/Parse, plate CEL-Seq2/MARS-seq/
  SCRB-seq/ICELL8/Quartz-Seq2), no_cell_barcode (bulk RNA, Smart-seq2/3, and 10x libraries whose deposit holds no barcode read).
Version (scored on 10x families only): the engine tag the library should run under (10x-3p-v1..v4, 10x-5p-v2 for 5' v1/v2,
10x-5p-v3, 10x-arc-gex, 10x-visium).  Rows whose version is not stated get version "".

Relabel rule, fixed before any detector run: a 10x-labeled row whose deposit holds no 10x barcode read is scored as
no_cell_barcode.  The relabel is applied by score.py from the panel column `deposit_check` = "barcode_read" and the head dump
(no slot reaches 5 percent hits on any 10x list at any offset or strand); 5 percent cannot arise by chance (random 16-mers hit a
3.7 M list about once per million reads).  The 9 identify rows with label_doubt keep their doubt note.

Split: grouped by library group (study; each Ding 10x run is its own group so short-R1 runs reach both halves), stratified by family, assigned by sha256 of the group key
so that the split does not depend on row order; groups alternate between halves within each family in hash order.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
H = "$HL"   # $W/data/heldout-launch on Clipper

FAM_MAP = {  # (identify family, protocol) -> (family, version)
    "droplet_3p": "10x_3p", "droplet_5p": "10x_5p", "multiome_gex": "10x_arc_gex", "multiome_atac": "atac", "atac": "atac",
    "visium": "visium", "feature_barcode": "feature_barcode", "bulk_rna": "no_cell_barcode", "smartseq": "no_cell_barcode",
    "plate_barcoded": "other_barcoded", "droplet_other": "other_barcoded",
}


def version_of(family: str, protocol: str) -> str:
    p = protocol.lower()
    if family == "10x_3p":
        for v in ("v1", "v2", "v3", "v4"):
            if p.startswith("10x-3p-" + v):
                return "10x-3p-" + v
        return ""
    if family == "10x_5p":
        return {"10x-5p-v1": "10x-5p-v2", "10x-5p-v2": "10x-5p-v2", "10x-5p-v3": "10x-5p-v3"}.get(p, "")
    if family == "10x_arc_gex":
        return "10x-arc-gex"
    if family == "visium":
        return "10x-visium"
    return ""


def map_identify(fam: str, proto: str) -> str:
    p = proto.lower()
    if p == "10x-flex":
        return "10x_flex"
    if p.startswith("10x-visium-") and "probe" in p:
        return "visium_probe"
    if p == "10x-5p-vdj":
        return "vdj"
    return FAM_MAP[fam]


def rows_identify():
    out = []
    src = {r["id"]: r for r in csv.DictReader(open(os.path.join(HERE, "src_identify_dev_sources.tsv")), delimiter="\t",
                                              fieldnames=["id", "source", "hint", "genome"])}
    for r in csv.DictReader(open(os.path.join(HERE, "src_identify_dev.tsv")), delimiter="\t"):
        fam = map_identify(r["family"], r["protocol"])
        group = r["id"].split("_")[0] + "_" + r["id"].split("_")[1] if r["id"].startswith(("mereu_", "ding_")) else r["id"]
        if r["id"].startswith("ding_"):
            group = "ding_" + r["id"].split("_")[1].split(".")[0]
            if "_10X_" in r["id"]:
                group = r["id"]                    # Ding 10x runs split one by one (the short-R1 layouts sit in few flowcells)
        out.append(dict(id=r["id"], source=src.get(r["id"], {}).get("source", r["source"]), family=fam, protocol=r["protocol"],
                        version=version_of(fam, r["protocol"]), species=r["species"], label_source=r["label_source"],
                        tier=r["tier"], origin="identify_dev", group=group,
                        deposit_check="barcode_read" if fam.startswith("10x") else "", note=r.get("label_doubt", "")))
    for r in csv.DictReader(open(os.path.join(HERE, "src_identify_heldout.tsv")), delimiter="\t"):
        fam = map_identify(r["family"], r["protocol"])
        out.append(dict(id=r["accession"], source=r["accession"], family=fam, protocol=r["protocol"],
                        version=version_of(fam, r["protocol"]), species="", label_source=r["label_source"][:160], tier="T2",
                        origin="identify_heldout", group=r["label_source"].split()[0] if r["label_source"] else r["accession"],
                        deposit_check="barcode_read" if fam.startswith("10x") else "", note=r.get("note", "")))
    return out


def rows_s2():
    out, cur = [], None
    for line in open(os.path.join(HERE, "src_s2_runmaps.txt")):
        line = line.rstrip("\n")
        if line.startswith("### "):
            cur = line[4:]
            continue
        f = line.split("\t")
        if cur not in ("ding2020_pbmc", "ding2020_cortex") or len(f) < 4 or not f[0].startswith("SRR"):
            continue
        title = f[3]
        ver = "10x-3p-v3" if "V3_10X" in title else "10x-3p-v2"
        sp = "Mus musculus" if cur == "ding2020_cortex" else "Homo sapiens"
        grp = f"{cur}_{f[0]}"                                # one group per run, see rows_identify
        out.append(dict(id=f"{cur}_{f[0]}", source=f[0], family="10x_3p", protocol=ver, version=ver, species=sp,
                        label_source=f"GSE132044 {f[2]} title {title}; Ding 2020 methods (10x v2 unless the title says V3)",
                        tier="D", origin="s2_heldout_launch", group=grp, deposit_check="barcode_read", note=""))
    return out


def rows_fixed():
    R = []
    def add(i, src, fam, proto, sp, lab, grp, tier="T2", dc=None):
        R.append(dict(id=i, source=src, family=fam, protocol=proto, version=version_of(fam, proto), species=sp, label_source=lab,
                      tier=tier, origin="s2_heldout_launch", group=grp,
                      deposit_check=("barcode_read" if fam.startswith("10x") else "") if dc is None else dc, note=""))
    for acc in ("SRR9170959", "SRR9170960", "SRR9170961", "SRR9170962", "SRR9171224", "SRR9171225", "SRR9171227", "SRR9171228"):
        add(f"ding2020_mixture_{acc}", acc, "10x_3p", "10x-3p-v2", "Homo sapiens+Mus musculus",
            "GSE132044 Mixture 10x runs; Ding 2020 methods (10x v2)", "ding_Mixture", "D")
    for acc in ("SRR19609899", "SRR19609900", "SRR19609901", "SRR19609902", "SRR19609903", "SRR19609904", "SRR19609905"):
        add(f"schihost_{acc}", acc, "10x_3p", "10x-3p-v3", "Homo sapiens", "GSE205796 methods: Chromium 3' v3.1", "schihost")
    for acc in ("SRR6844906", "SRR6844907"):
        add(f"gm12891_{acc}", acc, "10x_3p", "10x-3p-v2", "Homo sapiens", "GSE111912 methods: Chromium 3' v2", "gm12891")
    add("king_tonsil_gex", f"bam:{H}/king2021_tonsil/gex/H2FWCBGXB_BCP008_Total_S7_L001.bam", "10x_5p", "10x-5p-v1",
        "Homo sapiens", "E-MTAB-9005 SDRF and King 2021: Chromium 5' v1 GEX", "king_tonsil")
    add("king_tonsil_bcr", f"bam:{H}/king2021_tonsil/bcr/BCP8_Unsorted_ImmuneCells_Tonsil.bam", "vdj", "10x-5p-vdj-bcr",
        "Homo sapiens", "E-MTAB-9003 SDRF and King 2021: Chromium 5' V(D)J BCR enrichment", "king_tonsil", dc="")
    for n in (1, 2):
        add(f"snisor_FCtx{n}", f"bam:{H}/snisor_gse178175/illumina/FCtx{n}_10X_sc.bam", "10x_3p", "10x-3p-v3", "Homo sapiens",
            "GSE178175 / Snisor 2021 methods: Chromium 3' v3", "snisor")
    add("dlpfc_151673", ",".join(f"{H}/dlpfc_151673/fastq/151673_0_1_HVHCFDSXX_L001_{s}.fastq.gz" for s in ("R1", "R2", "I1")),
        "visium", "10x-visium", "Homo sapiens", "spatialLIBD 151673 (Maynard 2021): Visium v1", "dlpfc")
    return R


def rows_campaign(done_path):
    rows = [json.loads(l) for l in open(done_path)]
    ok = {"10x_3p_v2": "10x-3p-v2", "10x_3p_v3": "10x-3p-v3", "10x_3p_v4": "10x-3p-v4"}
    det = {"10x-3p-v2": "10xv2", "10x-3p-v3": "10xv3", "10x-3p-v4": "10xv4"}
    seen, out = set(), []
    good = [r for r in rows if r["verdict"] == "pass" and (r.get("mapping_rate") or 0) >= 0.85 and (r.get("cells") or 0) >= 2000
            and r.get("runs") and r["chemistry"] in ok]
    good.sort(key=lambda r: hashlib.sha256(r["gsm"].encode()).hexdigest())
    nv3 = 0
    for r in good:
        v = ok[r["chemistry"]]
        if (r.get("detection") or {}).get("detected_protocol") != det[v] or r["gse"] in seen:
            continue
        if v == "10x-3p-v3":
            if nv3 >= 30:
                continue
            nv3 += 1
        seen.add(r["gse"])
        out.append(dict(id=r["gsm"], source=sorted(r["runs"])[0], family="10x_3p", protocol=v, version=v, species="Homo sapiens",
                        label_source=f"{r['gse']} stated chemistry; campaign run mapping {r['mapping_rate']:.3f}, {r['cells']} cells",
                        tier="P", origin="campaign_pass", group=r["gse"], deposit_check="barcode_read", note=""))
    return out


def split(rows):
    by_fam = {}
    for r in rows:
        by_fam.setdefault(r["family"], {}).setdefault(r["group"], []).append(r)
    for fam, groups in by_fam.items():
        keys = sorted(groups, key=lambda g: hashlib.sha256(f"{fam}|{g}".encode()).hexdigest())
        # balance by row count: give each group to the half that has fewer rows of this family so far (hash order breaks ties)
        n = {"dev": 0, "heldout": 0}
        for k in keys:
            half = "dev" if n["dev"] <= n["heldout"] else "heldout"
            for r in groups[k]:
                r["split"] = half
            n[half] += len(groups[k])
    return rows


COLS = ["id", "source", "family", "protocol", "version", "species", "label_source", "tier", "origin", "group", "deposit_check",
        "note", "split"]


def main():
    rows = rows_identify()
    have = {r["source"] for r in rows}
    rows += [r for r in rows_s2() + rows_fixed() + rows_campaign(sys.argv[1]) if r["source"] not in have]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    srcs = [r["source"] for r in rows]
    dup = {s for s in srcs if srcs.count(s) > 1}
    assert not dup, f"duplicate sources {dup}"
    split(rows)
    for name, sel in (("panel.tsv", rows), ("dev.tsv", [r for r in rows if r["split"] == "dev"]),
                      ("heldout.tsv", [r for r in rows if r["split"] == "heldout"])):
        with open(os.path.join(HERE, name), "w", newline="\n") as f:
            f.write("\t".join(COLS) + "\n")
            for r in sel:
                f.write("\t".join(str(r.get(c, "")).replace("\t", " ") for c in COLS) + "\n")
    import collections
    c = collections.Counter((r["family"], r["split"]) for r in rows)
    for fam in sorted({r["family"] for r in rows}):
        print(f"{fam:16s} dev {c[(fam, 'dev')]:3d}  heldout {c[(fam, 'heldout')]:3d}")
    print("total", len(rows))


if __name__ == "__main__":
    main()
