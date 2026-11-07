"""Write provenance.json for the atac, multiome and spatial artifact directories (bench/public contract 1.0.0).

    python make_provenance.py ART_ROOT COMMIT BINARY [--status draft|final]

ART_ROOT/<capability>/ holds the artifacts of make_artifacts.py (the *_runs.json files give the resources).
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("commit"); ap.add_argument("binary")
ap.add_argument("--status", default="draft")
ap.add_argument("--base", default="", help="prod/modality-fixes commit the branch was merged onto for the build")
a = ap.parse_args()
ENV = "/mnt/projects/debruinz_project/debruinz/singlet/envs/atacbench/bin"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ver(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout.strip() or r.stderr.strip()).splitlines()[0]
    except Exception:
        return "unknown"


SINGLET = {"version": "1.0.0", "commit": a.commit, "branch": "claude/atac-spatial-benchmark-2ba7df",
           "binary": os.path.basename(a.binary), "binary_sha256": sha256(a.binary),
           "build_flags": f"CMake Release, -march=x86-64-v3; this branch merged onto prod/modality-fixes {a.base}".strip()}
CHROMAP = {"name": "chromap+snapatac2", "version": f"chromap {ver([f'{ENV}/chromap', '--version'])}; "
           f"snapatac2 {ver([f'{ENV}/python', '-c', 'import snapatac2; print(snapatac2.__version__)'])}; "
           f"MACS3 {ver([f'{ENV}/macs3', '--version'])}",
           "role": "competitor", "url": "https://github.com/haowenz/chromap",
           "params": "chromap --preset atac -t 16 -x GRCh38-2024-A.index -r genome.fa -1 R1 -2 R3 -b R2 "
                     "--barcode-whitelist WL [--read-format bc:<window of Singlet's detected layout>] "
                     "[--barcode-translate ARC pairs]; macs3 callpeak -f BED --nomodel --extsize 200 --shift -100 "
                     "--keep-dup all -g hs (Signac CallPeaks); snapatac2.pp.import_fragments + metrics.tsse; cells = "
                     "ArchR createArrowFiles defaults (minFrags 1000, minTSS 4)"}
SINGLET_TOOL = {"name": "singlet", "version": "1.0.0", "role": "singlet",
                "params": "singlet encode --reads R1 R2 [R3] --protocol 10x-atac|10x-visium; singlet x.1fq --ref-base REF "
                          "--genome-dir GRCh38/GRCm39-2024-A/star_2.7.11b --exons genes.gtf --threads 16 --pipeline "
                          "--output-format mtx --raw-matrix [--paired-gex GEX_RUN]"}


def resources(runs, tool_keys):
    """One row per tool and dataset: every step from FASTQs to the compared outputs, summed (wall, CPU), max RSS.
    Singlet: encode + pipeline; chromap+snapatac2: Chromap + MACS3 + SnapATAC2; starsolo: STARsolo."""
    out = []
    for r in runs:
        for key, tool in tool_keys:
            x = r.get(key) or {}
            steps = {k: x[k] for k in ("time_encode", "time", "comparators_time") if x.get(k)}
            if not steps:
                continue
            out.append({"tool": tool, "dataset": r["dataset"], "threads": 16,
                        "wall_s": round(sum(t["wall_s"] for t in steps.values()), 1),
                        "cpu_h": round(sum(t["cpu_h"] for t in steps.values()), 3),
                        "max_rss_gb": max(t["max_rss_gb"] for t in steps.values()),
                        "notes": "sum of " + ", ".join({"time_encode": "FASTQ encode", "time": "main run",
                                                        "comparators_time": "MACS3 + SnapATAC2"}[k] for k in steps)})
    return out


COHORT_DS = {
    "atac_pbmc_5k_nextgem": {"id": "atac_pbmc_5k_nextgem", "title": "10x scATAC PBMC 5k Next GEM (Cell Ranger ATAC 1.2)",
                             "accessions": {"url": "https://cf.10xgenomics.com/samples/cell-atac/1.2.0/atac_pbmc_5k_nextgem/"},
                             "truth": "Cell Ranger ATAC 1.2.0 published cell calls, fragments and peaks"},
    "pbmc_multiome10k": {"id": "pbmc_multiome10k", "title": "10x Multiome PBMC 10k, granulocytes sorted out (ARC 2.0)",
                         "accessions": {"url": "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/"},
                         "truth": "Cell Ranger ARC 2.0.0 published joint cell calls, barcode pairs, fragments and peaks"},
    "pbmc_multiome3k": {"id": "pbmc_multiome3k", "title": "10x Multiome PBMC 3k, granulocytes sorted out (ARC 2.0), development",
                        "accessions": {"url": "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/"},
                        "truth": "Cell Ranger ARC 2.0.0 published outputs"},
    "visium_ffpe_human_breast": {"id": "visium_ffpe_human_breast", "title": "10x Visium FFPE human breast cancer (Space Ranger 1.3)",
                                 "accessions": {"url": "https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/"},
                                 "truth": "Space Ranger 1.3.0 spot matrix (probe set); tissue positions from the H&E image"},
    "visium_mouse_brain_ant": {"id": "visium_mouse_brain_ant", "title": "10x Visium mouse brain, sagittal anterior section 1",
                               "accessions": {"url": "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Mouse_Brain_Sagittal_Anterior/"},
                               "truth": "Space Ranger 1.1.0 spot matrix; tissue positions from the H&E image"},
}

COMMON = [
    "Binary: this branch (ATAC/Visium code) merged onto prod/modality-fixes, not yet part of the singlet-campaign build "
    "the other capabilities use; the results describe the code that the merge will bring.",
    "Barcode lists (scATAC whitelist, ARC ATAC-GEX pairs, visium-v1 spots) were rebuilt from 10x public datasets "
    "(CC BY 4.0); the 737K-cratac-v1.txt bundled before this change is a copy of the 3' v2 list and matches ~1% of "
    "scATAC barcodes.",
]
CAVEATS = {
    "atac": COMMON + [
        "Development data: Multiome PBMC 3k. The ATAC-only cell cut (density valley), STAR Extend5pOfReads12 and peak "
        "q 0.05 were chosen on it. The 5k scATAC cell count of the first caller (6,895 vs Cell Ranger's 4,585) was seen "
        "before the cell caller changed; the change was made on the 3k evidence only.",
        "Cell Ranger ATAC 1.2 published the 5k outputs on hg19; its peaks and fragments are lifted to GRCh38 with UCSC's "
        "hg19ToHg38 chain before comparison (both tools are scored against the same lifted set).",
        "Peaks: on Multiome 10k Singlet recovers more Cell Ranger peaks (recall 0.786 vs 0.771) but fewer of its peaks "
        "are Cell Ranger peaks (precision 0.878 vs 0.916); F1 0.830 vs 0.837 fails.",
        "Runtime: Chromap is an approximate mapper specialised for chromatin; Singlet aligns with STAR as part of the "
        "full pipeline and is ~1.8x slower (resources, not gated).",
    ],
    "multiome": COMMON + [
        "The competitor intersects Singlet's own GEX call (EmptyDrops) with the ArchR filter on Chromap fragments; "
        "STARsolo/EmptyDrops GEX calls were not run here.",
    ],
    "spatial": COMMON + [
        "Spot counts use Singlet's general GEX counting. On GRCm39 it drops multimapping reads that overlap only one "
        "gene, so mt-Co2, mt-Atp6, mt-Co3 and mt-Nd3 (covered by the chr1 NUMT) are nearly absent (mt-Co3: 71 vs Space "
        "Ranger 1.7 M); STARsolo and Space Ranger keep them. This, not the Visium handling, drives the mouse count "
        "failures and affects every mouse sample; reported for the gex-counts owner.",
        "Tissue detection uses counts only; Space Ranger's call comes from the H&E image. Model constants were fixed on "
        "synthetic data before any Visium run.",
        "FFPE probe reads are antisense: STARsolo was run with --soloStrand Reverse; Singlet detected the strand itself.",
    ],
}

for cap in ("atac", "multiome", "spatial"):
    d = Path(a.root) / cap
    if not d.exists():
        continue
    runs = [json.load(open(p)) for p in sorted(d.glob("*_runs.json"))]
    dss = [r["dataset"] for r in runs]
    test = [x for x in dss if x != "pbmc_multiome3k"]
    if cap == "multiome":
        test = [x for x in test if "multiome" in x]
    prov = {"singlet": SINGLET, "status": a.status, "test_datasets": test,
            "dev_datasets": [x for x in dss if x == "pbmc_multiome3k"],
            "datasets": [dict(COHORT_DS[x], data_kind="real") for x in test],
            "caveats": CAVEATS[cap], "methods": (Path(__file__).parent / f"methods_{cap}.md").read_text(encoding="utf-8")}
    if cap == "spatial":
        prov["tools"] = [SINGLET_TOOL,
                         {"name": "starsolo", "version": "2.7.11b", "role": "competitor",
                          "params": "STAR --soloType CB_UMI_Simple --soloCBwhitelist visium-v1.txt --soloCBlen 16 "
                                    "--soloUMIstart 17 --soloUMIlen 12 --soloFeatures Gene --soloUMIdedup 1MM_CR "
                                    "--soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR "
                                    "--clipAdapterType CellRanger4 --outFilterScoreMin 30"},
                         {"name": "otsu-umi-threshold", "version": "n/a", "role": "competitor",
                          "params": "Otsu split of log10(UMIs + 1) over the 4,992 slide spots"},
                         {"name": "spaceranger", "version": "1.1.0", "role": "truth",
                          "url": "https://www.10xgenomics.com/datasets", "params": "10x published outputs"}]
        prov["resources"] = resources(runs, [("singlet", "singlet"), ("starsolo", "starsolo")])
    else:
        prov["tools"] = [SINGLET_TOOL, CHROMAP,
                         {"name": "cellranger-atac/arc", "version": "ATAC 1.2.0 / ARC 2.0.0", "role": "truth",
                          "url": "https://www.10xgenomics.com/datasets", "params": "10x published outputs"}]
        if cap == "multiome":
            prov["tools"].append({"name": "per-modality-calls", "version": "n/a", "role": "competitor",
                                  "params": "GEX cells (Singlet GEX run, called_v2 = EmptyDrops) AND ATAC cells (ArchR "
                                            "default filter: >= 1000 fragments, TSS enrichment >= 4, on Chromap + "
                                            "SnapATAC2 fragments)"})
        prov["resources"] = resources(runs, [("singlet", "singlet"), ("chromap", "chromap+snapatac2")])
    json.dump(prov, open(d / "provenance.json", "w"), indent=1)
    print(cap, test, len(prov["resources"]))
