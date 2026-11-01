"""Write provenance.json into the donor-demux, donor-snps and mtdna artifact directories (laptop side, after the artifacts
were pulled from Clipper): the exact commands, the commit of this code and the resource table.

    python bench/public/genotype/make_provenance.py ARTIFACT_ROOT CODE_COMMIT
ARTIFACT_ROOT holds donor-demux/, donor-snps/, mtdna/ (copies of $WORK/bench-public/artifacts/<capability>/).
"""
import json
import sys
from pathlib import Path

import pandas as pd

root, commit = Path(sys.argv[1]), sys.argv[2]
REF = "/mnt/projects/debruinz_project/cellarium/reference"
PANEL = "genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz"
common = {
    "code_commit": commit,
    "singlet_command": (f"singlet-campaign POOL.1fq --ref-base {REF} --genome-dir {REF}/GRCh38-2024-A/star_2.7.11b --exons "
                        f"{REF}/GRCh38-2024-A/genes/genes.gtf --threads 16 --pipeline --n-donors -1 --tagged-bam possorted.bam "
                        "--bam-sort-mode disk --output-format mtx --raw-matrix  (default SNP panel: 1000 Genomes phase 3, AF >= 5%, 7.35 M SNPs; "
                        "--n-donors K only for the told-K comparison runs)"),
    "cellsnp_command": (f"cellsnp-lite -s possorted.bam -b barcodes.txt -O cellsnp -R {PANEL} -p 16 --minMAF 0.1 --minCOUNT 20 "
                        "--gzip --cellTAG CB --UMItag UB   (barcodes = Singlet's called cells; the cellSNP/vireo manuals' settings for vireo)"),
    "vireo_command": ("cellsnp-lite as above, then vireo -c cellsnp -o OUT -N K -p 16 --randSeed {1,2,3} --noPlot (K = true K, defaults otherwise: "
                      "nInit 50, doublet detection on); K estimate: vireo -N 12 --randSeed 1, K = donors holding >= 1% of cells"),
    "souporcell_command": (f"souporcell_pipeline.py -i cells.bam -b barcodes.txt -f {REF}/GRCh38-2024-A/fasta/genome.fa -t 16 -o OUT -k K "
                           f"--common_variants {PANEL.replace('.vcf.gz', '.vcf')}  (K = true K; remapping with minimap2 and every other option at its default; "
                           "cells.bam = Singlet's tagged BAM restricted to the called cells)"),
    "pileup_command": ("cellsnp-lite counts as above, summed over the cells of each cluster (vireo's labels; all cells for one-donor libraries), "
                       "genotyped with the binomial model of the truth: error 0.01, flat prior, posterior >= 0.99"),
    "mgatk_command": ("mgatk bcall -i chrM.bam -n POOL -o out -c 8 -bt CB -b barcodes.txt -ub UB -g rCRS (chrM reads of Singlet's tagged BAM; "
                      "other options at their defaults: NHmax 1, NMmax 4, base quality 0)"),
}
res = pd.read_csv(root / "donor-demux" / "donor_demux_resources.tsv", sep="\t")
TOOL = {"singlet": "singlet", "singlet_told": "singlet", "cellsnp-lite": "cellsnp-lite", "vireo_told": "vireo", "vireo_n12": "vireo",
        "souporcell": "souporcell", "mgatk": "mgatk"}
NOTE = {"singlet": "end to end from reads incl. alignment, K estimated", "singlet_told": "K given", "cellsnp-lite": "pileup on the shared BAM and panel (vireo input)",
        "vireo_told": "K given, one seed, clustering only", "vireo_n12": "N = 12, clustering only",
        "souporcell": "remap + freebayes + vartrix + clustering, k given", "mgatk": "chrM reads of the shared BAM, 8 threads"}


def rows(datasets, tools, pool_map=None):
    out = []
    for _, r in res.iterrows():
        pool = (pool_map or {}).get(r["pool"], r["pool"])
        if pool not in datasets or TOOL[r["tool"]] not in tools:
            continue
        out.append({"tool": TOOL[r["tool"]], "dataset": pool, "wall_s": float(r["wall_s"]), "max_rss_gb": float(r["max_rss_gb"]),
                    "threads": int(r["threads"]), "cpu_h": round(float(r["wall_s"]) * int(r["threads"]) / 3600, 3), "notes": NOTE[r["tool"]]})
    return out


kang = ["kang2018_ctrl", "kang2018_stim", "kang2018_b1_a", "kang2018_b1_b", "kang2018_b1_c"]
(root / "donor-snps").mkdir(exist_ok=True); (root / "mtdna").mkdir(exist_ok=True)
json.dump({**common, "resources": rows(kang + ["osorio2019_lcl_mix"], {"singlet", "vireo", "souporcell", "cellsnp-lite"})},
          open(root / "donor-snps" / "provenance.json", "w"), indent=1)
json.dump({**common, "resources": rows(kang + ["maester_bpdcn712_3p", "maester_k562_bt142_3p"], {"singlet", "mgatk", "vireo"})},
          open(root / "mtdna" / "provenance.json", "w"), indent=1)
json.dump({**common, "resources": rows(kang + ["maester_bpdcn712_3p", "maester_k562_bt142_3p"] + ["osorio2019_lcl_mix"],
                                       {"singlet", "vireo", "souporcell", "cellsnp-lite"})},
          open(root / "donor-demux" / "provenance.json", "w"), indent=1)
print("provenance written for", commit)
