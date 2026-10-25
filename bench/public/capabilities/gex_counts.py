# SPDX-License-Identifier: MIT
"""gex-counts: Singlet's gene counts per cell against STARsolo, alevin-fry and Cell Ranger (public benchmark, schema 1.0.0).

    python bench/public/capabilities/gex_counts.py ARTIFACT_DIR [--out bench/public/results/gex-counts.json]

``compute(artifact_dir)`` reads the per-dataset tables written by ``extract/counts.py`` (<id>.cells.tsv.gz, <id>.genes.tsv.gz,
resources.json) and returns the capability document. The statistics and the gate are in ``_familycmp.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _familycmp as F  # noqa: E402
import _pubcommon as C  # noqa: E402

CAP = "gex-counts"
LABEL = {"singlet_exon": "Singlet exonic", "singlet_total": "Singlet exon+intron", "starsolo_gene": "STARsolo Gene",
         "starsolo_genefull": "STARsolo GeneFull", "alevinfry_total": "alevin-fry USA (S+U+A)",
         "cellranger": "Cell Ranger (published)"}
DECL = {"singlet_exon": "singlet", "singlet_total": "singlet", "starsolo_gene": "starsolo-gene",
        "starsolo_genefull": "starsolo-genefull", "alevinfry_total": "alevin-fry-usa", "cellranger": "cellranger-published"}
SPECS = [
    F.Spec("exonic", "singlet_exon", LABEL, DECL, "exonic UMI", "exonic"),
    F.Spec("total", "singlet_total", LABEL, DECL, "exon+intron UMI", "total"),
]

TRUTH = ("none for counts. Agreement between independent tools run on the same FASTQs and reference. "
         "Cell set: barcodes called by STARsolo EmptyDrops_CR.")
NOTES = {
    "pbmc5p_v1": "first 60 M read pairs; the published Cell Ranger matrix (full 448 M pairs) is not compared",
    "hgmm1k_v3": "human + mouse reference; genes matched by Ensembl id",
}

STAR_CMD = ("STAR --runThreadN 16 --genomeDir $GENOME --readFilesIn $R2 $R1 --readFilesCommand zcat --soloType CB_UMI_Simple "
            "--soloCBwhitelist $WHITELIST --soloCBlen 16 --soloUMIstart 17 --soloUMIlen $UMILEN --soloBarcodeReadLength 0 "
            "--soloStrand $STRAND --soloFeatures Gene GeneFull Velocyto SJ --soloCellFilter EmptyDrops_CR "
            "--soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR "
            "--clipAdapterType CellRanger4 --outFilterScoreMin 30 --outSAMtype None "
            "(the STARsolo manual's Cell Ranger 4+ emulation; $UMILEN 12 for v3, 10 for v2 and 5'; $STRAND Forward, Reverse for 5')")
AF_CMD = ("simpleaf quant --reads1 $R1 --reads2 $R2 --index af_idx/index --t2g-map af_idx/index/t2g_3col.tsv --chemistry "
          "$CHEM(10xv3|10xv2|10xv2-5p) --expected-ori fw --resolution cr-like --unfiltered-pl $WHITELIST --threads 16; "
          "index: simpleaf index --ref-type spliced+unspliced --rlen 91 on the same FASTA and GTF")
SINGLET_CMD = ("singlet encode --reads R1.fastq.gz R2.fastq.gz -o s.1fq; singlet s.1fq --ref-base $REF --genome-dir "
               "$REF/$SPECIES/star_2.7.11b --exons $REF/$SPECIES/genes/genes.gtf --out-prefix $OUT/ --threads 8 --pipeline "
               "--output-format mtx --raw-matrix")


def tools() -> list[dict]:
    return [
        C.tool("singlet", "1.0.0", "singlet", SINGLET_CMD, "https://singlet.bio"),
        C.tool("starsolo-gene", "2.7.11b", "competitor", STAR_CMD, "https://github.com/alexdobin/STAR",
               "Kaminow, Yunusov, Dobin. STARsolo: accurate, fast and versatile mapping/quantification of single-cell and single-nucleus RNA-seq data. bioRxiv 2021"),
        C.tool("starsolo-genefull", "2.7.11b", "competitor", "same STARsolo run as starsolo-gene; feature GeneFull (exons + introns)",
               "https://github.com/alexdobin/STAR"),
        C.tool("alevin-fry-usa", "0.18.3 (simpleaf 0.30)", "competitor", AF_CMD, "https://github.com/COMBINE-lab/alevin-fry",
               "He et al. Alevin-fry unlocks rapid, accurate and memory-frugal quantification of single-cell RNA-seq data. Nat Methods 2022"),
        C.tool("cellranger-published", "2.1.0 / 3.0.0 (as published by 10x)", "competitor",
               "not re-run: the filtered_feature_bc_matrix 10x publishes with each dataset (exonic UMIs; 10x reference of the time, "
               "not 2024-A). Cell Ranger 8 on the 2024-A reference needs the 10x EULA, which is pending.",
               "https://www.10xgenomics.com/support/software/cell-ranger"),
    ]


def _load(artifact_dir: Path):
    ids = [d for d in C.DATASET_ORDER if (artifact_dir / f"{d}.cells.tsv.gz").exists()]
    cells = {d: C.read_table(artifact_dir / f"{d}.cells.tsv.gz", index_col=0) for d in ids}
    genes = {d: C.read_table(artifact_dir / f"{d}.genes.tsv.gz", index_col=0) for d in ids}
    res = json.loads((artifact_dir / "resources.json").read_text()) if (artifact_dir / "resources.json").exists() else {}
    return ids, cells, genes, res


def compute(artifact_dir: Path) -> dict:
    artifact_dir = Path(artifact_dir)
    ids, cells, genes, res = _load(artifact_dir)
    stats = []
    for k, d in enumerate(ids):
        for spec in SPECS:
            s = F.collect(d, spec, cells[d], genes[d], k)
            if s is not None:
                stats.append(s)
    metrics = C.stamp(F.build_metrics(stats))
    resources = C.resource_rows(res, {"singlet": "singlet", "starsolo": "starsolo-gene", "alevinfry": "alevin-fry-usa"})
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAP,
        "title": "Gene counts per cell",
        "methods": _methods(),
        "status": "draft", "generated_at": C.now(), "owner": C.OWNER, "singlet": C.singlet_block(),
        "tools": tools(),
        "datasets": C.dataset_entries(ids, {d: TRUTH for d in ids}, {d: len(cells[d]) for d in ids}, NOTES),
        "metrics": metrics, "resources": resources,
        "figures": _figures(ids, stats, cells),
        "artifacts": C.artifact_entries(artifact_dir, CAP, {}),
        "code": C.code_entries({
            "bench/public/capabilities/gex_counts.py": "metric", "bench/public/capabilities/_familycmp.py": "metric",
            "bench/public/capabilities/_pubcommon.py": "metric", "bench/public/capabilities/extract/counts.py": "harness",
            "bench/public/capabilities/extract/pairstats.py": "harness",
            "bench/public/capabilities/extract/runio.py": "harness", "bench/public/capabilities/extract/run_extract.py": "harness",
            "bench/public/capabilities/clipper/starsolo_public.sbatch": "harness",
            "bench/public/capabilities/clipper/singlet_public.sbatch": "harness"}),
        "reproduce": {"module": "bench/public/capabilities/gex_counts.py", "function": "compute", "laptop_minutes": 3,
                      "clipper_command": "bash bench/public/capabilities/clipper/run_all.sh all"},
        "caveats": _caveats(),
    }
    doc["conclusion"] = _conclusion(stats, metrics, len(ids))
    return C.finish(doc)


def _rng(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return (min(v), max(v)) if v else (float("nan"), float("nan"))


def _pick(stats, fam, a, b, attr="raw_ratio"):
    out = []
    for s in stats:
        if s.spec.fam != fam:
            continue
        d = getattr(s, attr)
        if attr == "raw_ratio":
            if (a, b) in d:
                out.append(d[(a, b)])
        else:
            p = d.get((a, b)) or d.get((b, a))
            if p is not None:
                out.append(p.value)
    return out


def _conclusion(stats, metrics, n_ds) -> str:
    g = _rng(_pick(stats, "exonic", "singlet_exon", "starsolo_gene")), _rng(_pick(stats, "exonic", "singlet_exon", "starsolo_gene", "corr"))
    t = _rng(_pick(stats, "total", "singlet_total", "starsolo_genefull")), _rng(_pick(stats, "total", "singlet_total", "starsolo_genefull", "corr"))
    a = _rng(_pick(stats, "total", "singlet_total", "alevinfry_total"))
    gating = [m for m in metrics if m["rule"]["type"] != "informational"]
    n_pass = sum(m["pass"] == "pass" for m in gating)
    s = (f"On {n_ds} datasets (3' v2 and v3, 5', nuclei, mouse, barnyard), Singlet's exonic UMI per cell is {g[0][0]:.2f}-{g[0][1]:.2f} of STARsolo Gene's "
         f"(median ratio; gene correlation {g[1][0]:.3f}-{g[1][1]:.3f}). Exon+intron counts are {t[0][0]:.2f}-{t[0][1]:.2f} of STARsolo GeneFull's "
         f"(r {t[1][0]:.3f}-{t[1][1]:.3f}) and {a[0]:.2f}-{a[1]:.2f} of alevin-fry USA's. {n_pass} of {len(gating)} gating metrics pass: Singlet sits "
         f"inside the spread of the competitors")
    failing = [m for m in gating if m["pass"] == "fail"]
    return s + (f", except {len(failing)} metrics (see metrics)." if failing else ".")


def _methods() -> str:
    return (
        "**Data.** The cohort's 10x datasets, each from its FASTQs on its own 2024-A reference (GRCh38, GRCm39 or the human+mouse barnyard): "
        "PBMC 3' v3 and v2, mouse neurons and nuclei, the barnyard, PBMC with antibodies, and 5' PBMC (first 60 M read pairs). The same "
        "FASTQs and GTF feed Singlet, STARsolo and alevin-fry.\n\n"
        "**Cells and genes.** Cells are the barcodes STARsolo EmptyDrops_CR calls, so the cell set does not depend on Singlet. A family is "
        "compared on the cells all its tools have and the genes all of them annotate (Ensembl ids).\n\n"
        "**Families.** Exonic UMI: Singlet exon_counts (spliced + ambiguous), STARsolo Gene, and the Cell Ranger matrix 10x published. "
        "Exon+intron UMI: Singlet exon+intron, STARsolo GeneFull, alevin-fry USA (S+U+A).\n\n"
        "**Measures, for every pair of tools.** The distance of the median per-cell UMI ratio from 1 (|ln ratio|); the median over cells of "
        "the Pearson r of log1p counts over genes; the Spearman rho of pseudobulk gene counts.\n\n"
        "**Gate.** Singlet's mean value against its competitors must be as good as the competitors' value against each other, within the "
        "half-width of the paired bootstrap 95% CI of the difference. Singlet is gated as a member of the cloud of existing tools, not as "
        "better than each. Per-reference values are in each metric's notes. With one competitor (5' Cell Ranger) the threshold comes from the other datasets.\n\n"
        "**Not compared.** scBaseCount: no cohort dataset is in SRA, so no equal-depth comparison exists."
    )


def _caveats() -> list[str]:
    return [
        "Cell Ranger is the matrix 10x published with each dataset (versions 2.1.0 and 3.0.0, 10x references of the time, exonic UMIs), not a re-run on the 2024-A reference, so its comparisons also carry reference and annotation differences. Cell Ranger 8 is pending the 10x EULA.",
        "The 5' dataset is the first 60 M read pairs, so the published full-depth Cell Ranger matrix is left out for it; that dataset's exonic gate is a threshold borrowed from the other datasets, not a measured spread.",
        "There is no ground truth for counts: every metric is agreement between tools, and the tolerance is the spread between the competitors. A tool can sit inside that spread and still be wrong where the competitors are.",
        "The gate averages Singlet's distance to its competitors. It is looser than requiring Singlet to be at least as close to each competitor separately; the per-reference values are in the notes of each metric. Tool settings were fixed before any metric was computed and no Singlet parameter was tuned on these data.",
        "Singlet's exon+intron definition counts reads that STARsolo GeneFull discards (reads overlapping two gene bodies), so totals above 1 against GeneFull are expected.",
        "Wall time and memory in resources are from a shared cluster at differing thread counts (Singlet 8, STARsolo and alevin-fry 16) and are not a speed comparison.",
    ]


def _figures(ids, stats, cells):
    by = {(s.ds, s.spec.fam): s for s in stats}

    def ser(fam, a, b, name, tool, attr="raw_ratio"):
        xs, ys = [], []
        for d in ids:
            s = by.get((d, fam))
            if s is None:
                continue
            if attr == "raw_ratio":
                v = s.raw_ratio.get((a, b))
            else:
                p = getattr(s, attr).get((a, b)) or getattr(s, attr).get((b, a))
                v = p.value if p else None
            if v is not None:
                xs.append(d)
                ys.append(C.rnd(v))
        return {"name": name, "tool": tool, "x": xs, "y": ys}
    figs = [
        {"id": "ratio_by_dataset", "title": "Median per-cell UMI ratio", "kind": "grouped_bar",
         "caption": "Ratio of the first tool's per-cell UMI total to the second's, median over STARsolo-called cells. 1.0 is exact agreement.",
         "x": {"label": "dataset", "categories": ids}, "y": {"label": "ratio", "unit": "ratio", "scale": "linear"},
         "reference_lines": [{"axis": "y", "value": 1.0, "label": "equal"}],
         "series": [ser("exonic", "singlet_exon", "starsolo_gene", "Singlet exonic / STARsolo Gene", "singlet"),
                    ser("total", "singlet_total", "starsolo_genefull", "Singlet exon+intron / STARsolo GeneFull", "singlet"),
                    ser("total", "singlet_total", "alevinfry_total", "Singlet exon+intron / alevin-fry USA", "singlet"),
                    ser("total", "alevinfry_total", "starsolo_genefull", "alevin-fry USA / STARsolo GeneFull", "alevin-fry-usa"),
                    ser("exonic", "starsolo_gene", "cellranger", "STARsolo Gene / Cell Ranger", "starsolo-gene")]},
        {"id": "gene_corr_by_dataset", "title": "Median per-cell gene correlation", "kind": "grouped_bar",
         "caption": "Pearson r of log1p UMI over shared genes, median over cells.",
         "x": {"label": "dataset", "categories": ids}, "y": {"label": "r", "unit": "r", "scale": "linear", "min": 0.8, "max": 1},
         "series": [ser("exonic", "singlet_exon", "starsolo_gene", "Singlet exonic vs STARsolo Gene", "singlet", "corr"),
                    ser("total", "singlet_total", "starsolo_genefull", "Singlet exon+intron vs STARsolo GeneFull", "singlet", "corr"),
                    ser("total", "alevinfry_total", "starsolo_genefull", "alevin-fry USA vs STARsolo GeneFull", "alevin-fry-usa", "corr"),
                    ser("exonic", "starsolo_gene", "cellranger", "STARsolo Gene vs Cell Ranger", "starsolo-gene", "corr")]},
    ]
    d0 = ids[0] if ids else None
    if d0 and "total:n:singlet_total" in cells[d0].columns:
        c = cells[d0]
        x, y = c["total:n:starsolo_genefull"], c["total:n:singlet_total"]
        ok = x.notna() & y.notna() & (x > 0) & (y > 0)
        figs.append({"id": f"cell_totals_{d0}", "title": f"Per-cell exon+intron UMI, {d0}", "kind": "scatter",
                     "caption": "Each point is a STARsolo-called cell.",
                     "x": {"label": "STARsolo GeneFull UMI", "unit": "UMI", "scale": "log"},
                     "y": {"label": "Singlet exon+intron UMI", "unit": "UMI", "scale": "log"},
                     "series": [{"name": "cells", "tool": "singlet", "x": [int(v) for v in x[ok].to_numpy()[:4000]],
                                 "y": [int(v) for v in y[ok].to_numpy()[:4000]]}]})
    return figs


if __name__ == "__main__":
    C.cli(compute, CAP)
