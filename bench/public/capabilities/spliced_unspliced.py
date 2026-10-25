# SPDX-License-Identifier: MIT
"""spliced-unspliced: Singlet's unspliced and spliced+ambiguous layers against STARsolo Velocyto and alevin-fry USA.

    python bench/public/capabilities/spliced_unspliced.py ARTIFACT_DIR [--out bench/public/results/spliced-unspliced.json]

Why the gate is on *unspliced* and on *spliced + ambiguous*, not on spliced and ambiguous separately: the three tools draw the
spliced / ambiguous line differently by definition. STARsolo Velocyto and velocyto label a molecule against individual
transcripts, so an exonic read that is also compatible with an intron-retaining isoform makes it ambiguous. Singlet uses the
collapsed gene model (exons = union of the gene's exons, introns = gene body minus exons), where that read is spliced.
alevin-fry USA calls ambiguous every read compatible with both a spliced and an unspliced target. Unspliced molecules (reads
touching an intron of the gene) mean the same in all three, and spliced + ambiguous is everything else, so those two layers are
comparable. The ambiguous share per tool is reported (informational metric for Singlet, figure for all three) because it
explains the differences, not because anything is wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _familycmp as F  # noqa: E402
import _pubcommon as C  # noqa: E402

CAP = "spliced-unspliced"
LABEL = {"singlet_U": "Singlet unspliced", "starsolo_U": "STARsolo Velocyto unspliced", "alevinfry_U": "alevin-fry USA unspliced",
         "singlet_SA": "Singlet spliced+ambiguous", "starsolo_SA": "STARsolo Velocyto spliced+ambiguous",
         "alevinfry_SA": "alevin-fry USA spliced+ambiguous"}
DECL = {"singlet_U": "singlet", "singlet_SA": "singlet", "starsolo_U": "starsolo-velocyto", "starsolo_SA": "starsolo-velocyto",
        "alevinfry_U": "alevin-fry-usa", "alevinfry_SA": "alevin-fry-usa"}
SPECS = [
    F.Spec("unspliced", "singlet_U", LABEL, DECL, "unspliced UMI", "unspliced"),
    F.Spec("spliced_ambiguous", "singlet_SA", LABEL, DECL, "spliced+ambiguous UMI",
           "spliced_ambiguous"),
]
TRUTH = ("none. Agreement between independent tools run on the same FASTQs and reference. Unspliced and spliced+ambiguous are "
         "gated; spliced and ambiguous alone differ by definition and are reported.")

STAR_CMD = ("STAR --runThreadN 16 --genomeDir $GENOME --readFilesIn $R2 $R1 --readFilesCommand zcat --soloType CB_UMI_Simple "
            "--soloCBwhitelist $WHITELIST --soloCBlen 16 --soloUMIstart 17 --soloUMIlen $UMILEN --soloBarcodeReadLength 0 "
            "--soloStrand $STRAND --soloFeatures Gene GeneFull Velocyto SJ --soloCellFilter EmptyDrops_CR "
            "--soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR "
            "--clipAdapterType CellRanger4 --outFilterScoreMin 30 --outSAMtype None (Velocyto layers: spliced, unspliced, ambiguous)")
AF_CMD = ("simpleaf quant --reads1 $R1 --reads2 $R2 --index af_idx/index --t2g-map af_idx/index/t2g_3col.tsv --chemistry "
          "$CHEM --expected-ori fw --resolution cr-like --unfiltered-pl $WHITELIST --threads 16 (USA matrix: S, U, A blocks)")
SINGLET_CMD = ("singlet encode --reads R1.fastq.gz R2.fastq.gz -o s.1fq; singlet s.1fq --ref-base $REF --genome-dir "
               "$REF/$SPECIES/star_2.7.11b --exons $REF/$SPECIES/genes/genes.gtf --out-prefix $OUT/ --threads 8 --pipeline "
               "--output-format mtx --raw-matrix  (spliced = exon_counts - ambiguous_counts, unspliced = intron_counts)")


def tools() -> list[dict]:
    return [
        C.tool("singlet", "1.0.0", "singlet", SINGLET_CMD, "https://singlet.bio"),
        C.tool("starsolo-velocyto", "2.7.11b", "competitor", STAR_CMD, "https://github.com/alexdobin/STAR",
               "La Manno et al. RNA velocity of single cells. Nature 2018 (velocyto rules); Kaminow et al. bioRxiv 2021 (STARsolo)"),
        C.tool("alevin-fry-usa", "0.18.3 (simpleaf 0.30)", "competitor", AF_CMD, "https://github.com/COMBINE-lab/alevin-fry",
               "He et al. Nat Methods 2022"),
    ]


def _load(artifact_dir: Path):
    ids = [d for d in C.DATASET_ORDER if (artifact_dir / f"{d}.cells.tsv.gz").exists()]
    cells = {d: C.read_table(artifact_dir / f"{d}.cells.tsv.gz", index_col=0) for d in ids}
    genes = {d: C.read_table(artifact_dir / f"{d}.genes.tsv.gz", index_col=0) for d in ids}
    res = json.loads((artifact_dir / "resources.json").read_text()) if (artifact_dir / "resources.json").exists() else {}
    return ids, cells, genes, res


def shares(df) -> dict:
    """Pooled S / U / A share per tool over the cells every tool has."""
    out = {}
    for t in ("singlet", "starsolo", "alevinfry"):
        tot = {k: float(np.nansum(df[f"layers:n:{t}_{k}"].to_numpy(float))) for k in ("S", "U", "A")}
        n = sum(tot.values())
        out[t] = {k: v / n for k, v in tot.items()} if n else {}
    return out


def compute(artifact_dir: Path) -> dict:
    artifact_dir = Path(artifact_dir)
    ids, cells, genes, res = _load(artifact_dir)
    stats = []
    for k, d in enumerate(ids):
        for spec in SPECS:
            st = F.collect(d, spec, cells[d], genes[d], k)
            if st is not None:
                stats.append(st)
    metrics = C.stamp(F.build_metrics(stats))
    sh = {d: shares(cells[d]) for d in ids}
    for d in ids:
        s = sh[d]
        metrics.append(C.metric(
            "ambiguous_share", "Ambiguous share of molecules, Singlet", "fraction", "agreement", d, "none", s["singlet"].get("A"),
            None, [], {"type": "informational"},
            notes=f"STARsolo Velocyto {s['starsolo'].get('A', float('nan')):.3f}, alevin-fry USA {s['alevinfry'].get('A', float('nan')):.3f}: "
                  "the spliced / ambiguous boundary is defined differently by each tool (see methods); not gated."))
    resources = C.resource_rows(res, {"singlet": "singlet", "starsolo": "starsolo-velocyto", "alevinfry": "alevin-fry-usa"})
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAP,
        "title": "Spliced and unspliced layers",
        "methods": _methods(),
        "status": "draft", "generated_at": C.now(), "owner": C.OWNER, "singlet": C.singlet_block(),
        "tools": tools(),
        "datasets": C.dataset_entries(ids, {d: TRUTH for d in ids}, {d: len(cells[d]) for d in ids}),
        "metrics": metrics, "resources": resources, "figures": _figures(ids, sh),
        "artifacts": C.artifact_entries(artifact_dir, CAP, {}),
        "code": C.code_entries({
            "bench/public/capabilities/spliced_unspliced.py": "metric", "bench/public/capabilities/_familycmp.py": "metric",
            "bench/public/capabilities/_pubcommon.py": "metric", "bench/public/capabilities/extract/counts.py": "harness", "bench/public/capabilities/extract/pairstats.py": "harness",
            "bench/public/capabilities/extract/runio.py": "harness", "bench/public/capabilities/extract/run_extract.py": "harness",
            "bench/public/capabilities/clipper/starsolo_public.sbatch": "harness",
            "bench/public/capabilities/clipper/singlet_public.sbatch": "harness"}),
        "reproduce": {"module": "bench/public/capabilities/spliced_unspliced.py", "function": "compute", "laptop_minutes": 2,
                      "clipper_command": "bash bench/public/capabilities/clipper/run_all.sh spliced-unspliced"},
        "caveats": _caveats(),
    }
    doc["conclusion"] = _conclusion(stats, metrics, sh, len(ids))
    return C.finish(doc)


def _conclusion(stats, metrics, sh, n_ds) -> str:
    def rng(fam, a, b):
        v = [s.raw_ratio[(a, b)] for s in stats if s.spec.fam == fam and (a, b) in s.raw_ratio]
        return (min(v), max(v)) if v else (float("nan"), float("nan"))
    u1, u2 = rng("unspliced", "singlet_U", "starsolo_U"), rng("unspliced", "singlet_U", "alevinfry_U")
    sa1 = rng("spliced_ambiguous", "singlet_SA", "starsolo_SA")
    gating = [m for m in metrics if m["rule"]["type"] != "informational"]
    n_pass = sum(m["pass"] == "pass" for m in gating)
    a = {t: (min(sh[d][t].get("A", np.nan) for d in sh), max(sh[d][t].get("A", np.nan) for d in sh)) for t in ("singlet", "starsolo", "alevinfry")}
    return (f"On {n_ds} datasets Singlet's unspliced UMI per cell is {u1[0]:.2f}-{u1[1]:.2f} of STARsolo Velocyto's and "
            f"{u2[0]:.2f}-{u2[1]:.2f} of alevin-fry USA's (median ratio), and its spliced+ambiguous UMI {sa1[0]:.2f}-{sa1[1]:.2f} of STARsolo's; "
            f"{n_pass} of {len(gating)} gating metrics pass (Singlet inside the spread of the two competitors). "
            f"Spliced and ambiguous alone are defined differently by each tool (ambiguous share: Singlet {a['singlet'][0]:.1%}-{a['singlet'][1]:.1%}, "
            f"STARsolo {a['starsolo'][0]:.1%}-{a['starsolo'][1]:.1%}, alevin-fry {a['alevinfry'][0]:.1%}-{a['alevinfry'][1]:.1%}) and are not gated.")


def _methods() -> str:
    return (
        "**Data and cells.** The cohort's 10x datasets, each run from its FASTQs on its own 2024-A reference (see gex-counts). Cells "
        "are the barcodes STARsolo EmptyDrops_CR calls; a comparison uses the cells all three tools have and their shared genes.\n\n"
        "**Layers.** Singlet: unspliced = intron_counts, spliced + ambiguous = exon_counts (spliced = exon_counts - ambiguous_counts). "
        "STARsolo Velocyto: unspliced, and spliced + ambiguous. alevin-fry USA: U, and S + A.\n\n"
        "**Why spliced+ambiguous.** STARsolo labels a molecule against individual transcripts, so exonic reads compatible with an "
        "intron-retaining isoform become ambiguous; Singlet uses the collapsed gene model and calls them spliced; alevin-fry calls "
        "ambiguous every read compatible with both targets. Unspliced means the same in all three, and spliced + ambiguous is the "
        "rest of the molecules, so those two layers are gated. The ambiguous share of each tool is reported.\n\n"
        "**Measures, for every pair of tools.** The distance of the median per-cell UMI ratio from 1 (|ln ratio|); the median over cells of "
        "the Pearson r of log1p counts over genes; the Spearman rho of pseudobulk gene counts.\n\n"
        "**Gate.** Singlet's mean value against the two competitors (STARsolo Velocyto and alevin-fry USA) must be as good as the "
        "competitors' value against each other, within the half-width of the paired bootstrap 95% CI of the difference: Singlet is gated "
        "as a member of the cloud of existing tools. Per-reference values are in each metric's notes. All settings were fixed before the "
        "metrics were computed."
    )


def _caveats() -> list[str]:
    return [
        "Spliced-only and ambiguous-only layers are not gated: the three tools define the spliced / ambiguous boundary differently, so agreement there would measure definitions, not accuracy.",
        "There is no ground truth for velocity layers; agreement between STARsolo and alevin-fry sets the tolerance, so a metric where the two competitors happen to agree closely is a strict test.",
        "Singlet's collapsed gene model reports far fewer ambiguous molecules than STARsolo and alevin-fry; RNA velocity users who use only the spliced layer will see a different spliced / unspliced ratio.",
        "The 5' dataset is the first 60 M read pairs; alevin-fry used expected orientation fw (simpleaf's own default for 10xv2-5p).",
        "Wall time and memory in resources are from a shared cluster at differing thread counts and are not a speed comparison.",
    ]


def _figures(ids, sh):
    cats = ["singlet", "starsolo", "alevinfry"]
    names = {"singlet": "Singlet", "starsolo": "STARsolo Velocyto", "alevinfry": "alevin-fry USA"}
    toolkey = {"singlet": "singlet", "starsolo": "starsolo-velocyto", "alevinfry": "alevin-fry-usa"}
    figs = []
    for layer, title in (("A", "Ambiguous share of molecules"), ("U", "Unspliced share of molecules")):
        figs.append({"id": f"{'ambiguous' if layer == 'A' else 'unspliced'}_share", "title": title, "kind": "grouped_bar",
                     "caption": "Pooled over the common cells. The spliced / ambiguous line is drawn differently by each tool." if layer == "A" else "Pooled over the common cells.",
                     "x": {"label": "dataset", "categories": ids}, "y": {"label": "share of molecules", "unit": "fraction", "scale": "linear", "min": 0, "max": 1},
                     "series": [{"name": names[t], "tool": toolkey[t], "x": ids, "y": [C.rnd(sh[d][t].get(layer)) for d in ids]} for t in cats]})
    return figs


if __name__ == "__main__":
    C.cli(compute, CAP)
