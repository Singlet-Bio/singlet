# SPDX-License-Identifier: MIT
"""spliced-unspliced: Singlet's unspliced calls against long-read truth (primary), and against STARsolo Velocyto and alevin-fry USA (secondary).

Amendment (gate 3, after the long-read truth became available): the primary metric is the precision, recall and F1 of the
unspliced calls against the long-read truth (>= 10 aligned intronic bases of the gene, TRUTH-LR of expression-sota/longread_usa),
on PBMC-E (Singlet, STARsolo Velocyto, alevin-fry USA, match-or-beat the best) and on the MuSC dev cells (Singlet only,
reported). Agreement with Velocyto and alevin-fry USA (UMI ratio, correlation, rank correlation, below) is kept as secondary
and no longer gates, because two short-read tools cannot show which of them is right.

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


LR_TOOLS = {"singlet": ("Singlet", "singlet"), "starsolo": ("STARsolo Velocyto", "starsolo-velocyto"),
            "alevinfry": ("alevin-fry USA", "alevin-fry-usa")}
LR_STATS = (("precision", "Unspliced precision against long-read truth"), ("recall", "Unspliced recall against long-read truth"),
            ("f1", "Unspliced F1 against long-read truth"))
LR_BASIS = "(cell, gene) pairs with a long-read-truth molecule: does the cell have an unspliced molecule of the gene (truth: >= 10 intronic bases)"


def _longread_metrics(artifact_dir: Path) -> tuple[list[dict], dict, dict]:
    """Primary metrics from longread_usa.json: {"pbmc_e": score_usa.json, "musc_dev": score_musc.json}. Returns (metrics, n_cells, notes)."""
    p = artifact_dir / "longread_usa.json"
    if not p.exists():
        return [], {}, {}
    lr = json.loads(p.read_text())
    metrics, n_cells, notes = [], {}, {}
    pe = lr.get("pbmc_e")
    if pe:
        ds = "longread_pbmc_e"
        n_cells[ds] = int(pe.get("truth_cells", 0))
        notes[ds] = "PBMC-E: PacBio and ONT long reads of the same cells as the short reads; truth class per molecule = U if >= 10 aligned intronic bases."
        common = pe["clsLR"]["common"]
        val = {t: common[t]["truth_pairs_all"]["has_unspliced"] for t in LR_TOOLS if t in common}
        for key, name in LR_STATS:
            comps = [{"tool": LR_TOOLS[t][1], "value": val[t][key]} for t in ("starsolo", "alevinfry") if t in val]
            metrics.append(C.metric(f"unspliced_{key}_lr", name, "fraction", "accuracy", ds, "higher", val["singlet"][key], None, comps,
                                    C.match_rule(None, None),
                                    notes=f"Basis: {LR_BASIS}. Cells: the {n_cells[ds]} truth cells every tool reports. "
                                          + "; ".join(f"{LR_TOOLS[t][0]} {val[t][key]:.4f}" for t in val) + "."))
    mu = lr.get("musc_dev")
    if mu:
        ds = "longread_musc"
        n_cells[ds] = int(mu.get("truth_cells", 0))
        notes[ds] = (f"MuSC (mouse muscle stem cells, ONT long reads, 10x v2): {n_cells[ds]} dev cells. Recall is capped by the length of "
                     "the cDNA read (R2): an unspliced molecule whose intronic part is not on R2 cannot be seen from the short read.")
        st = mu["dev"]["mol"]
        for key, name in LR_STATS:
            metrics.append(C.metric(f"unspliced_{key}_lr", name, "fraction", "accuracy", ds, "higher", st[key], None, [],
                                    {"type": "informational"},
                                    notes="Molecule level (CB, UMI, gene) against the long-read truth, dev cells only; Singlet only (no other tool "
                                          "keeps a per-molecule label). Reported, not gated: 78 cells, no competitor."))
    return metrics, n_cells, notes


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
    secondary = F.build_metrics(stats)
    for m in secondary:   # agreement with the other short-read tools no longer gates
        m["rule"] = {"type": "informational"}
        m["competitors"] = []   # per-reference values stay in the notes
        m["notes"] = (m.get("notes", "") + " Secondary since the long-read amendment: agreement between short-read tools, not accuracy.").strip()
    lr_metrics, lr_n, lr_notes = _longread_metrics(artifact_dir)
    metrics = C.stamp(lr_metrics + secondary)
    sh = {d: shares(cells[d]) for d in ids}
    for d in ids:
        s = sh[d]
        metrics.append(C.metric(
            "ambiguous_share", "Ambiguous share of molecules, Singlet", "fraction", "agreement", d, "none", s["singlet"].get("A"),
            None, [], {"type": "informational"},
            notes=f"STARsolo Velocyto {s['starsolo'].get('A', float('nan')):.3f}, alevin-fry USA {s['alevinfry'].get('A', float('nan')):.3f}: "
                  "the spliced / ambiguous boundary is defined differently by each tool (see methods); not gated."))
    resources = C.resource_rows(res, {"singlet": "singlet", "starsolo": "starsolo-velocyto", "alevinfry": "alevin-fry-usa"}, ids)
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAP,
        "title": "Spliced and unspliced layers",
        "methods": _methods(),
        "status": "draft", "generated_at": C.now(), "owner": C.OWNER, "singlet": C.singlet_block(),
        "tools": tools(),
        "datasets": C.dataset_entries(ids + list(lr_n), {**{d: TRUTH for d in ids}, **{d: "long-read (PacBio/ONT) USA class per molecule" for d in lr_n}},
                                      {**{d: len(cells[d]) for d in ids}, **lr_n}, lr_notes),
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
        "caveats": _caveats() + C.cohort_gap_caveat(CAP, ids + list(lr_n)),
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
        "**Primary (amendment).** Long-read truth: for each molecule the class of its primary long-read alignment (U: >= 10 aligned intronic bases of "
        "the gene; S: not U and an annotated junction within 2 nt; A otherwise). Precision, recall and F1 of the call cell has an unspliced molecule "
        "of this gene over the (cell, gene) pairs the truth covers, on PBMC-E for Singlet, STARsolo Velocyto and alevin-fry USA (match-or-beat the "
        "best tool, strict), and at molecule level on the 78 MuSC dev cells for Singlet (reported). Recall is capped by R2 length."
        "\n\n"
        "**Secondary: agreement.** Singlet's mean value against the two competitors (STARsolo Velocyto and alevin-fry USA) must be as good as the "
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
