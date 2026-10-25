# SPDX-License-Identifier: MIT
"""cell-calling: Singlet's called_v2 against EmptyDrops_CR (STARsolo), CellBender and ground truth.

    python bench/public/capabilities/cell_calling.py ARTIFACT_DIR [--out bench/public/results/cell-calling.json]

Truth. For the 10x datasets: the filtered barcodes 10x publishes with the dataset (Cell Ranger 2.1 / 3.0; EmptyDrops-like or
OrdMag cutoffs, so a barcode beyond the truth set is not necessarily wrong). For the Kang 2018 control lane: the authors' demuxlet
table. For the barnyard (hgmm1k_v3): additionally the species composition of every called barcode (does it look like ambient?).
Competitors: STARsolo EmptyDrops_CR and CellBender 0.3.2 (neither knows the truth sets). Rules: match_or_beat the best competitor,
tolerance = half-width of the paired bootstrap 95% CI of the difference (barcodes resampled together).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pubcommon as C  # noqa: E402

CAP = "cell-calling"
CALLERS = {"singlet": "singlet", "starsolo_edcr": "starsolo-edcr", "cellbender": "cellbender"}
TRUTH_COL = {"kang2018_ctrl": "demuxlet"}
B = 300

STAR_CMD = ("STAR --runThreadN 16 --genomeDir $GENOME --readFilesIn $R2 $R1 --readFilesCommand zcat --soloType CB_UMI_Simple "
            "--soloCBwhitelist $WHITELIST --soloCBlen 16 --soloUMIstart 17 --soloUMIlen $UMILEN --soloBarcodeReadLength 0 --soloStrand $STRAND "
            "--soloFeatures Gene GeneFull Velocyto SJ --soloCellFilter EmptyDrops_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts "
            "--soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30 --outSAMtype None; "
            "calls = Solo.out/Gene/filtered (10x v1 for Kang: CB 14, UMI 10)")
CB_CMD = ("cellbender remove-background --cuda --input <STARsolo Solo.out/Gene/raw> --output cb.h5 --expected-cells $EXP "
          "--total-droplets-included $TOT --fpr 0.01 --epochs 150; EXP = barcodes at or above 1/10 of the 99th percentile of the top 3000 "
          "barcodes' UMI (Cell Ranger 2.2 rule from the raw curve), TOT = max(3 x EXP, 10000); calls = cb_cell_barcodes.csv")
SINGLET_CMD = ("singlet encode --reads R1.fastq.gz R2.fastq.gz -o s.1fq; singlet s.1fq --ref-base $REF --genome-dir $REF/$SPECIES/star_2.7.11b "
               "--exons $REF/$SPECIES/genes/genes.gtf --out-prefix $OUT/ --threads 8 --pipeline --output-format mtx --raw-matrix  "
               "(calls = called_v2 column of cell_calls.tsv: EmptyDrops on the stored counts, FDR 0.001, 10,000 draws)")


def tools(have_cb: bool) -> list[dict]:
    t = [
        C.tool("singlet", "1.0.0", "singlet", SINGLET_CMD, "https://singlet.bio"),
        C.tool("starsolo-edcr", "2.7.11b", "competitor", STAR_CMD, "https://github.com/alexdobin/STAR",
               "Lun et al. EmptyDrops: distinguishing cells from empty droplets in droplet-based single-cell RNA sequencing data. Genome Biol 2019 (EmptyDrops_CR is STARsolo's port of Cell Ranger's variant)"),
        C.tool("cellbender", "0.3.2", "competitor", CB_CMD, "https://github.com/broadinstitute/CellBender",
               "Fleming et al. Unsupervised removal of systematic background noise from droplet-based single-cell experiments using CellBender. Nat Methods 2023"),
        C.tool("cellranger-published", "2.1.0 / 3.0.0 (as published by 10x)", "truth",
               C.cellranger_params() + " (only the barcodes are used here)", "https://cf.10xgenomics.com/samples/"),
        C.tool("demuxlet-table", "GSE96583", "truth", "the authors' table GSE96583_batch2.total.tsne.df.tsv.gz, rows of the control lane (all demuxlet singlet, doublet and ambiguous calls)",
               "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583", "Kang et al. Multiplexed droplet single-cell RNA-sequencing using natural genetic variation. Nat Biotechnol 2018"),
    ]
    return t


def _load(artifact_dir: Path):
    ids = [d for d in C.DATASET_ORDER if (artifact_dir / f"{d}.barcodes.tsv.gz").exists()]
    tabs = {d: C.read_table(artifact_dir / f"{d}.barcodes.tsv.gz", index_col=0) for d in ids}
    res = json.loads((artifact_dir / "resources.json").read_text()) if (artifact_dir / "resources.json").exists() else {}
    v42 = json.loads((artifact_dir / "validation42.json").read_text()) if (artifact_dir / "validation42.json").exists() else None
    return ids, tabs, res, v42


def _prf(a: np.ndarray, t: np.ndarray, idx: np.ndarray | None = None):
    if idx is not None:
        a, t = a[idx], t[idx]
    tp = (a & t).sum(axis=-1)
    na, nt, nu = a.sum(axis=-1), t.sum(axis=-1), (a | t).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / nt, tp / na, tp / nu  # recall, precision, jaccard


def metrics_for(ds: str, k: int, df: pd.DataFrame) -> list[dict]:
    tcol = TRUTH_COL.get(ds, "cellranger")
    T = df[tcol].to_numpy().astype(bool)
    calls = {c: df[c].to_numpy().astype(bool) for c in CALLERS if c in df.columns}
    comps = [c for c in calls if c != "singlet"]
    idx = C.boot_indices(len(df), offset=k * 7 + 1, B=B)
    names = (("recall", "Recall of the truth cells"), ("precision", "Precision against the truth cells"), ("jaccard", "Jaccard with the truth cells"))
    full = {c: _prf(calls[c], T) for c in calls}
    boot = {c: _prf(calls[c], T, idx) for c in calls}
    out = []
    truth_label = "demuxlet table (control lane)" if tcol == "demuxlet" else "Cell Ranger filtered barcodes published by 10x"
    for j, (mid, nm) in enumerate(names):
        best = max(comps, key=lambda c: full[c][j])
        hw = C.half_width(C.ci95(boot["singlet"][j] - boot[best][j]))
        rule = C.match_rule(hw, f"tolerance = half-width of the paired bootstrap 95% CI (barcodes resampled) of Singlet minus the best competitor "
                                f"({CALLERS[best]}): {hw:.4f}")
        out.append(C.metric(mid, nm, "fraction", "accuracy", ds, "higher", full["singlet"][j], C.ci95(boot["singlet"][j]),
                            [{"tool": CALLERS[c], "value": full[c][j], "ci": C.ci95(boot[c][j])} for c in comps], rule,
                            notes=f"truth: {truth_label}, {int(T.sum())} barcodes; {int(calls['singlet'].sum())} called by Singlet, " +
                                  ", ".join(f"{int(calls[c].sum())} by {CALLERS[c]}" for c in comps)))
    # barnyard: share of called barcodes whose species mix looks like ambient
    if "species_class" in df.columns:
        amb = (df["species_class"] == "ambient_like").to_numpy()
        each = {c: calls[c] for c in calls}
        if "cellranger" in df.columns:
            each["cellranger"] = df["cellranger"].to_numpy().astype(bool)

        def frac(a, i=None):
            a2, m2 = (a, amb) if i is None else (a[i], amb[i])
            n = a2.sum(axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return (a2 & m2).sum(axis=-1) / n
        fullf = {c: frac(each[c]) for c in each}
        bootf = {c: frac(each[c], idx) for c in each}
        others = [c for c in each if c != "singlet"]
        best = min(others, key=lambda c: fullf[c])
        hw = C.half_width(C.ci95(bootf["singlet"] - bootf[best]))
        rule = C.match_rule(hw, f"tolerance = half-width of the paired bootstrap 95% CI (barcodes resampled) of Singlet minus the best competitor "
                                f"({CALLERS.get(best, 'cellranger-published')}): {hw:.4f}")
        out.append(C.metric("ambient_like_fraction", "Called barcodes whose species mix is ambient-like", "fraction", "accuracy", ds, "lower",
                            fullf["singlet"], C.ci95(bootf["singlet"]),
                            [{"tool": CALLERS.get(c, "cellranger-published"), "value": fullf[c], "ci": C.ci95(bootf[c])} for c in others], rule,
                            notes="barnyard (human + mouse): a called barcode is ambient-like when its human fraction is not significantly different (binomial, p >= 0.001) "
                                  "from that of uncalled barcodes with 1-100 UMI; in a mixed-species library such a barcode is most likely an empty droplet"))
    out.append(C.metric("n_called", "Barcodes called by Singlet", "barcodes", "coverage", ds, "none", int(calls["singlet"].sum()), None, [],
                        {"type": "informational"},
                        notes=f"truth {int(T.sum())}; " + ", ".join(f"{CALLERS[c]} {int(calls[c].sum())}" for c in comps)))
    return out


def validation_metrics(v: dict) -> list[dict]:
    notes = (f"{v['n_usable']} usable samples of {v['n_samples']} (18 public studies); truth is the authors' own Cell Ranger calls, available for "
             f"{v['n_author_samples']} samples in 9 studies; studies weighted equally; Singlet's called_v2 (bounded knee, tail p-values, FDR 0.001) re-scored "
             "by the production kernel. Not run on the campaign binary; see docs/called-v2-truth.md. Reference points from docs/called-v2.md: stored knee call "
             "J = 0.55, Cell Ranger 3-like on stored data J = 0.70.")
    return [C.metric(f"validation42_{x}", f"{nm} against the authors' Cell Ranger calls, 42 public samples", "fraction", "accuracy", "pooled", "none",
                     v[x], None, [], {"type": "informational"}, notes=notes)
            for x, nm in (("jaccard", "Jaccard"), ("recall", "Recall"), ("precision", "Precision"))]


def compute(artifact_dir: Path) -> dict:
    artifact_dir = Path(artifact_dir)
    ids, tabs, res, v42 = _load(artifact_dir)
    metrics = []
    for k, d in enumerate(ids):
        metrics += metrics_for(d, k, tabs[d])
    if v42:
        metrics += validation_metrics(v42)
    resources = C.resource_rows(res, {"singlet": "singlet", "starsolo": "starsolo-edcr", "cellbender": "cellbender"})
    have_cb = any("cellbender" in t.columns for t in tabs.values())
    truths = {d: ("the authors' demuxlet table for the control lane (GSE96583, 14,619 barcodes)" if d == "kang2018_ctrl" else
                  "the filtered barcodes 10x published with the dataset (Cell Ranger), a cutoff-based call" +
                  ("; barnyard species composition for ambient-like calls" if d == "hgmm1k_v3" else "")) for d in ids}
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAP,
        "title": "Cell calling",
        "methods": _methods(),
        "status": "draft", "generated_at": C.now(), "owner": C.OWNER, "singlet": C.singlet_block(),
        "tools": tools(have_cb),
        "datasets": C.dataset_entries(ids, truths, {d: int(tabs[d]["singlet"].sum()) for d in ids}),
        "metrics": metrics, "resources": resources, "figures": _figures(ids, metrics),
        "artifacts": C.artifact_entries(artifact_dir, CAP, {"validation42.json": "truth"}),
        "code": C.code_entries({
            "bench/public/capabilities/cell_calling.py": "metric", "bench/public/capabilities/_pubcommon.py": "metric",
            "bench/public/capabilities/extract/cellcalling.py": "harness", "bench/public/capabilities/extract/runio.py": "harness",
            "bench/public/capabilities/extract/run_extract.py": "harness",
            "bench/public/capabilities/clipper/starsolo_public.sbatch": "harness",
            "bench/public/capabilities/clipper/singlet_public.sbatch": "harness",
            "bench/public/capabilities/clipper/cellbender_public.sbatch": "harness",
            "include/singlet/pileup/called_v2.h": "implementation"}),
        "reproduce": {"module": "bench/public/capabilities/cell_calling.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "bash bench/public/capabilities/clipper/run_all.sh cell-calling"},
        "caveats": _caveats(ids, tabs) + C.cohort_gap_caveat(CAP, ids),
    }
    C.stamp(metrics)
    doc["conclusion"] = _conclusion(metrics, len(ids))
    return C.finish(doc)


def _conclusion(metrics, n_ds) -> str:
    def rng(mid):
        v = [m["singlet"]["value"] for m in metrics if m["id"] == mid and m["dataset"] != "pooled"]
        return (min(v), max(v)) if v else (float("nan"), float("nan"))
    j, r, p = rng("jaccard"), rng("recall"), rng("precision")
    gating = [m for m in metrics if m["rule"]["type"] != "informational"]
    n_pass = sum(m["pass"] == "pass" for m in gating)
    loses = sorted({f"{m['id']} on {m['dataset']}" for m in gating if m["pass"] == "fail"})
    s = (f"On {n_ds} datasets Singlet's called_v2 reaches Jaccard {j[0]:.2f}-{j[1]:.2f} with the truth set (recall {r[0]:.2f}-{r[1]:.2f}, "
         f"precision {p[0]:.2f}-{p[1]:.2f}), compared with STARsolo EmptyDrops_CR and CellBender; {n_pass} of {len(gating)} gating comparisons pass. "
         "The 10x truth sets come from Cell Ranger's own EmptyDrops-type call, which STARsolo's EmptyDrops_CR reproduces by construction; only the "
         "barnyard species check and the Kang demuxlet table are independent of that algorithm.")
    if loses:
        s += f" Singlet loses {len(loses)} comparisons (see metrics)."
    return s


def _methods() -> str:
    return (
        "**Data and truth.** The cohort's 10x datasets from FASTQs, plus the Kang 2018 control lane. Truth for 10x datasets is the filtered "
        "barcodes 10x published (Cell Ranger 2.1 / 3.0), for Kang the authors' demuxlet table. Both are cutoff-based, so a barcode beyond "
        "the truth is not necessarily wrong.\n\n"
        "**Calls.** Singlet: the called_v2 column (EmptyDrops on the stored exon+intron counts, plateau ambient bound, knee fitted within "
        "DropletUtils' curve bounds, FDR 0.001, 10,000 draws, union with the stored knee call). STARsolo 2.7.11b EmptyDrops_CR in Cell Ranger "
        "4+ emulation on the same reads. CellBender 0.3.2 on STARsolo's raw Gene matrix, expected cells and droplets read off the raw UMI "
        "curve (Cell Ranger 2.2 rule), other options at defaults.\n\n"
        "**Metrics.** Over the union of every tool's calls and the truth: recall, precision and Jaccard of each tool's calls against the truth. "
        "In the barnyard, also the share of each tool's called barcodes whose human fraction cannot be told from the ambient one.\n\n"
        "**Rules.** match_or_beat the best competitor on each metric; tolerance = half-width of the paired bootstrap 95% CI of the "
        "difference (barcodes resampled together). Precision and recall are both gated, so a method that trades one for the other will "
        "fail one of them.\n\n"
        "**Broader evidence.** The 42-sample validation of docs/called-v2.md (authors' calls in 9 public studies) is reported as an "
        "informational pooled metric; it was not run on the campaign binary."
    )


def _caveats(ids, tabs) -> list[str]:
    cv = [
        "called_v2's parameters (plateau ambient bound, knee bounds, tail p-values, FDR 0.001) were chosen and corrected while looking at pbmc1k_v3, hgmm1k_v3, neuron1k_v3, nuclei2k_v2, pbmc5p_v1 and the Kang lane (docs/called-v2-truth.md). Those datasets are development data for this method, so these scores are not held-out; the 42-sample validation is broader but was also used to choose among variants.",
        "The truth sets are themselves cutoff-based (Cell Ranger OrdMag / EmptyDrops-like, demuxlet on called cells). Barcodes called by Singlet beyond them are often real low-RNA cells (docs/called-v2-truth.md) and count against precision here.",
        "The Cell Ranger truth for each 10x dataset is the matrix 10x published, not a re-run on the 2024-A reference. It is itself the output of a cell-calling algorithm (EmptyDrops-like, or OrdMag for Cell Ranger 2), and STARsolo EmptyDrops_CR is a port of it, so its agreement with that truth is partly by construction. The independent real truths are the barnyard species composition (hgmm1k_v3) and the Kang demuxlet table (genotype-based).",
        "CellBender stopped with an encoder assertion (fewer than 4 cells in a minibatch) at its defaults and again at a lower learning rate on some datasets (see provenance.json); those datasets have no CellBender column and no CellBender competitor value.",
        "CellBender's expected-cells and total-droplets parameters were derived from the raw UMI curve with a fixed rule, not tuned; CellBender ran only on the datasets that have a cellbender column in the artifacts." if any("cellbender" in t.columns for t in tabs.values()) else "CellBender was not run on every dataset; where its column is absent, only STARsolo EmptyDrops_CR competes.",
        "Wall time and memory in resources are from a shared cluster at differing thread counts and are not a speed comparison.",
    ]
    return cv


def _figures(ids, metrics):
    figs = []
    for mid, title in (("jaccard", "Jaccard of called barcodes with the truth set"), ("recall", "Recall of the truth cells"),
                       ("precision", "Precision against the truth cells")):
        ser = {"singlet": ([], [])}
        for m in metrics:
            if m["id"] != mid or m["dataset"] == "pooled":
                continue
            ser["singlet"][0].append(m["dataset"])
            ser["singlet"][1].append(m["singlet"]["value"])
            for c in m["competitors"]:
                ser.setdefault(c["tool"], ([], []))
                ser[c["tool"]][0].append(m["dataset"])
                ser[c["tool"]][1].append(c["value"])
        figs.append({"id": f"{mid}_by_dataset", "title": title, "kind": "grouped_bar", "caption": "Higher is better.",
                     "x": {"label": "dataset", "categories": ids}, "y": {"label": mid, "unit": "fraction", "scale": "linear", "min": 0, "max": 1},
                     "series": [{"name": ("Singlet called_v2" if t == "singlet" else t), "tool": t, "x": xs, "y": ys} for t, (xs, ys) in ser.items()]})
    return figs


if __name__ == "__main__":
    C.cli(compute, CAP)
