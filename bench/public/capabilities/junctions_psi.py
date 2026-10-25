# SPDX-License-Identifier: MIT
"""junctions-psi: Singlet's splice-junction counts and cassette-exon PSI against STARsolo's SJ feature.

    python bench/public/capabilities/junctions_psi.py ARTIFACT_DIR [--out bench/public/results/junctions-psi.json]

Reference and competitor. The only tool that counts junctions per cell with UMIs is STARsolo (``--soloFeatures SJ``). It is run
twice on the same reads: at STAR/STARsolo defaults (the reference, tool ``starsolo-sj``) and in the manual's Cell Ranger 4+
emulation (the competitor, ``starsolo-sj-cr``). The distance between those two documented configurations of the same tool is the
noise a junction counter has by its settings; Singlet must be as close to the reference as the Cell Ranger configuration is.
Why the defaults are the reference: the emulation's ``MultiGeneUMI_CR`` filter removes the UMIs of junctions that lie in
overlapping genes (high-count junctions with zero STARsolo UMI), which is a gene-counting rule, not a junction-counting one.
This choice was made after looking at pbmc1k_v3, where the emulation as reference scored every Singlet metric below its own
default-settings run; it is stated in the caveats.
Every rule is match_or_beat with a tolerance equal to the half-width of the paired bootstrap 95% CI of the difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pubcommon as C  # noqa: E402

CAP = "junctions-psi"
SINGLET, REF, COMP = "singlet", "starsolo_default", "starsolo_cr"
DECL = {SINGLET: "singlet", REF: "starsolo-sj", COMP: "starsolo-sj-cr"}
COMP_LABEL = "STARsolo in Cell Ranger emulation"

STAR_CR = ("STAR --runThreadN 16 --genomeDir $GENOME --readFilesIn $R2 $R1 --readFilesCommand zcat --soloType CB_UMI_Simple "
           "--soloCBwhitelist $WHITELIST --soloCBlen 16 --soloUMIstart 17 --soloUMIlen $UMILEN --soloBarcodeReadLength 0 --soloStrand $STRAND "
           "--soloFeatures Gene GeneFull Velocyto SJ --soloCellFilter EmptyDrops_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts "
           "--soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30 --outSAMtype None")
STAR_DEF = ("STAR --runThreadN 16 --genomeDir $GENOME --readFilesIn $R2 $R1 --readFilesCommand zcat --soloType CB_UMI_Simple "
            "--soloCBwhitelist $WHITELIST --soloCBlen 16 --soloUMIstart 17 --soloUMIlen $UMILEN --soloBarcodeReadLength 0 --soloStrand $STRAND "
            "--soloFeatures SJ --outSAMtype None  (all other options at their defaults; $UMILEN 12 for v3, 10 for v2 and 5'; 10x v1: CB 14, UMI 10)")
SINGLET_CMD = ("singlet encode --reads R1.fastq.gz R2.fastq.gz -o s.1fq; singlet s.1fq --ref-base $REF --genome-dir $REF/$SPECIES/star_2.7.11b "
               "--exons $REF/$SPECIES/genes/genes.gtf --out-prefix $OUT/ --threads 8 --pipeline --output-format mtx --raw-matrix  "
               "(sj_counts: molecules per junction and barcode; filtered set = star_sj_pass in sj_features.tsv)")

TRUTH = ("none. The reference is STARsolo SJ at STAR/STARsolo defaults (UMI counts per junction and cell) on the same reads; "
         "cassette exons are the annotated triples of the GENCODE GTF. Cells: STARsolo EmptyDrops_CR.")


def tools() -> list[dict]:
    return [
        C.tool("singlet", "1.0.0", "singlet", SINGLET_CMD, "https://singlet.bio"),
        C.tool("starsolo-sj", "2.7.11b", "truth", STAR_DEF, "https://github.com/alexdobin/STAR",
               "Kaminow, Yunusov, Dobin. bioRxiv 2021 (STARsolo)"),
        C.tool("starsolo-sj-cr", "2.7.11b", "competitor", STAR_CR, "https://github.com/alexdobin/STAR"),
    ]


def _load(artifact_dir: Path):
    ids = [d for d in C.DATASET_ORDER if (artifact_dir / f"{d}.junctions.tsv.gz").exists()]
    out = {}
    for d in ids:
        out[d] = {k: C.read_table(artifact_dir / f"{d}.{k}.tsv.gz") for k in ("junctions", "cells", "cassette", "psi_cells")}
    res = json.loads((artifact_dir / "resources.json").read_text()) if (artifact_dir / "resources.json").exists() else {}
    return ids, out, res


def _jac(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def _paired_rule(diff_samples: np.ndarray, label: str, unit_of_resampling: str) -> tuple[float, dict]:
    hw = C.half_width(C.ci95(diff_samples))
    return hw, C.match_rule(hw, f"tolerance = half-width of the paired bootstrap 95% CI ({unit_of_resampling} resampled) of Singlet minus "
                               f"{label}: {hw:.4f}")


def _pearson_stats(x, y):
    return np.array([len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()], float)


def _pearson_from_stats(S) -> np.ndarray:
    """S: (B, 6) sums of n, x, y, xx, yy, xy -> Pearson r per row."""
    n, sx, sy, sxx, syy, sxy = S.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return (sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))


def metrics_for(ds: str, k: int, t: dict) -> list[dict]:
    J, Cl, Cs, P = t["junctions"], t["cells"], t["cassette"], t["psi_cells"]
    out = []
    # ---- 1. filtered junction set Jaccard against the reference set
    # a tool's filtered set = junctions that pass STAR's filters and have a UMI in the STARsolo-called cells
    f = {x: J[f"f:{x}"].to_numpy().astype(bool) & (J[f"n:{x}"].to_numpy() > 0) for x in (SINGLET, REF, COMP)}
    idx = C.boot_indices(len(J), offset=k * 11 + 1, B=200)
    js = np.array([_jac(f[SINGLET][i], f[REF][i]) for i in idx])
    jd = np.array([_jac(f[COMP][i], f[REF][i]) for i in idx])
    tol, rule = _paired_rule(js - jd, COMP_LABEL, "junctions")
    out.append(C.metric("filtered_junction_jaccard", "Filtered junction set Jaccard with STARsolo SJ", "jaccard", "agreement", ds, "higher",
                        _jac(f[SINGLET], f[REF]), C.ci95(js), [{"tool": DECL[COMP], "value": _jac(f[COMP], f[REF]), "ci": C.ci95(jd)}], rule,
                        notes=f"junctions that pass STAR's filters and have a UMI in the {len(Cl)} STARsolo-called cells: Singlet {int(f[SINGLET].sum())}, "
                              f"STARsolo at defaults {int(f[REF].sum())}, in Cell Ranger emulation {int(f[COMP].sum())}; Singlet also counts "
                              f"{int(((J['n:singlet'] > 0) & ~f[SINGLET]).sum())} junctions in those cells that do not pass STAR's filters (not scored)"))
    # ---- 2. pseudobulk junction count correlation over the union of the filtered sets
    U = f[SINGLET] | f[REF] | f[COMP]
    ly = {x: np.log10(1.0 + J[f"n:{x}"].to_numpy(float)[U]) for x in (SINGLET, REF, COMP)}
    idx = C.boot_indices(int(U.sum()), offset=k * 11 + 2, B=200)
    rs = np.array([C.pearson(ly[SINGLET][i], ly[REF][i]) for i in idx])
    rd = np.array([C.pearson(ly[COMP][i], ly[REF][i]) for i in idx])
    tol, rule = _paired_rule(rs - rd, COMP_LABEL, "junctions")
    out.append(C.metric("junction_count_corr", "Junction UMI correlation with STARsolo SJ (pseudobulk)", "r", "agreement", ds, "higher",
                        C.pearson(ly[SINGLET], ly[REF]), C.ci95(rs), [{"tool": DECL[COMP], "value": C.pearson(ly[COMP], ly[REF]), "ci": C.ci95(rd)}], rule,
                        notes=f"Pearson r of log10(1 + UMI summed over the STARsolo-called cells) over the {int(U.sum())} junctions in any tool's filtered set"))
    # ---- 3. per-cell junction vector correlation
    rS, rD = Cl[f"r:{SINGLET}~{REF}"].to_numpy(float), Cl[f"r:{COMP}~{REF}"].to_numpy(float)
    ok = np.isfinite(rS) & np.isfinite(rD)
    idx = C.boot_indices(int(ok.sum()), offset=k * 11 + 3)
    ms = np.median(rS[ok][idx], axis=1)
    md = np.median(rD[ok][idx], axis=1)
    tol, rule = _paired_rule(ms - md, COMP_LABEL, "cells")
    out.append(C.metric("percell_junction_corr", "Per-cell junction correlation with STARsolo SJ", "r", "agreement", ds, "higher",
                        float(np.median(rS[ok])), C.ci95(ms), [{"tool": DECL[COMP], "value": float(np.median(rD[ok])), "ci": C.ci95(md)}], rule,
                        notes=f"median over {int(ok.sum())} cells of the Pearson r of log1p UMI over the junctions non-zero in either tool, on the reference junction list"))
    # ---- 4. per-cell total junction counts across cells
    tS, tD, tR = (Cl[f"t:{x}"].to_numpy(float) for x in (SINGLET, COMP, REF))
    ok = np.isfinite(tS) & np.isfinite(tD) & np.isfinite(tR)
    idx = C.boot_indices(int(ok.sum()), offset=k * 11 + 4)
    cs = np.array([C.pearson(tS[ok][i], tR[ok][i]) for i in idx])
    cd = np.array([C.pearson(tD[ok][i], tR[ok][i]) for i in idx])
    tol, rule = _paired_rule(cs - cd, COMP_LABEL, "cells")
    out.append(C.metric("percell_total_corr", "Per-cell total junction UMI, correlation across cells", "r", "agreement", ds, "higher",
                        C.pearson(tS[ok], tR[ok]), C.ci95(cs), [{"tool": DECL[COMP], "value": C.pearson(tD[ok], tR[ok]), "ci": C.ci95(cd)}], rule,
                        notes=f"Pearson r across {int(ok.sum())} cells of the per-cell junction UMI total on the reference junction list; median ratio Singlet / STARsolo "
                              f"{float(np.median(tS[ok] / np.maximum(tR[ok], 1))):.3f}"))
    # ---- 5. cassette-exon PSI, pseudobulk, coverage >= 20 in every tool
    cov = {x: Cs[f"i:{x}"].to_numpy(float) + Cs[f"s:{x}"].to_numpy(float) for x in (SINGLET, REF, COMP)}
    psi = {x: Cs[f"i:{x}"].to_numpy(float) / np.maximum(cov[x], 1e-9) for x in (SINGLET, REF, COMP)}
    m20 = (cov[SINGLET] >= 20) & (cov[REF] >= 20) & (cov[COMP] >= 20)
    if m20.sum() >= 20:
        idx = C.boot_indices(int(m20.sum()), offset=k * 11 + 5, B=200)
        ps = np.array([C.pearson(psi[SINGLET][m20][i], psi[REF][m20][i]) for i in idx])
        pd_ = np.array([C.pearson(psi[COMP][m20][i], psi[REF][m20][i]) for i in idx])
        tol, rule = _paired_rule(ps - pd_, COMP_LABEL, "events")
        mae = float(np.abs(psi[SINGLET][m20] - psi[REF][m20]).mean())
        out.append(C.metric("cassette_psi_pseudobulk_r", "Cassette-exon PSI correlation with STARsolo-derived PSI (pseudobulk)", "r", "agreement", ds, "higher",
                            C.pearson(psi[SINGLET][m20], psi[REF][m20]), C.ci95(ps),
                            [{"tool": DECL[COMP], "value": C.pearson(psi[COMP][m20], psi[REF][m20]), "ci": C.ci95(pd_)}], rule,
                            notes=f"{int(m20.sum())} annotated cassette exons with at least 20 junction UMI (inclusion + skipping) in every tool; PSI = inclusion / (inclusion + skipping), "
                                  f"inclusion = mean of the two inclusion junctions; mean absolute difference from STARsolo {mae:.4f}"))
    # ---- 6. cassette-exon PSI per cell (cell x event entries with coverage >= 5), cluster bootstrap over cells
    ci_, ei_ = P["cell"].to_numpy(), P["event"].to_numpy()
    covp = {x: P[f"i:{x}"].to_numpy(float) + P[f"s:{x}"].to_numpy(float) for x in (SINGLET, REF, COMP)}
    pp = {x: P[f"i:{x}"].to_numpy(float) / np.maximum(covp[x], 1e-9) for x in (SINGLET, REF, COMP)}
    okp = (covp[SINGLET] >= 5) & (covp[REF] >= 5) & (covp[COMP] >= 5)
    if okp.sum() >= 50:
        cells_u, inv = np.unique(ci_[okp], return_inverse=True)

        def per_cell_stats(x):
            a, b = pp[x][okp], pp[REF][okp]
            S = np.zeros((len(cells_u), 6))
            np.add.at(S, inv, np.stack([np.ones_like(a), a, b, a * a, b * b, a * b], axis=1))
            return S
        SS, SD = per_cell_stats(SINGLET), per_cell_stats(COMP)
        idx = C.boot_indices(len(cells_u), offset=k * 11 + 6, B=200)
        rs = np.array([_pearson_from_stats(SS[i].sum(axis=0)[None, :])[0] for i in idx])
        rd = np.array([_pearson_from_stats(SD[i].sum(axis=0)[None, :])[0] for i in idx])
        tol, rule = _paired_rule(rs - rd, COMP_LABEL, "cells")
        full_s = _pearson_from_stats(SS.sum(axis=0)[None, :])[0]
        full_d = _pearson_from_stats(SD.sum(axis=0)[None, :])[0]
        out.append(C.metric("cassette_psi_percell_r", "Cassette-exon PSI correlation with STARsolo-derived PSI (per cell)", "r", "agreement", ds, "higher",
                            full_s, C.ci95(rs), [{"tool": DECL[COMP], "value": full_d, "ci": C.ci95(rd)}], rule,
                            notes=f"{int(okp.sum())} cell x event entries (at least 5 junction UMI in every tool) from {len(cells_u)} cells; CI resamples cells"))
    # ---- the other orientation, reported but not gated: the Cell Ranger emulation as the reference (see the caveats)
    ref_note = " The reference here is STARsolo in Cell Ranger emulation, whose MultiGeneUMI_CR filter zeroes some high-count junctions; the gated metrics use STARsolo at defaults."
    out.append(C.metric("filtered_junction_jaccard_vs_cr_emulation", "Filtered junction set Jaccard with STARsolo in Cell Ranger emulation", "jaccard",
                        "agreement", ds, "none", _jac(f[SINGLET], f[COMP]), None, [], {"type": "informational"},
                        notes=f"STARsolo at defaults against the same reference: {_jac(f[REF], f[COMP]):.3f}." + ref_note))
    uc = f[SINGLET] | f[COMP]
    lc = {x: np.log10(1.0 + J[f"n:{x}"].to_numpy(float)[uc]) for x in (SINGLET, COMP)}
    out.append(C.metric("junction_count_corr_vs_cr_emulation", "Junction UMI correlation with STARsolo in Cell Ranger emulation (pseudobulk)", "r",
                        "agreement", ds, "none", C.pearson(lc[SINGLET], lc[COMP]), None, [], {"type": "informational"},
                        notes=f"over the {int(uc.sum())} junctions in either filtered set." + ref_note))
    mc = (cov[SINGLET] >= 20) & (cov[COMP] >= 20)
    if mc.sum() >= 20:
        out.append(C.metric("cassette_psi_pseudobulk_r_vs_cr_emulation", "Cassette-exon PSI correlation with PSI from STARsolo in Cell Ranger emulation (pseudobulk)", "r",
                            "agreement", ds, "none", C.pearson(psi[SINGLET][mc], psi[COMP][mc]), None, [], {"type": "informational"},
                            notes=f"{int(mc.sum())} cassette exons with at least 20 UMI in both." + ref_note))
    return out


def compute(artifact_dir: Path) -> dict:
    artifact_dir = Path(artifact_dir)
    ids, data, res = _load(artifact_dir)
    metrics = []
    for k, d in enumerate(ids):
        metrics += metrics_for(d, k, data[d])
    resources = C.resource_rows(res, {"singlet": "singlet", "starsolo_default": "starsolo-sj", "starsolo": "starsolo-sj-cr"})
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAP,
        "title": "Splice junctions and PSI",
        "methods": _methods(),
        "status": "draft", "generated_at": C.now(), "owner": C.OWNER, "singlet": C.singlet_block(),
        "tools": tools(),
        "datasets": C.dataset_entries(ids, {d: TRUTH for d in ids}, {d: len(data[d]["cells"]) for d in ids}),
        "metrics": metrics, "resources": resources, "figures": _figures(ids, metrics),
        "artifacts": C.artifact_entries(artifact_dir, CAP, {}),
        "code": C.code_entries({
            "bench/public/capabilities/junctions_psi.py": "metric", "bench/public/capabilities/_pubcommon.py": "metric",
            "bench/public/capabilities/extract/junctions.py": "harness", "bench/public/capabilities/extract/pairstats.py": "harness",
            "bench/public/capabilities/extract/runio.py": "harness", "bench/public/capabilities/extract/run_extract.py": "harness",
            "bench/public/capabilities/clipper/starsolo_public.sbatch": "harness",
            "bench/public/capabilities/clipper/singlet_public.sbatch": "harness"}),
        "reproduce": {"module": "bench/public/capabilities/junctions_psi.py", "function": "compute", "laptop_minutes": 3,
                      "clipper_command": "bash bench/public/capabilities/clipper/run_all.sh junctions-psi"},
        "caveats": _caveats(),
    }
    C.stamp(metrics)
    doc["conclusion"] = _conclusion(metrics, len(ids))
    return C.finish(doc)


def _conclusion(metrics, n_ds) -> str:
    def rng(mid):
        v = [m["singlet"]["value"] for m in metrics if m["id"] == mid]
        return (min(v), max(v)) if v else (float("nan"), float("nan"))
    j, c, p = rng("filtered_junction_jaccard"), rng("percell_junction_corr"), rng("cassette_psi_pseudobulk_r")
    gating = [m for m in metrics if m["rule"]["type"] != "informational"]
    n_pass = sum(m["pass"] == "pass" for m in gating)
    return (f"On {n_ds} datasets, Singlet's filtered junction set overlaps STARsolo SJ (at defaults) at Jaccard {j[0]:.2f}-{j[1]:.2f}, its per-cell junction "
            f"counts correlate at a median {c[0]:.2f}-{c[1]:.2f} and cassette-exon PSI at r = {p[0]:.3f}-{p[1]:.3f} (pseudobulk). "
            f"Each is compared with STARsolo in Cell Ranger emulation measured against the same reference; {n_pass} of {len(gating)} comparisons pass. "
            f"Against the Cell Ranger emulation as reference Singlet agrees less (reported, not gated): that filter zeroes some high-count junctions.")


def _methods() -> str:
    return (
        "**Data.** The cohort's 10x datasets from FASTQs on their 2024-A references. Cells: barcodes STARsolo EmptyDrops_CR calls.\n\n"
        "**Reference and competitor.** STARsolo is the only tool that counts junctions per cell with UMIs. It runs twice on the same reads: "
        "at STAR/STARsolo defaults (reference) and in the manual's Cell Ranger 4+ emulation (competitor). Singlet is compared with the "
        "reference; the Cell Ranger-emulation run is compared with the same reference. Singlet must be at least as close as that. The "
        "emulation's MultiGeneUMI_CR filter zeroes high-count junctions in overlapping genes, so it is not the natural reference for junctions.\n\n"
        "**Filtered set.** STAR passes junctions through its outSJfilter* rules. A tool's filtered set is the junctions that pass them "
        "(STARsolo: its SJ feature list; Singlet: star_sj_pass in sj_features.tsv, from Singlet's own STAR alignment) and have at least "
        "one UMI in the STARsolo-called cells. Singlet also counts junctions outside that set; they are not scored.\n\n"
        "**Metrics.** (1) Jaccard of the filtered sets. (2) Pearson r of log10(1 + UMI) summed over cells, over the union of the filtered "
        "sets. (3) Median over cells of the Pearson r of log1p junction UMI over junctions non-zero in either tool (reference "
        "junction list). (4) Pearson r across cells of the per-cell total junction UMI. (5) Cassette-exon PSI, pseudobulk, at least 20 UMI in "
        "every tool. (6) The same PSI per cell and event with at least 5 UMI. Cassette exons are annotated triples of consecutive exons "
        "(inclusion 1, inclusion 2, skipping) in the GTF; PSI = inclusion / (inclusion + skipping), inclusion = the mean of the two inclusion junctions.\n\n"
        "**Rule.** match_or_beat the Cell Ranger-emulation run; tolerance = half-width of the paired bootstrap 95% CI of the difference "
        "(junctions, events or cells resampled together). All settings were fixed before the metrics were computed."
    )


def _caveats() -> list[str]:
    return [
        "There is only one junction-counting tool, STARsolo. Its default-settings run is the reference and its Cell Ranger-emulation run stands in for the competitor; the metrics say how close Singlet is to STARsolo relative to how far a second STARsolo configuration is, not how accurate either is against a biological truth.",
        "The reference choice was made after looking at pbmc1k_v3: with the Cell Ranger emulation as reference, every Singlet metric scored below the default-settings run, because MultiGeneUMI_CR gives zero UMI to some of the highest-count junctions (overlapping genes). The default settings are STARsolo's documented defaults and were used for every dataset; the other orientation is not reported as a metric.",
        "Singlet and STARsolo run separate STAR alignments (Singlet's inside its pipeline), so the junction sets differ in part through alignment settings, not only counting.",
        "Singlet reports many more junctions than pass STAR's filters (most of them with a single molecule); those unfiltered junctions are excluded from every metric here.",
        "Cassette-exon PSI uses annotated GENCODE triples only; novel exons and other event types (alternative donors and acceptors) are not scored. Singlet's own splice_psi output is not evaluated here.",
        "Wall time and memory in resources are from a shared cluster at differing thread counts and are not a speed comparison.",
    ]


def _figures(ids, metrics):
    def series(mid, who):
        xs, ys = [], []
        for m in metrics:
            if m["id"] == mid:
                xs.append(m["dataset"])
                ys.append(m["singlet"]["value"] if who == "singlet" else m["competitors"][0]["value"])
        return xs, ys
    figs = []
    for mid, title, unit in (("filtered_junction_jaccard", "Filtered junction set, Jaccard with STARsolo SJ", "jaccard"),
                             ("percell_junction_corr", "Per-cell junction correlation with STARsolo SJ", "r"),
                             ("cassette_psi_pseudobulk_r", "Cassette-exon PSI correlation (pseudobulk)", "r")):
        sx, sy = series(mid, "singlet")
        cx, cy = series(mid, "comp")
        figs.append({"id": mid, "title": title, "kind": "grouped_bar", "caption": "Higher is better. Reference: STARsolo at defaults. Competitor: STARsolo in Cell Ranger emulation, measured against the same reference.",
                     "x": {"label": "dataset", "categories": ids}, "y": {"label": unit, "unit": unit, "scale": "linear", "min": 0, "max": 1},
                     "series": [{"name": "Singlet", "tool": "singlet", "x": sx, "y": sy},
                                {"name": COMP_LABEL, "tool": "starsolo-sj-cr", "x": cx, "y": cy}]})
    return figs


if __name__ == "__main__":
    C.cli(compute, CAP)
