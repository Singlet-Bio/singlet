# SPDX-License-Identifier: MIT
"""doublets capability of the Singlet 1.0.0 public benchmark: doublet scores and calls vs scDblFinder, Scrublet and vireo (genotype).

    python bench/public/capabilities/doublets.py ARTIFACT_DIR [--out bench/public/results/doublets.json]

`compute(artifact_dir)` is both the writer of the published file and the function the notebook re-runs; it reads only the per-cell
tables `<dataset>_doublets.tsv.gz` and `resources_<dataset>.json` that capabilities/harness/{barnyard,kang}_artifacts.py write.

Truth. (1) Genotype-multiplexed PBMC pools (Kang 2018): demuxlet's `doublet` vs `singlet` labels from the donors' genotypes; cells it
labels `ambs` are not scored. (2) Human + mouse mixtures: a cell is a doublet when the minor species supplies >= 20% of its UMIs.
Only heterotypic doublets (two donors / two species) are visible in either truth, so every precision here is a lower bound on the
real precision of an expression-based caller, and recall is recall of heterotypic doublets only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _adcommon as C  # noqa: E402

CAP = "doublets"
MIN_POS = 30  # a library with fewer truth doublets is shown in a table, not gated: AUPRC and F1 on a handful of cells are noise
MINOR_CUT = 0.2

# (cohort id, kind, development data?, title)
DATASETS = [
    ("kang2018_ctrl", "geno", True, "Kang 2018 PBMC, batch 2 control lane (8 donors, 10x 3' v1)"),
    ("kang2018_b1_c", "geno", False, "Kang 2018 PBMC, batch 1 lane C (8 donors, 10x 3' v1)"),
    ("kang2018_b1_a", "geno", False, "Kang 2018 PBMC, batch 1 lane A (4 donors, 10x 3' v1)"),
    ("kang2018_b1_b", "geno", False, "Kang 2018 PBMC, batch 1 lane B (4 donors, 10x 3' v1)"),
    ("hgmm1k_v3", "barnyard", True, "10x HEK293T + NIH3T3, 1k cells, 3' v3"),
    ("hgmm10k_v3", "barnyard", False, "10x HEK293T + NIH3T3, 11k cells, 3' v3"),
    ("hgmm1k_v2", "barnyard", False, "10x HEK293T + NIH3T3, 1k cells, 3' v2"),
]
ACCESSIONS = {
    "kang2018_ctrl": {"gse": "GSE96583", "gsm": ["GSM2560248"], "srr": ["SRR5398238"]},
    "kang2018_b1_c": {"gse": "GSE96583", "gsm": ["GSM2560247"], "srr": ["SRR5398237"]},
    "kang2018_b1_a": {"gse": "GSE96583", "gsm": ["GSM2560245"], "srr": ["SRR5398235"]},
    "kang2018_b1_b": {"gse": "GSE96583", "gsm": ["GSM2560246"], "srr": ["SRR5398236"]},
    "hgmm1k_v3": {"url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/hgmm_1k_v3/hgmm_1k_v3_fastqs.tar"},
    "hgmm10k_v3": {"url": "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/3.0.0/hgmm_10k_v3/hgmm_10k_v3_fastqs.tar"},
    "hgmm1k_v2": {"url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/hgmm_1k_v2/hgmm_1k_v2_fastqs.tar"},
}
EXPR = {"singlet": "singlet", "scrublet": "Scrublet", "scdblfinder": "scDblFinder"}   # column prefix -> tool name
GENO = {"singlet_gt": "singlet", "vireo": "vireo"}
KINDS = {  # metric id -> (name, kind of score)
    "auroc": ("AUROC", "ranking"), "auprc": ("AUPRC", "ranking"), "f1_native": ("F1 at the tool's own calls", "call")}


def _score_cols(prefix: str) -> tuple[str, str]:
    return (f"{prefix}_pdbl", f"{prefix}_call") if prefix in GENO else (f"{prefix}_score", f"{prefix}_call")


def _load(art: Path, ds: str, kind: str):
    df = pd.read_csv(art / f"{ds}_doublets.tsv.gz", sep="\t")
    if kind == "geno":
        keep = df.truth.isin(["singlet", "doublet"]).values
        y = (df.truth == "doublet").values[keep].astype(int)
    else:
        N = (df.human_umi + df.mouse_umi).values.astype(float)
        minor = np.minimum(df.mouse_umi.values, df.human_umi.values) / np.maximum(N, 1)
        keep = np.ones(len(df), bool)
        y = (minor >= MINOR_CUT).astype(int)
    return df[keep].reset_index(drop=True), y


def _metric_fn(mid: str):
    if mid == "auroc":
        return lambda y, s, c: C.auroc(y, s)
    if mid == "auprc":
        return lambda y, s, c: C.auprc(y, s)
    return lambda y, s, c: C.f1(y, c)


def _boot(fn, y, s, c, B):
    return np.array([fn(y[ix], s[ix], c[ix]) for ix in B])


def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    present = [x for x in DATASETS if (art / f"{x[0]}_doublets.tsv.gz").exists()]
    if not present:
        raise FileNotFoundError(f"no <dataset>_doublets.tsv.gz in {art}")
    datasets, metrics, caveats, rows_info, sens = [], [], [], [], []
    fig = {"expr_auroc": {}, "expr_auprc": {}, "geno_auroc": {}, "geno_auprc": {}}
    for ds, kind, dev, title in present:
        df, y = _load(art, ds, kind)
        n, npos = len(df), int(y.sum())
        datasets.append({
            "id": ds, "data_kind": "real", "title": title, "accessions": ACCESSIONS[ds], "n_cells": int(len(df)),
            "truth": ("demuxlet singlet / doublet labels from the donors' genotypes (Kang 2018); cells labelled ambs are not scored."
                      if kind == "geno" else f"Species mixture: a doublet is a called cell whose minor species supplies >= {int(MINOR_CUT*100)}% of its UMIs."),
            "notes": ("DEVELOPMENT DATA: Singlet's doublet threshold / genotype-doublet model were developed while looking at this library (PR #36). "
                      if dev else "Held out: not looked at while developing Singlet's doublet callers. ")
                     + f"{n} scored cells, {npos} truth doublets ({100*npos/n:.1f}%)."})
        gated = npos >= MIN_POS
        calls = {}
        for fam, cols_map, label in (("expr", EXPR, "expression"), ("geno", GENO, "genotype")):
            if fam == "geno" and kind != "geno":
                continue
            tools = [p for p in cols_map if _score_cols(p)[0] in df]
            sing = "singlet" if fam == "expr" else "singlet_gt"
            if sing not in tools:
                continue
            B = C.boot_indices(n, (CAP, ds)) if gated else None
            arrays = {}
            for p in tools:
                sc, cc = _score_cols(p)
                col = df[sc]
                arrays[p] = (col.fillna(col.min() if col.notna().any() else 0).values.astype(float), df[cc].fillna(0).values.astype(int))
            res = {}
            for mid in KINDS:
                fn = _metric_fn(mid)
                for p, (s, c) in arrays.items():
                    res[(mid, p)] = (fn(y, s, c), _boot(fn, y, s, c, B) if gated else None)
            if fam == "expr":
                calls = {TOOLNAME(p): int(arrays[p][1].sum()) for p in tools}
            for mid, (mname, _) in KINDS.items():
                pt = {p: res[(mid, p)][0] for p in tools}
                for p in tools:
                    sens.append([ds, label, TOOLNAME(p), mid, round(float(pt[p]), 4)])
                comp_ps = [p for p in tools if p != sing]
                if not gated or not comp_ps:
                    continue
                comps = [{"tool": TOOLNAME(p), "value": C.r6(pt[p]), "ci": C.ci95(res[(mid, p)][1])} for p in comp_ps]
                best_p = max(comp_ps, key=lambda p: pt[p])
                tol = C.half_width(res[(mid, sing)][1] - res[(mid, best_p)][1])
                metrics.append({
                    "id": f"{fam}_{mid}", "name": f"{label.capitalize()} doublets: {mname}", "unit": "score" if mid != "f1_native" else "F1",
                    "category": "accuracy" if mid != "f1_native" else "agreement", "dataset": ds, "direction": "higher",
                    "singlet": {"value": C.r6(pt[sing]), "ci": C.ci95(res[(mid, sing)][1])}, "competitors": comps,
                    "rule": {"type": "match_or_beat", "against": "best", "tolerance": {"abs": round(tol, 6)},
                             "rationale": f"half-width of the paired bootstrap 95% CI (1000 resamples of the {n} scored cells) of the difference between Singlet and the best competitor ({TOOLNAME(best_p)}) on this metric: {tol:.4f}"},
                    "notes": ("development data. " if dev else "held-out. ") + f"Best competitor: {TOOLNAME(best_p)}."})
                if mid in ("auroc", "auprc"):
                    fig[f"{fam}_{mid}"].setdefault(ds, {}).update({TOOLNAME(p): round(float(pt[p]), 4) for p in tools})
        rows_info.append([ds, "development" if dev else "held out", kind, n, npos, calls.get("singlet"), calls.get("Scrublet"), calls.get("scDblFinder")])
        metrics.append({"id": "n_truth_doublets", "name": "Truth doublets among scored cells", "unit": "cells", "category": "coverage", "dataset": ds,
                        "direction": "none", "singlet": {"value": npos}, "rule": {"type": "informational"},
                        "notes": "gated" if gated else f"fewer than {MIN_POS} truth doublets: metrics are shown in the table figure and not gated"})
        if not gated:
            caveats.append(f"{ds} has only {npos} truth doublets (< {MIN_POS}); its scores are in the per-dataset table but nothing is gated on it.")
        if kind == "barnyard":
            for cut in (0.1, 0.3):
                N = (df.human_umi + df.mouse_umi).values.astype(float)
                mn = np.minimum(df.mouse_umi.values, df.human_umi.values) / np.maximum(N, 1)
                yy = (mn >= cut).astype(int)
                sens.append([ds, "truth threshold", f"minor species >= {cut}", "n_truth_doublets", int(yy.sum())])

    # figures
    def bar(fid, title, key, caption):
        tools = sorted({t for v in fig[key].values() for t in v}, key=lambda t: (t != "singlet", t))
        cats = [d for d in fig[key]]
        return {"id": fid, "title": title, "kind": "grouped_bar", "caption": caption,
                "x": {"label": "dataset", "categories": cats}, "y": {"label": key.split("_")[1].upper(), "unit": "score", "min": 0, "max": 1},
                "series": [{"name": t, "tool": t, "x": [d for d in cats if t in fig[key][d]], "y": [fig[key][d][t] for d in cats if t in fig[key][d]]} for t in tools]}
    figures = []
    for fid, title, key, cap in (
            ("expr_auroc", "Expression-based doublet score: AUROC", "expr_auroc", "Gated datasets only (>= 30 truth doublets). Kang lanes are genotype truth; hgmm libraries are species truth."),
            ("expr_auprc", "Expression-based doublet score: AUPRC", "expr_auprc", "Average precision; the truth contains heterotypic doublets only, so it is a lower bound on real precision."),
            ("geno_auroc", "Genotype doublets: AUROC", "geno_auroc", "Singlet donor_assignments prob_doublet vs vireo prob_doublet. vireo is told the true donor count; Singlet estimates it."),
            ("geno_auprc", "Genotype doublets: AUPRC", "geno_auprc", "Same comparison, average precision.")):
        if fig[key]:
            figures.append(bar(fid, title, key, cap))
    figures.append({"id": "all_scores", "title": "Every score on every library (gated or not)", "kind": "table",
                    "caption": "dataset, score family, tool, metric, value. Libraries with fewer than 30 truth doublets are listed here only.",
                    "columns": ["dataset", "family / sensitivity", "tool", "metric", "value"], "rows": sens})
    figures.append({"id": "native_calls", "title": "Cells and calls per library", "kind": "table",
                    "caption": "Number of scored cells, truth doublets, and cells each expression-based tool calls at its own threshold.",
                    "columns": ["dataset", "role", "truth kind", "cells", "truth doublets", "Singlet calls", "Scrublet calls", "scDblFinder calls"],
                    "rows": rows_info})

    resources = []
    for p in sorted(art.glob("resources_*.json")):
        resources += json.loads(p.read_text())
    ids = {x[0] for x in present}
    resources = [r for r in resources if r["dataset"] in ids]
    for r in resources:
        r["tool"] = TOOLNAME(r["tool"])
    artifacts = []
    for p in sorted(art.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        ds = next((i for i in ids if p.name.startswith(i + "_") or p.name == f"resources_{i}.json"), None)
        a = {"name": p.name, "url": f"https://data.singlet.bio/benchmarking/1.0.0/{CAP}/{p.name}", "sha256": C.sha256_file(p),
             "bytes": p.stat().st_size, "kind": "derived" if p.suffix == ".json" else "singlet_output", "format": "json" if p.suffix == ".json" else "tsv"}
        if ds:
            a["dataset"] = ds
        artifacts.append(a)

    held_gated = [m for m in metrics if m["rule"]["type"] != "informational" and not next(x for x in present if x[0] == m["dataset"])[2]]
    doc = {
        "schema": C.contract.SCHEMA_CAPABILITY, "schema_version": C.contract.SCHEMA_VERSION, "capability": CAP, "title": "Doublets",
        "conclusion": _conclusion(metrics, held_gated), "methods": METHODS, "status": _status(art, present), "generated_at": C.now(), "owner": "pipeline-benchmark",
        "singlet": C.SINGLET, "tools": _tools(), "datasets": datasets, "metrics": metrics, "resources": resources, "figures": figures,
        "artifacts": artifacts, "code": _code(),
        "reproduce": {"module": "bench/public/capabilities/doublets.py", "function": "compute", "laptop_minutes": 8,
                      "clipper_command": "genotyped lanes: clipper/fetch_kang.sbatch; harness/kang_singlet.sbatch ID LANE.1fq; harness/kang_tools.sbatch ID K HARNESS_DIR; harness/artifacts.sbatch kang ID TRUTH_TSV LANE_FILTER. Mixtures: as for ambient-rna (clipper/run_singlet_10x.sbatch, harness/barnyard_tools.sbatch, harness/artifacts.sbatch barnyard ID). Then python bench/public/capabilities/doublets.py $WORK/bench-public/artifacts/doublets"},
        "caveats": _caveats(present) + sorted(set(caveats)),
    }
    return C.finish(doc)


def _status(art: Path, present) -> str:
    """`final` only when every listed library has artifacts; otherwise `draft`."""
    return "final" if {x[0] for x in DATASETS} <= {x[0] for x in present} else "draft"


def TOOLNAME(p: str) -> str:
    return {"singlet": "singlet", "singlet_gt": "singlet", "scrublet": "Scrublet", "scdblfinder": "scDblFinder", "vireo": "vireo"}.get(p, p)


def _conclusion(metrics: list[dict], held: list[dict]) -> str:
    ev = lambda ms: sum(1 for m in ms if C.contract.evaluate_metric(m) == "pass")  # noqa: E731
    g = [m for m in metrics if m["rule"]["type"] != "informational"]

    def pick(mid, ds_kind):
        ms = [m for m in held if m["id"] == mid and (m["dataset"].startswith("kang") == (ds_kind == "kang"))]
        return ms
    out = [f"On held-out libraries Singlet passes {ev(held)} of {len(held)} gating doublet metrics ({ev(g)} of {len(g)} including development data)."]
    for mid, label in (("expr_auroc", "expression AUROC"), ("geno_auroc", "genotype AUROC")):
        ms = pick(mid, "kang")
        if ms:
            s = np.mean([m["singlet"]["value"] for m in ms])
            best = {}
            for m in ms:
                for c in m["competitors"]:
                    best.setdefault(c["tool"], []).append(c["value"])
            comp = ", ".join(f"{k} {np.mean(v):.2f}" for k, v in best.items())
            out.append(f"Mean held-out Kang-lane {label}: Singlet {s:.2f}; {comp}.")
    return " ".join(out)


def _tools() -> list[dict]:
    return [
        {"name": "singlet", "version": "1.0.0", "role": "singlet", "url": "https://github.com/Singlet-Bio/singlet",
         "params": "singlet-campaign s.1fq --pipeline --output-format mtx --raw-matrix (+ --tagged-bam on genotyped lanes); doublet_scores.tsv (scrublet-style simulated-doublet score, call rate adapted to cell count) and donor/donor_assignments.tsv prob_doublet (donor count K estimated by held-out likelihood; default 1000G SNP panel). No doublet or donor flags"},
        {"name": "Scrublet", "version": "0.2.3", "role": "competitor", "url": "https://github.com/swolock/scrublet", "citation": "Wolock et al. 2019, Cell Systems 8:281",
         "params": "scr.Scrublet(counts, expected_doublet_rate=0.06, random_state=0).scrub_doublets()  (documented defaults) on the called cells' gene counts"},
        {"name": "scDblFinder", "version": "1.23.4", "role": "competitor", "url": "https://bioconductor.org/packages/scDblFinder", "citation": "Germain et al. 2022, F1000Research 10:979",
         "params": "scDblFinder(SingleCellExperiment(assays = list(counts = toc))), set.seed(1); all defaults, on the called cells' gene counts"},
        {"name": "vireo", "version": "0.5.9", "role": "competitor", "url": "https://github.com/single-cell-genetics/vireo", "citation": "Huang et al. 2019, Genome Biology 20:273",
         "params": "cellsnp-lite -s possorted.bam -b barcodes.txt -R genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz -p 16 --minMAF 0.1 --minCOUNT 20 --cellTAG CB --UMItag UB; vireo -c cellsnp -o out -N K -p 16 --randSeed 42, K = the true donor count of the lane (vireo cannot choose it)"},
        {"name": "cellsnp-lite", "version": "1.2.3", "role": "helper", "url": "https://github.com/single-cell-genetics/cellsnp-lite", "params": "pileup for vireo (see vireo)"},
        {"name": "demuxlet-kang2018", "version": "GSE96583", "role": "truth", "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583",
         "params": "multiplets = singlet / doublet / ambs from GSE96583_batch{1,2}.total.tsne.df.tsv.gz (demuxlet with the donors' genotypes)"},
        {"name": "barnyard-species-truth", "version": "n/a", "role": "truth", "params": f"doublet = minor species >= {int(MINOR_CUT*100)}% of the cell's UMIs"},
    ]


def _code() -> list[dict]:
    p = Path(__file__).resolve().parent / "CODE_COMMIT"
    sha = p.read_text().strip() if p.exists() else C.SINGLET["commit"]
    base = f"{C.REPO_URL}/blob/{sha}/bench/public/capabilities"
    rows = [("doublets.py", "metric"), ("_adcommon.py", "metric"), ("harness/kang_singlet.sbatch", "harness"), ("harness/kang_tools.sbatch", "harness"),
            ("harness/kang_prep.py", "harness"), ("harness/kang_artifacts.py", "harness"), ("harness/scrublet_run.py", "harness"),
            ("harness/barnyard_tools.R", "harness"), ("harness/barnyard_prep.py", "harness"), ("harness/barnyard_tools.sbatch", "harness"),
            ("harness/barnyard_artifacts.py", "harness"), ("clipper/run_singlet_10x.sbatch", "harness"), ("clipper/fetch_kang.sbatch", "harness")]
    out = [{"path": f"bench/public/capabilities/{p}", "url": f"{base}/{p}", "role": r} for p, r in rows]
    for f in ("doublet_detect.h", "donor_demux.h"):
        out.append({"path": f"include/singlet/pileup/{f}", "role": "implementation", "url": f"{C.REPO_URL}/blob/{C.SINGLET['commit']}/include/singlet/pileup/{f}"})
    return out


def _caveats(present) -> list[str]:
    dev = [x[0] for x in present if x[2]]
    return [
        f"{', '.join(dev)} are development data: Singlet's doublet threshold, simulated-doublet score and genotype-doublet model were built while looking at them (PR #36). The headline numbers are the held-out libraries.",
        "The Kang held-out lanes share the 10x v1 chemistry, the donor population and the demuxlet truth method with the development lane; batch 1 lanes are different libraries and in part different donors, but this is not a different technology. No other public genotype-multiplexed pool with raw reads and independent labels is in the cohort yet.",
        "Truth contains heterotypic doublets only (two donors or two species). Homotypic doublets, which expression-based callers also try to find, count as singlets, so precision and F1 are lower bounds for expression-based tools and do not measure the genotype tool's blind spot.",
        "demuxlet labels are themselves a genotype-based call with its own error, and Singlet's genotype model and demuxlet use the same donor genotypes in spirit; vireo shares Singlet's SNP panel. Truth is not an independent measurement of doublets (no hashing, no imaging).",
        "vireo is given the true donor count; Singlet estimates it. Scrublet and scDblFinder are run at their documented defaults without tuning to the library's expected doublet rate.",
        "Every tool is scored on the cells Singlet called (called_v2) that carry a truth label; tools with their own cell calling were not given it. Scrublet and scDblFinder score the gene-level counts Singlet exports.",
        "Resource rows compare Singlet's whole pipeline with one competitor step; they are not a speed comparison and are not gating.",
    ]


METHODS = """**Data.** Genotype-multiplexed PBMC pools from Kang et al. 2018 (GSE96583): batch 2 control lane (development data), and batch 1 lanes C (8 donors), A and B (4 donors each) as held-out libraries. The submitted Cell Ranger BAMs were converted back to FASTQ-equivalent reads and re-processed by Singlet against GRCh38-2024-A. Species mixtures: 10x HEK293T + NIH3T3 libraries (hgmm1k_v3 development; hgmm10k_v3, hgmm1k_v2 held out) on the joint GRCh38 + GRCm39 reference.

**Truth.** Kang lanes: demuxlet's singlet / doublet labels computed from donor genotypes (cells labelled ambs are dropped). Mixtures: a called cell is a doublet when the minor species supplies at least 20% of its UMIs (10% and 30% listed as sensitivity). Both truths see heterotypic doublets only.

**Tools.** Singlet 1.0.0 at pipeline defaults, with the donor count estimated: an expression score (`doublet_scores.tsv`) and a genotype doublet probability (`donor_assignments.tsv`). Competitors at documented defaults on the same cells: Scrublet and scDblFinder on Singlet's gene counts; vireo (after cellsnp-lite) on Singlet's tagged BAM and the same 1000 Genomes SNP list, given the true donor count. Exact commands are in `tools[].params`.

**Metrics.** Per library and score family (expression, genotype): AUROC and average precision (AUPRC) of the continuous score, and F1 of the tool's own doublet calls. Ties count half in AUROC; AUPRC processes tied scores together. Every competitor is scored on the identical cell set.

**Pass rule.** Singlet must match or beat the best competitor on each metric and library within the half-width of the paired bootstrap 95% CI (1000 resamples of the scored cells) of the difference between Singlet and that competitor. Libraries with fewer than 30 truth doublets (the 1k mixtures) are tabulated and not gated, because AUPRC and F1 on a handful of cells are noise. A metric that fails is reported as failing; speed is not gated.""".strip()


if __name__ == "__main__":
    sys.exit(C.write_main(compute, __doc__, str(Path(__file__).resolve().parents[1] / "results" / "doublets.json")))
