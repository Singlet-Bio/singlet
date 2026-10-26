"""Capability ``crispr-guides``: guide UMI counting and per-cell guide assignment.

Callers, all on Singlet's guide counts and the same called cells:
- Singlet (``singlet.crispr.assign.call_ambient``): ambient-calibrated Poisson test per (guide, cell);
- a Cell Ranger-style per-guide two-component GMM on log10 UMIs (reimplemented, ``call_gmm``);
- a Poisson-Gaussian mixture per guide (Replogle 2022 / Norman 2019 style, ``call_poisson_gaussian``).
Published calls: Cell Ranger 6.0.0's protospacer calls (10x A549) and the authors' assignments (Replogle 2022).

Truth that depends on no caller: the expression of each guide's target gene (CRISPRi / knockout) in the cells a caller
assigns, relative to control cells. Artifacts (``bench/crispr/compare_crispr.py --artifacts``), per dataset ``<ds>``:
``<ds>.calls.<caller>.csv.gz``, ``<ds>.reference.csv.gz``, ``<ds>.guides.csv.gz``, ``<ds>.target_expr.csv.gz``,
``<ds>.expr_cells.txt``, ``<ds>.meta.json``; ``provenance.json``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _common import REPO_URL, artifact_entries, code_url, contract, load_provenance, now_iso, r

CAP = "crispr-guides"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "crispr"))
sys.path.insert(0, str(ROOT / "python"))
COMPETITORS = {"cellranger_gmm": "gmm-cellranger-style", "poisson_gaussian": "poisson-gaussian-mixture",
               "cellranger_published": "cellranger"}


def _metric(mid, name, unit, cat, ds, direc, s, comps, rule=None, notes=None):
    m = {"id": mid, "name": name, "unit": unit, "category": cat, "dataset": ds, "direction": direc,
         "singlet": {"value": r(s)}, "competitors": [{"tool": t, "value": r(v)} for t, v in comps if v is not None],
         "rule": rule or {"type": "match_or_beat", "against": "best"}}
    if notes:
        m["notes"] = notes
    return m


def compute(artifact_dir: Path) -> dict:
    import compare_crispr as CC

    artifact_dir = Path(artifact_dir)
    C = contract()
    prov = load_provenance(artifact_dir)
    metrics, figures = [], []
    for ds in prov["datasets"]:
        labs, ref, level, expr, kd_cells, gt, gk, meta = CC.read_artifacts(str(artifact_dir), ds)
        mode = meta.get("mode", "i")
        res = CC.evaluate(labs, ref, level, expr, kd_cells, gt, gk, mode)
        moved = "induced (CRISPRa)" if mode == "a" else "lower"
        S = res["singlet"]
        callers = [c for c in COMPETITORS if c in res and c != "cellranger_published"]
        ref_name = "published calls" if level == "target" else "Cell Ranger's protospacer calls"
        unit_ = "target" if level == "target" else "guide"
        metrics.append(_metric(
            "single_call_f1", f"Single-{unit_} calls against the {ref_name}: F1 of recall (reference single cells given "
            f"the same call) and precision (single calls the reference shares)", "fraction", "agreement", ds, "higher",
            S["f1_single"], [(COMPETITORS[c], res[c]["f1_single"]) for c in callers],
            notes="recall / precision: " + "; ".join(f"{n} {res[c]['recall_single']:.3f} / {res[c]['precision_single']:.3f}"
                                                     for c, n in [("singlet", "singlet")] + [(c, COMPETITORS[c]) for c in callers])))
        kd_tools = [c for c in COMPETITORS if c in res and "kd_cells_scored" in res[c]]
        if "kd_cells_scored" in S:
            metrics.append(_metric("knockdown_cells", "Single-guide cells of scored guides (usable perturbed cells)",
                                   "cells", "coverage", ds, "higher", S["kd_cells_scored"],
                                   [(COMPETITORS[c], res[c]["kd_cells_scored"]) for c in kd_tools]))
            nsig = lambda m_: round(m_.get("kd_frac_guides_significant", 0) * m_.get("kd_guides_scored", 0))
            metrics.append(_metric(
                "knockdown_guides_significant", f"Guides whose target is significantly {moved} in their single-guide "
                "cells (one-sided Mann-Whitney vs controls, BH 0.05; guides with >= 5 cells)", "guides", "accuracy", ds,
                "higher", nsig(S), [(COMPETITORS[c], nsig(res[c])) for c in kd_tools],
                notes="a count, not a share: a caller that assigns more cells scores more guides; shares: " + "; ".join(
                    f"{n} {res[c]['kd_frac_guides_significant']:.3f} of {res[c]['kd_guides_scored']}"
                    for c, n in [("singlet", "singlet")] + [(c, COMPETITORS[c]) for c in kd_tools])))
            thr_kd = 1.0 if mode == "a" else 0.5
            metrics.append(_metric(
                "knockdown_median", "Median effect on the target over scored guides (" +
                ("log2 induction, CRISPRa" if mode == "a" else "1 - expression / control") + ")",
                "log2" if mode == "a" else "fraction", "accuracy", ds, "higher", S["kd_median_knockdown"], [],
                rule={"type": "threshold", "value": thr_kd,
                      "rationale": "a working screen and correct calls: half of the target removed (CRISPRi / knockout) or "
                                   "doubled (CRISPRa) in the median guide"},
                notes="other callers: " + "; ".join(f"{COMPETITORS[c]} {res[c]['kd_median_knockdown']:.3f}" for c in kd_tools)))
        mc = res.get("multiplet_check")
        if mc and mc["ref_extra_guides_in_those_cells"]["n"] >= 20:
            metrics.append(_metric(
                "extra_guides_target_expression", "Cells Singlet calls single and Cell Ranger multiple: target "
                "expression of the guides only Cell Ranger adds, relative to non-targeting cells (1 = no knockdown)",
                "ratio", "accuracy", ds, "higher", mc["ref_extra_guides_in_those_cells"]["mean_rel_expr"], [],
                rule={"type": "threshold", "value": 0.8,
                      "rationale": "real guides knock their target down to ~0.3 here; an added guide whose target stays "
                                   ">= 0.8 of control is an ambient molecule, so Singlet was right to drop it"},
                notes=f"n = {mc['ref_extra_guides_in_those_cells']['n']}; the guide Singlet keeps leaves its target at "
                      f"{mc['singlet_guide_in_ref_multiple']['mean_rel_expr']:.2f}, clean single cells at "
                      f"{mc['singlet_single_ref_single']['mean_rel_expr']:.2f}"))
        cnt = meta.get("counting_vs_cellranger")
        if cnt:
            metrics.append(_metric("counting_umi_ratio", "Guide UMIs vs Cell Ranger's on the same cells and guides",
                                   "ratio", "agreement", ds, "none", cnt["umi_ratio"], [],
                                   rule={"type": "equivalence", "target": 1.0, "tolerance": {"rel": 0.03},
                                         "rationale": "the ADT counting core reproduces Cell Ranger to 0.02%; 3% "
                                                      "covers guide-window differences (anchor search vs pattern)"}))
        if "cellranger_thresholds_on_singlet_counts" in res:
            metrics.append(_metric("cellranger_rule_on_singlet_counts", "Cell Ranger's single calls reproduced by "
                                   "Cell Ranger's own per-guide UMI thresholds applied to Singlet's counts", "fraction",
                                   "agreement", ds, "higher",
                                   res["cellranger_thresholds_on_singlet_counts"]["recall_single"], [],
                                   rule={"type": "threshold", "value": 0.98,
                                         "rationale": "counting is equivalent when the same rule gives the same calls; "
                                                      "2% allows thresholds that fall on a count Singlet differs by one"}))
        cats = ["single_recall", "single_precision", "knockdown_significant_share"]
        figures.append({"id": f"{ds}_callers", "title": f"Guide assignment, {ds}", "kind": "grouped_bar",
                        "x": {"label": "metric", "categories": cats},
                        "y": {"label": "fraction", "unit": "fraction", "min": 0, "max": 1},
                        "series": [{"name": n, "tool": n, "x": cats,
                                    "y": [r(res[c].get(k)) for k in ("recall_single", "precision_single",
                                                                     "kd_frac_guides_significant")]}
                                   for c, n in [("singlet", "singlet")] + [(c, COMPETITORS[c]) for c in callers]]})
    commit = prov["commit"]
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAP,
        "title": "CRISPR guide assignment",
        "conclusion": "",
        "methods": (
            "Guide UMIs are counted from the guide library's FASTQs by singlet.crispr (protospacer cut before the scaffold "
            "anchor, Cell Ranger feature-barcode logic for barcodes, guides and UMIs, 3' v3 barcode translation decided "
            "from the reads). Three callers run on the same counts and called cells: Singlet's ambient-calibrated test "
            "(the ambient profile from the library's empty droplets), a Cell Ranger-style per-guide GMM and a "
            "Poisson-Gaussian mixture. A549: Cell Ranger 6.0.0's published protospacer calls are the reference and a "
            "caller of their own. K562 (Replogle 2022, gem group 41): the authors' assignments; dual guides are "
            "compared as target genes; the protospacers were rebuilt from the sgRNA ids on hg19 (98% seen in reads). "
            "Knockdown: CP10k of each guide's target in its single-guide cells vs non-targeting cells (A549) or the "
            "cells of other targets (K562), one-sided Mann-Whitney, BH."),
        "status": "draft", "generated_at": now_iso(), "owner": "vdj-crispr",
        "singlet": {"version": "1.0.0", "commit": commit, "branch": "claude/vdj-crispr-benchmark-721d3a"},
        "tools": [
            {"name": "singlet", "version": "1.0.0", "role": "singlet",
             "params": "singlet.crispr.assign.call_ambient(min_umis=3, alpha=1e-3, min_share=0.05)"},
            {"name": "gmm-cellranger-style", "version": "n/a", "role": "competitor",
             "params": "per guide: 2-component sklearn GaussianMixture on log10 UMIs of cells with UMIs; upper component and >= 3 UMIs"},
            {"name": "poisson-gaussian-mixture", "version": "n/a", "role": "competitor",
             "params": "per guide: EM for Poisson background + Gaussian on log2(UMI+1); posterior > 0.5"},
            {"name": "cellranger", "version": "6.0.0 (A549) / 4.0.0 (A375)", "role": "competitor",
             "url": "https://cf.10xgenomics.com/samples/cell-exp/6.0.0/SC3_v3_NextGem_DI_CRISPR_A549_5K_SC3_v3_NextGem_DI_CRISPR_A549_5K/",
             "params": "published protospacer_calls_per_cell.csv (10x outputs; Cell Ranger not run)"},
            {"name": "published-assignments", "version": "Replogle 2022", "role": "truth",
             "params": "K562_essential_raw_singlecell_01.h5ad obs gene / sgID_AB"},
        ],
        "datasets": [
            {"id": "tenx_crispr_a549_5k", "title": "10x A549 5K CRISPR (3' v3.1, direct capture, 93 guides)", "data_kind": "real",
             "accessions": {"url": "https://cf.10xgenomics.com/samples/cell-exp/6.0.0/SC3_v3_NextGem_DI_CRISPR_A549_5K_Multiplex/"},
             "truth": "Cell Ranger 6.0.0 protospacer calls; target expression"},
            {"id": "tenx_crispr_a375_10k", "title": "10x A375 CRISPR 10k (3' v3.1, direct capture, 2 guides)", "data_kind": "real",
             "accessions": {"url": "https://cf.10xgenomics.com/samples/cell-exp/4.0.0/SC3_v3_NextGem_DI_CRISPR_10K/"},
             "truth": "Cell Ranger 4.0.0 protospacer calls; target expression"},
            {"id": "norman2019_gem1", "title": "Norman 2019 K562 CRISPRa Perturb-seq, gem group 1 guide-barcode library",
             "data_kind": "real", "accessions": {"gse": "GSE133344", "gsm": ["GSM3906028"], "srr": ["SRR9602551", "SRR9602552"]},
             "truth": "authors' cell identities (held-out half: guide-barcode names learnt on the other half); target induction",
             "notes": "GEO has no guide-barcode sequences: the barcode -> identity table is learnt from the training half"},
            {"id": "replogle2022_k562_gg41", "title": "Replogle 2022 K562 essential-scale, gem group 41 guide library",
             "data_kind": "real", "accessions": {"srr": ["SRR19331010", "SRR19331011", "SRR19331013", "SRR19331014"],
                            "url": "https://plus.figshare.com/articles/dataset/20029387"},
             "truth": "published cell-to-guide assignments; target expression (CRISPRi)"},
        ],
        "metrics": metrics, "resources": prov.get("resources", []), "figures": figures,
        "artifacts": artifact_entries(artifact_dir, CAP, {p.name: ("singlet_output" if ".calls.singlet." in p.name else
                                                                     "competitor_output" if ".calls." in p.name else
                                                                     "truth" if ("reference" in p.name or "expr" in p.name)
                                                                     else "derived") for p in sorted(artifact_dir.iterdir())}),
        "code": [{"path": "python/singlet/crispr", "url": f"{REPO_URL}/tree/{commit}/python/singlet/crispr", "role": "implementation"},
                 {"path": "bench/crispr", "url": f"{REPO_URL}/tree/{commit}/bench/crispr", "role": "harness"},
                 {"path": "bench/crispr/compare_crispr.py", "url": code_url("bench/crispr/compare_crispr.py", commit), "role": "metric"}],
        "reproduce": {"module": "bench/public/capabilities/crispr_guides.py", "function": "compute", "laptop_minutes": 2,
                      "clipper_command": "sbatch --export=ALL,DS=<a549|replogle_k562_gg41> bench/crispr/hpc/run.sbatch"},
        "caveats": [
            "Cell Ranger was not run here (no licence accepted); its published A549 calls are compared, and its "
            "assignment rule is approximated by a reimplemented per-guide GMM; its published per-guide thresholds applied "
            "to Singlet's counts check the counting.",
            "The K562 published assignments keep only cells with one guide pair (the authors filtered the rest), so "
            "multiple-guide rates cannot be compared there.",
            "K562 non-targeting guides have no coordinate in their ids: their windows are counted as 'unidentified' guides.",
        ],
    }
    C.finalize(doc)
    return doc
