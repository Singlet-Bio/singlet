"""Capability `non-host`: Singlet's microbial / viral screen vs Kraken2 (and a minimap2 oracle).

Data: SARS-CoV-2-infected and mock human bronchial epithelial cells (GSE166766, 10x 3' v3, first 50 M read pairs of each run) and
a clean PBMC sample (10x PBMC 1k v3). Truth: the mock and PBMC samples carry no infection; the infected sample carries
SARS-CoV-2, and a read-level minimap2 alignment of the unmapped reads to NC_045512.2 is the per-cell oracle.

Artifacts (made by bench/public/harness/non_host/collect.py from the Clipper runs; all in one directory):
  percell_<dataset>.tsv.gz  per called cell: barcode, <tool>_sars2_umi and <tool>_called_umi for singlet, kraken2_pluspf,
                            kraken2_viral and oracle_sars2_umi (UMIs of SARS-CoV-2 / of any called species)
  species.tsv               dataset, tool, taxid, name, domain, reads: the species each tool found at sample level
                            (Singlet: the EM-confirmed species; Kraken2: every non-human species)
  meta.tsv                  dataset, screened_reads, n_cells, kingdom hits of Singlet
  resources.tsv             tool, dataset, wall_s, max_rss_gb, threads, cpu_s, node, note

A species is *called* at sample level when it reaches Singlet's own reporting floor, max(500, 0.5 % of screened reads) reads
(Singlet applies it through its EM; here it is applied to Kraken2 as well, so that no tool is penalised for reporting
every species that has a single read). A cell is *positive* for a tool when it has at least one UMI of a called species.

Run:  python bench/public/capabilities/non_host.py --artifacts DIR [--out bench/public/results/non-host.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _perf as C  # noqa: E402

CAPABILITY = "non-host"
# Commit of the harness and metric code that produced the published numbers (the code URLs below point at it).
CODE_COMMIT = "0000000000000000000000000000000000000000"
DPI2, MOCK, PBMC = "ravindra2021_dpi2", "ravindra2021_mock", "pbmc1k_v3"  # cohort ids (GSE166766)
TOOL_S, TOOL_KP, TOOL_KV, TOOL_O = "singlet", "kraken2-pluspf", "kraken2-viral", "minimap2-oracle"
KRAKEN = (TOOL_KP, TOOL_KV)
COL = {TOOL_S: "singlet", TOOL_KP: "kraken2_pluspf", TOOL_KV: "kraken2_viral", TOOL_O: "oracle"}
B = 200  # bootstrap resamples
SEED = 20261025


def floor_reads(screened_reads: float) -> float:
    """Singlet's detection floor: a species needs at least this many assigned reads (src/pipeline/singlet.cpp)."""
    return max(500.0, 0.005 * float(screened_reads))


def is_sars2(name: str, taxid=None) -> bool:
    return "Severe acute respiratory syndrome" in str(name) or "NC_045512" in str(name)


# ------------------------------------------------------------------------------------------------ helpers
def _metric(mid, name, unit, cat, ds, direction, s, comps, rule, notes=None, ci=None):
    m = {"id": mid, "name": name, "unit": unit, "category": cat, "dataset": ds, "direction": direction,
         "singlet": {"value": C.finite(s)}, "competitors": [{"tool": t, "value": C.finite(v)} for t, v in comps], "rule": rule}
    if ci is not None:
        m["singlet"]["ci"] = [float(ci[0]), float(ci[1])]
    if notes:
        m["notes"] = notes
    return m


def _boot_idx(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def _paired_boot(n: int, stats, rng) -> np.ndarray:
    """B bootstrap resamples of cells; stats(idx) returns an array of statistics for each resample."""
    return np.array([stats(_boot_idx(n, rng)) for _ in range(B)])


# ------------------------------------------------------------------------------------------------ compute
def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    pc = {ds: pd.read_csv(art / f"percell_{ds}.tsv.gz", sep="\t") for ds in (DPI2, MOCK, PBMC)}
    sp = pd.read_csv(art / "species.tsv", sep="\t")
    meta = pd.read_csv(art / "meta.tsv", sep="\t").set_index("dataset")
    res = pd.read_csv(art / "resources.tsv", sep="\t")
    rng = np.random.default_rng(SEED)
    metrics: list[dict] = []

    # ---------------------------------------------------------------- sample level: which virus, how many reads
    def called(ds, tool):
        s = sp[(sp.dataset == ds) & (sp.tool == tool)]
        if tool == TOOL_S:
            return s  # the EM-confirmed species
        return s[s.reads >= floor_reads(meta.loc[ds, "screened_reads"])]

    def top_virus_is_sars2(ds, tool) -> float:
        s = called(ds, tool)
        s = s[s.domain.astype(str).str.contains("Virus|VIRAL", case=False)]
        return float(len(s) > 0 and is_sars2(s.sort_values("reads", ascending=False).iloc[0]["name"]))

    metrics.append(_metric(
        "correct_virus", "SARS-CoV-2 is the top called virus", "flag", "accuracy", DPI2, "higher",
        top_virus_is_sars2(DPI2, TOOL_S), [(t, top_virus_is_sars2(DPI2, t)) for t in KRAKEN],
        {"type": "match_or_beat", "against": "best"},
        "1 when the called viral species with the most reads is SARS-CoV-2 (infected sample)."))

    def sars_reads(ds, tool) -> float:
        s = sp[(sp.dataset == ds) & (sp.tool == tool)]
        return float(s[s.name.map(is_sars2)].reads.sum())

    r_s, r_kp, r_kv, r_o = (sars_reads(DPI2, t) for t in (TOOL_S, TOOL_KP, TOOL_KV, TOOL_O))
    tol_rel = round(abs(r_kv - r_o) / max(r_o, 1.0) + 1e-3, 3)
    metrics.append(_metric(
        "sars2_reads", "Reads assigned to SARS-CoV-2", "reads", "agreement", DPI2, "none", r_s,
        [(TOOL_KP, r_kp), (TOOL_KV, r_kv), (TOOL_O, r_o)],
        {"type": "equivalence", "target": TOOL_KV, "tolerance": {"rel": tol_rel},
         "rationale": f"Kraken2 (viral database) and the minimap2 oracle differ by {abs(r_kv - r_o) / max(r_o, 1.0):.1%} on the same reads, "
                      "so a count within that band of Kraken2 is as close as two independent methods get."},
        "Reads of the host-unmapped reads assigned to SARS-CoV-2; the oracle counts reads aligned over >= 60 nt at >= 90 % identity. "
        "The target is the viral-database Kraken2: the PlusPF index is capped at 8 GB and assigns fewer SARS-CoV-2 reads at species rank."))

    # ---------------------------------------------------------------- per cell, infected sample
    d = pc[DPI2].fillna(0)
    n = len(d)
    ora = d[f"{COL[TOOL_O]}_sars2_umi"].to_numpy(float)
    x = {t: d[f"{COL[t]}_sars2_umi"].to_numpy(float) for t in (TOOL_S, TOOL_KP, TOOL_KV)}
    rho = {t: C.spearman(x[t], ora) for t in x}
    best_k = max(KRAKEN, key=lambda t: rho[t])
    boots = _paired_boot(n, lambda i: [C.spearman(x[t][i], ora[i]) for t in (TOOL_S, *KRAKEN)], rng)
    diff = boots[:, 0] - boots[:, [1 + KRAKEN.index(best_k)]].ravel()
    hw = C.half_width(diff)
    metrics.append(_metric(
        "percell_spearman_vs_oracle", "Per-cell SARS-CoV-2 UMIs vs the oracle (Spearman)", "rho", "accuracy", DPI2, "higher",
        rho[TOOL_S], [(t, rho[t]) for t in KRAKEN],
        {"type": "match_or_beat", "against": list(KRAKEN), "tolerance": {"abs": round(hw, 4)},
         "rationale": f"half-width of the paired bootstrap 95% CI of the difference to {best_k} over {n} cells ({B} resamples)"},
        f"{n} called cells of the infected sample; molecules = distinct (cell, UMI) of SARS-CoV-2.",
        ci=C.ci(boots[:, 0])))
    rho_sk = C.spearman(x[TOOL_S], x[TOOL_KV])
    rho_ko = rho[TOOL_KV]
    metrics.append(_metric(
        "percell_spearman_vs_kraken2", "Per-cell SARS-CoV-2 UMIs vs Kraken2, viral database (Spearman)", "rho", "agreement", DPI2, "higher",
        rho_sk, [],
        {"type": "threshold", "value": round(rho_ko, 4),
         "rationale": f"Kraken2 (viral database) agrees with the independent oracle at rho = {rho_ko:.3f}; Singlet must agree with it at least that well."},
        "Singlet and Kraken2 classify the same reads with different methods; the oracle is the independent reference."))

    # recall / precision of "cell has >= 1 SARS-CoV-2 UMI" against the oracle
    o_pos = ora >= 1
    pos = {t: x[t] >= 1 for t in x}

    def rp(t, i=slice(None)):
        tp = (pos[t][i] & o_pos[i]).sum()
        return tp / max(o_pos[i].sum(), 1), tp / max(pos[t][i].sum(), 1)

    def f1(t, i=slice(None)):
        r_, p_ = rp(t, i)
        return 0.0 if r_ + p_ == 0 else 2 * r_ * p_ / (r_ + p_)

    v = {t: f1(t) for t in x}
    best_k = max(KRAKEN, key=lambda t: v[t])
    bs = _paired_boot(n, lambda i: [f1(t, i) for t in (TOOL_S, best_k)], rng)
    hw = C.half_width(bs[:, 0] - bs[:, 1])
    rr = {t: rp(t) for t in x}
    metrics.append(_metric(
        "cell_f1_vs_oracle", "Infected-cell calls vs the oracle (F1 of >= 1 SARS-CoV-2 UMI)", "fraction", "accuracy", DPI2, "higher",
        v[TOOL_S], [(t, v[t]) for t in KRAKEN],
        {"type": "match_or_beat", "against": list(KRAKEN), "tolerance": {"abs": round(hw, 4)},
         "rationale": f"half-width of the paired bootstrap 95% CI of the difference to {best_k} over {n} cells ({B} resamples)"},
        f"A cell is positive with >= 1 SARS-CoV-2 UMI; oracle-positive cells: {int(o_pos.sum())} of {n}. Recall / precision: "
        + "; ".join(f"{t} {rr[t][0]:.3f} / {rr[t][1]:.3f}" for t in x) + ".", ci=C.ci(bs[:, 0])))

    metrics.append(_metric(
        "dpi2_cells_with_sars2", "Infected-sample cells with >= 1 SARS-CoV-2 UMI", "fraction", "coverage", DPI2, "none",
        float(pos[TOOL_S].mean()), [], {"type": "informational"},
        f"Oracle: {o_pos.mean():.3f}; Kraken2 PlusPF {pos[TOOL_KP].mean():.3f}. Not every cell of the infected sample is infected, "
        "which bounds the best possible infected-vs-mock AUROC at the oracle's value (see auroc_infected_vs_mock)."))

    # ---------------------------------------------------------------- infected vs mock, per cell
    m = pc[MOCK].fillna(0)
    nm = len(m)

    def au(tool_col, idx_d=None, idx_m=None):
        sd = d[tool_col].to_numpy(float) if idx_d is None else d[tool_col].to_numpy(float)[idx_d]
        sm = m[tool_col].to_numpy(float) if idx_m is None else m[tool_col].to_numpy(float)[idx_m]
        return C.auroc(np.concatenate([sd, sm]), np.concatenate([np.ones(len(sd)), np.zeros(len(sm))]).astype(int))

    cols = {t: f"{COL[t]}_sars2_umi" for t in (TOOL_S, TOOL_KP, TOOL_KV, TOOL_O)}
    a = {t: au(c) for t, c in cols.items()}
    best_k = max(KRAKEN, key=lambda t: a[t])
    bs = np.array([[au(cols[t], rng_d, rng_m) for t in (TOOL_S, best_k)]
                   for rng_d, rng_m in ((_boot_idx(n, rng), _boot_idx(nm, rng)) for _ in range(B))])
    hw = C.half_width(bs[:, 0] - bs[:, 1])
    metrics.append(_metric(
        "auroc_infected_vs_mock", "Infected vs mock cells, AUROC of the SARS-CoV-2 UMI count", "auroc", "accuracy", "pooled", "higher",
        a[TOOL_S], [(t, a[t]) for t in KRAKEN],
        {"type": "match_or_beat", "against": list(KRAKEN), "tolerance": {"abs": round(hw, 4)},
         "rationale": f"half-width of the bootstrap 95% CI of the difference to {best_k} ({B} resamples of {n} infected and {nm} mock cells)"},
        f"Ceiling: the minimap2 oracle reaches {a[TOOL_O]:.3f} because only {o_pos.mean():.1%} of infected cells carry a SARS-CoV-2 read.",
        ci=C.ci(bs[:, 0])))

    # ---------------------------------------------------------------- false positives
    def fp_cells(ds, tool) -> float:
        t = pc[ds].fillna(0)
        called_umi = t[f"{COL[tool]}_called_umi"].to_numpy(float)
        o = t["oracle_sars2_umi"].to_numpy(float)
        return float(((called_umi > 0) & (o == 0)).sum())

    f = {t: fp_cells(MOCK, t) for t in (TOOL_S, *KRAKEN)}
    tol = abs(f[TOOL_KP] - f[TOOL_KV])
    metrics.append(_metric(
        "mock_false_positive_cells", "Mock cells with a called species and no oracle SARS-CoV-2 read", "cells", "robustness", MOCK, "lower",
        f[TOOL_S], [(t, f[t]) for t in KRAKEN],
        {"type": "match_or_beat", "against": "best", **({"tolerance": {"abs": float(tol)},
         "rationale": "Kraken2 with the PlusPF and with the viral-only database differ by this many cells on the same reads"} if tol else {})},
        f"{nm} called mock cells. The oracle finds a few real SARS-CoV-2 reads in the mock (index hopping or ambient RNA); those cells are not counted."))
    for ds, mid, nm_ in ((MOCK, "mock_false_positive_species", "Mock: called species"), (PBMC, "pbmc_false_positive_species", "PBMC: called species")):
        v = {t: float(len(called(ds, t))) for t in (TOOL_S, *KRAKEN)}
        metrics.append(_metric(mid, nm_, "species", "robustness", ds, "lower", v[TOOL_S], [(t, v[t]) for t in KRAKEN],
                               {"type": "match_or_beat", "against": "best"},
                               "Species reaching the reporting floor in a sample that carries no virus or other non-host organism the screen should find."))
    pf = {t: float(((pc[PBMC].fillna(0)[f"{COL[t]}_called_umi"].to_numpy(float)) > 0).sum()) for t in (TOOL_S, *KRAKEN)}
    metrics.append(_metric("pbmc_false_positive_cells", "PBMC cells with a UMI of a called species", "cells", "robustness", PBMC, "lower",
                           pf[TOOL_S], [(t, pf[t]) for t in KRAKEN], {"type": "match_or_beat", "against": "best"},
                           f"{len(pc[PBMC])} called PBMC cells."))

    # ---------------------------------------------------------------- resources
    resources = [{k: (None if pd.isna(v) else v) for k, v in r.items() if k in ("tool", "dataset", "wall_s", "max_rss_gb", "threads", "cpu_h", "node", "notes")}
                 for r in res.assign(cpu_h=res.cpu_s / 3600.0, notes=res.note).to_dict("records")]
    for r in resources:
        r["threads"] = int(r["threads"])
        r["wall_s"] = round(float(r["wall_s"]), 1)
        r["max_rss_gb"] = round(float(r["max_rss_gb"]), 2)
        if r.get("cpu_h") is not None:
            r["cpu_h"] = round(float(r["cpu_h"]), 4)

    fig_species = {
        "id": "sars2_umis_per_cell", "title": "SARS-CoV-2 UMIs per cell, Singlet vs Kraken2 (infected sample)", "kind": "scatter",
        "caption": "Each point is a called cell of the infected sample; both axes log1p(UMIs). Spearman between the two tools "
                   f"{rho_sk:.3f}.",
        "x": {"label": "Kraken2 (viral database), log1p UMIs"}, "y": {"label": "Singlet, log1p UMIs"},
        "series": [{"name": "cells", "tool": TOOL_S, "x": np.round(np.log1p(x[TOOL_KV]), 3).tolist()[:5000],
                    "y": np.round(np.log1p(x[TOOL_S]), 3).tolist()[:5000]}]}
    fig_auc = {
        "id": "auroc_by_tool", "title": "Infected vs mock: AUROC of the per-cell SARS-CoV-2 UMI count", "kind": "bar",
        "caption": "The minimap2 oracle shows the ceiling: only a fraction of infected cells carry virus reads.",
        "x": {"label": "tool", "categories": [TOOL_S, TOOL_KP, TOOL_KV, TOOL_O]}, "y": {"label": "AUROC", "unit": "auroc", "min": 0.5, "max": 0.8},
        "series": [{"name": "infected vs mock", "x": [TOOL_S, TOOL_KP, TOOL_KV, TOOL_O], "y": [round(a[t], 4) for t in (TOOL_S, TOOL_KP, TOOL_KV, TOOL_O)]}]}

    prov = json.loads((art / "provenance.json").read_text(encoding="utf-8"))
    commit = prov["bench_commit"]
    byid = {(m["id"], m["dataset"]): m for m in metrics}

    def sv(mid, ds):
        return byid[(mid, ds)]["singlet"]["value"]

    def cv(mid, ds, tool):
        return next(c["value"] for c in byid[(mid, ds)]["competitors"] if c["tool"] == tool)

    conclusion = (
        f"Singlet names SARS-CoV-2 in the infected sample and assigns {sv('sars2_reads', DPI2):,.0f} reads to it, against "
        f"{cv('sars2_reads', DPI2, TOOL_KP):,.0f} for Kraken2 (PlusPF) and {cv('sars2_reads', DPI2, TOOL_O):,.0f} for the minimap2 oracle; "
        f"per-cell UMIs correlate with the oracle at rho {sv('percell_spearman_vs_oracle', DPI2):.3f} (Kraken2 {cv('percell_spearman_vs_oracle', DPI2, TOOL_KP):.3f}) "
        f"and separate infected from mock cells at AUROC {sv('auroc_infected_vs_mock', 'pooled'):.3f} (Kraken2 {cv('auroc_infected_vs_mock', 'pooled', TOOL_KP):.3f}, ceiling {a[TOOL_O]:.3f}). "
        f"In the mock and PBMC samples Singlet calls {sv('mock_false_positive_species', MOCK):.0f} and {sv('pbmc_false_positive_species', PBMC):.0f} species.")
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAPABILITY,
        "title": "Non-host (viral and microbial) detection", "conclusion": conclusion, "methods": prov["methods"],
        "status": prov.get("status", "final"), "generated_at": prov["generated_at"], "owner": "pipeline-benchmark",
        "singlet": prov["singlet"], "tools": prov["tools"], "datasets": prov["datasets"],
        "metrics": metrics, "resources": resources, "figures": [fig_species, fig_auc],
        "artifacts": C.artifact_entries(art, CAPABILITY, {"percell_": "derived", "species": "derived", "meta": "derived", "resources": "derived",
                                                          "provenance": "derived"}),
        "code": C.code_entries(commit, [
            ("bench/public/capabilities/non_host.py", "metric"), ("bench/public/harness/non_host/10_singlet.sbatch", "harness"),
            ("bench/public/harness/non_host/20_kraken.sbatch", "harness"), ("bench/public/harness/non_host/21_kraken_percell.sbatch", "harness"),
            ("bench/public/harness/non_host/30_oracle.sbatch", "harness"), ("bench/public/harness/non_host/kraken_percell.py", "harness"),
            ("bench/public/harness/non_host/oracle_percell.py", "harness"), ("bench/public/harness/non_host/collect.py", "harness")]),
        "reproduce": {"module": "bench/public/capabilities/non_host.py", "function": "compute", "laptop_minutes": 2,
                      "clipper_command": "sbatch bench/public/harness/collect_and_compute.sbatch non-host"},
        "caveats": prov["caveats"],
    }
    for d_ in doc["datasets"]:
        d_.setdefault("data_kind", "real")
    return doc


if __name__ == "__main__":
    C.cli(compute)
