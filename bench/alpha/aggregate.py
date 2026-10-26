"""Turn the Research Alpha benchmark outputs into results documents (public results contract 1.1.0) and figures.

    python -P aggregate.py samples COMPARE_JSON_DIR OUT_DIR        # A: per-sample JSONs -> alpha_samples.csv
    python -P aggregate.py docs OUT_DIR [ENGINE_JSON_DIR]           # A (+B): documents + figures from OUT_DIR/alpha_samples.csv
      env CONTRACT = path of bench/public/contract.py (branch claude/bench-public-cohort-485cf7); CODE_SHA = commit of this code

Every document carries engine "alpha (published atlas)" and a dataset_scope, so the website can show the Alpha half
next to the Singlet 1.0.0 half. `compute(artifact_dir)` rebuilds the A documents from the alpha_samples.csv artifact.
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHA = os.environ.get("CODE_SHA", "0" * 40)
REPO = "https://github.com/Singlet-Bio/singlet"
ENGINE = "alpha (published atlas)"
SINGLET = {"version": "0.3.0 (Research Alpha atlas engine)", "commit": "8e1803ba5ac9cb8038e666f06df446c9a85550b0",
           "branch": "v0.3.0-pilot-freeze", "binary": "singlet (campaign build, cellarium full_campaign)"}
OWNER = "alpha-bench"
STRATA_ORDER = [f"{s}|{c}|{z}" for s in ("human", "mouse") for c in ("3p_v2", "3p_v3", "5p", "nuclei") for z in ("small", "large")]


def load_contract():
    p = os.environ.get("CONTRACT") or str(HERE.parent / "public" / "contract.py")
    spec = importlib.util.spec_from_file_location("contract", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ A: flatten
def flatten(d: dict) -> dict:
    g = lambda *ks: _get(d, ks)  # noqa: E731
    sra_all, cat = g("sra", "spots_all_runs"), g("catalog", "read_count")
    used = g("singlet", "reads_used")
    avail = sra_all or cat
    lay = g("gene_level", "layer_matched")
    return {
        "gsm": d["gsm"], "gse": d["gse"], "stratum": d["stratum"], "species": d["species"], "chem": d["chem"], "size": d["size"],
        "protocol_label": d.get("protocol_label"), "status": "error" if d.get("error") else ("excluded" if d.get("excluded_reason") else "scored"),
        "note": d.get("error") or d.get("excluded_reason") or "",
        "singlet_called": g("calls", "singlet") if "calls" in d else g("singlet", "n_called"),
        "author_cells": g("calls", "author") if "calls" in d else g("author", "n_barcodes"),
        "recovery": g("calls", "recovery"), "precision": g("calls", "precision"), "jaccard": g("calls", "jaccard"),
        "author_in_singlet_matrix": g("calls", "author_in_singlet_matrix"),
        "missed_median_author_umi": g("calls", "missed_author_cells_median_author_umi"),
        "shared_median_author_umi": g("calls", "shared_cells_median_author_umi"),
        "umi_ratio_exon": g("umi_ratio", "exon_over_author", "median"),
        "umi_ratio_exon_intron": g("umi_ratio", "exon_intron_over_author", "median"),
        "umi_ratio_matched": g("headline", "umi_ratio"), "layer_matched": lay,
        "cell_corr_raw": g("gene_level", lay, "cell_corr") if lay else None,
        "cell_corr_depth_matched": g("headline", "cell_corr_depth_matched"), "noise_floor": g("headline", "noise_floor"),
        "gap_to_floor": g("headline", "gap_to_floor"), "pseudobulk_spearman": g("headline", "pseudobulk_spearman"),
        "gene_match": d.get("gene_match"), "genes_common": g("genes", "common"),
        "reads_used": used, "reads_available": avail, "reads_available_source": "NCBI runinfo, all runs" if sra_all else ("catalog read_count" if cat else None),
        "reads_first_run": g("sra", "spots_first_run"), "n_runs": g("sra", "n_runs"), "catalog_read_count": cat,
        "frac_reads_used": (used / avail) if (used and avail) else None,
        "frac_first_run_used": (used / g("sra", "spots_first_run")) if (used and g("sra", "spots_first_run")) else None,
        "wrong_strand_frac": g("singlet", "wrong_strand_frac"), "singlet_git_sha": g("singlet", "git_sha"),
        "singlet_median_umi": g("singlet", "median_umi_per_cell"), "author_median_umi": g("author", "median_umi"),
        "author_files": ";".join(g("author", "files") or []),
    }


def _get(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k is None or k not in d:
            return None
        d = d[k]
    return d


# ------------------------------------------------------------------------------------------------ stats
def boot_median(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return float(np.median(x)), [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def r4(v):
    return None if v is None else float(round(v, 4))


# ------------------------------------------------------------------------------------------------ A: documents
def compute(artifact_dir) -> dict:
    """Notebook entry point: the cell-calling document from the alpha_samples.csv artifact."""
    return build_docs(pd.read_csv(Path(artifact_dir) / "alpha_samples.csv"))["alpha-cell-calling"]


def datasets_for(S):
    out = []
    for st in STRATA_ORDER:
        sub = S[S.stratum == st]
        if not len(sub):
            continue
        out.append({"id": "alpha_" + st.replace("|", "_"), "title": f"Research Alpha samples: {st.replace('|', ', ')}",
                    "accessions": {"gsm": sub.gsm.tolist(), "other": sorted(set(sub.gse))},
                    "truth": "the authors' filtered Cell Ranger matrix deposited in the GEO sample record",
                    "data_kind": "real", "n_cells": int(sub.author_cells.fillna(0).sum()),
                    "notes": f"{len(sub)} samples, one per study; Singlet values are the published bundles as served by data.singlet.bio"})
    return out


def common(capability, title):
    return {"schema": "singlet.benchmark.capability", "schema_version": "1.1.0", "capability": capability, "title": title,
            "engine": ENGINE,
            "dataset_scope": "Research Alpha: samples as currently published on singlet.bio (atlas engine v0.3.0, first SRA run, first 30M reads), "
                             "stratified human/mouse x 10x 3' v2 / 3' v3 / 5' / nuclei x small/large, restricted to samples whose GEO record holds the authors' filtered matrix",
            "status": "final", "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "owner": OWNER,
            "singlet": SINGLET,
            "tools": [{"name": "singlet", "version": "0.3.0", "role": "singlet",
                       "params": "published Research Alpha bundle: singlet download --max-reads 30000000 (first SRR); singlet --threads 4 --pipeline --output-format 1pz; cell calls = STARsolo CellRanger2.2 knee"},
                      {"name": "cellranger-authors", "version": "as deposited (Cell Ranger 2-8, per study)", "role": "truth",
                       "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/ (GEO sample supplementary files; exact files in alpha_samples.csv)",
                       "params": "filtered feature-barcode matrix as deposited by the authors"}],
            "code": [{"path": f"bench/alpha/{f}", "url": f"{REPO}/blob/{SHA}/bench/alpha/{f}", "role": r}
                     for f, r in (("select_samples.py", "harness"), ("alpha_compare.py", "metric"), ("aggregate.py", "metric"))],
            "reproduce": {"module": "bench/alpha/aggregate.py", "function": "compute", "laptop_minutes": 1,
                          "clipper_command": "sbatch --array=0-158%12 bench/alpha/alpha_compare.sbatch"}}


def strat_metric(S, col, mid, name, unit, cat, direction, rule, competitor_col=None, competitor_tool=None, ci_rule=None):
    """One pooled metric plus one per stratum (median over samples, bootstrap 95% CI)."""
    out = []
    groups = [("pooled", S)] + [("alpha_" + st.replace("|", "_"), S[S.stratum == st]) for st in STRATA_ORDER if (S.stratum == st).any()]
    for dset, sub in groups:
        v, ci = boot_median(sub[col].values)
        if v is None:
            continue
        m = {"id": mid, "name": name, "unit": unit, "category": cat, "dataset": dset, "direction": direction,
             "singlet": {"value": r4(v), "ci": [r4(c) for c in ci]}, "rule": dict(rule),
             "notes": f"median over {int(sub[col].notna().sum())} samples; CI = bootstrap over samples"}
        if competitor_col:
            cv, cci = boot_median(sub[competitor_col].values)
            m["competitors"] = [{"tool": competitor_tool, "value": r4(cv), "ci": [r4(c) for c in cci]}]
            if ci_rule:
                d, dci = boot_median((sub[competitor_col] - sub[col]).values)
                half = max(0.0, (dci[1] - dci[0]) / 2)
                m["rule"]["tolerance"] = {"abs": r4(half)}
                m["rule"]["rationale"] = f"half-width of the bootstrap 95% CI of the median per-sample difference ({r4(d)})"
        out.append(m)
    return out


def build_docs(S, contract=None):
    S = S.copy()
    sc = S[S.status == "scored"]
    n_tot, n_sc = len(S), len(sc)
    docs = {}
    # ---------------- cell calling
    d = common("cell-calling", "Cell calls of the published atlas vs the authors' Cell Ranger cells (Research Alpha)")
    d["datasets"] = datasets_for(sc)
    thr = "Cell Ranger and STARsolo EmptyDrops_CR agree at Jaccard 0.95-0.99 on identical reads (docs/pipeline-validation.md 4.5), so a caller that reproduces the authors' Cell Ranger cells should find at least 95% of them"
    d["metrics"] = (strat_metric(sc, "recovery", "cell_recovery", "Authors' cells recovered", "fraction", "accuracy", "higher",
                                 {"type": "threshold", "value": 0.95, "rationale": thr})
                    + strat_metric(sc, "precision", "cell_precision", "Singlet cells that are authors' cells", "fraction", "accuracy", "higher",
                                   {"type": "threshold", "value": 0.95, "rationale": thr})
                    + strat_metric(sc, "author_in_singlet_matrix", "author_cells_in_barcode_matrix", "Authors' cells present in Singlet's barcode matrix (>=100 reads)",
                                   "fraction", "coverage", "higher", {"type": "informational"}))
    rec, rci = boot_median(sc.recovery)
    prec, _ = boot_median(sc.precision)
    d["conclusion"] = (f"On {n_sc} Research Alpha samples, the published atlas recovers a median {rec:.0%} (95% CI {rci[0]:.0%}-{rci[1]:.0%}) of the cells "
                       f"the authors called with Cell Ranger, and {prec:.0%} of its own calls are authors' cells. The calls are precise but miss "
                       "low-UMI cells: a knee filter on reads capped at 30M of the first SRA run.")
    d["methods"] = METHODS_A
    d["caveats"] = caveats(S)
    d["resources"] = RESOURCES
    d["figures"] = [fig_strata(sc, "recovery", "cell_recovery_by_stratum", "Authors' cells recovered, by stratum"),
                    fig_strata(sc, "precision", "cell_precision_by_stratum", "Precision of Singlet's calls, by stratum")]
    d["artifacts"] = ARTIFACTS
    docs["alpha-cell-calling"] = d
    # ---------------- counts
    d = common("gex-counts", "Gene counts of the published atlas vs the authors' Cell Ranger matrices (Research Alpha)")
    d["datasets"] = datasets_for(sc)
    d["tools"] = d["tools"] + [{"name": "authors-thinning-noise-floor", "version": "n/a", "role": "truth",
                                "params": "two independent binomial thinnings of the authors' per-cell profile to the shared depth; their correlation is the best any tool can reach at that depth"}]
    d["metrics"] = (strat_metric(sc, "cell_corr_depth_matched", "cell_corr_depth_matched", "Per-cell gene correlation at equal depth (log1p Pearson)",
                                 "pearson r", "agreement", "higher", {"type": "match_or_beat", "against": "best"},
                                 competitor_col="noise_floor", competitor_tool="authors-thinning-noise-floor", ci_rule=True)
                    + strat_metric(sc, "umi_ratio_matched", "umi_per_cell_ratio", "UMIs per cell, Singlet / authors (matched layer, shared cells)",
                                   "ratio", "agreement", "none",
                                   {"type": "equivalence", "target": 1.0, "tolerance": {"rel": 0.1},
                                    "rationale": "on identical reads the counting engine's exonic counts are within -10% to +5% of 10x's published Cell Ranger matrices (docs/pipeline-validation.md 4.3)"})
                    + strat_metric(sc, "pseudobulk_spearman", "pseudobulk_spearman", "Pseudobulk Spearman rho over common genes", "rho", "agreement", "higher",
                                   {"type": "threshold", "value": 0.95,
                                    "rationale": "on identical reads Singlet's exonic pseudobulk vs published Cell Ranger reaches rho 0.96-0.98 (docs/pipeline-validation.md 4.3)"})
                    + strat_metric(sc, "frac_reads_used", "reads_used_over_available", "Reads used / reads available in SRA", "fraction", "coverage", "higher",
                                   {"type": "informational"}))
    dm, _ = boot_median(sc.cell_corr_depth_matched)
    fl, _ = boot_median(sc.noise_floor)
    ur, uci = boot_median(sc.umi_ratio_matched)
    fr, _ = boot_median(sc.frac_reads_used)
    ps, _ = boot_median(sc.pseudobulk_spearman)
    d["conclusion"] = (f"At equal depth the published atlas's per-cell profiles match the authors' Cell Ranger matrices almost to the sampling-noise floor "
                       f"(median r {dm:.3f} vs floor {fl:.3f}; pseudobulk rho {ps:.3f}, {n_sc} samples). Absolute counts are low: a median {ur:.2f}x the authors' "
                       f"UMIs per cell (95% CI {uci[0]:.2f}-{uci[1]:.2f}), because the atlas used a median {fr:.0%} of each sample's SRA reads.")
    d["methods"] = METHODS_A
    d["caveats"] = caveats(S)
    d["resources"] = RESOURCES
    d["figures"] = [fig_scatter(sc), fig_strata(sc, "umi_ratio_matched", "umi_ratio_by_stratum", "UMIs per cell, Singlet / authors, by stratum")]
    d["artifacts"] = ARTIFACTS
    docs["alpha-gex-counts"] = d
    if contract is not None:
        for k in docs:
            docs[k] = contract.finalize(docs[k])
    return docs


def fig_strata(sc, col, fid, title):
    cats = [st for st in STRATA_ORDER if (sc.stratum == st).any()]
    med = [r4(boot_median(sc[sc.stratum == st][col].values)[0]) for st in cats]
    return {"id": fid, "title": title, "kind": "box", "x": {"label": "stratum", "categories": cats},
            "y": {"label": col, "scale": "linear"},
            "series": [{"name": "Singlet (published)", "tool": "singlet", "x": cats, "y": med}],
            "caption": "Median over samples per stratum; the per-sample values are in the alpha_samples.csv artifact."}


def fig_scatter(sc):
    s = sc.dropna(subset=["frac_reads_used", "umi_ratio_matched"])
    return {"id": "umi_ratio_vs_reads_used", "title": "UMI ratio to the authors rises with the share of reads used", "kind": "scatter",
            "x": {"label": "reads used / reads available", "scale": "log"}, "y": {"label": "UMIs per cell, Singlet / authors", "scale": "log"},
            "series": [{"name": "sample", "tool": "singlet", "x": [r4(v) for v in s.frac_reads_used], "y": [r4(v) for v in s.umi_ratio_matched],
                        "text": s.gsm.tolist()}],
            "reference_lines": [{"axis": "y", "value": 1.0, "label": "equal depth"}]}


def caveats(S):
    ex = S[S.status != "scored"]
    c = [f"{len(ex)} of {len(S)} picked samples were not scored: " + "; ".join(f"{n} {r}" for r, n in ex.note.str.split(":").str[0].value_counts().items()) if len(ex) else "every picked sample was scored",
         "Samples were limited to those whose GEO sample record holds a Cell Ranger-style filtered matrix, which favours newer, better-documented studies; the 16 strata are equally weighted, not weighted by their share of the atlas.",
         "The authors' matrices are not uniform truth: Cell Ranger versions differ (exon-only before 7.0, introns after), and some authors filtered cells further (QC, doublets) before depositing, which lowers recovery and is not Singlet's error. The Singlet layer compared (exon or exon+intron) is chosen per sample from which one fits the authors' pseudobulk better.",
         "Nuclei and 5' strata were assigned from GEO text (regular expressions on the sample record); some may be mislabelled.",
         "Mouse samples were not screened for wrong-strand counting before Alpha; wrong_strand_frac is reported per sample.",
         "reads_available counts every run of the sample in SRA (NCBI runinfo), including non-GEX runs of the same GSM where they exist; it overstates the GEX reads for multi-library samples.",
         "The depth-matched correlation removes depth, not counting rules; it says the profiles agree at the atlas's depth, not that the atlas has the authors' depth."]
    return c


METHODS_A = (
    "**Samples.** Research Alpha = samples in published bundles of studies that are not withheld, minus the samples excluded by the Alpha policy "
    "(wrong strand, non-RNA library, empty matrix), in studies downloadable now, usable and QC tier 0 (47,702 samples). From these, candidates were drawn at random "
    "(seed 7, one per study) for human and mouse, labelled 10x 3' v2, 3' v3 or 5', and kept when the GEO sample record holds a filtered matrix (.h5 or "
    "matrix/barcodes/features, no 'raw' in the name). Nuclei were recognised from the GEO text. Up to 10 samples per stratum (species x chemistry x Singlet "
    "called cells <3,000 or >=3,000), one per study.\n\n"
    "**Singlet** values are read from the bundles exactly as data.singlet.bio serves them (exon_counts, intron_counts, cell_calls, summary.json), by HTTP "
    "ranges. **Truth** is the authors' deposited Cell Ranger matrix. Barcodes are matched on the 16-mer; genes on Ensembl ID (symbol when the deposit has none).\n\n"
    "**Metrics.** Recovery = Singlet-called & author / author; precision = Singlet-called & author / Singlet-called. On cells both called: UMIs per cell "
    "(Singlet / author), per-cell gene correlation (log1p Pearson, up to 2,000 cells) after thinning both profiles of each cell to the smaller total, against "
    "a noise floor of two independent thinnings of the authors' profile; pseudobulk Spearman over common genes. Singlet's exon or exon+intron layer is used, "
    "whichever fits the authors' pseudobulk better (Cell Ranger <7 counts exons only). reads_used / reads_available = Singlet's input reads over all SRA "
    "spots of the sample (NCBI runinfo). Values are medians over samples with bootstrap 95% CIs, pooled and per stratum.")
RESOURCES = []
ARTIFACTS = []


def main():
    mode = sys.argv[1]
    if mode == "samples":
        src, out = sys.argv[2], Path(sys.argv[3])
        rows = [flatten(json.load(open(p))) for p in sorted(glob.glob(os.path.join(src, "GSM*.json")))]
        S = pd.DataFrame(rows)
        S.to_csv(out / "alpha_samples.csv", index=False)
        print(S.status.value_counts().to_dict())
        return
    out = Path(sys.argv[2])
    S = pd.read_csv(out / "alpha_samples.csv")
    C = load_contract()
    art = out / "alpha_samples.csv"
    ARTIFACTS.append({"name": "alpha_samples.csv", "kind": "derived", "format": "csv",
                      "url": "https://data.singlet.bio/benchmarking/alpha/alpha_samples.csv",
                      "sha256": hashlib.sha256(art.read_bytes()).hexdigest(), "bytes": art.stat().st_size})
    eng = json.load(open(out / "alpha_engine_resources.json")) if (out / "alpha_engine_resources.json").exists() else None
    if eng:
        RESOURCES.extend(eng)
    else:
        RESOURCES.append({"tool": "singlet", "dataset": "pooled", "wall_s": 1051.8, "max_rss_gb": 49.0, "threads": 4,
                          "notes": "campaign median wall time per sample at 4 threads (27,578 GRCh38 samples); RSS from summary.json peak_rss_gb of a typical 30M-read sample"})
    docs = build_docs(S, C)
    cohort = {}
    for k, d in docs.items():
        for ds in d["datasets"]:
            cohort[ds["id"]] = {"data_kind": "real"}
        errs = C.validate(d, None)
        d_path = out / f"{k}.json"
        d_path.write_text(json.dumps(d, indent=1))
        print(k, "valid" if not errs else errs, C.capability_status(d))


if __name__ == "__main__":
    main()
