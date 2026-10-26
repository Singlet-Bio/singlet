"""mtdna: mitochondrial variants from 10x 3' scRNA-seq reads, scored on real data with real truth.

Artifacts (counts and summaries only; no site, allele count or haplotype is published) are written by
bench/public/genotype/lib/mt_eval.py:
  mt_haplotype_counts.tsv   Kang lanes: per pool, tool and donor, predicted / truth / true-positive homoplasmic variants
  mt_haplotype_pairs.tsv    Kang lanes: donor pairs separable by the truth haplotypes and by each tool's
  mt_voi_summary.tsv        MAESTER BPDCN712: per anonymised clone-defining variant, AUROC of per-cell VAF (Singlet, mgatk)
  mt_linemix_stats.tsv      MAESTER K562 + BT142: sufficient statistics of tool-vs-MAESTER allele fractions per line

    python bench/public/capabilities/mtdna.py ARTIFACT_DIR OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import contract  # noqa: E402
from _gcommon import (SINGLET_BINARY, SINGLET_BINARY_SHA256, SINGLET_COMMIT, artifact_entry, boot, code_url, now, r6)  # noqa: E402

CAP = "mtdna"
ART_BASE = "https://data.singlet.bio/benchmarking/1.0.0/mtdna"
DEV = "kang2018_ctrl"
KANG = {
    "kang2018_ctrl": ("Kang 2018 batch 2 control lane (8 donors)", {"gse": "GSE96583", "gsm": ["GSM2560248"], "srr": ["SRR5398238"]}),
    "kang2018_stim": ("Kang 2018 batch 2 IFN-beta lane (8 donors)", {"gse": "GSE96583", "gsm": ["GSM2560249"], "srr": ["SRR5398239"]}),
    "kang2018_b1_a": ("Kang 2018 batch 1 lane A (4 donors)", {"gse": "GSE96583", "gsm": ["GSM2560245"], "srr": ["SRR5398235"]}),
    "kang2018_b1_b": ("Kang 2018 batch 1 lane B (4 donors)", {"gse": "GSE96583", "gsm": ["GSM2560246"], "srr": ["SRR5398236"]}),
    "kang2018_b1_c": ("Kang 2018 batch 1 lane C (8 donors)", {"gse": "GSE96583", "gsm": ["GSM2560247"], "srr": ["SRR5398237"]}),
}
HAP_TRUTH = "homoplasmic mt variants of each donor in a second library of the same donors (cells labelled by demuxlet; mgatk counts; AF >= 0.9, depth >= 20 vs rCRS)"
TOOL_MAP = {"singlet": "singlet", "mgatk+vireo": "mgatk"}


def _hap(df):
    npred, tp, ntruth = df["n_pred_covered"].sum(), df["tp"].sum(), df["n_truth"].sum()
    return (tp / npred if npred else None), (tp / ntruth if ntruth else None)


def _corr(x):
    n = x["n"]
    num = n * x["sxy"] - x["sx"] * x["sy"]
    den = np.sqrt(max(n * x["sxx"] - x["sx"] ** 2, 0) * max(n * x["syy"] - x["sy"] ** 2, 0))
    return float(num / den) if den > 0 else None


def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    prov = json.loads((art / "provenance.json").read_text())
    hap = pd.read_csv(art / "mt_haplotype_counts.tsv", sep="\t")
    pairs = pd.read_csv(art / "mt_haplotype_pairs.tsv", sep="\t")
    voi = pd.read_csv(art / "mt_voi_summary.tsv", sep="\t")
    lm = pd.read_csv(art / "mt_linemix_stats.tsv", sep="\t")
    metrics = []

    def M(id_, name, unit, cat, ds, direction, sing, comps, rule, ci=None, notes=None):
        if sing is None:
            return
        m = {"id": id_, "name": name, "unit": unit, "category": cat, "dataset": ds, "direction": direction,
             "singlet": {"value": r6(sing)}, "rule": rule}
        if ci:
            m["singlet"]["ci"] = [r6(ci[0]), r6(ci[1])]
        cl = [{"tool": k, "value": r6(v)} for k, v in comps.items() if v is not None]
        if cl:
            m["competitors"] = cl
        if notes:
            m["notes"] = notes
        metrics.append(m)

    rule = {"type": "match_or_beat", "against": "best"}
    ds_k = [k for k in KANG if k in set(hap["pool"])]
    held = [k for k in ds_k if k != DEV]
    for ds in ds_k + (["pooled"] if len(held) > 1 else []):
        sel = held if ds == "pooled" else [ds]
        sub = {t: hap[(hap["pool"].isin(sel)) & (hap["tool"] == t)] for t in TOOL_MAP}
        sp, sr = _hap(sub["singlet"]); mp_, mr = _hap(sub["mgatk+vireo"]) if len(sub["mgatk+vireo"]) else (None, None)
        ci_p = ci_r = None
        if ds == "pooled":
            x = sub["singlet"].reset_index(drop=True)
            ci_p = boot(lambda i: x.iloc[i]["tp"].sum() / max(x.iloc[i]["n_pred_covered"].sum(), 1), len(x))
            ci_r = boot(lambda i: x.iloc[i]["tp"].sum() / max(x.iloc[i]["n_truth"].sum(), 1), len(x))
        note = "Pooled over the donors of the held-out lanes (development lane excluded); CI resamples donors." if ds == "pooled" else (
            "Development lane: Singlet's mt caller was developed here (PR #36)." if ds == DEV else None)
        M("haplotype_precision", "Donor mt-haplotype precision against replicate-lane truth", "fraction", "accuracy", ds, "higher", sp,
          {"mgatk": mp_}, rule, ci=ci_p, notes=note)
        M("haplotype_recall", "Donor mt-haplotype recall against replicate-lane truth", "fraction", "accuracy", ds, "higher", sr,
          {"mgatk": mr}, rule, ci=ci_r, notes=note)
    if len(pairs):
        pp = pairs[pairs["pool"].isin(held)] if held else pairs
        agr = {t: (pp[pp["tool"] == t]["agree"].mean() if (pp["tool"] == t).any() else None) for t in TOOL_MAP}
        M("donor_pair_separation_agreement", "Donor pairs separated by mt haplotype exactly when the truth haplotypes separate them",
          "fraction", "agreement", "pooled", "higher", agr["singlet"], {"mgatk": agr["mgatk+vireo"]}, rule,
          notes=f"{int((pp['tool'] == 'singlet').sum())} donor pairs in the held-out lanes.")

    if len(voi):
        ev = voi.dropna(subset=["auroc_singlet", "auroc_mgatk"])
        if len(ev):
            a_s, a_m = ev["auroc_singlet"].values, ev["auroc_mgatk"].values
            M("clonal_variant_auroc", "Per-cell VAF separates truth-positive cells of a clone-defining variant (mean AUROC over variants)",
              "AUROC", "accuracy", "maester_bpdcn712_3p", "higher", float(a_s.mean()), {"mgatk": float(a_m.mean())}, rule,
              ci=boot(lambda i: float(a_s[i].mean()), len(a_s)),
              notes=f"{len(ev)} of {len(voi)} author-validated clone-defining variants have >= 10 positive and >= 10 negative cells with >= 3 reads in the standard 3' library.")
            M("clonal_variants_evaluable", "Clone-defining variants with enough 3' coverage to score", "variants", "coverage",
              "maester_bpdcn712_3p", "none", float(len(ev)), {}, {"type": "informational"})
        M("clonal_variants_in_variant_list", "Clone-defining variants listed in Singlet's heteroplasmic variant list", "fraction", "coverage",
          "maester_bpdcn712_3p", "none", float(voi["singlet_listed"].mean()), {}, {"type": "informational"},
          notes="Singlet-only: mgatk has no variant list without its R step.")
        if "singlet_call_tp" in voi:
            tp, fp = voi["singlet_call_tp"].sum(), voi["singlet_call_fp"].sum()
            npos, nneg = voi["n_pos_eval"].sum(), voi["n_neg_eval"].sum()
            M("clonal_call_sensitivity", "Share of truth-positive cells (>= 3 reads) Singlet flags heteroplasmic at the variant", "fraction",
              "accuracy", "maester_bpdcn712_3p", "none", float(tp / npos) if npos else None, {}, {"type": "informational"})
            M("clonal_call_false_positive_rate", "Share of other clone cells (>= 3 reads) Singlet flags heteroplasmic at the variant", "fraction",
              "accuracy", "maester_bpdcn712_3p", "none", float(fp / nneg) if nneg else None, {}, {"type": "informational"})

    if len(lm):
        for line in ("K562", "BT142"):
            st = {t: lm[(lm["line"] == line) & (lm["tool"] == t)] for t in ("singlet", "mgatk")}
            if all(len(v) for v in st.values()):
                s, g = st["singlet"].iloc[0], st["mgatk"].iloc[0]
                if s["n"] >= 5:
                    M(f"line_af_pearson_{line.lower()}", f"Agreement of 3' allele fractions with MAESTER truth in {line} (Pearson r)",
                      "r", "accuracy", "maester_k562_bt142_3p", "higher", _corr(s), {"mgatk": _corr(g)}, rule,
                      notes=f"{int(s['n'])} variant-line pairs with MAESTER AF >= 0.02 and >= 100 3' reads in both tools.")
                    M(f"line_af_mae_{line.lower()}", f"Mean absolute error of 3' allele fractions against MAESTER truth in {line}",
                      "AF", "accuracy", "maester_k562_bt142_3p", "lower", float(s["sabs"] / s["n"]), {"mgatk": float(g["sabs"] / g["n"])}, rule)

    ds_ids = [(k, KANG[k]) for k in ds_k]
    datasets = [{"id": k, "title": v[0], "accessions": v[1], "truth": HAP_TRUTH, "data_kind": "real",
                 **({"notes": "development data"} if k == DEV else {})} for k, v in ds_ids]
    if len(voi):
        datasets.append({"id": "maester_bpdcn712_3p", "title": "MAESTER BPDCN712 bone marrow, standard 10x 3' library", "data_kind": "real",
                         "accessions": {"gse": "GSE182685", "srr": ["SRR15598776"]},
                         "truth": "26 clone-defining mt variants with positive cells from the paired MAESTER (mt-enriched) library of the same cells, authors' calls"})
    if len(lm):
        datasets.append({"id": "maester_k562_bt142_3p", "title": "MAESTER K562 + BT142 mixture, standard 10x 3' library", "data_kind": "real",
                         "accessions": {"gse": "GSE182685", "srr": ["SRR15598773"]},
                         "truth": "per-line allele fractions of mt variants from the paired MAESTER library, authors' cell-line labels"})
    doc = {
        "schema": contract.SCHEMA_CAPABILITY, "schema_version": contract.SCHEMA_VERSION, "capability": CAP,
        "title": "Mitochondrial variants and donor haplotypes", "conclusion": _conclusion(metrics), "methods": METHODS,
        "status": "final", "generated_at": now(), "owner": "pipeline-benchmark",
        "singlet": {"version": "1.0.0", "commit": SINGLET_COMMIT, "branch": "prod/campaign", "binary": SINGLET_BINARY,
                    "binary_sha256": SINGLET_BINARY_SHA256},
        "tools": [
            {"name": "singlet", "version": "1.0.0", "role": "singlet", "params": prov["singlet_command"]},
            {"name": "mgatk", "version": "0.9.0", "role": "competitor", "url": "https://github.com/caleblareau/mgatk",
             "citation": "Lareau et al. 2021, Nat Biotechnol 39:451", "params": prov["mgatk_command"]},
            {"name": "vireo", "version": "0.5.9", "role": "helper", "params": prov["vireo_command"]},
            {"name": "maester-authors", "version": "n/a", "role": "truth", "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182685",
             "params": "Miller et al. 2022: clone-defining variants, positive cells, per-line allele fractions"},
            {"name": "kang-replicate-lane", "version": "n/a", "role": "truth", "params": HAP_TRUTH},
        ],
        "datasets": datasets, "metrics": metrics, "resources": prov.get("resources", []),
        "figures": _figures(hap, voi, ds_k),
        "artifacts": [artifact_entry(art / f, f, "derived" if f.endswith(".tsv") else "singlet_output", ART_BASE)
                      for f in ("mt_haplotype_counts.tsv", "mt_haplotype_pairs.tsv", "mt_voi_summary.tsv", "mt_linemix_stats.tsv", "provenance.json")],
        "reach": [
            {"tool": "singlet", "output": "per-donor homoplasmic mt haplotype without donor labels", "supported": "yes", "note": "private/mt/donor*_mt_variants.vcf, from its own clusters"},
            {"tool": "mgatk", "output": "per-donor homoplasmic mt haplotype without donor labels", "supported": "no", "note": "needs cluster labels from another tool (vireo here)"},
            {"tool": "singlet", "output": "per-cell heteroplasmy matrix from 3' scRNA-seq", "supported": "yes", "note": "mt_heteroplasmy"},
            {"tool": "mgatk", "output": "per-cell heteroplasmy matrix from 3' scRNA-seq", "supported": "yes", "note": "allele counts; variant calling needs its R step"},
        ],
        "code": [
            {"path": "include/singlet/pileup/mt_heteroplasmy.h", "role": "implementation", "url": code_url("include/singlet/pileup/mt_heteroplasmy.h", SINGLET_COMMIT)},
            {"path": "bench/public/capabilities/mtdna.py", "role": "metric", "url": code_url("bench/public/capabilities/mtdna.py", prov["code_commit"])},
            {"path": "bench/public/genotype", "role": "harness", "url": code_url("bench/public/genotype", prov["code_commit"], tree=True)},
        ],
        "reproduce": {"module": "bench/public/capabilities/mtdna.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "sbatch bench/public/genotype/hpc/export.sbatch mt"},
        "caveats": [
            "Haplotypes, variant positions, allele counts and per-site allele fractions are not published; the artifacts hold counts per donor, per anonymised variant and sufficient statistics only.",
            "The Kang haplotype truth comes from a second library of the same donors, counted by mgatk, so mgatk-based predictions share its counting rules; recall is measured against every truth variant, including positions the tested lane barely covers.",
            "Singlet's mt caller was developed on the Kang batch 2 control lane (PR #36); that lane is reported but kept out of the pooled numbers.",
            "mgatk is run with its documented 10x settings (bcall, CB and UB tags, rCRS); its variant calling is an R step that was not run, so heteroplasmic variant lists are not compared, only its counts.",
            "In standard 3' libraries mt reads cluster at transcript 3' ends; only the clone-defining variants with enough coverage are scorable, and the other variants of the MAESTER truth cannot be tested with this data type.",
        ],
    }
    return contract.finalize(doc)


METHODS = (
    "Real data with real truth. (1) Kang 2018 lanes: each donor's homoplasmic mtDNA haplotype is taken from a second library of the "
    "same donors (cells labelled by the authors' demuxlet calls, mgatk allele counts, AF >= 0.9 and depth >= 20 against rCRS). "
    "Singlet's genotype-free per-donor mt variant files (from its own clusters) are compared with it, as are mgatk counts pooled "
    "by vireo's clusters, the pipeline a user would assemble. Precision is the share of predicted variants at covered truth "
    "positions that are in the truth; recall is the share of truth variants recovered. (2) MAESTER BPDCN712: 26 clone-defining mt "
    "variants whose positive cells were established from the paired mt-enriched library of the same cells; for each, the AUROC "
    "of the per-cell VAF from the standard 3' library separates positive cells from the other clone cells (both tools on the same "
    "alignments and cells; mgatk counts UMI-deduplicated molecules, Singlet counts reads). (3) MAESTER K562 + BT142: agreement of "
    "per-line allele fractions from the 3' library with the MAESTER fractions. Singlet is run end to end (default pipeline, "
    "automatic donor count); mgatk 0.9.0 runs on the chrM reads of Singlet's BAM for the same cells. Rules are match-or-beat "
    "against mgatk with zero tolerance; informational metrics have no competitor. Variants and counts stay private."
)


def _conclusion(metrics):
    def get(i, ds):
        for m in metrics:
            if m["id"] == i and m["dataset"] == ds:
                return m
    p, r_ = get("haplotype_precision", "pooled"), get("haplotype_recall", "pooled")
    a = get("clonal_variant_auroc", "maester_bpdcn712_3p")
    s = []
    if p and r_:
        mg = {c["tool"]: c["value"] for c in p.get("competitors", [])}.get("mgatk"), {c["tool"]: c["value"] for c in r_.get("competitors", [])}.get("mgatk")
        s.append(f"Singlet's genotype-free per-donor mt haplotypes reach precision {p['singlet']['value']:.2f} and recall {r_['singlet']['value']:.2f} "
                 f"against replicate-lane truth on held-out Kang lanes (mgatk on vireo clusters: {mg[0]:.2f} / {mg[1]:.2f})." if None not in mg else
                 f"Singlet's per-donor mt haplotypes reach precision {p['singlet']['value']:.2f} and recall {r_['singlet']['value']:.2f} on held-out Kang lanes.")
    if a:
        mg = {c["tool"]: c["value"] for c in a.get("competitors", [])}.get("mgatk")
        s.append(f"For MAESTER clone-defining variants the 3' per-cell VAF has mean AUROC {a['singlet']['value']:.2f} (mgatk {mg:.2f})." if mg is not None else "")
    return " ".join(x for x in s if x)


def _figures(hap, voi, ds_k):
    figs = []
    if len(voi):
        ev = voi.dropna(subset=["auroc_singlet", "auroc_mgatk"])
        if len(ev):
            figs.append({"id": "voi_auroc", "title": "Clone-defining variants: per-cell VAF AUROC", "kind": "scatter",
                         "caption": "One point per MAESTER BPDCN712 clone-defining variant (anonymised); above the diagonal Singlet separates truth-positive cells better than mgatk.",
                         "x": {"label": "mgatk AUROC", "min": 0.4, "max": 1}, "y": {"label": "Singlet AUROC", "min": 0.4, "max": 1},
                         "series": [{"name": "variants", "tool": "singlet", "x": [r6(v) for v in ev["auroc_mgatk"]], "y": [r6(v) for v in ev["auroc_singlet"]]}]})
    return figs


def main(argv=None):
    a = argv or sys.argv[1:]
    doc = compute(Path(a[0]))
    Path(a[1]).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(a[1], contract.capability_status(doc))


if __name__ == "__main__":
    main()
