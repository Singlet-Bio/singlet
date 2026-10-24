# SPDX-License-Identifier: MIT
"""Tables for docs/called-v2.md from every ``<GSM>.v2.json`` (validate.py).

* variant ranking against the authors' Cell Ranger calls (studies weighted equally: per-sample metrics averaged within a study
  first), with a leave-one-study-out choice so the reported agreement is out of sample;
* hashtag check (GSE108313), species purity and profile-test share of the added barcodes;
* the prototype's defaults per sample and per protocol family; exon/intron double counting per sample.

Usage: ``python -P aggregate.py RESULTS --out REPORT`` -> ``REPORT/summary.json``, ``REPORT/tables.md``, ``REPORT/samples.csv``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

EXCLUDE_GSE = {"GSE145370": "TCR libraries", "GSE233574": "Parse sub-libraries (broken stored call)"}


def usable(R):
    uc = R.get("umi_called") or {}
    return R["gse"] not in EXCLUDE_GSE and R["n_called"] >= 200 and (uc.get("0.5") or 0) >= 100


def vkey(run, v):
    return (run["matrix"], str(run["lower_rule"]), v["mode"], v["test_min"], str(v["retain"]), v["fdr"])


def kstr(k):
    m, l, mode, t, r, f = k
    return f"{m} lower={l} {mode} test>{t} retain={r} fdr={f}"


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def study_mean(per_sample: dict, field: str):
    """per_sample: {(gse, gsm): metrics} -> mean over studies of the within-study mean of metrics['author'][field]."""
    by = defaultdict(list)
    for (gse, _), m in per_sample.items():
        a = m.get("author")
        if a:
            by[gse].append(a[field])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None, {g: float(np.mean(v)) for g, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Rs = []
    for f in sorted(glob.glob(os.path.join(a.results, "GSE*", "GSM*.v2.json"))):
        with open(f) as fh:
            Rs.append(json.load(fh))
    S = {"n_samples": len(Rs), "n_usable": sum(usable(R) for R in Rs)}
    md = []

    # ------------------------------------------------------------------ variants x samples
    table = defaultdict(dict)  # key -> {(gse, gsm): metrics}
    for R in Rs:
        if not usable(R):
            continue
        for run in R["runs"]:
            for v in run.get("variants", []):
                table[vkey(run, v)][(R["gse"], R["gsm"])] = v
        for name in ("stored", "ordmag_stored_counts", "cr3_like", "cr3_like_union"):
            c = R["comparators"].get(name)
            if c and "error" not in c:
                table[("comparator", name, "", "", "", "")][(R["gse"], R["gsm"])] = c
        table[("prototype", "", "", "", "", "")][(R["gse"], R["gsm"])] = R["prototype"]

    # GSE213957's deposited metadata lists only the cells that passed the authors' QC: not a call
    auth_samples = {(R["gse"], R["gsm"]) for R in Rs if usable(R) and R.get("author") and R["gse"] != "GSE213957"}
    studies = sorted({g for g, _ in auth_samples})
    S["author_samples"] = sorted(f"{g}/{s}" for g, s in auth_samples)

    def score(key, keep_studies):
        ps = {k: m for k, m in table[key].items() if k in auth_samples and k[0] in keep_studies}
        if len({g for g, _ in ps}) < len(keep_studies):
            return None
        return study_mean(ps, "jaccard")[0]

    ranking = []
    for key in table:
        j = score(key, studies)
        if j is None:
            continue
        ps = {k: m for k, m in table[key].items() if k in auth_samples}
        ranking.append({"variant": kstr(key) if key[0] != "comparator" and key[0] != "prototype" else f"{key[0]} {key[1]}",
                        "key": list(key), "jaccard": j, "recall": study_mean(ps, "recall")[0],
                        "precision": study_mean(ps, "precision")[0], "per_study_jaccard": study_mean(ps, "jaccard")[1]})
    ranking.sort(key=lambda r: -r["jaccard"])
    S["ranking_top"] = ranking[:40]
    S["ranking_comparators"] = [r for r in ranking if r["key"][0] in ("comparator", "prototype")]

    # leave one study out: choose the best EmptyDrops variant on the other studies, score it on the held-out one
    cand = [k for k in table if k[0] not in ("comparator", "prototype")]
    loso = []
    for held in studies:
        rest = [s for s in studies if s != held]
        best = max((k for k in cand if score(k, rest) is not None), key=lambda k: score(k, rest))
        ps = {k: m for k, m in table[best].items() if k[0] == held and k in auth_samples}
        base = {k: m for k, m in table[("comparator", "stored", "", "", "", "")].items() if k[0] == held and k in auth_samples}
        cr3 = {k: m for k, m in table.get(("comparator", "cr3_like_union", "", "", "", ""), {}).items() if k[0] == held and k in auth_samples}
        proto = {k: m for k, m in table[("prototype", "", "", "", "", "")].items() if k[0] == held and k in auth_samples}
        row = {"held_out": held, "chosen": kstr(best)}
        for nm, d in (("chosen", ps), ("stored", base), ("cr3_like_union", cr3), ("prototype", proto)):
            for fld in ("jaccard", "recall", "precision"):
                row[f"{nm}_{fld}"] = study_mean(d, fld)[0] if d else None
        loso.append(row)
    S["loso"] = loso

    # ------------------------------------------------------------------ hashtags (GSE108313)
    hto = [R for R in Rs if R.get("hto")]
    if hto:
        R = hto[0]
        rows = [{"variant": "stored", **R["hto"]["stored_call"]}]
        for nm in ("cr3_like_union", "cr3_like", "ordmag_stored_counts"):
            c = R["comparators"].get(nm)
            if c and "hto" in c:
                rows.append({"variant": nm, "n": c["n"], **c["hto"]})
        rows.append({"variant": "prototype", "n": R["prototype"]["n"], **R["prototype"]["hto"]})
        for run in R["runs"]:
            for v in run.get("variants", []):
                if v["mode"] == "union" and v["retain"] == "knee":
                    rows.append({"variant": kstr(vkey(run, v)), "n": v["n"], **v["hto"]})
        S["hto"] = {"sample": f"{R['gse']}/{R['gsm']}", **{k: R["hto"][k] for k in R["hto"] if k != "stored_call"}, "rows": rows}

    # ------------------------------------------------------------------ prototype per sample, double counting
    rows = []
    for R in Rs:
        p = R["prototype"]
        pa = p.get("params", {})
        dc = R.get("double_counting", {})
        rows.append({
            "sample": f"{R['gse']}/{R['gsm']}", "protocol": R.get("protocol"), "family": R.get("family"), "usable": usable(R),
            "n_barcodes": R["n_barcodes"], "called": R["n_called"], "called_v2": p["n"], "added": p["added"],
            "dropped": p["dropped"], "added_umi_med": p.get("added_umi_med"), "lower": pa.get("lower_value"),
            "retain": pa.get("retain_value"), "alpha": pa.get("alpha"), "n_ambient": pa.get("n_ambient"),
            "author_recall_stored": (R.get("stored_call") or {}).get("recall"),
            "author_recall_v2": (p.get("author") or {}).get("recall"),
            "author_precision_stored": (R.get("stored_call") or {}).get("precision"),
            "author_precision_v2": (p.get("author") or {}).get("precision"),
            "author_jaccard_stored": (R.get("stored_call") or {}).get("jaccard"),
            "author_jaccard_v2": (p.get("author") or {}).get("jaccard"),
            "cr3_union_added": (R["comparators"].get("cr3_like_union") or {}).get("added"),
            "species_added_pure_frac": (p.get("species") or {}).get("added_pure_frac"),
            "profile_added_sig_frac": (p.get("profile") or {}).get("added_sig_frac"),
            "intron_dup_share_ambient": dc.get("ambient_intron_dup_share"),
            "frac_1e1i_one_molecule": dc.get("frac_1e1i_one_molecule"),
            "called_min_share_med": (dc.get("called_min_share_upper_bound") or {}).get("0.5"),
            "seconds": R.get("seconds"),
        })
    with open(os.path.join(a.out, "samples.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fam = defaultdict(list)
    for r in rows:
        if r["usable"]:
            fam[r["family"]].append(r)
    S["by_family"] = {f: {"samples": len(v), "added_med": med([r["added"] for r in v]),
                          "added_rel_med": med([r["added"] / max(1, r["called"]) for r in v]),
                          "dropped_med": med([r["dropped"] for r in v]), "added_umi_med": med([r["added_umi_med"] for r in v]),
                          "lower_med": med([r["lower"] for r in v]),
                          "species_added_pure_frac_med": med([r["species_added_pure_frac"] for r in v]),
                          "profile_added_sig_frac_med": med([r["profile_added_sig_frac"] for r in v])} for f, v in fam.items()}
    us = [r for r in rows if r["usable"]]
    S["double_counting"] = {"intron_dup_share_ambient_med": med([r["intron_dup_share_ambient"] for r in us]),
                            "intron_dup_share_ambient_range": [min(r["intron_dup_share_ambient"] or 0 for r in us),
                                                               max(r["intron_dup_share_ambient"] or 0 for r in us)],
                            "frac_1e1i_one_molecule_med": med([r["frac_1e1i_one_molecule"] for r in us]),
                            "called_min_share_med": med([r["called_min_share_med"] for r in us])}
    S["prototype_overall"] = {"added_med": med([r["added"] for r in us]), "added_range": [min(r["added"] for r in us), max(r["added"] for r in us)],
                              "dropped_total": int(sum(r["dropped"] for r in us)), "seconds_med": med([r["seconds"] for r in us])}

    # matrix choice summary: best variant per matrix
    for m in ("sum", "exon", "max"):
        best = [r for r in ranking if r["key"][0] == m]
        S[f"best_{m}"] = best[0] if best else None

    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(S, fh, indent=1)

    # ------------------------------------------------------------------ markdown
    def f3(x):
        return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

    md.append("## Ranking vs authors (study-weighted)\n\n| variant | jaccard | recall | precision |\n|---|---|---|---|")
    for r in ranking[:25] + S["ranking_comparators"]:
        md.append(f"| {r['variant']} | {f3(r['jaccard'])} | {f3(r['recall'])} | {f3(r['precision'])} |")
    md.append("\n## Leave one study out\n\n| held out | chosen | chosen J/R/P | prototype J/R/P | stored J/R/P | CR3-like J/R/P |\n|---|---|---|---|---|---|")
    for r in loso:
        jrp = lambda n: "/".join(f3(r[f"{n}_{x}"]) for x in ("jaccard", "recall", "precision"))  # noqa: E731
        md.append(f"| {r['held_out']} | {r['chosen']} | {jrp('chosen')} | {jrp('prototype')} | {jrp('stored')} | {jrp('cr3_like_union')} |")
    if "hto" in S:
        md.append("\n## Hashtags\n\n| variant | n | singlet called | added singlet | added negative | added measured | singlet recall >=20 |\n|---|---|---|---|---|---|---|")
        for r in S["hto"]["rows"][:60]:
            md.append(f"| {r['variant']} | {r.get('n', '')} | {r['singlet_called']} | {r.get('added_singlet', '')} | {r.get('added_negative', '')} | "
                      f"{r.get('added_measured', '')} | {f3(r.get('singlet_recall_ge20'))} |")
    md.append("\n## Prototype per sample\n\n| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]))
    for r in rows:
        md.append("| " + " | ".join(f3(v) if not isinstance(v, str) else v for v in r.values()) + " |")
    with open(os.path.join(a.out, "tables.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(json.dumps({k: S[k] for k in ("n_samples", "n_usable", "loso", "double_counting", "prototype_overall")}, indent=1))


if __name__ == "__main__":
    main()
