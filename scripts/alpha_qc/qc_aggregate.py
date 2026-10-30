# SPDX-License-Identifier: MIT
"""Research Alpha QC: aggregate the per-study results of a run into small TSV tables and a JSON summary.

    python qc_aggregate.py --run DIR --selection selection.tsv --out OUTDIR [--cl cl-basic.obo] [--sweep sweep.json]
"""

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

IMMUNE_L1 = {"CL:0000988", "CL:0000738", "CL:0000842", "CL:0000542"}  # hematopoietic cell, leukocyte, mononuclear, lymphocyte


def obo_names(path):
    names = {}
    if not path or not os.path.exists(path):
        return names
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("id: "):
            cur = line[4:].strip()
        elif line.startswith("name: ") and cur:
            names[cur] = line[6:].strip()
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cl", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = obo_names(a.cl)
    sel = {}
    with open(a.selection) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(hdr, line.rstrip("\n").split("\t")))
            sel[r["gse"]] = r
    res = defaultdict(list)
    for f in glob.glob(os.path.join(a.run, "*.json")):
        r = json.load(open(f))
        res[r["gse"]].append(r)
    checks = defaultdict(lambda: {"pass": 0, "fail": 0, "warn": 0, "studies_fail": set(), "studies_warn": set()})
    fails, mats, labs, sizes, guard, markers, mtref, prov, fatal = [], [], [], [], [], Counter(), defaultdict(Counter), [], []
    done = set()
    tissue = defaultdict(list)
    sex_amb = []
    for g, parts in sorted(res.items()):
        sh = sel.get(g, {})
        nsh = int(sh.get("n_shards") or 1)
        got = sorted(p["shard"] for p in parts)
        if len(got) < nsh:
            fatal.append([g, f"only {len(got)} of {nsh} shards"])
        for p in parts:
            if "fatal" in p:
                fatal.append([g, p["shard"], p["fatal"][-400:]])
                continue
            ck = p.get("checks") or {}
            gp = ck.get("3_guard_private_refused")
            if gp:  # 404 is also a refusal (the guard answers 404 for members of excluded samples)
                ok404 = [it for it in gp["fail_items"] if it.get("status") in (403, 404)]
                gp["fail"] -= len(ok404)
                gp["pass"] += len(ok404)
                gp["fail_items"] = [it for it in gp["fail_items"] if it not in ok404]
            for c, v in ck.items():
                C = checks[c]
                C["pass"] += v["pass"]
                C["fail"] += v["fail"]
                C["warn"] += v["warn"]
                if v["fail"]:
                    C["studies_fail"].add(g)
                if v["warn"]:
                    C["studies_warn"].add(g)
                for it in v["fail_items"]:
                    fails.append([g, p["shard"], c, "FAIL", json.dumps(it, default=str)[:1500]])
                for it in v["warn_items"][:3]:
                    fails.append([g, p["shard"], c, "warn", json.dumps(it, default=str)[:600]])
            stud = p.get("study") or {}
            if p["shard"].startswith("0/"):
                done.add(g)
                sizes.append([g, stud.get("species"), stud.get("n_samples_manifest"), stud.get("s1_bytes_total"),
                              (stud.get("legacy") or {}).get("bytes"), (stud.get("legacy_head") or {}).get("bytes")])
                gp = stud.get("guard_probes") or {}
                guard.append([g, json.dumps(gp.get("private")), json.dumps(gp.get("excluded_sample")),
                              json.dumps(stud.get("guard_full_get")), stud.get("n_legacy_excluded"),
                              stud.get("legacy_genotype_members"), json.dumps(stud.get("legacy_layer_counts"))])
                for h in (stud.get("manifest_marker_hits") or []) + (stud.get("study_file_marker_hits") or []):
                    markers[(h["marker"], h["where"].split(":")[0], h["context"][:80])] += 1
            for gsm, s in (p.get("samples") or {}).items():
                for mem, d in (s.get("matrices") or {}).items():
                    mats.append([g, gsm, mem, all(d.get(k, True) for k in ("shape", "nnz", "indptr", "indices", "values",
                                                                            "rownames", "colnames", "kv")),
                                 d.get("shape_legacy"), d.get("nnz_legacy"), d.get("legacy_vt_code"), d.get("s1_value_type"),
                                 d.get("t_legacy_s"), d.get("t_s1_decode_s"), d.get("kind", 1)])
                for h in s.get("marker_hits") or []:
                    w = h["where"].split(":", 1)[1] if ":" in h["where"] else h["where"]
                    markers[(h["marker"], w.split(":")[0] if w.startswith("cells") else w, h.get("context", "")[:80])] += 1
                if s.get("mt_reference"):
                    mtref[stud.get("reference_build")][s["mt_reference"]["sha256"][:16]] += 1
                pv = s.get("provenance") or {}
                prov.append([g, gsm, pv.get("reads_used"), pv.get("reads_used_source"), pv.get("wrong_strand_fraction"),
                             pv.get("strand_class"), pv.get("library_type")])
                lb = s.get("labels") or {}
                nc = s.get("n_called_cells_table")
                if nc is None:
                    continue
                sc = lb.get("sex_counts") or {}
                amb = sc.get("ambiguous", 0) / nc if nc else None
                l1 = lb.get("level1_counts") or {}
                tl1 = sum(l1.values())
                imm = sum(v for k, v in l1.items() if k in IMMUNE_L1) / tl1 if tl1 else None
                top1 = max(l1, key=l1.get) if l1 else ""
                labs.append([g, gsm, stud.get("species"), sh.get("tissues", ""), nc, s.get("n_called_ed"),
                             round(lb.get("cell_type_frac_called", 0), 4) if "cell_type_frac_called" in lb else "",
                             round(lb.get("sex_frac_called", 0), 4) if "sex_frac_called" in lb else "",
                             round(amb, 4) if amb is not None else "", json.dumps(sc),
                             round(lb.get("ancestry_frac_called", 0), 4) if "ancestry_frac_called" in lb else "",
                             json.dumps(lb.get("ancestry_counts") or {}), top1, names.get(top1, ""),
                             round(imm, 4) if imm is not None else "", lb.get("cell_type_on_uncalled", "")])
                if amb is not None and stud.get("species") in ("human", "mouse"):
                    sex_amb.append((amb, nc))
                try:
                    tg = json.loads(sh.get("tissues") or "[]")
                except ValueError:
                    tg = []
                for t in tg or ["(none)"]:
                    if imm is not None:
                        tissue[t].append(imm)

    def w(name, header, rows):
        with open(os.path.join(a.out, name), "w") as fh:
            fh.write("\t".join(header) + "\n")
            for r in rows:
                fh.write("\t".join("" if x is None else str(x) for x in r) + "\n")

    w("check_summary.tsv", ["check", "pass", "fail", "warn", "n_studies_fail", "n_studies_warn", "studies_fail_first"],
      [[c, v["pass"], v["fail"], v["warn"], len(v["studies_fail"]), len(v["studies_warn"]),
        ",".join(sorted(v["studies_fail"])[:15])] for c, v in sorted(checks.items())])
    w("failures.tsv", ["gse", "shard", "check", "level", "evidence"], fails)
    w("matrices.tsv", ["gse", "gsm", "member", "equal", "shape", "nnz", "legacy_vt", "s1_vt", "t_legacy_s", "t_s1_s", "kind"], mats)
    w("labels.tsv", ["gse", "gsm", "species", "tissues", "n_called", "n_called_ed", "cell_type_frac", "sex_frac", "sex_ambiguous_frac",
                     "sex_counts", "ancestry_frac", "ancestry_counts", "top_level1", "top_level1_name", "immune_frac_level1",
                     "cell_type_on_uncalled"], labs)
    w("size.tsv", ["gse", "species", "n_samples", "s1_bytes", "legacy_bytes_manifest", "legacy_bytes_head"], sizes)
    w("guard.tsv", ["gse", "private_probes", "excluded_sample_probes", "full_get", "n_legacy_excluded", "legacy_genotype_members",
                    "legacy_layer_counts"], guard)
    w("markers.tsv", ["marker", "where", "context", "count"], [[k[0], k[1], k[2].replace("\t", " ").replace("\n", " "), v]
                                                               for k, v in markers.most_common()])
    w("provenance.tsv", ["gse", "gsm", "reads_used", "reads_used_source", "wrong_strand_fraction", "strand_class", "library_type"], prov)
    w("tissue_immune.tsv", ["tissue_group", "n_samples", "median_immune_frac", "q10", "q90"],
      [[t, len(v), round(statistics.median(v), 3), round(sorted(v)[len(v) // 10], 3), round(sorted(v)[(9 * len(v)) // 10], 3)]
       for t, v in sorted(tissue.items(), key=lambda kv: -len(kv[1]))])
    s1b = sum(r[3] or 0 for r in sizes)
    lgb = sum(r[5] or r[4] or 0 for r in sizes)
    tot_amb = sum(x * n for x, n in sex_amb) / max(1, sum(n for _x, n in sex_amb))
    summ = {
        "studies_with_results": len(res), "studies_selected": len(sel), "studies_shard0_done": len(done),
        "missing": sorted(set(sel) - set(res)), "fatal": fatal,
        "checks": {c: {k: (len(v[k]) if isinstance(v[k], set) else v[k]) for k in v} for c, v in sorted(checks.items())},
        "n_matrices": len(mats), "n_matrices_equal": sum(1 for m in mats if m[3]),
        "matrices_by_member": dict(Counter(m[2] for m in mats)),
        "size": {"s1_bytes": s1b, "legacy_bytes": lgb, "ratio": s1b / lgb if lgb else None},
        "sex_ambiguous_frac_cellweighted": tot_amb,
        "sex_ambiguous_frac_median_sample": statistics.median([x for x, _ in sex_amb]) if sex_amb else None,
        "mt_reference_distinct_per_build": {b: len(c) for b, c in mtref.items()},
        "mt_reference_top": {b: c.most_common(3) for b, c in mtref.items()},
    }
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: summ[k] for k in ("studies_with_results", "studies_selected", "n_matrices", "n_matrices_equal", "size")}))


if __name__ == "__main__":
    main()
