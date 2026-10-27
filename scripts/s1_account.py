# SPDX-License-Identifier: MIT
"""Final accounting of a .s1 transcode campaign (read-only: R2 reads, local files; nothing is written to R2 or D1).

    python -P scripts/s1_account.py --listing plan.tsv --policy alpha-policy.json --runs RUN_DIR [RUN_DIR ...] \\
        [--cidx-dir $WORK/state/s1/<cidx run>] [--sample-objects 300] --out account.json --tsv account.tsv

1. Every study of the frozen listing (``MODE=plan``) gets one final status from the latest results row of the runs (later runs win):
   ``verified`` (written, verified, and the manifest in R2 is the one the run verified), ``withheld`` (release policy),
   ``corrupt`` (the legacy bundle fails its CRC/size checks: nothing to transcode), ``failed`` (with the reason) or ``missing``.
2. Release promises over every manifest in R2 (``s1/m/``): no withheld study, no excluded accession anywhere in the text, no member
   of a withheld layer (:func:`singlet.s1.policy.omitted_layer`) or restricted member in any sample or study file; plus the member
   lists read from a random sample of the sample objects themselves, which must match their manifest entries.
3. With ``--cidx-dir``: the cell index report, the D1 exports checked with the rules of the website's ``s1:labels-sql`` loader,
   label coverage, and where chromosomal sex stays ambiguous (per sample: the website gives a sample a sex fact only when at least
   20 of its cells are sexed).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import s1_campaign as camp  # noqa: E402

ACC = re.compile(r"\b(GS[EM]\d+)\b")
CORRUPT = re.compile(r"CRC/size mismatch|bad CRC|BadZipFile|truncated|not a zip", re.IGNORECASE)
NUCLEI = re.compile(r"nucle(us|i|ar)|snRNA|sn-RNA|snSeq|single[- ]nuc", re.IGNORECASE)
# the website loader (singlet-website scripts/s1/labels-sql.ts): rows that fail these stop the load
ITEM_RE = {"cell_type": re.compile(r"^CL:\d{7}$"), "sex": re.compile(r"^PATO:\d{7}$"), "ancestry": re.compile(r"^HANCESTRO:\d{4}$")}
FEMALE, MALE = "PATO:0000383", "PATO:0000384"
MIN_CELLS, SEX_MIN_SHARE = 20, 0.2


def latest_rows(run_dirs):
    rows, source = {}, {}
    for d in run_dirs:
        for p in sorted(Path(d).glob("results-*.jsonl"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
            for ln in p.read_text("utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    rows[r["gse"]] = r
                    source[r["gse"]] = str(d)
    return rows, source


def study_log_tail(run_dir: str, gse: str, n: int = 4000) -> str:
    p = Path(run_dir) / "study-logs" / f"{gse}.log"
    return p.read_text("utf-8", errors="replace")[-n:] if p.exists() else ""


def gsm_meta(m: dict) -> dict:
    from singlet.s1 import atlas

    return atlas._study_gsm_meta(m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listing", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories, oldest first")
    ap.add_argument("--cidx-dir")
    ap.add_argument("--sample-objects", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--bucket", default="singlet-data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tsv", required=True)
    a = ap.parse_args(argv)

    from singlet._restricted import is_restricted_member
    from singlet.s1.policy import Policy, omitted_layer

    pol = Policy.load(a.policy)
    bad_acc = set(pol.excluded) | set(pol.withheld)
    listing = [ln.split("\t")[0] for ln in Path(a.listing).read_text().splitlines()[1:] if ln.strip()]
    rows, source = latest_rows(a.runs)
    st = camp.S3Store()
    manifests = {m.group(1): k for k, _ in st.list_keys(a.bucket, "s1/m/") if (m := camp.RE_MANIFEST.match(k))}
    # older prefixes (the v2 pilot, the short-lived store/ name) are scanned too: they are public while the bucket serves them
    other = sorted(k for pre in ("store/m/", "v2/m/") for k, _ in st.list_keys(a.bucket, pre) if camp.RE_MANIFEST.match(k))
    rep: dict = {"listing": len(listing), "manifests_in_r2": len(manifests), "older_prefix_manifests": len(other)}

    # ---- 1. one final status per listed study
    final, per = Counter(), []
    docs: dict = {}
    for g in listing:
        r = rows.get(g)
        st_, reason = "missing", "no results row"
        if pol.withholds(g):
            st_, reason = "withheld", "release policy"
        elif r is not None and r["status"] == "ok":
            if not (r.get("verify") or {}).get("ok"):
                st_, reason = "failed", "not verified"
            elif g not in manifests:
                st_, reason = "failed", "no manifest in R2"
            else:
                raw = st.get_bytes(a.bucket, manifests[g])
                docs[g] = raw
                d = camp.digest_bytes(raw)
                st_, reason = ("verified", "") if d.sha256 == r.get("manifest_sha256") else ("failed", "manifest in R2 differs from the verified one")
        elif r is not None and r["status"] == "skipped_done":
            st_, reason = "failed", "skipped_done without an ok row"
        elif r is not None:
            msg = f"{r['status']} at {r.get('stage')}: {r.get('error') or ''}"
            tail = study_log_tail(source[g], g)
            if CORRUPT.search(msg) or CORRUPT.search(tail):
                st_ = "corrupt"
                hit = CORRUPT.search(tail) or CORRUPT.search(msg)
                line = next((x for x in tail.splitlines() if hit and hit.group(0) in x), msg)
                reason = line.strip()[:240]
            else:
                st_, reason = "failed", msg[:240]
        final[st_] += 1
        warn = [w for w in (r or {}).get("warnings") or [] if "crc" in json.dumps(w).lower()]
        per.append({
            "gse": g, "status": st_, "reason": reason, "run": Path(source.get(g, "-")).name,
            "n_samples": (r or {}).get("n_samples") if st_ == "verified" else None,
            "n_cells_called": (r or {}).get("n_cells_called") if st_ == "verified" else None,
            "legacy_bytes": (r or {}).get("legacy_bytes"), "s1_bytes": (r or {}).get("s1_bytes") if st_ == "verified" else None,
            "crc_repaired": bool(warn), "label_notes": "; ".join((r or {}).get("label_notes") or []),
            "labels": (r or {}).get("labels") if st_ == "verified" else None,
        })
    rep["final"] = dict(final)
    ver = [p for p in per if p["status"] == "verified"]
    rep["totals_verified"] = {
        "studies": len(ver),
        "samples": sum(p["n_samples"] or 0 for p in ver),
        "cells_called": sum(p["n_cells_called"] or 0 for p in ver),
        "legacy_bytes_in": sum(p["legacy_bytes"] or 0 for p in ver),
        "s1_bytes_out": sum(p["s1_bytes"] or 0 for p in ver),
        "s1_label_bytes": sum((rows[p["gse"]].get("s1_label_bytes") or 0) for p in ver),
        "s1_sample_bytes": sum((rows[p["gse"]].get("s1_sample_bytes") or 0) for p in ver),
    }
    lab_s, lab_c = Counter(), Counter()
    for p in ver:
        lb = p["labels"] or {}
        lab_s.update(lb.get("samples") or {})
        lab_c.update(lb.get("cells") or {})
    rep["labels_from_results"] = {"samples": dict(lab_s), "cells": dict(lab_c)}
    rep["not_verified"] = [p for p in per if p["status"] not in ("verified", "withheld")]
    rep["crc_repaired_verified"] = [p["gse"] for p in ver if p["crc_repaired"]]
    rep["label_notes"] = [(p["gse"], p["label_notes"]) for p in ver if p["label_notes"]]
    rep["stray_manifests"] = sorted(set(manifests) - {p["gse"] for p in ver})
    print(json.dumps({k: rep[k] for k in ("listing", "manifests_in_r2", "final", "totals_verified")}, indent=1), flush=True)

    # ---- 2. release promises over every manifest in R2
    problems, n_members, all_samples = [], 0, []
    nuclei_gsm, meta_org, pack_ids = set(), {}, set()
    older_hits = []
    for k in other:
        g = camp.RE_MANIFEST.match(k).group(1)
        found = sorted(set(ACC.findall(st.get_bytes(a.bucket, k).decode("utf-8"))) & bad_acc)
        if found or pol.withholds(g):
            older_hits.append(f"{k}: {len(found)} withheld/excluded accessions{' (withheld study)' if pol.withholds(g) else ''}")
    rep["older_prefix_scan"] = {"manifests": len(other), "hits": older_hits}
    for g, k in sorted(manifests.items()):
        raw = docs.get(g) or st.get_bytes(a.bucket, k)
        txt = raw.decode("utf-8")
        found = sorted(set(ACC.findall(txt)) & bad_acc)
        if found or pol.withholds(g):
            problems.append(f"{g}: manifest names {len(found)} withheld/excluded accessions{' (withheld study)' if pol.withholds(g) else ''}")
        m = json.loads(txt)
        for v in (m.get("packs") or {}).values():
            for x in v if isinstance(v, list) else [v]:
                if isinstance(x, str) and re.fullmatch(r"[0-9a-f]{32}", x):
                    pack_ids.add(x)
        for name in (m.get("study_files") or {}):
            if omitted_layer(name) or is_restricted_member(name):
                problems.append(f"{g}: study file {name} is a withheld layer")
        gm = gsm_meta(m)
        title = json.dumps((m.get("study") or {}).get("title") or "")
        for s in m.get("samples") or []:
            all_samples.append((g, s["gsm"], s["key"], sorted((s.get("members") or {}).keys())))
            for name in s.get("members") or {}:
                n_members += 1
                if omitted_layer(name) or is_restricted_member(name):
                    problems.append(f"{g}/{s['gsm']}: member {name} is a withheld layer")
            meta = gm.get(s["gsm"]) or {}
            meta_org[s["gsm"]] = str(meta.get("organism") or "")
            if NUCLEI.search(json.dumps(meta)) or NUCLEI.search(title):
                nuclei_gsm.add(s["gsm"])
    rep["policy_scan"] = {"manifests": len(manifests), "samples": len(all_samples), "members": n_members, "problems": problems}

    # reference packs: every pack a manifest names is listed in s1/ref/index.json and present under s1/ref/
    idx = json.loads(st.get_bytes(a.bucket, camp.REFS_INDEX_KEY))
    listed = {p["id"]: p for p in idx.get("packs", [])}
    missing_idx, missing_obj = sorted(pack_ids - set(listed)), []
    for pid in sorted(pack_ids | set(listed)):
        h = st.head(a.bucket, f"s1/ref/{pid}.pack")
        if h is None or (pid in listed and h.size != listed[pid]["bytes"]):
            missing_obj.append(pid)
    rep["refs"] = {
        "index_format": idx.get("format"), "index_packs": len(listed), "packs_named_by_manifests": len(pack_ids),
        "named_but_not_in_index": missing_idx, "missing_or_wrong_size_objects": missing_obj,
        "kinds": dict(Counter(str(p.get("kind") or p.get("type") or "?") for p in listed.values())),
    }
    problems += [f"pack {x} named by a manifest is not in s1/ref/index.json" for x in missing_idx]
    problems += [f"pack {x} is missing from s1/ref/ (or has the wrong size)" for x in missing_obj]
    print(json.dumps(rep["refs"])[:2000], flush=True)
    print(f"policy scan: {len(manifests)} manifests, {len(all_samples)} samples, {n_members} members, {len(problems)} problems", flush=True)

    # random sample objects: the member list inside the object must match its manifest entry and hold no withheld layer
    from singlet.s1 import packs as P
    from singlet.s1.sample import S1Sample

    packs = P.default_store()
    packs.download = True
    rnd = random.Random(a.seed)
    pick = rnd.sample(all_samples, min(a.sample_objects, len(all_samples)))
    obj_problems, nbytes = [], 0
    for g, gsm, key, names in pick:
        obj = st.get_bytes(a.bucket, key)
        nbytes += len(obj)
        inside = sorted(S1Sample(obj, packs).reader.members)
        if inside != names:
            obj_problems.append(f"{gsm}: members differ from the manifest ({len(inside)} vs {len(names)})")
        for name in inside:
            if omitted_layer(name) or is_restricted_member(name):
                obj_problems.append(f"{gsm}: object holds withheld member {name}")
        if ACC.findall(" ".join(inside)) and set(ACC.findall(" ".join(inside))) & bad_acc:
            obj_problems.append(f"{gsm}: member names carry an excluded accession")
    rep["object_sample"] = {"objects": len(pick), "bytes": nbytes, "problems": obj_problems}
    print(f"object sample: {len(pick)} objects, {nbytes / 1e9:.2f} GB, {len(obj_problems)} problems", flush=True)

    # ---- 3. the cell index and its D1 exports
    if a.cidx_dir:
        cd = Path(a.cidx_dir)
        cr = json.loads((cd / "cidx-report.json").read_text())
        rep["cidx"] = {k: cr.get(k) for k in ("version", "stats", "n_errors", "n_skipped")}
        rep["cidx"]["errors"] = cr.get("errors")
        rep["cidx"]["skipped"] = cr.get("skipped")
        vdir = cd / "cidx" / "s1" / "cidx" / "v"
        dpath = vdir / f"{cr['version']}.json"
        directory = json.loads(dpath.read_text())
        dicts = directory.get("dicts") or {}
        ct = [e for e in dicts.get("cell_type", []) if e.get("id")]
        rep["cidx"]["directory"] = {
            "path": str(dpath), "keys": sorted(directory),
            "cell_type_terms": len(ct), "cell_type_terms_with_ancestors": sum(1 for e in ct if e.get("anc")),
            "ancestry_terms": [e.get("id") for e in dicts.get("ancestry", []) if e.get("id")],
            "organisms": [e.get("label") for e in dicts.get("organism", [])],
        }
        samples = {}
        with open(cd / "cidx-d1" / "cidx_sample.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                samples[r["gsm_id"]] = r
        sexed = defaultdict(lambda: [0, 0])
        cov_s, cov_c, rejected, nrows = defaultdict(set), Counter(), [], 0
        with open(cd / "cidx-d1" / "sample_cell_summary.csv", newline="", encoding="utf-8") as fh:
            for i, r in enumerate(csv.DictReader(fh), start=2):
                subj = r["subject"]
                if subj not in ITEM_RE:
                    continue
                nrows += 1
                n = int(r["n_cells"]) if r["n_cells"].isdigit() else -1
                if not re.match(r"^GSM\d+$", r["gsm_id"]) or not re.match(r"^GSE\d+$", r["gse_id"]) or not ITEM_RE[subj].match(r["item"]) or n < 0 or not r["version"]:
                    if len(rejected) < 50:
                        rejected.append(f"line {i}: {list(r.values())[:6]}")
                    continue
                cov_s[subj].add(r["gsm_id"])
                cov_c[subj] += n
                if subj == "sex":
                    sexed[r["gsm_id"]][0 if r["item"] == FEMALE else 1] += n
        rep["cidx"]["d1_check"] = {"label_rows": nrows, "rejected_by_loader_rules": rejected}
        rep["label_coverage_index"] = {
            "samples_in_index": len(samples),
            "cells_in_index": sum(int(s["n_called"]) for s in samples.values()),
            "samples": {k: len(v) for k, v in cov_s.items()},
            "cells": dict(cov_c),
        }
        orgs = Counter(s["organism"] for s in samples.values())
        rep["cidx"]["organism_counts"] = dict(orgs.most_common(12))

        # ---- sex: where a sample gets no sex fact (website rule: >= 20 sexed cells), and where cells stay ambiguous
        def org_of(gsm):
            o = (samples.get(gsm) or {}).get("organism") or ""
            return o if o and o != "unknown" else meta_org.get(gsm, "")

        sexable = re.compile(r"homo sapiens|mus musculus", re.IGNORECASE)
        cause, amb_cells, lab_cells = Counter(), 0, 0
        by_umi, by_nuc, by_mix = defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0])
        indexed = set(samples)
        for g, gsm, key, names in all_samples:
            s = samples.get(gsm)
            org = org_of(gsm)
            if s is None:
                cause["not in the index (no count matrix)" if gsm not in indexed else "?"] += 1
                continue
            n = int(s["n_called"])
            f, mm = sexed.get(gsm, (0, 0))
            if n == 0:
                cause["0 called cells"] += 1
                continue
            if not sexable.search(org):
                cause[f"no sex model for the organism ({org or 'unknown'})"] += 1
                continue
            if s["labeled"] in ("0", ""):
                cause["human/mouse but not labelled (skipped, failed or over budget)"] += 1
                continue
            amb = max(0, n - f - mm)
            amb_cells += amb
            lab_cells += n
            umi = int(s.get("umi_q50") or 0)
            ub = "<500" if umi < 500 else "500-1k" if umi < 1000 else "1k-2k" if umi < 2000 else "2k-5k" if umi < 5000 else ">=5k"
            nuc = "nuclei" if gsm in nuclei_gsm else "cells"
            mixed = f + mm > 0 and min(f, mm) >= SEX_MIN_SHARE * (f + mm)
            fact = (f + mm) >= MIN_CELLS
            for bucket, key_ in ((by_umi, ub), (by_nuc, nuc), (by_mix, "mixed-sex pool" if mixed else "single sex")):
                bucket[key_][0] += 1
                bucket[key_][1] += 0 if fact else 1
                bucket[key_][2] += amb
            if fact:
                cause["sex fact"] += 1
            elif n < MIN_CELLS:
                cause["fewer than 20 called cells"] += 1
            else:
                cause[">= 20 cells but < 20 sexed (ambiguous)"] += 1
        tot = sum(cause.values())
        rep["sex"] = {
            "samples": tot,
            "causes": dict(cause.most_common()),
            "samples_without_sex_fact_share": round(1 - cause["sex fact"] / tot, 4) if tot else None,
            "labelled_human_mouse_cells": lab_cells,
            "ambiguous_cells": amb_cells,
            "ambiguous_cell_share": round(amb_cells / lab_cells, 4) if lab_cells else None,
            "by_median_umi": {k: {"samples": v[0], "no_fact": v[1], "ambiguous_cells": v[2]} for k, v in sorted(by_umi.items())},
            "by_nuclei_heuristic": {k: {"samples": v[0], "no_fact": v[1], "ambiguous_cells": v[2]} for k, v in by_nuc.items()},
            "by_pool": {k: {"samples": v[0], "no_fact": v[1], "ambiguous_cells": v[2]} for k, v in by_mix.items()},
        }
        print(json.dumps(rep["sex"], indent=1), flush=True)

    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    with open(a.tsv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["gse", "status", "reason", "run", "n_samples", "n_cells_called", "legacy_bytes", "s1_bytes", "label_notes"])
        for p in per:
            w.writerow([p["gse"], p["status"], p["reason"], p["run"], p["n_samples"] or "", p["n_cells_called"] or "",
                        p["legacy_bytes"] or "", p["s1_bytes"] or "", p["label_notes"]])
    bad = rep["policy_scan"]["problems"] or rep["object_sample"]["problems"]
    print(f"-> {a.out}, {a.tsv}; final {dict(final)}; release problems {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
