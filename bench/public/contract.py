"""Singlet public benchmark results contract: validation, pass rules, index.

One JSON document per capability (``bench/public/results/<capability>.json``) and one index
(``bench/public/results/index.json``). The specification is ``bench/public/CONTRACT.md``; the JSON
Schemas in ``bench/public/schema/`` describe the same shape for non-Python readers (the website).
This module is the reference implementation: the harness, the notebook and CI all call it, and a
document is valid only if ``validate()`` returns no errors.

Standard library only, so it runs in the notebook, on Clipper and in CI without extra packages.

    python bench/public/contract.py validate bench/public/results/*.json [--cohort bench/public/cohort.tsv]
    python bench/public/contract.py index bench/public/results          # writes results/index.json
    python bench/public/contract.py diff published.json recomputed.json  # reproduction check
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA_CAPABILITY = "singlet.benchmark.capability"
SCHEMA_INDEX = "singlet.benchmark.index"
SCHEMA_VERSION = "1.1.0"  # semver: a major bump breaks readers, minor adds optional fields
# 1.1.0: datasets[].data_kind + metrics[].corroborated_by (real-data rule), reach[], figure kind "pareto",
#        10x comparators must name the source of their published outputs

CAPABILITIES: dict[str, str] = {
    # id -> owning session / person (who writes results/<id>.json)
    "gex-counts": "pipeline-benchmark",
    "spliced-unspliced": "pipeline-benchmark",
    "junctions-psi": "pipeline-benchmark",
    "cell-calling": "pipeline-benchmark",
    "ambient-rna": "pipeline-benchmark",
    "doublets": "pipeline-benchmark",
    "donor-demux": "pipeline-benchmark",
    "donor-snps": "pipeline-benchmark",
    "mtdna": "pipeline-benchmark",
    "non-host": "pipeline-benchmark",
    "alignment-performance": "pipeline-benchmark",
    "download": "pipeline-benchmark",
    "adt": "adt-modality",
    "hto": "adt-modality",
    "cell-type": "ensemble-classifier",
    "vdj": "vdj-crispr",
    "crispr-guides": "vdj-crispr",
    "atac": "atac-multiome-spatial",
    "multiome": "atac-multiome-spatial",
    "spatial": "atac-multiome-spatial",
    "chromosomal-sex": "bench-public",
    "genetic-ancestry": "bench-public",
}

DIRECTIONS = {"higher", "lower", "none"}
RULE_TYPES = {"match_or_beat", "threshold", "equivalence", "informational"}
CATEGORIES = {"accuracy", "agreement", "coverage", "resources", "robustness"}
TOOL_ROLES = {"singlet", "competitor", "truth", "helper"}
FIGURE_KINDS = {"bar", "grouped_bar", "scatter", "line", "box", "histogram", "heatmap", "table", "pareto"}
# real: reads or values as deposited (read caps and format conversion allowed). derived: real reads recombined or
# thinned (in-silico pools, downsampling). simulated: synthetic data. derived and simulated results must be corroborated
# by a metric on real data, and a capability must rest on at least one gating metric on real data.
DATA_KINDS = {"real", "derived", "simulated"}
REACH = {"yes", "no", "partial"}
TENX_TOOLS = re.compile(r"^(cell ?ranger|space ?ranger)", re.I)
STATUSES = {"draft", "final"}
ARTIFACT_KINDS = {"singlet_output", "competitor_output", "truth", "derived"}
MAX_CONCLUSION_WORDS = 100
MAX_METHODS_WORDS = 400
MAX_INLINE_POINTS = 5000
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CODE_URL_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/(blob|tree)/[0-9a-f]{7,40}/")


# --------------------------------------------------------------------------------------------- rules
def _tol(tolerance: dict | None, ref: float) -> float:
    """Allowed slack around a reference value: {"abs": a} and/or {"rel": r} (the larger wins)."""
    if not tolerance:
        return 0.0
    return max(float(tolerance.get("abs", 0.0)), float(tolerance.get("rel", 0.0)) * abs(ref))


def _best(values: Iterable[float], direction: str) -> float:
    vals = list(values)
    return max(vals) if direction == "higher" else min(vals)


def evaluate_metric(metric: dict) -> str:
    """Return "pass", "fail" or "na" for one metric from its values and rule alone.

    match_or_beat  Singlet must be at least as good as the reference competitor(s), less the
                   tolerance: higher-is-better s >= r - tol(r); lower-is-better s <= r + tol(r).
                   against = "best" (the best competitor value, the default) or a list of tool
                   names, each of which must be matched.
    threshold      s >= value (higher) or s <= value (lower); for truth-only metrics.
    equivalence    |s - target| <= tol(target); target is a number or a tool name.
    informational  never gates; returns "na". Not allowed when competitor values are present.
    A null Singlet value or a missing reference value gives "na" (counts as incomplete).
    """
    rule = metric.get("rule") or {}
    rtype = rule.get("type")
    s = (metric.get("singlet") or {}).get("value")
    if rtype == "informational" or s is None:
        return "na"
    direction = metric.get("direction")
    comps = {c["tool"]: c.get("value") for c in metric.get("competitors", [])}
    tol = rule.get("tolerance")
    if rtype == "match_or_beat":
        against = rule.get("against", "best")
        if against == "best":
            vals = [v for v in comps.values() if v is not None]
            if not vals:
                return "na"
            refs = [_best(vals, direction)]
        else:
            refs = [comps.get(t) for t in against]
            if any(r is None for r in refs):
                return "na"
        ok = all((s >= r - _tol(tol, r)) if direction == "higher" else (s <= r + _tol(tol, r)) for r in refs)
        return "pass" if ok else "fail"
    if rtype == "threshold":
        v = float(rule["value"])
        return "pass" if ((s >= v) if direction == "higher" else (s <= v)) else "fail"
    if rtype == "equivalence":
        target = rule["target"]
        t = comps.get(target) if isinstance(target, str) else target
        if t is None:
            return "na"
        return "pass" if abs(s - t) <= _tol(tol, t) + 1e-12 else "fail"
    raise ValueError(f"unknown rule type {rtype!r}")


def capability_status(doc: dict) -> dict:
    """Summarise a capability: pass only when every gating metric passes and the doc is final."""
    gating = [m for m in doc.get("metrics", []) if (m.get("rule") or {}).get("type") != "informational"]
    res = [evaluate_metric(m) for m in gating]
    n_pass, n_fail, n_na = res.count("pass"), res.count("fail"), res.count("na")
    if n_fail:
        status = "fail"
    elif n_na or doc.get("status") != "final" or not gating:
        status = "incomplete"
    else:
        status = "pass"
    return {"status": status, "n_metrics": len(gating), "n_pass": n_pass, "n_fail": n_fail, "n_na": n_na}


# ---------------------------------------------------------------------------------------- validation
def _words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_cohort(path: str | Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t")]
    return {r["id"]: r for r in rows}


def validate(doc: dict, cohort: dict[str, dict] | None = None) -> list[str]:
    """Return a list of error strings; empty means the document satisfies the contract."""
    E: list[str] = []
    err = E.append

    def need(obj: dict, key: str, where: str, typ: type | tuple = str) -> Any:
        if key not in obj or obj[key] is None or obj[key] == "":
            err(f"{where}: missing '{key}'")
            return None
        if not isinstance(obj[key], typ):
            err(f"{where}.{key}: expected {typ}, got {type(obj[key]).__name__}")
            return None
        return obj[key]

    if doc.get("schema") != SCHEMA_CAPABILITY:
        err(f"schema must be '{SCHEMA_CAPABILITY}'")
    sv = str(doc.get("schema_version", ""))
    if sv.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        err(f"schema_version {sv!r} is not compatible with {SCHEMA_VERSION}")
    cap = need(doc, "capability", "doc")
    if cap and cap not in CAPABILITIES:
        err(f"capability {cap!r} is not registered in contract.CAPABILITIES (add it there first)")
    need(doc, "title", "doc")
    concl = need(doc, "conclusion", "doc")
    if concl and _words(concl) > MAX_CONCLUSION_WORDS:
        err(f"conclusion has {_words(concl)} words (max {MAX_CONCLUSION_WORDS})")
    meth = need(doc, "methods", "doc")
    if meth and _words(meth) > MAX_METHODS_WORDS:
        err(f"methods has {_words(meth)} words (max {MAX_METHODS_WORDS})")
    if doc.get("status") not in STATUSES:
        err(f"status must be one of {sorted(STATUSES)}")
    ga = need(doc, "generated_at", "doc")
    if ga:
        try:
            _dt.datetime.fromisoformat(ga.replace("Z", "+00:00"))
        except ValueError:
            err("generated_at is not ISO 8601")
    need(doc, "owner", "doc")

    sg = need(doc, "singlet", "doc", dict) or {}
    if sg:
        need(sg, "version", "singlet")
        c = need(sg, "commit", "singlet")
        if c and not SHA_RE.match(c):
            err("singlet.commit must be a full 40-hex git SHA")
        if sg.get("binary_sha256") and not SHA256_RE.match(sg["binary_sha256"]):
            err("singlet.binary_sha256 must be 64 hex chars")

    tools = need(doc, "tools", "doc", list) or []
    tool_names: dict[str, str] = {}
    for i, t in enumerate(tools):
        w = f"tools[{i}]"
        n = need(t, "name", w)
        need(t, "version", w)
        if t.get("role") not in TOOL_ROLES:
            err(f"{w}.role must be one of {sorted(TOOL_ROLES)}")
        if n:
            if n in tool_names:
                err(f"{w}: duplicate tool name {n!r}")
            tool_names[n] = t.get("role", "")
    if "singlet" not in tool_names.values():
        err("tools: one tool with role 'singlet' is required")

    datasets = need(doc, "datasets", "doc", list) or []
    ds_kind: dict[str, str] = {}
    ds_ids = set()
    for i, d in enumerate(datasets):
        w = f"datasets[{i}]"
        did = need(d, "id", w)
        if did:
            ds_ids.add(did)
            if cohort is not None and did not in cohort:
                err(f"{w}: dataset {did!r} is not in the cohort table")
        acc = need(d, "accessions", w, dict) or {}
        if acc and not any(acc.get(k) for k in ("gse", "gsm", "srr", "url", "other")):
            err(f"{w}.accessions: give at least one of gse/gsm/srr/url/other")
        need(d, "truth", w)
        kind = d.get("data_kind", "real")
        if kind not in DATA_KINDS:
            err(f"{w}.data_kind must be one of {sorted(DATA_KINDS)}")
        if did:
            ds_kind[did] = kind
            if cohort is not None and did in cohort and (cohort[did].get("data_kind") or "real") != kind:
                err(f"{w}: data_kind {kind!r} but the cohort says {cohort[did].get('data_kind')!r}")

    metrics = need(doc, "metrics", "doc", list) or []
    seen = set()
    for i, m in enumerate(metrics):
        w = f"metrics[{i}]"
        mid = need(m, "id", w)
        if mid:
            if not ID_RE.match(mid):
                err(f"{w}.id {mid!r}: lowercase letters, digits, '_', '.', '-'")
            key = (mid, m.get("dataset"))
            if key in seen:
                err(f"{w}: duplicate (id, dataset) {key}")
            seen.add(key)
        need(m, "name", w)
        need(m, "unit", w)
        if m.get("category") not in CATEGORIES:
            err(f"{w}.category must be one of {sorted(CATEGORIES)}")
        dset = need(m, "dataset", w)
        if dset and dset != "pooled" and dset not in ds_ids:
            err(f"{w}.dataset {dset!r} is neither 'pooled' nor listed in datasets")
        direction = m.get("direction")
        if direction not in DIRECTIONS:
            err(f"{w}.direction must be one of {sorted(DIRECTIONS)}")
        sv_ = m.get("singlet")
        if not isinstance(sv_, dict) or "value" not in sv_:
            err(f"{w}.singlet: object with 'value' required (null allowed while incomplete)")
        elif sv_["value"] is not None and not _is_num(sv_["value"]):
            err(f"{w}.singlet.value must be a finite number or null")
        comps = m.get("competitors", [])
        if not isinstance(comps, list):
            err(f"{w}.competitors must be a list")
            comps = []
        for j, c in enumerate(comps):
            if c.get("tool") not in tool_names:
                err(f"{w}.competitors[{j}].tool {c.get('tool')!r} not declared in tools")
            elif tool_names[c["tool"]] not in ("competitor", "truth"):
                err(f"{w}.competitors[{j}]: tool {c['tool']!r} has role {tool_names[c['tool']]!r}")
            if c.get("value") is not None and not _is_num(c.get("value")):
                err(f"{w}.competitors[{j}].value must be a finite number or null")
        rule = m.get("rule")
        if not isinstance(rule, dict) or rule.get("type") not in RULE_TYPES:
            err(f"{w}.rule.type must be one of {sorted(RULE_TYPES)}")
            continue
        rt = rule["type"]
        if rt == "informational":
            if comps:
                err(f"{w}: an informational metric cannot carry competitor values; give it a rule")
        else:
            if direction == "none" and rt != "equivalence":
                err(f"{w}: direction 'none' is only valid with an equivalence rule")
            if rt == "match_or_beat":
                if not comps:
                    err(f"{w}: match_or_beat needs at least one competitor value")
                ag = rule.get("against", "best")
                if ag != "best" and not (isinstance(ag, list) and all(a in tool_names for a in ag)):
                    err(f"{w}.rule.against must be 'best' or a list of declared tool names")
            if rt == "threshold" and not _is_num(rule.get("value")):
                err(f"{w}.rule.value (number) required for threshold")
            if rt == "equivalence":
                tg = rule.get("target")
                if not (_is_num(tg) or (isinstance(tg, str) and tg in tool_names)):
                    err(f"{w}.rule.target must be a number or a declared tool name")
            tol = rule.get("tolerance") or {}
            if tol:
                if set(tol) - {"abs", "rel"} or not all(_is_num(v) and v >= 0 for v in tol.values()):
                    err(f"{w}.rule.tolerance: only non-negative 'abs'/'rel' numbers")
                if any(v > 0 for v in tol.values()) and not rule.get("rationale"):
                    err(f"{w}.rule: a non-zero tolerance needs a 'rationale' (why that margin counts as a match)")
            if rt == "threshold" and not rule.get("rationale"):
                err(f"{w}.rule: a threshold needs a 'rationale' (where the number comes from)")
        if "pass" in m:
            try:
                got = evaluate_metric(m)
            except (KeyError, TypeError, ValueError) as e:
                err(f"{w}: rule cannot be evaluated ({e})")
            else:
                if m["pass"] != got:
                    err(f"{w}.pass is {m['pass']!r} but the rule gives {got!r}")

    # ---- real-data rule: derived/simulated metrics need corroboration on real data; one gating metric must be real
    def kind_of(dset):
        if dset == "pooled":
            vals = set(ds_kind.values()) or {"real"}
            return "real" if "real" in vals else ("derived" if "derived" in vals else "simulated")
        return ds_kind.get(dset, "real")

    keys = {(m.get("id"), m.get("dataset")) for m in metrics}
    real_gating = False
    for i, m in enumerate(metrics):
        k = kind_of(m.get("dataset"))
        gating = (m.get("rule") or {}).get("type") not in (None, "informational")
        if k == "real" and gating:
            real_gating = True
        if k != "real" and gating:
            refs = m.get("corroborated_by") or []
            if not refs:
                err(f"metrics[{i}] ({m.get('id')}@{m.get('dataset')}): {k} data needs corroborated_by "
                    "(metrics on real data that show the same effect)")
            for ref in refs:
                if "/" in str(ref):
                    continue  # another capability's metric: checked by build_index
                mid, _, dset = str(ref).partition("@")
                if (mid, dset) not in keys:
                    err(f"metrics[{i}].corroborated_by {ref!r}: no such metric (use 'id@dataset' or 'capability/id@dataset')")
                elif kind_of(dset) != "real":
                    err(f"metrics[{i}].corroborated_by {ref!r}: the corroborating metric is not on real data")
    if metrics and not real_gating:
        err("no gating metric rests on real data; derived or simulated results can only supplement real ones")

    for i, rch in enumerate(doc.get("reach", [])):
        w = f"reach[{i}]"
        if rch.get("tool") not in tool_names:
            err(f"{w}.tool {rch.get('tool')!r} not declared in tools")
        need(rch, "output", w)
        if rch.get("supported") not in REACH:
            err(f"{w}.supported must be one of {sorted(REACH)}")

    for i, t in enumerate(tools):
        if t.get("role") == "competitor" and TENX_TOOLS.match(str(t.get("name", ""))) and not t.get("url"):
            err(f"tools[{i}] ({t.get('name')}): a 10x comparator must give the source of its published outputs in url")

    res = need(doc, "resources", "doc", list) or []
    for i, r in enumerate(res):
        w = f"resources[{i}]"
        if r.get("tool") not in tool_names:
            err(f"{w}.tool {r.get('tool')!r} not declared in tools")
        d = r.get("dataset")
        if d and d != "pooled" and d not in ds_ids:
            err(f"{w}.dataset {d!r} not listed in datasets")
        for k in ("wall_s", "max_rss_gb", "threads"):
            if not _is_num(r.get(k)):
                err(f"{w}.{k} (number) required")
        if r.get("cpu_h") is not None and not _is_num(r.get("cpu_h")):
            err(f"{w}.cpu_h must be a number")

    for i, f in enumerate(doc.get("figures", [])):
        w = f"figures[{i}]"
        need(f, "id", w)
        need(f, "title", w)
        kind = f.get("kind")
        if kind not in FIGURE_KINDS:
            err(f"{w}.kind must be one of {sorted(FIGURE_KINDS)}")
        if kind == "heatmap":
            for k in ("rows", "cols", "z"):
                if not isinstance(f.get(k), list):
                    err(f"{w}.{k} (list) required for heatmap")
        elif kind == "table":
            if not isinstance(f.get("columns"), list) or not isinstance(f.get("rows"), list):
                err(f"{w}: table needs 'columns' and 'rows' lists")
        elif kind in FIGURE_KINDS:
            if kind == "pareto":
                for ax in ("x", "y"):
                    if not (f.get(ax) or {}).get("label"):
                        err(f"{w}.{ax}.label required for a pareto figure (cost on x, quality on y)")
            series = f.get("series")
            if not isinstance(series, list) or (not series and not f.get("data_file")):
                err(f"{w}.series (non-empty list) or data_file required")
                series = []
            for j, s in enumerate(series):
                need(s, "name", f"{w}.series[{j}]")
                ys = s.get("y")
                if not isinstance(ys, list):
                    err(f"{w}.series[{j}].y (list) required")
                    continue
                if isinstance(s.get("x"), list) and len(s["x"]) != len(ys):
                    err(f"{w}.series[{j}]: x and y lengths differ")
                if len(ys) > MAX_INLINE_POINTS:
                    err(f"{w}.series[{j}]: {len(ys)} points inline (max {MAX_INLINE_POINTS}); use data_file")

    for i, a in enumerate(doc.get("artifacts", [])):
        w = f"artifacts[{i}]"
        need(a, "name", w)
        need(a, "url", w)
        if a.get("kind") not in ARTIFACT_KINDS:
            err(f"{w}.kind must be one of {sorted(ARTIFACT_KINDS)}")
        if a.get("sha256") and not SHA256_RE.match(a["sha256"]):
            err(f"{w}.sha256 must be 64 hex chars")
        if doc.get("status") == "final" and not a.get("sha256"):
            err(f"{w}.sha256 required in a final document")

    code = need(doc, "code", "doc", list) or []
    for i, c in enumerate(code):
        w = f"code[{i}]"
        need(c, "path", w)
        u = need(c, "url", w)
        if u and not CODE_URL_RE.match(u):
            err(f"{w}.url must be a GitHub blob/tree URL pinned to a commit")

    rep = need(doc, "reproduce", "doc", dict) or {}
    if rep:
        need(rep, "module", "reproduce")
        need(rep, "function", "reproduce")
    if doc.get("status") == "final":
        if not metrics:
            err("a final document needs metrics")
        if not res:
            err("a final document needs resources (runtime and memory)")
        if any((m.get("singlet") or {}).get("value") is None for m in metrics):
            err("a final document cannot have null Singlet values")
    if not isinstance(doc.get("caveats", []), list):
        err("caveats must be a list of strings")
    return E


def pareto_figure(doc: dict, metric_id: str, dataset: str, resource_field: str = "cpu_h", fig_id: str | None = None) -> dict:
    """Quality against cost: one point per tool, x = resources[resource_field] on `dataset` (or 'pooled'), y = the metric.

    Tool names must match between metrics[].competitors[].tool, resources[].tool and 'singlet'. A point is on the frontier
    (series[].frontier) when no other tool is at most as costly and at least as good, and strictly better in one of them."""
    m = next(x for x in doc["metrics"] if x["id"] == metric_id and x.get("dataset") == dataset)
    hi = m["direction"] != "lower"
    q = {"singlet": m["singlet"]["value"], **{c["tool"]: c.get("value") for c in m.get("competitors", [])}}
    cost = {r["tool"]: r.get(resource_field) for r in doc.get("resources", []) if r.get("dataset") in (dataset, "pooled")}
    pts = [(t, cost[t], v) for t, v in q.items() if v is not None and cost.get(t) is not None]

    def better_or_equal(a, b):
        return (a >= b) if hi else (a <= b)

    def dominated(p):
        return any(o[0] != p[0] and o[1] <= p[1] and better_or_equal(o[2], p[2]) and (o[1] < p[1] or o[2] != p[2])
                   for o in pts)

    unit = {"cpu_h": "CPU-hours", "wall_s": "seconds", "max_rss_gb": "GB"}.get(resource_field, resource_field)
    return {"id": fig_id or f"pareto_{metric_id}_{dataset}", "kind": "pareto",
            "title": f"{m['name']} against {unit} ({dataset})",
            "x": {"label": resource_field, "unit": unit, "scale": "log"}, "y": {"label": m["name"], "unit": m["unit"]},
            "series": [{"name": t, "tool": t, "x": [x], "y": [y], "frontier": not dominated((t, x, y))} for t, x, y in pts]}


# ---------------------------------------------------------------------------------------------- index
def finalize(doc: dict) -> dict:
    """Fill each metric's 'pass' from its rule (the only way 'pass' should be written)."""
    for m in doc.get("metrics", []):
        m["pass"] = evaluate_metric(m)
    return doc


def build_index(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    caps, singlet = [], {}
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "index.json":
            continue
        raw = p.read_bytes()
        doc = json.loads(raw)
        st = capability_status(doc)
        errs = validate(doc)
        caps.append({
            "capability": doc.get("capability"),
            "title": doc.get("title"),
            "file": p.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "owner": doc.get("owner"),
            "doc_status": doc.get("status"),
            "valid": not errs,
            "generated_at": doc.get("generated_at"),
            "conclusion": doc.get("conclusion"),
            "n_caveats": len(doc.get("caveats", [])),
            **st,
            "status": st["status"] if not errs else "invalid",
        })
        if doc.get("singlet"):
            singlet.setdefault(doc["singlet"].get("commit"), doc["singlet"].get("version"))
    present = {c["capability"] for c in caps}
    docs_by_cap = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.name != "index.json":
            d = json.loads(p.read_text(encoding="utf-8"))
            docs_by_cap[d.get("capability")] = d
    for c in caps:
        d = docs_by_cap.get(c["capability"]) or {}
        missing = []
        for m in d.get("metrics", []):
            for ref in m.get("corroborated_by") or []:
                if "/" in str(ref):
                    cap, _, rest = str(ref).partition("/")
                    mid, _, dset = rest.partition("@")
                    other = docs_by_cap.get(cap) or {}
                    if not any(x.get("id") == mid and x.get("dataset") == dset for x in other.get("metrics", [])):
                        missing.append(ref)
        if missing:
            c["unresolved_corroboration"] = missing
            if c["status"] == "pass":
                c["status"] = "incomplete"
    for cid in CAPABILITIES:
        if cid not in present:
            caps.append({"capability": cid, "title": None, "file": None, "owner": CAPABILITIES[cid],
                         "status": "missing", "n_metrics": 0, "n_pass": 0, "n_fail": 0, "n_na": 0})
    order = list(CAPABILITIES)
    caps.sort(key=lambda c: order.index(c["capability"]) if c["capability"] in order else len(order))
    statuses = {c["status"] for c in caps}
    overall = "pass" if statuses == {"pass"} else ("fail" if statuses & {"fail", "invalid"} else "incomplete")
    return {
        "schema": SCHEMA_INDEX,
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "singlet": [{"commit": k, "version": v} for k, v in singlet.items()],
        "overall": overall,
        "n_capabilities": len(caps),
        "n_pass": sum(c["status"] == "pass" for c in caps),
        "capabilities": caps,
    }


# ------------------------------------------------------------------------------------------ diffing
def diff(published: dict, recomputed: dict, rel: float = 1e-6, abs_: float = 1e-9) -> list[str]:
    """Metric values that differ between a published document and a re-computation."""
    def key(m):
        return (m["id"], m.get("dataset"))

    out = []
    pub = {key(m): m for m in published.get("metrics", [])}
    rec = {key(m): m for m in recomputed.get("metrics", [])}
    for k in sorted(set(pub) | set(rec), key=str):
        if k not in rec or k not in pub:
            out.append(f"{k}: only in {'published' if k in pub else 'recomputed'}")
            continue
        a, b = pub[k], rec[k]
        pairs = [("singlet", a["singlet"].get("value"), b["singlet"].get("value"))]
        ca = {c["tool"]: c.get("value") for c in a.get("competitors", [])}
        cb = {c["tool"]: c.get("value") for c in b.get("competitors", [])}
        pairs += [(t, ca.get(t), cb.get(t)) for t in sorted(set(ca) | set(cb))]
        for who, x, y in pairs:
            if x is None and y is None:
                continue
            if x is None or y is None or not math.isclose(x, y, rel_tol=rel, abs_tol=abs_):
                out.append(f"{k} {who}: published {x} vs recomputed {y}")
    return out


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("files", nargs="+")
    v.add_argument("--cohort")
    ix = sub.add_parser("index")
    ix.add_argument("results_dir")
    ix.add_argument("--check", action="store_true", help="fail unless every capability passes")
    d = sub.add_parser("diff")
    d.add_argument("published")
    d.add_argument("recomputed")
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        cohort = load_cohort(a.cohort) if a.cohort else None
        bad = 0
        for f in a.files:
            if Path(f).name == "index.json":
                continue
            doc = json.loads(Path(f).read_text(encoding="utf-8"))
            errs = validate(doc, cohort)
            st = capability_status(doc)
            print(f"{f}: {'OK' if not errs else 'INVALID'}  {st['status']}  "
                  f"({st['n_pass']}/{st['n_metrics']} pass, {st['n_fail']} fail, {st['n_na']} n/a)")
            for e in errs:
                print(f"  - {e}")
            bad += bool(errs)
        return 1 if bad else 0
    if a.cmd == "index":
        idx = build_index(a.results_dir)
        out = Path(a.results_dir) / "index.json"
        out.write_text(json.dumps(idx, indent=2) + "\n", encoding="utf-8")
        print(f"{out}: overall {idx['overall']}, {idx['n_pass']}/{idx['n_capabilities']} capabilities pass")
        return 0 if (not a.check or idx["overall"] == "pass") else 1
    if a.cmd == "diff":
        out = diff(json.loads(Path(a.published).read_text()), json.loads(Path(a.recomputed).read_text()))
        print("\n".join(out) if out else "identical metric values")
        return 1 if out else 0
    return 2


if __name__ == "__main__":
    sys.exit(_main())
