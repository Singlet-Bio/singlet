#!/usr/bin/env python3
"""The demand vector of an arm and the node rate it predicts. Tier 1 measures the demands of one run alone with low noise.

Demand vector of one run (alone):
  hold_prefill, hold_handoff, hold_solo, hold_pileup, hold_nonhost   device hold seconds by phase (engine log lines)
  hold_s          seconds a run holds a device lock, from the sampler (the lock held number of the tables)
  core_s_run      core-seconds of one run (/usr/bin/time)
  core_s_per_M    core-seconds per million pairs (cgroup)
  pairs_M         million pairs per run
  dev_peak_gb, host_peak_gb, wall_s

Limits on the node rate in M pairs per second (sustained, ranking use only, never an absolute prediction):
  core limit    = allocated cores / core-s per M
  lock limit    = slots * pairs per run / hold s
  memory limit  = runs that fit in (device - 2 GB) * pairs per run / hold s. Runs that fit are floor((device - 2) / device peak).
                  It is the lock limit with the slots capped by memory, so it binds only when fewer runs fit than there are slots.
The predicted node rate is the smallest of the three. The binding limit is the one that is smallest.

Provisional rule (demand rule, tier 1): PROMOTE when the predicted node rate rises by RULE["node_gain"] or more,
identity kept and no fallback. Validity rules V1 to V6 of abcore come first.
"""
import math
import sys

import abcore

PHASES = ("prefill", "handoff", "solo", "pileup", "nonhost")
FIELDS = ("hold_prefill", "hold_handoff", "hold_solo", "hold_pileup", "hold_nonhost", "hold_s", "core_s_run", "core_s_per_M", "pairs_M", "dev_peak_gb",
          "host_peak_gb", "wall_s")
# The provisional demand rule. One place. ab.py can override each value.
RULE = {"node_gain": 0.05, "respect_spread": True, "cores": 28.0, "device_gb": 93.6}
DEFAULT_PAIRS_M = 66.6


def leg_demand(P, metrics):
    """Demand vector of one leg from its point.json (dict) and its metrics. Means over the runs that exited 0."""
    ok = [r for r in P.get("runs", []) if r.get("rc") == 0]
    if not ok:
        return {}
    m = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    d = {"pairs_M": m([(r.get("pairs") or 0) / 1e6 for r in ok]), "wall_s": m([r["wall"] for r in ok if r.get("wall") is not None]),
         "core_s_run": m([r["core_s"] for r in ok if r.get("core_s") is not None]),
         "hold_s": m([r["sampler_hold_s"] for r in ok if r.get("sampler_hold_s") is not None])}
    for ph in PHASES:
        d["hold_" + ph] = m([((r.get("locks") or {}).get(ph) or {}).get("held") or 0.0 for r in ok])
    d["core_s_per_M"] = abcore.num((metrics or {}).get("core_s_per_M")) or (d["core_s_run"] / d["pairs_M"] if d["pairs_M"] and d["core_s_run"] else None)
    d["dev_peak_gb"] = abcore.num((metrics or {}).get("dev_peak_gb"))
    d["host_peak_gb"] = abcore.num((metrics or {}).get("host_peak_gb"))
    return d


def arm_demand(res, arm):
    """Mean demand vector over the legs of an arm. Peaks take the maximum. Legs without a demand vector fall back to their metrics."""
    legs = abcore.arm_legs(res, arm)
    rows = []
    for l in legs:
        d = dict(l.get("demand") or {})
        mt = l.get("metrics", {})
        d.setdefault("hold_s", abcore.num(mt.get("lock_held_s")))
        d.setdefault("core_s_per_M", abcore.num(mt.get("core_s_per_M")))
        d.setdefault("dev_peak_gb", abcore.num(mt.get("dev_peak_gb")))
        d.setdefault("host_peak_gb", abcore.num(mt.get("host_peak_gb")))
        d.setdefault("wall_s", abcore.num(mt.get("wall_s")))
        rows.append(d)
    out = {}
    for f in FIELDS:
        v = [abcore.num(r.get(f)) for r in rows if abcore.num(r.get(f)) is not None]
        out[f] = (max(v) if f.endswith("peak_gb") else sum(v) / len(v)) if v else None
    out["pairs_M"] = out["pairs_M"] or DEFAULT_PAIRS_M
    slots = int(res.get("params", {}).get("slots") or 1)
    if res.get("tier") == 2 and out["dev_peak_gb"] and slots > 1:
        out["dev_peak_gb"] /= slots   # in a loaded leg the device peak is the sum over the holders. Per run it is about the peak over the slots
    out["spread"] = {"core_s_per_M": abcore.spread([abcore.num(r.get("core_s_per_M")) for r in rows]),
                     "hold_s": abcore.spread([abcore.num(r.get("hold_s")) for r in rows])}
    return out


def limits(d, cores, slots, device_gb):
    """Limits on the node rate for one demand vector. Returns core, lock, memory limits, runs that fit, node rate and the binding limit."""
    pairs, hold, cs = d.get("pairs_M") or DEFAULT_PAIRS_M, d.get("hold_s"), d.get("core_s_per_M")
    core = cores / cs if cs else None
    lock = slots * pairs / hold if hold else None
    peak = d.get("dev_peak_gb")
    fit = int(math.floor((device_gb - 2.0) / peak)) if peak else None
    mem = max(fit, 0) * pairs / hold if fit is not None and hold else None
    cand = [(v, n) for v, n in ((core, "cores"), (lock, "lock"), (mem, "memory")) if v is not None]
    node, bind = (min(cand)[0], min(cand)[1]) if cand else (None, None)
    # Device memory left when all slots hold at once. A small margin predicts the device out of memory fallback seen with E20 (2.2 GB at 2 slots).
    headroom = (device_gb - 2.0) - slots * peak if peak else None
    return {"core": core, "lock": lock, "memory": mem, "fit": fit, "node": node, "binding": bind, "headroom_gb": headroom}


def predicted_change(res, rule=None, slots=None):
    rule = dict(RULE, **(rule or {}))
    slots = slots or int(res.get("params", {}).get("slots") or 2)
    dA, dB = arm_demand(res, "A"), arm_demand(res, "B")
    lA, lB = limits(dA, rule["cores"], slots, rule["device_gb"]), limits(dB, rule["cores"], slots, rule["device_gb"])
    ch = (lB["node"] / lA["node"] - 1.0) if lA["node"] and lB["node"] else None
    per = {k: ((lB[k] / lA[k] - 1.0) if lA[k] and lB[k] else None) for k in ("core", "lock", "memory")}
    return {"A": dA, "B": dB, "limits_A": lA, "limits_B": lB, "change": ch, "change_by_limit": per, "slots": slots, "cores": rule["cores"], "device_gb": rule["device_gb"]}


def stacked_change(results, rule=None):
    """Predicted node rate change of an arm that stacks several tier 1 changes. results: tier 1 results, each with arm A (baseline) and B (one change).
    The demand ratios B over A multiply on the demand of the first baseline. A single result gives its own predicted change."""
    rule = dict(RULE, **(rule or {}))
    slots = int(results[0].get("params", {}).get("slots") or 2)
    base = arm_demand(results[0], "A")
    d = dict(base)
    for res in results:
        a, b = arm_demand(res, "A"), arm_demand(res, "B")
        for k in ("core_s_per_M", "hold_s", "dev_peak_gb"):
            if a.get(k) and b.get(k) and d.get(k):
                d[k] = d[k] * b[k] / a[k]
    l0, l1 = limits(base, rule["cores"], slots, rule["device_gb"]), limits(d, rule["cores"], slots, rule["device_gb"])
    return (l1["node"] / l0["node"] - 1.0) if l0["node"] and l1["node"] else None


def tier1_demand_verdict(res, p=None, intended_change=False):
    """Rules after the validity rules V1 to V6:
      T1D-1 the predicted node rate could not be formed                      -> INCONCLUSIVE
      T1D-2 the predicted rise is at or below the run-to-run spread of the binding input -> INCONCLUSIVE
      T1D-3 the predicted node rate rises by node_gain or more                -> PROMOTE
      T1D-4 anything else                                                     -> INCONCLUSIVE"""
    rule = dict(RULE, **(p or {}))
    A, B = abcore.arm_summary(res, "A"), abcore.arm_summary(res, "B")
    out = {"tier": 1, "rule_kind": "demand", "params": {**rule, "intended_change": intended_change}, "A": A, "B": B}
    pc = predicted_change(res, rule)
    out["demand"] = pc
    out["gain"] = {"node_rate": pc["change"], "core_s_per_M": abcore.gain_lower_better(A["core_s_per_M"], B["core_s_per_M"]),
                   "lock_held_s": abcore.gain_lower_better(A["lock_held_s"], B["lock_held_s"])}
    out["spread"] = {"core_s_per_M": max([x for x in (A["core_s_per_M_spread"], B["core_s_per_M_spread"]) if x is not None] or [0.0]),
                     "lock_held_s": max([x for x in (A["lock_held_s_spread"], B["lock_held_s_spread"]) if x is not None] or [0.0])}
    v = abcore.validity(res, A, B, intended_change)
    if v:
        out["verdict"], out["rule"], out["why"] = v
        return out
    ch = pc["change"]
    bind = pc["limits_A"]["binding"]
    key = {"cores": "core_s_per_M", "lock": "lock_held_s", "memory": "lock_held_s"}.get(bind, "core_s_per_M")
    desc = f"predicted node rate {abcore.pct(ch)} (A binds on {bind}, B on {pc['limits_B']['binding']})"
    if ch is None:
        out.update(verdict="INCONCLUSIVE", rule="T1D-1", why="the predicted node rate could not be formed")
    elif ch >= rule["node_gain"] and rule["respect_spread"] and ch <= out["spread"][key]:
        out.update(verdict="INCONCLUSIVE", rule="T1D-2", why=f"{desc} is inside the run-to-run spread {100 * out['spread'][key]:.1f}% of {key}")
    elif ch >= rule["node_gain"]:
        out.update(verdict="PROMOTE", rule="T1D-3", why=f"{desc}, at least {100 * rule['node_gain']:.0f}%. Identity kept, no fallback")
    else:
        out.update(verdict="INCONCLUSIVE", rule="T1D-4", why=f"{desc}, below {100 * rule['node_gain']:.0f}%")
    return out


def load_result(out_dir):
    """result.json of a fast lane out directory. Demand vectors are rebuilt from the saved raw legs (legs/<leg>/point.json) when they are there."""
    import json
    import os
    res = json.load(open(os.path.join(out_dir, "result.json")))
    for l in res.get("legs", []):
        pj = os.path.join(out_dir, "legs", l.get("dir", ""), "point.json")
        if os.path.exists(pj) and os.path.getsize(pj) > 10:
            l["demand"] = leg_demand(json.load(open(pj)), l.get("metrics"))
    return res


def render_demand(res, verdict):
    """Markdown sections for the demand vector and the limits. Used by abcore.render for tier 1."""
    pc = verdict.get("demand") or predicted_change(res)
    f = lambda x, d=1: "NA" if x is None else f"{x:.{d}f}"  # noqa: E731
    A, B = pc["A"], pc["B"]
    L = ["", "Demand per run, alone", "", "| demand | A | B | B vs A |", "|---|---|---|---|"]
    rows = [("hold prefill s", "hold_prefill", 1, 1), ("hold handoff s", "hold_handoff", 1, 1), ("hold solo s", "hold_solo", 2, 1),
            ("hold pileup s", "hold_pileup", 1, 1), ("hold nonhost s", "hold_nonhost", 1, 1), ("hold total s (sampler)", "hold_s", 1, 1),
            ("core-s per run", "core_s_run", 0, 1), ("core-s per M", "core_s_per_M", 2, 1), ("device peak GB", "dev_peak_gb", 1, 1),
            ("host peak GB", "host_peak_gb", 1, 1), ("wall s", "wall_s", 1, 1)]
    for lab, k, dd, _ in rows:
        a, b = A.get(k), B.get(k)
        L.append(f"| {lab} | {f(a, dd)} | {f(b, dd)} | {abcore.pct(abcore.gain_lower_better(a, b))} |")
    L += ["", f"Node rate limits in M pairs/s ({pc['cores']:g} cores, {pc['slots']} slots, {pc['device_gb']:g} GB device). Ranking screen only.", "",
          "| limit | A | B | B vs A |", "|---|---|---|---|"]
    for k, lab in (("core", "core limit"), ("lock", "lock limit"), ("memory", f"memory limit (fit A {pc['limits_A']['fit']}, B {pc['limits_B']['fit']})")):
        L.append(f"| {lab} | {f(pc['limits_A'][k], 3)} | {f(pc['limits_B'][k], 3)} | {abcore.pct(pc['change_by_limit'][k])} |")
    L.append(f"| device headroom with all slots held, GB | {f(pc['limits_A']['headroom_gb'], 1)} | {f(pc['limits_B']['headroom_gb'], 1)} | |")
    L.append(f"| **node rate (smallest)** | {f(pc['limits_A']['node'], 3)} ({pc['limits_A']['binding']}) | {f(pc['limits_B']['node'], 3)} ({pc['limits_B']['binding']}) | **{abcore.pct(pc['change'])}** |")
    return "\n".join(L)


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
